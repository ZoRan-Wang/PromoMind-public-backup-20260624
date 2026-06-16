"""Train an XGBoost learning-to-rank coupon-response model.

The model uses household-campaign groups and binary coupon-response labels:
an item is relevant if it is bought within five days after campaign start.
XGBoost is optional and can use CUDA when available.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in [str(SCRIPT_ROOT), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_coupon_response_neural_ranker as neural  # noqa: E402
import run_coupon_response_ranker as base  # noqa: E402
from promomind.data import schema  # noqa: E402

MODEL_NAME = "coupon_response_xgboost_ranker"
HEURISTIC_BLEND_WEIGHTS = {
    "repeat_signal": 0.45,
    "cadence_signal": 0.25,
    "category_signal": 0.20,
    "global_signal": 0.10,
}
XGB_DERIVED_FEATURES = [
    "repeat_cadence_signal",
    "base_repeat_signal",
    "base_cadence_signal",
    "global_repeat_signal",
    "category_repeat_signal",
    "interval_ratio_log",
    "interval_abs_log_error",
    "days_since_last_missing",
    "median_interval_missing",
    "base_signal_pct_rank",
    "repeat_signal_pct_rank",
    "cadence_signal_pct_rank",
    "global_signal_pct_rank",
    "discount_signal_pct_rank",
]
XGB_EXTRA_FEATURES = [
    "product_response_prior",
    "category_response_prior",
    "campaign_type_response_prior",
    "product_positive_count_log",
    "category_positive_count_log",
    "campaign_type_positive_count_log",
]
XGB_CONTENT_FEATURES = [
    "department_affinity",
    "brand_affinity",
    "category_affinity_exact",
    "product_type_affinity",
    "content_match_signal",
]
XGB_VALUE_FEATURES = [
    "product_spend_signal",
    "product_avg_sales_log",
    "product_quantity_log",
    "product_discount_rate",
    "product_coupon_discount_rate",
    "household_avg_sales_log",
    "household_discount_rate",
    "household_coupon_discount_rate",
    "household_history_depth_log",
    "household_value_match_signal",
    "household_discount_match_signal",
    "household_coupon_product_signal",
    "campaign_duration_days_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost coupon-response ranker.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--base-candidates", type=Path, default=REPO_ROOT / "outputs" / "candidates_sota_ensemble.csv")
    parser.add_argument("--max-global-products", type=int, default=1000)
    parser.add_argument("--max-candidates-per-event", type=int, default=600)
    parser.add_argument("--base-k", type=int, default=50)
    parser.add_argument("--top-categories", type=int, default=3)
    parser.add_argument("--category-products", type=int, default=250)
    parser.add_argument("--eval-k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--objective", default="rank:ndcg", choices=["rank:ndcg", "rank:pairwise", "rank:map"])
    parser.add_argument("--positive-train-events-only", action="store_true")
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--search", action="store_true", help="Tune XGBoost configs on validation before final test.")
    parser.add_argument("--wide-search", action="store_true", help="Search a wider XGBoost grid on validation.")
    parser.add_argument("--search-objectives", action="store_true", help="Also search rank:pairwise/rank:map objectives.")
    parser.add_argument("--search-score-blend", action="store_true", help="Tune a validation-selected blend of XGBoost and repeat-cadence scores.")
    parser.add_argument("--blend-step", type=float, default=0.05, help="Grid step for --search-score-blend.")
    parser.add_argument("--primary-metric", default="recall_at_20")
    parser.add_argument("--selection-tolerance", type=float, default=0.001)
    parser.add_argument("--use-response-priors", action="store_true", help="Add train-period product/category response priors.")
    parser.add_argument("--use-content-features", action="store_true", help="Add product metadata affinity features.")
    parser.add_argument("--use-derived-features", action="store_true", help="Add event-relative rank and interaction features.")
    parser.add_argument("--use-value-features", action="store_true", help="Add historical price, spend, and discount-sensitivity features.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_columns(
    use_response_priors: bool = False,
    use_content_features: bool = False,
    use_derived_features: bool = False,
    use_value_features: bool = False,
) -> list[str]:
    columns = list(neural.FEATURE_COLUMNS)
    if use_derived_features:
        columns.extend(XGB_DERIVED_FEATURES)
    if use_value_features:
        columns.extend(XGB_VALUE_FEATURES)
    if use_response_priors:
        columns.extend(XGB_EXTRA_FEATURES)
    if use_content_features:
        columns.extend(XGB_CONTENT_FEATURES)
    return columns


def _grouped_xy(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    ordered = frame.sort_values([base.EVENT_COL, schema.PRODUCT_ID]).reset_index(drop=True)
    groups = ordered.groupby(base.EVENT_COL, sort=False).size().astype(int).tolist()
    x = ordered[feature_columns]
    y = ordered["label"].astype(float)
    return x, y, groups, ordered


def _rank_scores(frame: pd.DataFrame, scores, k: int) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["final_score"] = scores
    ranked = ranked.sort_values([base.EVENT_COL, "final_score", schema.PRODUCT_ID], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby(base.EVENT_COL).cumcount() + 1
    return ranked[ranked["rank"] <= k].copy()


def _normalize_scores_by_event(frame: pd.DataFrame, scores) -> np.ndarray:
    scored = frame[[base.EVENT_COL]].copy()
    scored["_score"] = np.asarray(scores, dtype=np.float32)
    grouped = scored.groupby(base.EVENT_COL)["_score"]
    min_score = grouped.transform("min")
    max_score = grouped.transform("max")
    denom = (max_score - min_score).replace(0.0, 1.0)
    return ((scored["_score"] - min_score) / denom).to_numpy(dtype=np.float32)


def _heuristic_scores(frame: pd.DataFrame, device: str) -> np.ndarray:
    return base.compute_weighted_score(frame, HEURISTIC_BLEND_WEIGHTS, device=device)


def _blend_scores(
    frame: pd.DataFrame,
    xgb_scores,
    heuristic_scores,
    xgb_blend_weight: float,
) -> np.ndarray:
    xgb_normalized = _normalize_scores_by_event(frame, xgb_scores)
    heuristic_normalized = _normalize_scores_by_event(frame, heuristic_scores)
    weight = float(np.clip(xgb_blend_weight, 0.0, 1.0))
    return (weight * xgb_normalized) + ((1.0 - weight) * heuristic_normalized)


def _candidate_blend_weights(enabled: bool, step: float) -> list[float]:
    if not enabled:
        return [1.0]
    step = float(step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("--blend-step must be in the interval (0, 1].")
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    return [float(np.clip(round(weight, 6), 0.0, 1.0)) for weight in weights]


def _select_blend_weight(
    frame: pd.DataFrame,
    xgb_scores,
    heuristic_scores,
    val_events: pd.DataFrame,
    val_truth: pd.DataFrame,
    eval_ks: list[int],
    primary_metric: str,
    blend_weights: list[float],
) -> tuple[float, dict[str, float]]:
    best_weight = 1.0
    best_metrics: dict[str, float] | None = None
    for weight in blend_weights:
        blended = _blend_scores(frame, xgb_scores, heuristic_scores, weight)
        ranked = _rank_scores(frame, blended, max(eval_ks))
        metrics = base.evaluate_ranked(ranked, val_truth, val_events, eval_ks)
        metric_value = float(metrics.get(primary_metric, 0.0))
        best_value = float(best_metrics.get(primary_metric, 0.0)) if best_metrics else -np.inf
        if (metric_value > best_value) or (
            math.isclose(metric_value, best_value, abs_tol=1e-12) and weight > best_weight
        ):
            best_weight = weight
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = {}
    return best_weight, best_metrics


def _xgb_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def add_xgb_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for column in [
        "base_signal",
        "repeat_signal",
        "cadence_signal",
        "category_signal",
        "global_signal",
        "discount_signal",
        "days_since_last",
        "median_interval_days",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    out["days_since_last_missing"] = out["days_since_last"].isna().astype(float)
    out["median_interval_missing"] = out["median_interval_days"].isna().astype(float)
    days = out["days_since_last"].fillna(365.0).clip(lower=0.0)
    interval = out["median_interval_days"].fillna(365.0).clip(lower=0.0)
    out["interval_ratio_log"] = np.log1p(days) - np.log1p(interval)
    out["interval_abs_log_error"] = out["interval_ratio_log"].abs()

    for column in ["base_signal", "repeat_signal", "cadence_signal", "category_signal", "global_signal", "discount_signal"]:
        out[column] = out[column].fillna(0.0)

    out["repeat_cadence_signal"] = out["repeat_signal"] * out["cadence_signal"]
    out["base_repeat_signal"] = out["base_signal"] * out["repeat_signal"]
    out["base_cadence_signal"] = out["base_signal"] * out["cadence_signal"]
    out["global_repeat_signal"] = out["global_signal"] * out["repeat_signal"]
    out["category_repeat_signal"] = out["category_signal"] * out["repeat_signal"]

    for source, target in [
        ("base_signal", "base_signal_pct_rank"),
        ("repeat_signal", "repeat_signal_pct_rank"),
        ("cadence_signal", "cadence_signal_pct_rank"),
        ("global_signal", "global_signal_pct_rank"),
        ("discount_signal", "discount_signal_pct_rank"),
    ]:
        out[target] = out.groupby(base.EVENT_COL)[source].rank(method="average", pct=True)

    for column in XGB_DERIVED_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _smoothed_prior(
    frame: pd.DataFrame,
    group_cols: list[str],
    global_rate: float,
    smoothing: float,
) -> pd.DataFrame:
    stats = frame.groupby(group_cols)["label"].agg(["sum", "count"]).reset_index()
    stats["prior"] = (stats["sum"] + global_rate * smoothing) / (stats["count"] + smoothing)
    stats["positive_count_log"] = np.log1p(stats["sum"])
    return stats


def add_xgb_response_prior_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    train = out[out["split"] == "train"].copy()
    global_rate = float(train["label"].mean()) if not train.empty else float(out["label"].mean())
    if not np.isfinite(global_rate):
        global_rate = 0.0

    mappings = [
        (["product_id"], "product_response_prior", "product_positive_count_log", 30.0),
        (["product_category"], "category_response_prior", "category_positive_count_log", 50.0),
        (["campaign_type"], "campaign_type_response_prior", "campaign_type_positive_count_log", 100.0),
    ]
    for group_cols, prior_col, count_col, smoothing in mappings:
        stats = _smoothed_prior(train, group_cols, global_rate, smoothing)
        stats = stats.rename(columns={"prior": prior_col, "positive_count_log": count_col})
        out = out.merge(stats[group_cols + [prior_col, count_col]], on=group_cols, how="left")
        out[prior_col] = out[prior_col].fillna(global_rate)
        out[count_col] = out[count_col].fillna(0.0)

    for column in XGB_EXTRA_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _clean_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("UNKNOWN", index=frame.index)
    return frame[column].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")


def add_content_affinity_features(features: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = features.copy()
    products = sources["products"][
        [schema.PRODUCT_ID, "department", "brand", "product_category", "product_type"]
    ].drop_duplicates(schema.PRODUCT_ID).copy()
    for column in ["department", "brand", "product_category", "product_type"]:
        products[column] = _clean_text_column(products, column)

    candidate_meta = products.rename(
        columns={
            "department": "candidate_department",
            "brand": "candidate_brand",
            "product_category": "candidate_product_category",
            "product_type": "candidate_product_type",
        }
    )
    out = out.merge(candidate_meta, on=schema.PRODUCT_ID, how="left")
    for column in ["candidate_department", "candidate_brand", "candidate_product_category", "candidate_product_type"]:
        out[column] = _clean_text_column(out, column)

    transactions = sources["transactions"][[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]].copy()
    transactions = transactions.merge(products, on=schema.PRODUCT_ID, how="left")
    for column in ["department", "brand", "product_category", "product_type"]:
        transactions[column] = _clean_text_column(transactions, column)

    for feature in XGB_CONTENT_FEATURES:
        out[feature] = 0.0

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index]
        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        households = set(group[schema.HOUSEHOLD_ID].astype(int))
        history = transactions[
            transactions[schema.HOUSEHOLD_ID].astype(int).isin(households)
            & (transactions["transaction_timestamp"] < start)
        ].copy()
        if history.empty:
            continue

        totals = history.groupby(schema.HOUSEHOLD_ID).size().to_dict()
        counts = {
            "department_affinity": history.groupby([schema.HOUSEHOLD_ID, "department"]).size().to_dict(),
            "brand_affinity": history.groupby([schema.HOUSEHOLD_ID, "brand"]).size().to_dict(),
            "category_affinity_exact": history.groupby([schema.HOUSEHOLD_ID, "product_category"]).size().to_dict(),
            "product_type_affinity": history.groupby([schema.HOUSEHOLD_ID, "product_type"]).size().to_dict(),
        }
        key_columns = {
            "department_affinity": "candidate_department",
            "brand_affinity": "candidate_brand",
            "category_affinity_exact": "candidate_product_category",
            "product_type_affinity": "candidate_product_type",
        }
        household_values = group[schema.HOUSEHOLD_ID].astype(int).to_numpy()
        for feature, count_map in counts.items():
            attr_values = group[key_columns[feature]].astype(str).to_numpy()
            values = [
                float(count_map.get((int(household), str(attr)), 0)) / float(max(1, totals.get(int(household), 0)))
                for household, attr in zip(household_values, attr_values, strict=False)
            ]
            out.loc[index, feature] = values

    out["content_match_signal"] = out[
        ["department_affinity", "brand_affinity", "category_affinity_exact", "product_type_affinity"]
    ].max(axis=1)
    for feature in XGB_CONTENT_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_value_features(features: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add historical value and discount-sensitivity signals.

    Features are recomputed at each campaign start from prior transactions only.
    They are optional because the held-out campaign split can drift sharply.
    """

    out = features.copy()
    transactions = sources["transactions"][
        [
            schema.HOUSEHOLD_ID,
            schema.PRODUCT_ID,
            schema.BASKET_ID,
            schema.SALES_VALUE,
            schema.QUANTITY,
            schema.RETAIL_DISC,
            schema.COUPON_DISC,
            schema.COUPON_MATCH_DISC,
            "transaction_timestamp",
        ]
    ].copy()
    for column in [
        schema.SALES_VALUE,
        schema.QUANTITY,
        schema.RETAIL_DISC,
        schema.COUPON_DISC,
        schema.COUPON_MATCH_DISC,
    ]:
        transactions[column] = pd.to_numeric(transactions[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    transactions["discount_value"] = (
        transactions[schema.RETAIL_DISC] + transactions[schema.COUPON_DISC] + transactions[schema.COUPON_MATCH_DISC]
    )
    transactions["coupon_discount_value"] = transactions[schema.COUPON_DISC] + transactions[schema.COUPON_MATCH_DISC]
    transactions["gross_value_proxy"] = transactions[schema.SALES_VALUE] + transactions["discount_value"]

    campaign_dates = sources["campaigns"][["campaign_id", "start_date", "end_date"]].drop_duplicates("campaign_id")
    campaign_dates = campaign_dates.set_index("campaign_id")

    for feature in XGB_VALUE_FEATURES:
        out[feature] = 0.0

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index]
        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        history = transactions[transactions["transaction_timestamp"] < start].copy()
        if history.empty:
            continue

        product_stats = history.groupby(schema.PRODUCT_ID).agg(
            total_sales=(schema.SALES_VALUE, "sum"),
            mean_sales=(schema.SALES_VALUE, "mean"),
            mean_quantity=(schema.QUANTITY, "mean"),
            total_discount=("discount_value", "sum"),
            total_coupon_discount=("coupon_discount_value", "sum"),
            gross_value=("gross_value_proxy", "sum"),
        )
        max_spend = float(np.log1p(product_stats["total_sales"].max())) if product_stats["total_sales"].max() > 0 else 1.0
        product_stats["product_spend_signal"] = np.log1p(product_stats["total_sales"]) / max_spend
        product_stats["product_avg_sales_log"] = np.log1p(product_stats["mean_sales"].clip(lower=0.0))
        product_stats["product_quantity_log"] = np.log1p(product_stats["mean_quantity"].clip(lower=0.0))
        denominator = product_stats["gross_value"].clip(lower=1e-6)
        product_stats["product_discount_rate"] = (product_stats["total_discount"] / denominator).clip(lower=0.0, upper=1.0)
        product_stats["product_coupon_discount_rate"] = (
            product_stats["total_coupon_discount"] / denominator
        ).clip(lower=0.0, upper=1.0)

        households = set(group[schema.HOUSEHOLD_ID].astype(int))
        household_history = history[history[schema.HOUSEHOLD_ID].astype(int).isin(households)].copy()
        if household_history.empty:
            household_stats = pd.DataFrame()
        else:
            household_stats = household_history.groupby(schema.HOUSEHOLD_ID).agg(
                mean_sales=(schema.SALES_VALUE, "mean"),
                total_discount=("discount_value", "sum"),
                total_coupon_discount=("coupon_discount_value", "sum"),
                gross_value=("gross_value_proxy", "sum"),
                transaction_count=(schema.PRODUCT_ID, "size"),
                basket_count=(schema.BASKET_ID, "nunique"),
            )
            denominator = household_stats["gross_value"].clip(lower=1e-6)
            household_stats["household_avg_sales_log"] = np.log1p(household_stats["mean_sales"].clip(lower=0.0))
            household_stats["household_discount_rate"] = (
                household_stats["total_discount"] / denominator
            ).clip(lower=0.0, upper=1.0)
            household_stats["household_coupon_discount_rate"] = (
                household_stats["total_coupon_discount"] / denominator
            ).clip(lower=0.0, upper=1.0)
            household_stats["household_history_depth_log"] = np.log1p(household_stats["transaction_count"].clip(lower=0.0))

        product_ids = group[schema.PRODUCT_ID].astype(int)
        household_ids = group[schema.HOUSEHOLD_ID].astype(int)
        p_spend = product_ids.map(product_stats["product_spend_signal"]).fillna(0.0).to_numpy(dtype=float)
        p_sales = product_ids.map(product_stats["product_avg_sales_log"]).fillna(0.0).to_numpy(dtype=float)
        p_quantity = product_ids.map(product_stats["product_quantity_log"]).fillna(0.0).to_numpy(dtype=float)
        p_discount = product_ids.map(product_stats["product_discount_rate"]).fillna(0.0).to_numpy(dtype=float)
        p_coupon = product_ids.map(product_stats["product_coupon_discount_rate"]).fillna(0.0).to_numpy(dtype=float)

        if household_stats.empty:
            h_sales = np.zeros(len(group), dtype=float)
            h_discount = np.zeros(len(group), dtype=float)
            h_coupon = np.zeros(len(group), dtype=float)
            h_depth = np.zeros(len(group), dtype=float)
        else:
            h_sales = household_ids.map(household_stats["household_avg_sales_log"]).fillna(0.0).to_numpy(dtype=float)
            h_discount = household_ids.map(household_stats["household_discount_rate"]).fillna(0.0).to_numpy(dtype=float)
            h_coupon = household_ids.map(household_stats["household_coupon_discount_rate"]).fillna(0.0).to_numpy(dtype=float)
            h_depth = household_ids.map(household_stats["household_history_depth_log"]).fillna(0.0).to_numpy(dtype=float)

        value_match = np.exp(-np.abs(p_sales - h_sales))
        discount_match = np.clip(1.0 - np.abs(p_discount - h_discount), 0.0, 1.0)
        coupon_product = p_coupon * h_coupon

        duration_days = 0.0
        if int(campaign_id) in campaign_dates.index:
            row = campaign_dates.loc[int(campaign_id)]
            duration_days = max(0.0, (pd.Timestamp(row["end_date"]) - pd.Timestamp(row["start_date"])).days)

        out.loc[index, "product_spend_signal"] = p_spend
        out.loc[index, "product_avg_sales_log"] = p_sales
        out.loc[index, "product_quantity_log"] = p_quantity
        out.loc[index, "product_discount_rate"] = p_discount
        out.loc[index, "product_coupon_discount_rate"] = p_coupon
        out.loc[index, "household_avg_sales_log"] = h_sales
        out.loc[index, "household_discount_rate"] = h_discount
        out.loc[index, "household_coupon_discount_rate"] = h_coupon
        out.loc[index, "household_history_depth_log"] = h_depth
        out.loc[index, "household_value_match_signal"] = value_match
        out.loc[index, "household_discount_match_signal"] = discount_match
        out.loc[index, "household_coupon_product_signal"] = coupon_product
        out.loc[index, "campaign_duration_days_log"] = math.log1p(duration_days)

    for feature in XGB_VALUE_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _candidate_configs(args: argparse.Namespace) -> list[dict[str, float | int | str | bool]]:
    if not args.search:
        return [
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "objective": args.objective,
                "positive_train_events_only": args.positive_train_events_only,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            }
        ]
    configs = [
        {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "objective": objective,
            "positive_train_events_only": positive_only,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
        for n_estimators, learning_rate, max_depth, objective, positive_only in [
            (120, 0.03, 2, "rank:ndcg", False),
            (180, 0.03, 2, "rank:ndcg", False),
            (250, 0.03, 2, "rank:ndcg", False),
            (120, 0.05, 3, "rank:ndcg", False),
            (180, 0.05, 2, "rank:ndcg", False),
            (250, 0.015, 3, "rank:ndcg", False),
        ]
    ]
    if args.wide_search:
        configs.extend(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "objective": "rank:ndcg",
                "positive_train_events_only": False,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
            for n_estimators, learning_rate, max_depth, subsample, colsample_bytree in [
                (80, 0.05, 2, 0.9, 0.9),
                (120, 0.02, 2, 0.9, 0.9),
                (180, 0.02, 2, 0.9, 0.9),
                (320, 0.015, 2, 0.9, 0.9),
                (400, 0.01, 2, 0.9, 0.9),
                (120, 0.04, 2, 0.8, 0.9),
                (180, 0.04, 2, 0.8, 0.8),
                (250, 0.02, 3, 0.8, 0.8),
                (320, 0.01, 3, 0.8, 0.8),
                (120, 0.03, 4, 0.8, 0.8),
                (180, 0.02, 4, 0.8, 0.8),
            ]
        )
    if args.search_objectives:
        configs.extend(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "objective": objective,
                "positive_train_events_only": positive_only,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            }
            for n_estimators, learning_rate, max_depth, objective, positive_only in [
                (180, 0.03, 2, "rank:pairwise", False),
                (250, 0.03, 2, "rank:pairwise", False),
                (180, 0.03, 2, "rank:map", False),
                (250, 0.03, 2, "rank:map", False),
                (180, 0.03, 2, "rank:ndcg", True),
                (250, 0.03, 2, "rank:ndcg", True),
                (180, 0.03, 2, "rank:pairwise", True),
            ]
        )
    return configs


def _xgb_eval_metric(objective: str) -> str:
    if objective == "rank:map":
        return "map@10"
    return "ndcg@10"


def _fit_ranker(xgb, config: dict[str, float | int | str | bool], device: str, seed: int, train_x, train_y, train_groups):
    objective = str(config.get("objective", "rank:ndcg"))
    ranker = xgb.XGBRanker(
        objective=objective,
        eval_metric=_xgb_eval_metric(objective),
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        device=device,
    )
    ranker.fit(train_x, train_y, group=train_groups, verbose=False)
    return ranker


def _evaluate_config(
    xgb,
    config: dict[str, float | int | str | bool],
    device: str,
    seed: int,
    train_x,
    train_y,
    train_groups,
    val_x,
    val_ordered: pd.DataFrame,
    val_heuristic_scores: np.ndarray,
    val_events: pd.DataFrame,
    val_truth: pd.DataFrame,
    eval_ks: list[int],
    primary_metric: str,
    blend_weights: list[float],
) -> tuple[object, dict[str, float]]:
    ranker = _fit_ranker(xgb, config, device, seed, train_x, train_y, train_groups)
    val_scores = ranker.predict(val_x)
    blend_weight, metrics = _select_blend_weight(
        val_ordered,
        val_scores,
        val_heuristic_scores,
        val_events,
        val_truth,
        eval_ks,
        primary_metric,
        blend_weights,
    )
    row = {"model_name": MODEL_NAME, "split": "validation", "device": device}
    row.update(config)
    row["xgb_blend_weight"] = blend_weight
    row.update(metrics)
    return ranker, row


def _filter_positive_event_groups(frame: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return frame
    positive_events = frame.groupby(base.EVENT_COL)["label"].sum()
    keep_events = set(positive_events[positive_events > 0].index)
    filtered = frame[frame[base.EVENT_COL].isin(keep_events)].copy()
    if filtered.empty:
        return frame
    return filtered


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    eval_ks = sorted(set(args.eval_k))
    max_k = max(eval_ks)

    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("xgboost is required. Install with: python -m pip install xgboost") from exc

    device = _xgb_device(args.device)
    print(f"XGBoost device: {device}")

    sources = base.load_sources(args)
    events = neural.make_all_events(sources)
    features, truth = neural.load_or_build_features(args, sources, events)
    features = neural.attach_labels(neural.add_model_features(features), truth)
    if args.use_derived_features:
        features = add_xgb_derived_features(features)
    if args.use_value_features:
        features = add_value_features(features, sources)
    if args.use_response_priors:
        features = add_xgb_response_prior_features(features)
    if args.use_content_features:
        features = add_content_affinity_features(features, sources)
    feature_columns = _feature_columns(
        args.use_response_priors,
        args.use_content_features,
        args.use_derived_features,
        args.use_value_features,
    )

    train = features[features["split"] == "train"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    test = features[features["split"] == "test"].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Train/validation/test features are required.")

    val_x, val_y, val_groups, val_ordered = _grouped_xy(validation, feature_columns)
    test_x, test_y, test_groups, test_ordered = _grouped_xy(test, feature_columns)
    val_heuristic_scores = _heuristic_scores(val_ordered, device)
    test_heuristic_scores = _heuristic_scores(test_ordered, device)
    blend_weights = _candidate_blend_weights(args.search_score_blend, args.blend_step)

    val_events = events[events["split"] == "validation"].copy()
    test_events = events[events["split"] == "test"].copy()
    val_truth = truth[truth[base.EVENT_COL].isin(set(val_events[base.EVENT_COL]))].copy()
    test_truth = truth[truth[base.EVENT_COL].isin(set(test_events[base.EVENT_COL]))].copy()

    search_rows = []
    configs = _candidate_configs(args)
    for config in configs:
        config_train = _filter_positive_event_groups(train, bool(config.get("positive_train_events_only", False)))
        train_x, train_y, train_groups, _ = _grouped_xy(config_train, feature_columns)
        _, row = _evaluate_config(
            xgb,
            config,
            device,
            args.seed,
            train_x,
            train_y,
            train_groups,
            val_x,
            val_ordered,
            val_heuristic_scores,
            val_events,
            val_truth,
            eval_ks,
            args.primary_metric,
            blend_weights,
        )
        search_rows.append(row)

    if not search_rows:
        raise RuntimeError("No XGBoost configs were trained.")

    search_frame = pd.DataFrame(search_rows)
    best_primary = float(search_frame[args.primary_metric].max())
    eligible = search_frame[
        search_frame[args.primary_metric] >= best_primary - max(0.0, args.selection_tolerance)
    ].copy()
    eligible = eligible.sort_values(
        ["max_depth", "n_estimators", "learning_rate", args.primary_metric, "ndcg_at_10"],
        ascending=[True, True, True, False, False],
    )
    best_row = eligible.iloc[0].to_dict()
    best_config = {
        key: best_row[key]
        for key in [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "objective",
            "positive_train_events_only",
            "subsample",
            "colsample_bytree",
        ]
    }
    best_blend_weight = float(best_row.get("xgb_blend_weight", 1.0))
    best_train = _filter_positive_event_groups(train, bool(best_config.get("positive_train_events_only", False)))
    train_x, train_y, train_groups, _ = _grouped_xy(best_train, feature_columns)
    best_ranker = _fit_ranker(xgb, best_config, device, args.seed, train_x, train_y, train_groups)
    val_scores = best_ranker.predict(val_x)
    val_final_scores = _blend_scores(val_ordered, val_scores, val_heuristic_scores, best_blend_weight)
    val_ranked = _rank_scores(val_ordered, val_final_scores, max_k)

    final_train = pd.concat([train, validation], ignore_index=True)
    final_train = _filter_positive_event_groups(final_train, bool(best_config.get("positive_train_events_only", False)))
    final_train_x, final_train_y, final_train_groups, _ = _grouped_xy(final_train, feature_columns)
    final_ranker = _fit_ranker(xgb, best_config, device, args.seed, final_train_x, final_train_y, final_train_groups)
    test_scores = final_ranker.predict(test_x)
    test_final_scores = _blend_scores(test_ordered, test_scores, test_heuristic_scores, best_blend_weight)
    test_ranked = _rank_scores(test_ordered, test_final_scores, max_k)

    comparison_rows = []
    for split, ranked, split_events, split_truth, trained_on in [
        ("validation", val_ranked, val_events, val_truth, "train"),
        ("test", test_ranked, test_events, test_truth, "train_plus_validation"),
    ]:
        row = {
            "model_name": MODEL_NAME,
            "split": split,
            "device": device,
            "model_selection": "validation" if args.search else "fixed",
            "trained_on": trained_on,
            "primary_metric": args.primary_metric,
            "selection_tolerance": args.selection_tolerance if args.search else 0.0,
            "xgb_blend_weight": best_blend_weight,
        }
        row.update(best_config)
        row.update(base.evaluate_ranked(ranked, split_truth, split_events, eval_ks))
        comparison_rows.append(row)

    ranked_all = pd.concat([val_ranked, test_ranked], ignore_index=True)
    ranked_out = base.attach_product_metadata(ranked_all, sources["products"], truth)
    ranked_out["model_name"] = MODEL_NAME
    ranked_out.to_csv(args.outputs_dir / "candidates_coupon_response_xgboost_ranker.csv", index=False)
    ranked_out.to_csv(args.outputs_dir / "reranked_recommendations.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.outputs_dir / "coupon_response_xgboost_model_comparison.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.outputs_dir / "coupon_response_xgboost_search.csv", index=False)

    comparison_parts = []
    for filename in [
        "coupon_response_model_comparison.csv",
        "coupon_response_neural_model_comparison.csv",
    ]:
        path = args.outputs_dir / filename
        if path.exists():
            comparison_parts.append(pd.read_csv(path))
    comparison_parts.append(comparison)
    pd.concat(comparison_parts, ignore_index=True, sort=False).to_csv(
        args.outputs_dir / "coupon_response_final_model_comparison.csv",
        index=False,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": final_ranker.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.outputs_dir / "coupon_response_xgboost_feature_importance.csv", index=False)

    print(f"Wrote {args.outputs_dir / 'candidates_coupon_response_xgboost_ranker.csv'} ({len(ranked_out)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_xgboost_search.csv'} ({len(search_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_xgboost_model_comparison.csv'} ({len(comparison_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_final_model_comparison.csv'} ({sum(len(part) for part in comparison_parts)} rows)")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print("Top feature importances:")
    print(importance.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
