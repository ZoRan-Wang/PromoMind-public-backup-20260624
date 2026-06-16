"""Train an XGBoost learning-to-rank coupon-response model.

The model uses household-campaign groups and coupon-response labels:
an item is relevant if it is bought within five days after campaign start.
The final variant can use graded pull-forward timing labels, where purchases
near the household's expected repurchase cadence receive higher relevance.
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
NO_RANK_FUSION = {
    "rank_fusion_method": "none",
    "rank_fusion_c": 0.0,
    "rank_fusion_xgb_weight": 1.0,
    "rank_fusion_heuristic_weight": 0.0,
    "rank_fusion_base_weight": 0.0,
    "rank_fusion_global_weight": 0.0,
}
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
XGB_COUPON_FAMILY_FEATURES = [
    "coupon_family_size_log",
    "coupon_family_global_signal",
    "coupon_family_repeat_signal",
    "coupon_family_count_log",
    "coupon_family_match",
    "coupon_family_substitute_signal",
    "product_coupon_upc_count_log",
]
XGB_REDEMPTION_FEATURES = [
    "household_redemption_count_log",
    "household_coupon_upc_redemption_log",
    "household_product_redemption_log",
    "household_category_redemption_log",
    "product_redemption_signal",
    "category_redemption_signal",
    "coupon_upc_redemption_signal",
    "household_redemption_match_signal",
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
    parser.add_argument("--search-rank-fusion", action="store_true", help="Tune validation-selected rank fusion over XGBoost and heuristic ranks.")
    parser.add_argument(
        "--label-scheme",
        choices=["binary", "pull_forward_timing", "pull_forward_interval", "expected_lead_timing"],
        default="binary",
        help="Training label scheme. Use pull_forward_interval or expected_lead_timing for graded timing relevance.",
    )
    parser.add_argument("--timing-grade-early-end-days", type=float, default=1.0)
    parser.add_argument("--timing-grade-middle-end-days", type=float, default=3.0)
    parser.add_argument("--pull-forward-min-days", type=float, default=-1.0)
    parser.add_argument("--pull-forward-max-days", type=float, default=2.0)
    parser.add_argument("--expected-lead-min-days", type=float, default=1.0)
    parser.add_argument("--expected-lead-max-days", type=float, default=2.0)
    parser.add_argument("--primary-metric", default="recall_at_20")
    parser.add_argument("--selection-tolerance", type=float, default=0.001)
    parser.add_argument("--use-response-priors", action="store_true", help="Add train-period product/category response priors.")
    parser.add_argument("--use-content-features", action="store_true", help="Add product metadata affinity features.")
    parser.add_argument("--use-derived-features", action="store_true", help="Add event-relative rank and interaction features.")
    parser.add_argument("--use-value-features", action="store_true", help="Add historical price, spend, and discount-sensitivity features.")
    parser.add_argument("--use-coupon-family-features", action="store_true", help="Add campaign coupon-UPC family repeat features.")
    parser.add_argument("--use-redemption-features", action="store_true", help="Add historical coupon redemption propensity features.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_columns(
    use_response_priors: bool = False,
    use_content_features: bool = False,
    use_derived_features: bool = False,
    use_value_features: bool = False,
    use_coupon_family_features: bool = False,
    use_redemption_features: bool = False,
) -> list[str]:
    columns = list(neural.FEATURE_COLUMNS)
    if use_derived_features:
        columns.extend(XGB_DERIVED_FEATURES)
    if use_value_features:
        columns.extend(XGB_VALUE_FEATURES)
    if use_coupon_family_features:
        columns.extend(XGB_COUPON_FAMILY_FEATURES)
    if use_redemption_features:
        columns.extend(XGB_REDEMPTION_FEATURES)
    if use_response_priors:
        columns.extend(XGB_EXTRA_FEATURES)
    if use_content_features:
        columns.extend(XGB_CONTENT_FEATURES)
    return columns


def apply_label_scheme(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    scheme: str,
    early_end_days: float = 1.0,
    middle_end_days: float = 3.0,
    pull_forward_min_days: float = -1.0,
    pull_forward_max_days: float = 2.0,
    expected_lead_min_days: float = 1.0,
    expected_lead_max_days: float = 2.0,
) -> pd.DataFrame:
    """Transform binary labels into optional integer relevance grades."""

    if scheme == "binary":
        return features
    if scheme not in {"pull_forward_timing", "pull_forward_interval", "expected_lead_timing"}:
        raise ValueError(f"Unsupported label scheme: {scheme}")
    if scheme == "pull_forward_timing" and not (0.0 <= early_end_days < middle_end_days <= 5.0):
        raise ValueError("Timing grade boundaries must satisfy 0 <= early < middle <= 5 days.")
    if scheme == "pull_forward_interval" and pull_forward_min_days > pull_forward_max_days:
        raise ValueError("Pull-forward boundaries must satisfy min <= max days.")
    if scheme == "expected_lead_timing" and expected_lead_min_days > expected_lead_max_days:
        raise ValueError("Expected-lead boundaries must satisfy min <= max days.")

    out = features.copy()
    labels = truth[[base.EVENT_COL, schema.PRODUCT_ID, "observed_purchase_time"]].drop_duplicates().copy()
    labels["observed_purchase_time"] = pd.to_datetime(
        labels["observed_purchase_time"],
        errors="coerce",
        format="mixed",
    )
    timing = out[[base.EVENT_COL, schema.PRODUCT_ID, "coupon_start_date"]].merge(
        labels,
        on=[base.EVENT_COL, schema.PRODUCT_ID],
        how="left",
    )
    start = pd.to_datetime(timing["coupon_start_date"], errors="coerce")
    days_after_coupon = (
        (timing["observed_purchase_time"] - start).dt.total_seconds() / 86400.0
    ).fillna(np.inf)
    positive = out["label"].to_numpy(dtype=float) > 0
    grade = np.zeros(len(out), dtype=np.float32)
    grade[positive] = 2.0

    if scheme == "pull_forward_timing":
        middle = positive & (days_after_coupon.to_numpy() > early_end_days) & (
            days_after_coupon.to_numpy() <= middle_end_days
        )
    elif scheme == "pull_forward_interval":
        actual_interval = out["days_since_last"].to_numpy(dtype=float) + days_after_coupon.to_numpy(dtype=float)
        pull_forward_days = out["median_interval_days"].to_numpy(dtype=float) - actual_interval
        finite = np.isfinite(actual_interval) & np.isfinite(pull_forward_days)
        middle = positive & finite & (pull_forward_days >= pull_forward_min_days) & (
            pull_forward_days <= pull_forward_max_days
        )
    else:
        expected_lead_days = out["median_interval_days"].to_numpy(dtype=float) - out["days_since_last"].to_numpy(dtype=float)
        finite = np.isfinite(expected_lead_days)
        middle = positive & finite & (expected_lead_days >= expected_lead_min_days) & (
            expected_lead_days <= expected_lead_max_days
        )
    grade[middle] = 3.0
    out["label"] = grade
    return out


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


def _rank_array_by_event(frame: pd.DataFrame, scores) -> np.ndarray:
    ranked = frame[[base.EVENT_COL]].copy()
    ranked["_score"] = np.asarray(scores, dtype=np.float32)
    return ranked.groupby(base.EVENT_COL)["_score"].rank(method="first", ascending=False).to_numpy(dtype=np.float32)


def _rank_fusion_configs(enabled: bool) -> list[dict[str, float | str]]:
    if not enabled:
        return [NO_RANK_FUSION.copy()]

    configs: list[dict[str, float | str]] = []
    methods = [("rrf", 60.0), ("rrf", 100.0), ("exp", 30.0)]
    weight_sets = [
        (0.9, 0.1, 0.0, 0.0),
        (0.8, 0.2, 0.0, 0.0),
        (0.7, 0.3, 0.0, 0.0),
        (0.7, 0.2, 0.0, 0.1),
        (0.6, 0.3, 0.1, 0.0),
    ]
    for method, c_value in methods:
        for xgb_weight, heuristic_weight, base_weight, global_weight in weight_sets:
            configs.append(
                {
                    "rank_fusion_method": method,
                    "rank_fusion_c": c_value,
                    "rank_fusion_xgb_weight": xgb_weight,
                    "rank_fusion_heuristic_weight": heuristic_weight,
                    "rank_fusion_base_weight": base_weight,
                    "rank_fusion_global_weight": global_weight,
                }
            )
    return configs


def _rank_fusion_scores(
    frame: pd.DataFrame,
    xgb_scores,
    heuristic_scores,
    config: dict[str, float | str],
) -> np.ndarray:
    method = str(config.get("rank_fusion_method", "none"))
    if method == "none":
        return np.asarray(xgb_scores, dtype=np.float32)

    rank_sources = {
        "xgb": _rank_array_by_event(frame, xgb_scores),
        "heuristic": _rank_array_by_event(frame, heuristic_scores),
        "base": _rank_array_by_event(frame, frame["base_signal"].to_numpy(dtype=np.float32)),
        "global": _rank_array_by_event(frame, frame["global_signal"].to_numpy(dtype=np.float32)),
    }
    weights = {
        "xgb": float(config.get("rank_fusion_xgb_weight", 0.0)),
        "heuristic": float(config.get("rank_fusion_heuristic_weight", 0.0)),
        "base": float(config.get("rank_fusion_base_weight", 0.0)),
        "global": float(config.get("rank_fusion_global_weight", 0.0)),
    }
    c_value = max(float(config.get("rank_fusion_c", 60.0)), 1e-6)
    scores = np.zeros(len(frame), dtype=np.float32)
    for source, ranks in rank_sources.items():
        weight = weights[source]
        if weight == 0.0:
            continue
        if method == "rrf":
            scores += weight / (c_value + ranks)
        elif method == "exp":
            scores += weight * np.exp(-(ranks - 1.0) / c_value)
        else:
            raise ValueError(f"Unsupported rank fusion method: {method}")
    return scores


def _select_rank_fusion(
    frame: pd.DataFrame,
    xgb_scores,
    heuristic_scores,
    val_events: pd.DataFrame,
    val_truth: pd.DataFrame,
    eval_ks: list[int],
    primary_metric: str,
    fusion_configs: list[dict[str, float | str]],
) -> tuple[dict[str, float | str], dict[str, float]]:
    best_config = NO_RANK_FUSION.copy()
    best_metrics: dict[str, float] | None = None
    for config in fusion_configs:
        fused_scores = _rank_fusion_scores(frame, xgb_scores, heuristic_scores, config)
        ranked = _rank_scores(frame, fused_scores, max(eval_ks))
        metrics = base.evaluate_ranked(ranked, val_truth, val_events, eval_ks)
        metric_value = float(metrics.get(primary_metric, 0.0))
        best_value = float(best_metrics.get(primary_metric, 0.0)) if best_metrics else -np.inf
        if (metric_value > best_value) or (
            math.isclose(metric_value, best_value, abs_tol=1e-12)
            and float(config.get("rank_fusion_xgb_weight", 0.0)) > float(best_config.get("rank_fusion_xgb_weight", 0.0))
        ):
            best_config = config.copy()
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = {}
    return best_config, best_metrics


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


def add_coupon_family_features(features: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add campaign-local coupon family history.

    A coupon UPC can map to multiple product IDs in a campaign. These features
    capture whether a household bought any product in the same coupon family
    before the campaign starts, not only the exact candidate product.
    """

    out = features.copy()
    coupons = sources["coupons"][["campaign_id", "coupon_upc", schema.PRODUCT_ID]].drop_duplicates().copy()
    transactions = sources["transactions"][
        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]
    ].copy()
    transactions[schema.PRODUCT_ID] = pd.to_numeric(transactions[schema.PRODUCT_ID], errors="coerce").astype("Int64")
    transactions = transactions.dropna(subset=[schema.PRODUCT_ID])
    transactions[schema.PRODUCT_ID] = transactions[schema.PRODUCT_ID].astype(int)

    for feature in XGB_COUPON_FAMILY_FEATURES:
        out[feature] = 0.0

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index]
        campaign_coupons = coupons[coupons["campaign_id"] == int(campaign_id)].copy()
        if campaign_coupons.empty:
            continue

        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        history = transactions[transactions["transaction_timestamp"] < start].copy()
        if history.empty:
            continue

        coupon_family_size = campaign_coupons.groupby("coupon_upc")[schema.PRODUCT_ID].nunique()
        product_coupon_count = campaign_coupons.groupby(schema.PRODUCT_ID)["coupon_upc"].nunique()

        matched_history = history.merge(campaign_coupons, on=schema.PRODUCT_ID, how="inner")
        if matched_history.empty:
            family_global_counts = pd.Series(dtype=float)
            family_household_counts = pd.Series(dtype=float)
        else:
            family_global_counts = matched_history.groupby("coupon_upc").size().astype(float)
            family_household_counts = matched_history.groupby([schema.HOUSEHOLD_ID, "coupon_upc"]).size().astype(float)

        max_global = float(np.log1p(family_global_counts.max())) if not family_global_counts.empty else 1.0
        max_global = max(max_global, 1.0)

        row_family = campaign_coupons.merge(
            group[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "user_product_count"]],
            on=schema.PRODUCT_ID,
            how="inner",
        )
        if row_family.empty:
            continue

        family_rows = []
        for row in row_family.itertuples(index=False):
            household_id = int(getattr(row, schema.HOUSEHOLD_ID))
            product_id = int(getattr(row, schema.PRODUCT_ID))
            coupon_upc = int(getattr(row, "coupon_upc"))
            family_count = float(family_household_counts.get((household_id, coupon_upc), 0.0))
            exact_count = float(getattr(row, "user_product_count"))
            family_rows.append(
                {
                    schema.HOUSEHOLD_ID: household_id,
                    schema.PRODUCT_ID: product_id,
                    "coupon_family_size_log": math.log1p(float(coupon_family_size.get(coupon_upc, 0.0))),
                    "coupon_family_global_signal": math.log1p(float(family_global_counts.get(coupon_upc, 0.0))) / max_global,
                    "coupon_family_count_log": math.log1p(family_count),
                    "coupon_family_match": float(family_count > 0.0),
                    "coupon_family_substitute_signal": math.log1p(max(0.0, family_count - exact_count)),
                    "product_coupon_upc_count_log": math.log1p(float(product_coupon_count.get(product_id, 0.0))),
                }
            )

        family_features = pd.DataFrame(family_rows)
        if family_features.empty:
            continue
        aggregated = family_features.groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID], as_index=False).agg(
            coupon_family_size_log=("coupon_family_size_log", "max"),
            coupon_family_global_signal=("coupon_family_global_signal", "max"),
            coupon_family_count_log=("coupon_family_count_log", "max"),
            coupon_family_match=("coupon_family_match", "max"),
            coupon_family_substitute_signal=("coupon_family_substitute_signal", "max"),
            product_coupon_upc_count_log=("product_coupon_upc_count_log", "max"),
        )
        max_family_count = max(float(aggregated["coupon_family_count_log"].max()), 1.0)
        aggregated["coupon_family_repeat_signal"] = aggregated["coupon_family_count_log"] / max_family_count

        update = group[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]].merge(
            aggregated,
            on=[schema.HOUSEHOLD_ID, schema.PRODUCT_ID],
            how="left",
        )
        for feature in XGB_COUPON_FAMILY_FEATURES:
            out.loc[index, feature] = update[feature].fillna(0.0).to_numpy(dtype=float)

    for feature in XGB_COUPON_FAMILY_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_redemption_features(features: pd.DataFrame, sources: dict[str, pd.DataFrame], raw_dir: Path) -> pd.DataFrame:
    """Add historical coupon-redemption propensity features.

    Only redemptions before the current campaign start are used. The current
    campaign's future redemption outcomes are not used for ranking.
    """

    out = features.copy()
    redemptions_path = raw_dir / "coupon_redemptions.csv"
    for feature in XGB_REDEMPTION_FEATURES:
        out[feature] = 0.0
    if not redemptions_path.exists():
        return out

    redemptions = pd.read_csv(redemptions_path, parse_dates=["redemption_date"])
    if redemptions.empty:
        return out
    redemptions = base._normalize_ids(redemptions, [schema.HOUSEHOLD_ID, "campaign_id", "coupon_upc"])

    coupons = sources["coupons"][["campaign_id", "coupon_upc", schema.PRODUCT_ID]].drop_duplicates().copy()
    products = sources["products"][[schema.PRODUCT_ID, "product_category"]].drop_duplicates(schema.PRODUCT_ID).copy()
    products["product_category"] = products["product_category"].fillna("UNKNOWN").astype(str)
    coupon_products = coupons.merge(products, on=schema.PRODUCT_ID, how="left")

    redeemed_products = redemptions.merge(coupon_products, on=["campaign_id", "coupon_upc"], how="left")
    redeemed_products["product_category"] = redeemed_products["product_category"].fillna("UNKNOWN").astype(str)

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index]
        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        history = redemptions[redemptions["redemption_date"] < start].copy()
        product_history = redeemed_products[redeemed_products["redemption_date"] < start].copy()
        if history.empty and product_history.empty:
            continue

        campaign_coupons = coupons[coupons["campaign_id"] == int(campaign_id)].copy()
        if campaign_coupons.empty:
            continue
        candidate_coupon_counts = campaign_coupons.groupby(schema.PRODUCT_ID)["coupon_upc"].nunique()
        candidate_coupon_lists = campaign_coupons.groupby(schema.PRODUCT_ID)["coupon_upc"].apply(lambda values: set(map(int, values)))

        household_redemptions = history.groupby(schema.HOUSEHOLD_ID).size()
        household_coupon_redemptions = history.groupby([schema.HOUSEHOLD_ID, "coupon_upc"]).size()
        coupon_upc_redemptions = history.groupby("coupon_upc").size()
        max_coupon_upc = float(np.log1p(coupon_upc_redemptions.max())) if not coupon_upc_redemptions.empty else 1.0
        max_coupon_upc = max(max_coupon_upc, 1.0)

        if product_history.empty:
            household_product_redemptions = pd.Series(dtype=float)
            household_category_redemptions = pd.Series(dtype=float)
            product_redemptions = pd.Series(dtype=float)
            category_redemptions = pd.Series(dtype=float)
        else:
            household_product_redemptions = product_history.groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID]).size()
            household_category_redemptions = product_history.groupby([schema.HOUSEHOLD_ID, "product_category"]).size()
            product_redemptions = product_history.groupby(schema.PRODUCT_ID).size()
            category_redemptions = product_history.groupby("product_category").size()

        max_product = float(np.log1p(product_redemptions.max())) if not product_redemptions.empty else 1.0
        max_category = float(np.log1p(category_redemptions.max())) if not category_redemptions.empty else 1.0
        max_product = max(max_product, 1.0)
        max_category = max(max_category, 1.0)

        row_values: list[dict[str, float | int]] = []
        for row in group[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "product_category"]].itertuples(index=False):
            household_id = int(getattr(row, schema.HOUSEHOLD_ID))
            product_id = int(getattr(row, schema.PRODUCT_ID))
            category = str(getattr(row, "product_category"))
            coupon_upcs = candidate_coupon_lists.get(product_id, set())
            household_coupon_count = sum(
                float(household_coupon_redemptions.get((household_id, coupon_upc), 0.0))
                for coupon_upc in coupon_upcs
            )
            coupon_upc_count = sum(float(coupon_upc_redemptions.get(coupon_upc, 0.0)) for coupon_upc in coupon_upcs)
            row_values.append(
                {
                    schema.HOUSEHOLD_ID: household_id,
                    schema.PRODUCT_ID: product_id,
                    "household_redemption_count_log": math.log1p(float(household_redemptions.get(household_id, 0.0))),
                    "household_coupon_upc_redemption_log": math.log1p(household_coupon_count),
                    "household_product_redemption_log": math.log1p(
                        float(household_product_redemptions.get((household_id, product_id), 0.0))
                    ),
                    "household_category_redemption_log": math.log1p(
                        float(household_category_redemptions.get((household_id, category), 0.0))
                    ),
                    "product_redemption_signal": math.log1p(float(product_redemptions.get(product_id, 0.0))) / max_product,
                    "category_redemption_signal": math.log1p(float(category_redemptions.get(category, 0.0))) / max_category,
                    "coupon_upc_redemption_signal": math.log1p(coupon_upc_count) / max_coupon_upc,
                    "household_redemption_match_signal": float(household_coupon_count > 0.0)
                    * math.log1p(float(candidate_coupon_counts.get(product_id, 0.0))),
                }
            )

        redemption_features = pd.DataFrame(row_values)
        if redemption_features.empty:
            continue
        update = group[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]].merge(
            redemption_features,
            on=[schema.HOUSEHOLD_ID, schema.PRODUCT_ID],
            how="left",
        )
        for feature in XGB_REDEMPTION_FEATURES:
            out.loc[index, feature] = update[feature].fillna(0.0).to_numpy(dtype=float)

    for feature in XGB_REDEMPTION_FEATURES:
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
    rank_fusion_configs: list[dict[str, float | str]],
) -> tuple[object, dict[str, float]]:
    ranker = _fit_ranker(xgb, config, device, seed, train_x, train_y, train_groups)
    val_scores = ranker.predict(val_x)
    rank_fusion_config = NO_RANK_FUSION.copy()
    if rank_fusion_configs and str(rank_fusion_configs[0].get("rank_fusion_method", "none")) != "none":
        rank_fusion_config, metrics = _select_rank_fusion(
            val_ordered,
            val_scores,
            val_heuristic_scores,
            val_events,
            val_truth,
            eval_ks,
            primary_metric,
            rank_fusion_configs,
        )
        blend_weight = 1.0
    else:
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
    row.update(rank_fusion_config)
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
    if args.search_score_blend and args.search_rank_fusion:
        raise ValueError("Use either --search-score-blend or --search-rank-fusion, not both.")

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
    features = apply_label_scheme(
        features,
        truth,
        args.label_scheme,
        args.timing_grade_early_end_days,
        args.timing_grade_middle_end_days,
        args.pull_forward_min_days,
        args.pull_forward_max_days,
        args.expected_lead_min_days,
        args.expected_lead_max_days,
    )
    if args.use_derived_features:
        features = add_xgb_derived_features(features)
    if args.use_value_features:
        features = add_value_features(features, sources)
    if args.use_coupon_family_features:
        features = add_coupon_family_features(features, sources)
    if args.use_redemption_features:
        features = add_redemption_features(features, sources, args.raw_dir)
    if args.use_response_priors:
        features = add_xgb_response_prior_features(features)
    if args.use_content_features:
        features = add_content_affinity_features(features, sources)
    feature_columns = _feature_columns(
        args.use_response_priors,
        args.use_content_features,
        args.use_derived_features,
        args.use_value_features,
        args.use_coupon_family_features,
        args.use_redemption_features,
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
    rank_fusion_configs = _rank_fusion_configs(args.search_rank_fusion)

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
            rank_fusion_configs,
        )
        row["label_scheme"] = args.label_scheme
        row["timing_grade_early_end_days"] = args.timing_grade_early_end_days
        row["timing_grade_middle_end_days"] = args.timing_grade_middle_end_days
        row["pull_forward_min_days"] = args.pull_forward_min_days
        row["pull_forward_max_days"] = args.pull_forward_max_days
        row["expected_lead_min_days"] = args.expected_lead_min_days
        row["expected_lead_max_days"] = args.expected_lead_max_days
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
    best_rank_fusion_config = {
        "rank_fusion_method": best_row.get("rank_fusion_method", "none"),
        "rank_fusion_c": float(best_row.get("rank_fusion_c", 0.0)),
        "rank_fusion_xgb_weight": float(best_row.get("rank_fusion_xgb_weight", 1.0)),
        "rank_fusion_heuristic_weight": float(best_row.get("rank_fusion_heuristic_weight", 0.0)),
        "rank_fusion_base_weight": float(best_row.get("rank_fusion_base_weight", 0.0)),
        "rank_fusion_global_weight": float(best_row.get("rank_fusion_global_weight", 0.0)),
    }
    best_train = _filter_positive_event_groups(train, bool(best_config.get("positive_train_events_only", False)))
    train_x, train_y, train_groups, _ = _grouped_xy(best_train, feature_columns)
    best_ranker = _fit_ranker(xgb, best_config, device, args.seed, train_x, train_y, train_groups)
    val_scores = best_ranker.predict(val_x)
    if str(best_rank_fusion_config.get("rank_fusion_method", "none")) != "none":
        val_final_scores = _rank_fusion_scores(val_ordered, val_scores, val_heuristic_scores, best_rank_fusion_config)
    else:
        val_final_scores = _blend_scores(val_ordered, val_scores, val_heuristic_scores, best_blend_weight)
    val_ranked = _rank_scores(val_ordered, val_final_scores, max_k)

    final_train = pd.concat([train, validation], ignore_index=True)
    final_train = _filter_positive_event_groups(final_train, bool(best_config.get("positive_train_events_only", False)))
    final_train_x, final_train_y, final_train_groups, _ = _grouped_xy(final_train, feature_columns)
    final_ranker = _fit_ranker(xgb, best_config, device, args.seed, final_train_x, final_train_y, final_train_groups)
    test_scores = final_ranker.predict(test_x)
    if str(best_rank_fusion_config.get("rank_fusion_method", "none")) != "none":
        test_final_scores = _rank_fusion_scores(test_ordered, test_scores, test_heuristic_scores, best_rank_fusion_config)
    else:
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
            "label_scheme": args.label_scheme,
            "timing_grade_early_end_days": args.timing_grade_early_end_days,
            "timing_grade_middle_end_days": args.timing_grade_middle_end_days,
            "pull_forward_min_days": args.pull_forward_min_days,
            "pull_forward_max_days": args.pull_forward_max_days,
            "expected_lead_min_days": args.expected_lead_min_days,
            "expected_lead_max_days": args.expected_lead_max_days,
        }
        row.update(best_rank_fusion_config)
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
