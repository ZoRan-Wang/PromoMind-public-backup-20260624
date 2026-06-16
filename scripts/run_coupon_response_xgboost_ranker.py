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
XGB_TEXT_EMBEDDING_FEATURES = [
    "text_embedding_similarity",
    "text_embedding_profile_norm",
    "text_embedding_history_count_log",
    "text_embedding_has_profile",
]
XGB_TEXT_MATCH_FEATURES = [
    "text_match_max_similarity",
    "text_match_top3_similarity",
    "text_match_recent_max_similarity",
    "text_match_history_count_log",
    "text_match_has_profile",
]
XGB_CATEGORY_EMBEDDING_FEATURES = [
    "category_embedding_profile_similarity",
    "category_embedding_max_similarity",
    "category_embedding_profile_norm",
    "category_embedding_history_count_log",
    "category_embedding_has_profile",
]
XGB_EVENT_CATEGORY_FEATURES = [
    "event_category_count_log",
    "event_category_share",
    "event_category_global_mean",
    "event_category_global_max",
    "event_category_global_rank_pct",
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
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--search", action="store_true", help="Tune XGBoost configs on validation before final test.")
    parser.add_argument("--wide-search", action="store_true", help="Search a wider XGBoost grid on validation.")
    parser.add_argument("--search-objectives", action="store_true", help="Also search rank:pairwise/rank:map objectives.")
    parser.add_argument("--ensemble-top-n", type=int, default=1, help="Average the top-N validation-selected XGBoost rankers.")
    parser.add_argument(
        "--final-train-scope",
        choices=["train", "train_plus_validation"],
        default="train_plus_validation",
        help="Data used to fit the final test-scoring model after validation selection.",
    )
    parser.add_argument("--search-score-blend", action="store_true", help="Tune a validation-selected blend of XGBoost and repeat-cadence scores.")
    parser.add_argument("--blend-step", type=float, default=0.05, help="Grid step for --search-score-blend.")
    parser.add_argument("--search-rank-fusion", action="store_true", help="Tune validation-selected rank fusion over XGBoost and heuristic ranks.")
    parser.add_argument(
        "--label-scheme",
        choices=[
            "binary",
            "pull_forward_timing",
            "pull_forward_interval",
            "pull_forward_interval_new_high",
            "expected_lead_timing",
        ],
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
    parser.add_argument("--use-text-embedding-features", action="store_true", help="Add TF-IDF/SVD product-text profile features.")
    parser.add_argument("--text-embedding-components", type=int, default=32)
    parser.add_argument("--text-max-features", type=int, default=4096)
    parser.add_argument(
        "--use-text-match-features",
        action="store_true",
        help="Add direct TF-IDF product-text match features against recent household history.",
    )
    parser.add_argument("--text-match-max-features", type=int, default=8192)
    parser.add_argument("--text-match-history-products", type=int, default=60)
    parser.add_argument("--text-match-recent-products", type=int, default=10)
    parser.add_argument("--use-category-embedding-features", action="store_true", help="Add train-period category co-occurrence embedding features.")
    parser.add_argument("--category-embedding-components", type=int, default=24)
    parser.add_argument("--use-event-category-features", action="store_true", help="Add within-event coupon category concentration features.")
    parser.add_argument("--use-derived-features", action="store_true", help="Add event-relative rank and interaction features.")
    parser.add_argument("--use-value-features", action="store_true", help="Add historical price, spend, and discount-sensitivity features.")
    parser.add_argument("--use-coupon-family-features", action="store_true", help="Add campaign coupon-UPC family repeat features.")
    parser.add_argument("--use-redemption-features", action="store_true", help="Add historical coupon redemption propensity features.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_columns(
    use_response_priors: bool = False,
    use_content_features: bool = False,
    use_text_embedding_features: bool = False,
    use_text_match_features: bool = False,
    use_category_embedding_features: bool = False,
    use_event_category_features: bool = False,
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
    if use_text_embedding_features:
        columns.extend(XGB_TEXT_EMBEDDING_FEATURES)
    if use_text_match_features:
        columns.extend(XGB_TEXT_MATCH_FEATURES)
    if use_category_embedding_features:
        columns.extend(XGB_CATEGORY_EMBEDDING_FEATURES)
    if use_event_category_features:
        columns.extend(XGB_EVENT_CATEGORY_FEATURES)
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
    if scheme not in {
        "pull_forward_timing",
        "pull_forward_interval",
        "pull_forward_interval_new_high",
        "expected_lead_timing",
    }:
        raise ValueError(f"Unsupported label scheme: {scheme}")
    if scheme == "pull_forward_timing" and not (0.0 <= early_end_days < middle_end_days <= 5.0):
        raise ValueError("Timing grade boundaries must satisfy 0 <= early < middle <= 5 days.")
    if scheme in {"pull_forward_interval", "pull_forward_interval_new_high"} and pull_forward_min_days > pull_forward_max_days:
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
    elif scheme in {"pull_forward_interval", "pull_forward_interval_new_high"}:
        actual_interval = out["days_since_last"].to_numpy(dtype=float) + days_after_coupon.to_numpy(dtype=float)
        pull_forward_days = out["median_interval_days"].to_numpy(dtype=float) - actual_interval
        finite = np.isfinite(actual_interval) & np.isfinite(pull_forward_days)
        middle = positive & finite & (pull_forward_days >= pull_forward_min_days) & (
            pull_forward_days <= pull_forward_max_days
        )
        if scheme == "pull_forward_interval_new_high":
            user_product_count = out["user_product_count"].to_numpy(dtype=float)
            new_to_household = positive & (~np.isfinite(user_product_count) | (user_product_count <= 0.0))
            middle = middle | new_to_household
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


def _build_product_text(products: pd.DataFrame) -> pd.Series:
    text_columns = ["department", "product_category", "product_type", "brand", "package_size"]
    parts = []
    for column in text_columns:
        parts.append(_clean_text_column(products, column).str.replace("UNKNOWN", "", regex=False))
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


def _product_text_embeddings(
    products: pd.DataFrame,
    components: int,
    max_features: int,
) -> tuple[pd.DataFrame, list[str]]:
    if components < 1:
        raise ValueError("--text-embedding-components must be >= 1.")
    if max_features < 10:
        raise ValueError("--text-max-features must be >= 10.")

    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for --use-text-embedding-features.") from exc

    text_columns = ["department", "brand", "product_category", "product_type", "package_size"]
    available_columns = [column for column in text_columns if column in products.columns]
    product_meta = products[[schema.PRODUCT_ID, *available_columns]].drop_duplicates(schema.PRODUCT_ID).copy()
    for column in text_columns:
        if column not in product_meta.columns:
            product_meta[column] = "UNKNOWN"
    product_text = _build_product_text(product_meta).replace("", "unknown product")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2 if len(product_meta) >= 50 else 1,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/&:-]+\b",
        lowercase=True,
    )
    tfidf = vectorizer.fit_transform(product_text)
    if tfidf.shape[1] == 0:
        embedding = np.zeros((len(product_meta), 1), dtype=np.float32)
    elif min(tfidf.shape) <= 1:
        embedding = normalize(tfidf, norm="l2", copy=True).toarray().astype(np.float32)
    else:
        n_components = min(int(components), max(1, min(tfidf.shape) - 1))
        embedding = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(tfidf).astype(np.float32)
        embedding = normalize(embedding, norm="l2", copy=False).astype(np.float32)

    columns = [f"text_emb_{idx}" for idx in range(embedding.shape[1])]
    embeddings = pd.DataFrame(embedding, columns=columns)
    embeddings.insert(0, schema.PRODUCT_ID, product_meta[schema.PRODUCT_ID].astype(int).to_numpy())
    return embeddings, columns


def _product_text_tfidf(products: pd.DataFrame, max_features: int):
    if max_features < 10:
        raise ValueError("--text-match-max-features must be >= 10.")

    try:
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        raise RuntimeError("scikit-learn and scipy are required for --use-text-match-features.") from exc

    text_columns = ["department", "brand", "product_category", "product_type", "package_size"]
    available_columns = [column for column in text_columns if column in products.columns]
    product_meta = products[[schema.PRODUCT_ID, *available_columns]].drop_duplicates(schema.PRODUCT_ID).copy()
    for column in text_columns:
        if column not in product_meta.columns:
            product_meta[column] = "UNKNOWN"
    product_meta[schema.PRODUCT_ID] = pd.to_numeric(product_meta[schema.PRODUCT_ID], errors="coerce")
    product_meta = product_meta.dropna(subset=[schema.PRODUCT_ID]).copy()
    product_meta[schema.PRODUCT_ID] = product_meta[schema.PRODUCT_ID].astype(int)
    product_text = _build_product_text(product_meta).replace("", "unknown product")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2 if len(product_meta) >= 50 else 1,
        ngram_range=(1, 2),
        analyzer="word",
        token_pattern=r"(?u)\b[\w/&:-]+\b",
        lowercase=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(product_text)
    if matrix.shape[1] == 0:
        matrix = sparse.csr_matrix((len(product_meta), 1), dtype=np.float32)
    return product_meta[[schema.PRODUCT_ID]], matrix.tocsr().astype(np.float32)


def add_text_match_features(
    features: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    max_features: int = 8192,
    history_products: int = 60,
    recent_products: int = 10,
) -> pd.DataFrame:
    """Add direct product-text similarity features against prior household history."""

    if history_products < 1:
        raise ValueError("--text-match-history-products must be >= 1.")
    if recent_products < 1:
        raise ValueError("--text-match-recent-products must be >= 1.")

    out = features.copy()
    for feature in XGB_TEXT_MATCH_FEATURES:
        out[feature] = 0.0

    product_ids, matrix = _product_text_tfidf(sources["products"], max_features)
    product_row = {
        int(product_id): row_idx for row_idx, product_id in enumerate(product_ids[schema.PRODUCT_ID].astype(int))
    }

    transactions = sources["transactions"][
        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]
    ].copy()
    transactions[schema.HOUSEHOLD_ID] = pd.to_numeric(transactions[schema.HOUSEHOLD_ID], errors="coerce")
    transactions[schema.PRODUCT_ID] = pd.to_numeric(transactions[schema.PRODUCT_ID], errors="coerce")
    transactions["transaction_timestamp"] = pd.to_datetime(
        transactions["transaction_timestamp"],
        errors="coerce",
        format="mixed",
    )
    transactions["product_text_row"] = transactions[schema.PRODUCT_ID].map(product_row)
    transactions = transactions.dropna(
        subset=[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp", "product_text_row"]
    ).copy()
    transactions[schema.HOUSEHOLD_ID] = transactions[schema.HOUSEHOLD_ID].astype(int)
    transactions["product_text_row"] = transactions["product_text_row"].astype(int)
    transactions = transactions.sort_values([schema.HOUSEHOLD_ID, "transaction_timestamp"])

    out["_product_text_row"] = pd.to_numeric(out[schema.PRODUCT_ID].map(product_row), errors="coerce")
    history_limit = int(history_products)
    recent_limit = min(int(recent_products), history_limit)

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index]
        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        households = set(group[schema.HOUSEHOLD_ID].astype(int))
        history = transactions[
            transactions[schema.HOUSEHOLD_ID].isin(households)
            & (transactions["transaction_timestamp"] < start)
        ]
        if history.empty:
            continue

        for household_id, candidate_index in group.groupby(schema.HOUSEHOLD_ID, sort=False).groups.items():
            household_history = history[history[schema.HOUSEHOLD_ID].eq(int(household_id))]
            if household_history.empty:
                continue
            candidate_rows = out.loc[candidate_index, "_product_text_row"]
            valid_mask = candidate_rows.notna().to_numpy()
            if not valid_mask.any():
                continue

            history_rows = household_history["product_text_row"].tail(history_limit).to_numpy(dtype=int)
            if len(history_rows) == 0:
                continue
            candidate_matrix = matrix[candidate_rows[valid_mask].astype(int).to_numpy()]
            similarity = candidate_matrix @ matrix[history_rows].T
            similarity_array = similarity.toarray() if hasattr(similarity, "toarray") else np.asarray(similarity)
            max_similarity = similarity_array.max(axis=1)
            top_n = min(3, similarity_array.shape[1])
            if top_n == similarity_array.shape[1]:
                top3_similarity = similarity_array.mean(axis=1)
            else:
                top_values = np.partition(similarity_array, -top_n, axis=1)[:, -top_n:]
                top3_similarity = top_values.mean(axis=1)

            recent_rows = history_rows[-recent_limit:]
            recent_similarity = candidate_matrix @ matrix[recent_rows].T
            recent_array = recent_similarity.toarray() if hasattr(recent_similarity, "toarray") else np.asarray(recent_similarity)
            recent_max_similarity = recent_array.max(axis=1)

            valid_index = pd.Index(candidate_index)[valid_mask]
            out.loc[valid_index, "text_match_max_similarity"] = np.clip(max_similarity, 0.0, 1.0)
            out.loc[valid_index, "text_match_top3_similarity"] = np.clip(top3_similarity, 0.0, 1.0)
            out.loc[valid_index, "text_match_recent_max_similarity"] = np.clip(recent_max_similarity, 0.0, 1.0)
            out.loc[valid_index, "text_match_history_count_log"] = math.log1p(float(len(household_history)))
            out.loc[valid_index, "text_match_has_profile"] = 1.0

    out = out.drop(columns=["_product_text_row"])
    for feature in XGB_TEXT_MATCH_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_text_embedding_features(
    features: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    components: int = 32,
    max_features: int = 4096,
) -> pd.DataFrame:
    """Add product-text profile features from leakage-safe household history."""

    out = features.copy()
    for feature in XGB_TEXT_EMBEDDING_FEATURES:
        out[feature] = 0.0

    embeddings, embedding_columns = _product_text_embeddings(sources["products"], components, max_features)
    product_embedding_columns = [f"product_{column}" for column in embedding_columns]
    profile_embedding_columns = [f"profile_{column}" for column in embedding_columns]

    product_embeddings = embeddings.rename(
        columns={column: f"product_{column}" for column in embedding_columns}
    )
    transaction_embeddings = embeddings.rename(
        columns={column: f"profile_{column}" for column in embedding_columns}
    )
    transactions = sources["transactions"][
        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]
    ].copy()
    transactions = transactions.merge(transaction_embeddings, on=schema.PRODUCT_ID, how="left")
    for column in profile_embedding_columns:
        transactions[column] = pd.to_numeric(transactions[column], errors="coerce").fillna(0.0)

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

        profile = history.groupby(schema.HOUSEHOLD_ID)[profile_embedding_columns].mean().reset_index()
        counts = history.groupby(schema.HOUSEHOLD_ID).size().rename("text_embedding_history_count").reset_index()
        profile = profile.merge(counts, on=schema.HOUSEHOLD_ID, how="left")
        update = group[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]].merge(profile, on=schema.HOUSEHOLD_ID, how="left")
        update = update.merge(product_embeddings, on=schema.PRODUCT_ID, how="left")

        profile_matrix = update[profile_embedding_columns].fillna(0.0).to_numpy(dtype=np.float32)
        product_matrix = update[product_embedding_columns].fillna(0.0).to_numpy(dtype=np.float32)
        profile_norm = np.linalg.norm(profile_matrix, axis=1)
        product_norm = np.linalg.norm(product_matrix, axis=1)
        denominator = np.maximum(profile_norm * product_norm, 1e-6)
        similarity = np.sum(profile_matrix * product_matrix, axis=1) / denominator
        history_count = pd.to_numeric(
            update["text_embedding_history_count"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)

        out.loc[index, "text_embedding_similarity"] = np.clip(similarity, -1.0, 1.0)
        out.loc[index, "text_embedding_profile_norm"] = profile_norm
        out.loc[index, "text_embedding_history_count_log"] = np.log1p(history_count)
        out.loc[index, "text_embedding_has_profile"] = (history_count > 0).astype(np.float32)

    for feature in XGB_TEXT_EMBEDDING_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _category_embeddings(
    sources: dict[str, pd.DataFrame],
    components: int,
) -> tuple[pd.DataFrame, list[str]]:
    if components < 1:
        raise ValueError("--category-embedding-components must be >= 1.")

    try:
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.preprocessing import normalize
    except Exception as exc:
        raise RuntimeError("scikit-learn and scipy are required for --use-category-embedding-features.") from exc

    products = sources["products"][[schema.PRODUCT_ID, "product_category"]].drop_duplicates(schema.PRODUCT_ID).copy()
    products["product_category"] = _clean_text_column(products, "product_category")
    val_start = pd.Timestamp(sources["val"]["transaction_timestamp"].min())
    history = sources["transactions"][
        sources["transactions"]["transaction_timestamp"] < val_start
    ][[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]].copy()
    history = history.merge(products, on=schema.PRODUCT_ID, how="left")
    history["product_category"] = _clean_text_column(history, "product_category")
    counts = history.groupby([schema.HOUSEHOLD_ID, "product_category"]).size().reset_index(name="count")
    if counts.empty:
        return pd.DataFrame({"product_category": [], "category_emb_0": []}), ["category_emb_0"]

    households = pd.Index(sorted(counts[schema.HOUSEHOLD_ID].astype(int).unique()))
    categories = pd.Index(sorted(counts["product_category"].astype(str).unique()))
    household_codes = households.get_indexer(counts[schema.HOUSEHOLD_ID].astype(int))
    category_codes = categories.get_indexer(counts["product_category"].astype(str))
    matrix = sparse.csr_matrix(
        (
            counts["count"].astype(float).to_numpy(),
            (household_codes, category_codes),
        ),
        shape=(len(households), len(categories)),
    )
    weighted = TfidfTransformer(sublinear_tf=True).fit_transform(matrix)
    if min(weighted.shape) <= 1:
        embedding = normalize(weighted.T, norm="l2", copy=True).toarray().astype(np.float32)
    else:
        n_components = min(int(components), max(1, min(weighted.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        # components_.T represents category coordinates in the latent household-category space.
        svd.fit(weighted)
        embedding = normalize(svd.components_.T, norm="l2", copy=False).astype(np.float32)

    columns = [f"category_emb_{idx}" for idx in range(embedding.shape[1])]
    out = pd.DataFrame(embedding, columns=columns)
    out.insert(0, "product_category", categories.astype(str).to_numpy())
    return out, columns


def add_category_embedding_features(
    features: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    components: int = 24,
) -> pd.DataFrame:
    """Add latent category-neighbor features from training-period category co-occurrence."""

    out = features.copy()
    for feature in XGB_CATEGORY_EMBEDDING_FEATURES:
        out[feature] = 0.0

    category_embeddings, embedding_columns = _category_embeddings(sources, components)
    products = sources["products"][[schema.PRODUCT_ID, "product_category"]].drop_duplicates(schema.PRODUCT_ID).copy()
    products["product_category"] = _clean_text_column(products, "product_category")
    product_categories = products.set_index(schema.PRODUCT_ID)["product_category"].to_dict()
    embedding_map = {
        str(row["product_category"]): row[embedding_columns].to_numpy(dtype=np.float32)
        for _, row in category_embeddings.iterrows()
    }
    zero_vector = np.zeros(len(embedding_columns), dtype=np.float32)
    transactions = sources["transactions"][
        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]
    ].copy()
    transactions = transactions.merge(products, on=schema.PRODUCT_ID, how="left")
    transactions["product_category"] = _clean_text_column(transactions, "product_category")

    for campaign_id, index in out.groupby("campaign_id", sort=True).groups.items():
        group = out.loc[index].copy()
        start = pd.Timestamp(group["coupon_start_date"].iloc[0])
        households = set(group[schema.HOUSEHOLD_ID].astype(int))
        history = transactions[
            transactions[schema.HOUSEHOLD_ID].astype(int).isin(households)
            & (transactions["transaction_timestamp"] < start)
        ].copy()
        if history.empty:
            continue

        category_counts = (
            history.groupby([schema.HOUSEHOLD_ID, "product_category"])
            .size()
            .reset_index(name="count")
        )
        profile_map: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for household_id, household_counts in category_counts.groupby(schema.HOUSEHOLD_ID, sort=False):
            categories = household_counts["product_category"].astype(str).tolist()
            vectors = np.vstack([embedding_map.get(category, zero_vector) for category in categories]).astype(np.float32)
            weights = np.log1p(household_counts["count"].to_numpy(dtype=np.float32))
            total_weight = float(weights.sum())
            if total_weight <= 0.0:
                profile = zero_vector.copy()
            else:
                profile = np.average(vectors, axis=0, weights=weights).astype(np.float32)
            profile_map[int(household_id)] = (profile, vectors, float(household_counts["count"].sum()))

        profile_similarity = np.zeros(len(group), dtype=np.float32)
        max_similarity = np.zeros(len(group), dtype=np.float32)
        profile_norms = np.zeros(len(group), dtype=np.float32)
        history_counts = np.zeros(len(group), dtype=np.float32)
        candidate_categories = group[schema.PRODUCT_ID].astype(int).map(product_categories).fillna("UNKNOWN").astype(str)
        candidate_vectors = np.vstack([embedding_map.get(category, zero_vector) for category in candidate_categories])

        for position, (household_id, candidate_vector) in enumerate(
            zip(group[schema.HOUSEHOLD_ID].astype(int).to_numpy(), candidate_vectors, strict=False)
        ):
            if int(household_id) not in profile_map:
                continue
            profile, history_vectors, history_count = profile_map[int(household_id)]
            candidate_norm = float(np.linalg.norm(candidate_vector))
            profile_norm = float(np.linalg.norm(profile))
            history_counts[position] = history_count
            profile_norms[position] = profile_norm
            if candidate_norm > 0.0 and profile_norm > 0.0:
                profile_similarity[position] = float(np.dot(candidate_vector, profile) / (candidate_norm * profile_norm))
            if candidate_norm > 0.0 and len(history_vectors) > 0:
                similarities = history_vectors @ candidate_vector / np.maximum(
                    np.linalg.norm(history_vectors, axis=1) * candidate_norm,
                    1e-6,
                )
                max_similarity[position] = float(np.max(similarities))

        out.loc[index, "category_embedding_profile_similarity"] = np.clip(profile_similarity, -1.0, 1.0)
        out.loc[index, "category_embedding_max_similarity"] = np.clip(max_similarity, -1.0, 1.0)
        out.loc[index, "category_embedding_profile_norm"] = profile_norms
        out.loc[index, "category_embedding_history_count_log"] = np.log1p(history_counts)
        out.loc[index, "category_embedding_has_profile"] = (history_counts > 0).astype(np.float32)

    for feature in XGB_CATEGORY_EMBEDDING_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_event_category_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add event-local category concentration features.

    These features describe the coupon candidate set available in one
    household-campaign event and do not use future purchase labels.
    """

    out = features.copy()
    category = _clean_text_column(out, "product_category")
    out["_event_category"] = category
    out["_global_signal"] = pd.to_numeric(out["global_signal"], errors="coerce").fillna(0.0)
    event_size = out.groupby(base.EVENT_COL)[schema.PRODUCT_ID].transform("size").clip(lower=1)
    category_group = out.groupby([base.EVENT_COL, "_event_category"])
    category_count = category_group[schema.PRODUCT_ID].transform("size")
    category_global_mean = category_group["_global_signal"].transform("mean")
    category_global_max = category_group["_global_signal"].transform("max")

    category_summary = (
        out[[base.EVENT_COL, "_event_category"]]
        .assign(_category_global_mean=category_global_mean)
        .drop_duplicates([base.EVENT_COL, "_event_category"])
    )
    category_summary["_event_category_rank"] = category_summary.groupby(base.EVENT_COL)["_category_global_mean"].rank(
        method="dense",
        ascending=False,
    )
    category_summary["_event_category_total"] = category_summary.groupby(base.EVENT_COL)["_event_category"].transform("size")
    category_summary["event_category_global_rank_pct"] = 1.0 - (
        (category_summary["_event_category_rank"] - 1.0)
        / category_summary["_event_category_total"].sub(1.0).replace(0.0, 1.0)
    )

    out["event_category_count_log"] = np.log1p(category_count.to_numpy(dtype=float))
    out["event_category_share"] = category_count.to_numpy(dtype=float) / event_size.to_numpy(dtype=float)
    out["event_category_global_mean"] = category_global_mean.to_numpy(dtype=float)
    out["event_category_global_max"] = category_global_max.to_numpy(dtype=float)
    out = out.merge(
        category_summary[[base.EVENT_COL, "_event_category", "event_category_global_rank_pct"]],
        on=[base.EVENT_COL, "_event_category"],
        how="left",
    )
    for feature in XGB_EVENT_CATEGORY_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.drop(columns=["_event_category", "_global_signal"])


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
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
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
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
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
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
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
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
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
        min_child_weight=float(config.get("min_child_weight", 2.0)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        random_state=seed,
        tree_method="hist",
        device=device,
    )
    ranker.fit(train_x, train_y, group=train_groups, verbose=False)
    return ranker


def _config_from_row(row: dict[str, object]) -> dict[str, float | int | str | bool]:
    return {
        "n_estimators": int(row["n_estimators"]),
        "learning_rate": float(row["learning_rate"]),
        "max_depth": int(row["max_depth"]),
        "objective": str(row["objective"]),
        "positive_train_events_only": bool(row["positive_train_events_only"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "min_child_weight": float(row.get("min_child_weight", 2.0)),
        "reg_lambda": float(row.get("reg_lambda", 1.0)),
    }


def _select_ensemble_configs(
    search_frame: pd.DataFrame,
    primary_metric: str,
    ensemble_top_n: int,
) -> list[dict[str, float | int | str | bool]]:
    if ensemble_top_n < 1:
        raise ValueError("--ensemble-top-n must be >= 1.")
    ordered = search_frame.sort_values(
        [primary_metric, "ndcg_at_10", "recall_at_10", "max_depth", "n_estimators"],
        ascending=[False, False, False, True, True],
    )
    return [_config_from_row(row) for row in ordered.head(ensemble_top_n).to_dict("records")]


def _average_ranker_scores(
    xgb,
    configs: list[dict[str, float | int | str | bool]],
    device: str,
    seed: int,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    score_x: pd.DataFrame,
    score_ordered: pd.DataFrame,
) -> tuple[np.ndarray, list[object]]:
    scores = []
    rankers = []
    for config_index, config in enumerate(configs):
        config_train = _filter_positive_event_groups(train_frame, bool(config.get("positive_train_events_only", False)))
        train_x, train_y, train_groups, _ = _grouped_xy(config_train, feature_columns)
        ranker = _fit_ranker(xgb, config, device, seed + config_index, train_x, train_y, train_groups)
        rankers.append(ranker)
        scores.append(_normalize_scores_by_event(score_ordered, ranker.predict(score_x)))
    if not scores:
        return np.zeros(len(score_ordered), dtype=np.float32), rankers
    return np.mean(np.vstack(scores), axis=0).astype(np.float32), rankers


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
    if args.ensemble_top_n < 1:
        raise ValueError("--ensemble-top-n must be >= 1.")
    if args.ensemble_top_n > 1 and (args.search_score_blend or args.search_rank_fusion):
        raise ValueError("--ensemble-top-n currently supports direct XGBoost score averaging only.")

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
    if args.use_text_embedding_features:
        features = add_text_embedding_features(
            features,
            sources,
            args.text_embedding_components,
            args.text_max_features,
        )
    if args.use_text_match_features:
        features = add_text_match_features(
            features,
            sources,
            args.text_match_max_features,
            args.text_match_history_products,
            args.text_match_recent_products,
        )
    if args.use_category_embedding_features:
        features = add_category_embedding_features(features, sources, args.category_embedding_components)
    if args.use_event_category_features:
        features = add_event_category_features(features)
    feature_columns = _feature_columns(
        args.use_response_priors,
        args.use_content_features,
        args.use_text_embedding_features,
        args.use_text_match_features,
        args.use_category_embedding_features,
        args.use_event_category_features,
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
        row["ensemble_top_n"] = args.ensemble_top_n
        row["final_train_scope"] = args.final_train_scope
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
            "min_child_weight",
            "reg_lambda",
        ]
    }
    best_blend_weight = float(best_row.get("xgb_blend_weight", 1.0))
    ensemble_configs = _select_ensemble_configs(search_frame, args.primary_metric, args.ensemble_top_n)
    best_rank_fusion_config = {
        "rank_fusion_method": best_row.get("rank_fusion_method", "none"),
        "rank_fusion_c": float(best_row.get("rank_fusion_c", 0.0)),
        "rank_fusion_xgb_weight": float(best_row.get("rank_fusion_xgb_weight", 1.0)),
        "rank_fusion_heuristic_weight": float(best_row.get("rank_fusion_heuristic_weight", 0.0)),
        "rank_fusion_base_weight": float(best_row.get("rank_fusion_base_weight", 0.0)),
        "rank_fusion_global_weight": float(best_row.get("rank_fusion_global_weight", 0.0)),
    }
    final_rankers: list[object]
    if args.ensemble_top_n > 1:
        val_final_scores, _ = _average_ranker_scores(
            xgb,
            ensemble_configs,
            device,
            args.seed,
            train,
            feature_columns,
            val_x,
            val_ordered,
        )
    else:
        best_train = _filter_positive_event_groups(train, bool(best_config.get("positive_train_events_only", False)))
        train_x, train_y, train_groups, _ = _grouped_xy(best_train, feature_columns)
        best_ranker = _fit_ranker(xgb, best_config, device, args.seed, train_x, train_y, train_groups)
        val_scores = best_ranker.predict(val_x)
        if str(best_rank_fusion_config.get("rank_fusion_method", "none")) != "none":
            val_final_scores = _rank_fusion_scores(val_ordered, val_scores, val_heuristic_scores, best_rank_fusion_config)
        else:
            val_final_scores = _blend_scores(val_ordered, val_scores, val_heuristic_scores, best_blend_weight)
    val_ranked = _rank_scores(val_ordered, val_final_scores, max_k)

    if args.final_train_scope == "train":
        final_train = train.copy()
    else:
        final_train = pd.concat([train, validation], ignore_index=True)
    if args.ensemble_top_n > 1:
        test_final_scores, final_rankers = _average_ranker_scores(
            xgb,
            ensemble_configs,
            device,
            args.seed,
            final_train,
            feature_columns,
            test_x,
            test_ordered,
        )
    else:
        final_train = _filter_positive_event_groups(final_train, bool(best_config.get("positive_train_events_only", False)))
        final_train_x, final_train_y, final_train_groups, _ = _grouped_xy(final_train, feature_columns)
        final_ranker = _fit_ranker(xgb, best_config, device, args.seed, final_train_x, final_train_y, final_train_groups)
        final_rankers = [final_ranker]
        test_scores = final_ranker.predict(test_x)
        if str(best_rank_fusion_config.get("rank_fusion_method", "none")) != "none":
            test_final_scores = _rank_fusion_scores(test_ordered, test_scores, test_heuristic_scores, best_rank_fusion_config)
        else:
            test_final_scores = _blend_scores(test_ordered, test_scores, test_heuristic_scores, best_blend_weight)
    test_ranked = _rank_scores(test_ordered, test_final_scores, max_k)

    comparison_rows = []
    for split, ranked, split_events, split_truth, trained_on in [
        ("validation", val_ranked, val_events, val_truth, "train"),
        ("test", test_ranked, test_events, test_truth, args.final_train_scope),
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
            "ensemble_top_n": args.ensemble_top_n,
            "final_train_scope": args.final_train_scope,
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

    importances = np.vstack([ranker.feature_importances_ for ranker in final_rankers])
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances.mean(axis=0),
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
