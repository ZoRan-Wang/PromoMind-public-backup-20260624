"""Train an XGBoost learning-to-rank coupon-response model.

The model uses household-campaign groups and binary coupon-response labels:
an item is relevant if it is bought within five days after campaign start.
XGBoost is optional and can use CUDA when available.
"""

from __future__ import annotations

import argparse
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
XGB_EXTRA_FEATURES = [
    "product_response_prior",
    "category_response_prior",
    "campaign_type_response_prior",
    "product_positive_count_log",
    "category_positive_count_log",
    "campaign_type_positive_count_log",
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
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--search", action="store_true", help="Tune XGBoost configs on validation before final test.")
    parser.add_argument("--primary-metric", default="recall_at_20")
    parser.add_argument("--use-response-priors", action="store_true", help="Add train-period product/category response priors.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_columns() -> list[str]:
    return neural.FEATURE_COLUMNS + XGB_EXTRA_FEATURES


def _grouped_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    ordered = frame.sort_values([base.EVENT_COL, schema.PRODUCT_ID]).reset_index(drop=True)
    groups = ordered.groupby(base.EVENT_COL, sort=False).size().astype(int).tolist()
    x = ordered[_feature_columns()]
    y = ordered["label"].astype(float)
    return x, y, groups, ordered


def _rank_scores(frame: pd.DataFrame, scores, k: int) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["final_score"] = scores
    ranked = ranked.sort_values([base.EVENT_COL, "final_score", schema.PRODUCT_ID], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby(base.EVENT_COL).cumcount() + 1
    return ranked[ranked["rank"] <= k].copy()


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


def add_zero_response_prior_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for column in XGB_EXTRA_FEATURES:
        out[column] = 0.0
    return out


def _candidate_configs(args: argparse.Namespace) -> list[dict[str, float | int]]:
    if not args.search:
        return [
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            }
        ]
    return [
        {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
        for n_estimators, learning_rate, max_depth in [
            (80, 0.08, 3),
            (120, 0.03, 2),
            (120, 0.03, 3),
            (120, 0.05, 3),
            (180, 0.03, 2),
            (180, 0.05, 2),
            (250, 0.015, 3),
            (250, 0.03, 2),
            (500, 0.03, 4),
        ]
    ]


def _fit_ranker(xgb, config: dict[str, float | int], device: str, seed: int, train_x, train_y, train_groups):
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
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
    config: dict[str, float | int],
    device: str,
    seed: int,
    train_x,
    train_y,
    train_groups,
    val_x,
    val_ordered: pd.DataFrame,
    val_events: pd.DataFrame,
    val_truth: pd.DataFrame,
    eval_ks: list[int],
) -> tuple[object, dict[str, float]]:
    ranker = _fit_ranker(xgb, config, device, seed, train_x, train_y, train_groups)
    val_scores = ranker.predict(val_x)
    val_ranked = _rank_scores(val_ordered, val_scores, max(eval_ks))
    row = {"model_name": MODEL_NAME, "split": "validation", "device": device}
    row.update(config)
    row.update(base.evaluate_ranked(val_ranked, val_truth, val_events, eval_ks))
    return ranker, row


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
    if args.use_response_priors:
        features = add_xgb_response_prior_features(features)
    else:
        features = add_zero_response_prior_features(features)

    train = features[features["split"] == "train"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    test = features[features["split"] == "test"].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Train/validation/test features are required.")

    train_x, train_y, train_groups, train_ordered = _grouped_xy(train)
    val_x, val_y, val_groups, val_ordered = _grouped_xy(validation)
    test_x, test_y, test_groups, test_ordered = _grouped_xy(test)

    val_events = events[events["split"] == "validation"].copy()
    test_events = events[events["split"] == "test"].copy()
    val_truth = truth[truth[base.EVENT_COL].isin(set(val_events[base.EVENT_COL]))].copy()
    test_truth = truth[truth[base.EVENT_COL].isin(set(test_events[base.EVENT_COL]))].copy()

    search_rows = []
    best_row = None
    best_ranker = None
    configs = _candidate_configs(args)
    for config in configs:
        ranker, row = _evaluate_config(
            xgb,
            config,
            device,
            args.seed,
            train_x,
            train_y,
            train_groups,
            val_x,
            val_ordered,
            val_events,
            val_truth,
            eval_ks,
        )
        search_rows.append(row)
        if best_row is None:
            best_row = row
            best_ranker = ranker
            continue
        best_key = (
            float(best_row.get(args.primary_metric, 0.0)),
            float(best_row.get("positive_event_hit_rate_at_10", 0.0)),
            float(best_row.get("recall_at_10", 0.0)),
        )
        row_key = (
            float(row.get(args.primary_metric, 0.0)),
            float(row.get("positive_event_hit_rate_at_10", 0.0)),
            float(row.get("recall_at_10", 0.0)),
        )
        if row_key > best_key:
            best_row = row
            best_ranker = ranker

    if best_row is None or best_ranker is None:
        raise RuntimeError("No XGBoost configs were trained.")

    best_config = {
        key: best_row[key]
        for key in ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"]
    }
    val_scores = best_ranker.predict(val_x)
    val_ranked = _rank_scores(val_ordered, val_scores, max_k)

    final_train = pd.concat([train, validation], ignore_index=True)
    final_train_x, final_train_y, final_train_groups, _ = _grouped_xy(final_train)
    final_ranker = _fit_ranker(xgb, best_config, device, args.seed, final_train_x, final_train_y, final_train_groups)
    test_scores = final_ranker.predict(test_x)
    test_ranked = _rank_scores(test_ordered, test_scores, max_k)

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
            "feature": _feature_columns(),
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
