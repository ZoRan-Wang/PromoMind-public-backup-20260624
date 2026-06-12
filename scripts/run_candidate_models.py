"""Run PromoMind candidate-generation models from processed interaction files.

Inputs:
  data/processed/train_interactions.csv
  data/processed/valid_interactions.csv or test_interactions.csv
  data/processed/product_features.csv

Outputs:
  outputs/candidates_MODEL.csv
  outputs/model_comparison.csv
  outputs/als_tuning_results.csv
  outputs/bpr_tuning_results.csv when requested
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402
from promomind.evaluation.ranking import coverage, diversity, ndcg_at_k, novelty, recall_at_k  # noqa: E402
from promomind.models.als import ImplicitALSRecommender  # noqa: E402
from promomind.models.baselines import (  # noqa: E402
    category_popularity_recommendations,
    popularity_recommendations,
)
from promomind.models.bpr import BPRRecommender  # noqa: E402
from promomind.models.candidates import ITEM_COL, RANK_COL, SCORE_COL, USER_COL  # noqa: E402
from promomind.models.itemknn import ItemKNNRecommender  # noqa: E402


CATEGORY_CANDIDATES = [
    "product_category",
    "commodity_desc",
    "sub_commodity_desc",
    "department",
    "category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PromoMind candidate-generation models.")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--train-file", default="train_interactions.csv")
    parser.add_argument("--eval-file", default="valid_interactions.csv")
    parser.add_argument("--product-features-file", default="product_features.csv")
    parser.add_argument(
        "--models",
        default="popularity,category,itemknn,als",
        help="Comma-separated list from popularity,category,itemknn,als,bpr,all.",
    )
    parser.add_argument("--k", type=int, default=50, help="Number of candidates per household.")
    parser.add_argument("--eval-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--weight-col", default=schema.QUANTITY)
    parser.add_argument("--category-col", default="auto")
    parser.add_argument("--itemknn-neighbors", type=int, default=100)
    parser.add_argument(
        "--als-grid",
        default="16:0.05:3:10,32:0.05:5:20",
        help="ALS grid as factors:regularization:iterations:alpha entries.",
    )
    parser.add_argument("--als-backend", choices=["auto", "implicit", "native"], default="auto")
    parser.add_argument(
        "--bpr-grid",
        default="16:0.03:0.01:3,32:0.03:0.01:5",
        help="BPR grid as factors:learning_rate:regularization:epochs entries.",
    )
    parser.add_argument("--bpr-samples-per-epoch", type=int, default=None)
    return parser.parse_args()


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_dataset.py or scripts/clean_completejourney.py first."
        )
    return pd.read_csv(path)


def _existing_weight_col(frame: pd.DataFrame, requested: str | None) -> str | None:
    return requested if requested and requested in frame.columns else None


def _parse_model_list(value: str) -> list[str]:
    models = [part.strip().lower() for part in value.split(",") if part.strip()]
    if "all" in models:
        return ["popularity", "category", "itemknn", "als", "bpr"]
    valid = {"popularity", "category", "itemknn", "als", "bpr"}
    invalid = sorted(set(models) - valid)
    if invalid:
        raise ValueError(f"Unknown models: {invalid}")
    return models


def _parse_als_grid(value: str) -> list[dict[str, float | int]]:
    params = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        factors, regularization, iterations, alpha = entry.split(":")
        params.append(
            {
                "factors": int(factors),
                "regularization": float(regularization),
                "iterations": int(iterations),
                "alpha": float(alpha),
            }
        )
    return params


def _parse_bpr_grid(value: str) -> list[dict[str, float | int]]:
    params = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        factors, learning_rate, regularization, epochs = entry.split(":")
        params.append(
            {
                "factors": int(factors),
                "learning_rate": float(learning_rate),
                "regularization": float(regularization),
                "epochs": int(epochs),
            }
        )
    return params


def _evaluation_users(train: pd.DataFrame, eval_frame: pd.DataFrame) -> list:
    train_users = set(train[schema.HOUSEHOLD_ID].dropna().unique())
    eval_users = set(eval_frame[schema.HOUSEHOLD_ID].dropna().unique())
    users = sorted(train_users & eval_users, key=lambda value: str(value))
    if not users:
        users = sorted(train_users, key=lambda value: str(value))
    return users


def _canonical_truth(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]]
        .dropna()
        .drop_duplicates()
        .rename(columns={schema.HOUSEHOLD_ID: USER_COL, schema.PRODUCT_ID: ITEM_COL})
    )


def _canonical_train(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={schema.HOUSEHOLD_ID: USER_COL, schema.PRODUCT_ID: ITEM_COL})


def _load_item_features(path: Path, train: pd.DataFrame, category_col: str) -> tuple[pd.DataFrame, str]:
    if path.exists():
        features = pd.read_csv(path)
    else:
        features = train[[schema.PRODUCT_ID]].drop_duplicates().copy()
        features["category"] = "unknown"

    if schema.PRODUCT_ID not in features.columns and ITEM_COL in features.columns:
        features = features.rename(columns={ITEM_COL: schema.PRODUCT_ID})
    if schema.PRODUCT_ID not in features.columns:
        raise ValueError(f"{path} must contain {schema.PRODUCT_ID} or {ITEM_COL}.")

    if category_col == "auto":
        selected = next((col for col in CATEGORY_CANDIDATES if col in features.columns), None)
    else:
        selected = category_col if category_col in features.columns else None
    if selected is None:
        selected = "category"
        features[selected] = "unknown"

    features[selected] = features[selected].fillna("unknown").astype(str)
    return features, selected


def _features_for_metrics(features: pd.DataFrame, category_col: str) -> pd.DataFrame:
    return (
        features[[schema.PRODUCT_ID, category_col]]
        .drop_duplicates()
        .rename(columns={schema.PRODUCT_ID: ITEM_COL, category_col: "category"})
    )


def _evaluate(
    model_name: str,
    candidates: pd.DataFrame,
    eval_frame: pd.DataFrame,
    train: pd.DataFrame,
    item_features: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {"model_name": model_name, "n_candidates": len(candidates)}
    truth = _canonical_truth(eval_frame)
    train_canonical = _canonical_train(train)
    catalog = train_canonical[ITEM_COL].dropna().unique()

    for k in [10, 20]:
        row[f"recall_at_{k}"] = recall_at_k(candidates, truth, k=k)
        row[f"ndcg_at_{k}"] = ndcg_at_k(candidates, truth, k=k)
    row["coverage_at_20"] = coverage(candidates, catalog, k=20)
    row["diversity_at_20"] = diversity(candidates, item_features, k=20, category_col="category")
    row["novelty_at_20"] = novelty(candidates, train_canonical, k=20)
    return row


def _write_candidates(candidates: pd.DataFrame, model_name: str, path: Path) -> None:
    out = candidates.rename(
        columns={
            USER_COL: schema.HOUSEHOLD_ID,
            ITEM_COL: schema.PRODUCT_ID,
            SCORE_COL: "base_score",
            RANK_COL: "base_rank",
        }
    ).copy()
    out["model_name"] = model_name
    ordered = [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "base_score", "model_name", "base_rank"]
    extras = [col for col in out.columns if col not in ordered]
    out = out[ordered + extras]
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


def _select_best(rows: list[dict[str, object]]) -> int:
    scores = [
        (
            float(row.get("ndcg_at_10", 0.0)),
            float(row.get("recall_at_10", 0.0)),
            -idx,
        )
        for idx, row in enumerate(rows)
    ]
    return -max(scores)[2]


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    train = _read_required_csv(args.processed_dir / args.train_file)
    eval_frame = _read_required_csv(args.processed_dir / args.eval_file)
    product_features, category_col = _load_item_features(
        args.processed_dir / args.product_features_file,
        train,
        args.category_col,
    )
    metric_features = _features_for_metrics(product_features, category_col)

    users = _evaluation_users(train, eval_frame)
    weight_col = _existing_weight_col(train, args.weight_col)
    models = _parse_model_list(args.models)
    comparison_rows: list[dict[str, object]] = []

    if "popularity" in models:
        candidates = popularity_recommendations(
            train,
            users=users,
            k=args.k,
            user_col=schema.HOUSEHOLD_ID,
            item_col=schema.PRODUCT_ID,
            weight_col=weight_col,
        )
        _write_candidates(candidates, "popularity", args.outputs_dir / "candidates_popularity.csv")
        comparison_rows.append(_evaluate("popularity", candidates, eval_frame, train, metric_features))

    if "category" in models:
        candidates = category_popularity_recommendations(
            train,
            product_features,
            users=users,
            k=args.k,
            user_col=schema.HOUSEHOLD_ID,
            item_col=schema.PRODUCT_ID,
            category_col=category_col,
            weight_col=weight_col,
        )
        _write_candidates(
            candidates,
            "category_popularity",
            args.outputs_dir / "candidates_category_popularity.csv",
        )
        comparison_rows.append(
            _evaluate("category_popularity", candidates, eval_frame, train, metric_features)
        )

    if "itemknn" in models:
        model = ItemKNNRecommender(max_similar_items=args.itemknn_neighbors)
        model.fit(
            train,
            user_col=schema.HOUSEHOLD_ID,
            item_col=schema.PRODUCT_ID,
            weight_col=weight_col,
        )
        candidates = model.recommend(users, k=args.k)
        _write_candidates(candidates, "itemknn", args.outputs_dir / "candidates_itemknn.csv")
        comparison_rows.append(_evaluate("itemknn", candidates, eval_frame, train, metric_features))

    if "als" in models:
        tuning_rows: list[dict[str, object]] = []
        candidate_runs: list[pd.DataFrame] = []
        for params in _parse_als_grid(args.als_grid):
            model = ImplicitALSRecommender(**params, backend=args.als_backend)
            model.fit(
                train,
                user_col=schema.HOUSEHOLD_ID,
                item_col=schema.PRODUCT_ID,
                weight_col=weight_col,
            )
            candidates = model.recommend(users, k=args.k)
            row = _evaluate("als", candidates, eval_frame, train, metric_features)
            row.update(params)
            row["backend"] = model.backend_
            tuning_rows.append(row)
            candidate_runs.append(candidates)

        best_idx = _select_best(tuning_rows)
        best_candidates = candidate_runs[best_idx]
        _write_candidates(best_candidates, "als", args.outputs_dir / "candidates_als.csv")
        comparison_rows.append({"model_name": "als", **tuning_rows[best_idx]})
        pd.DataFrame(tuning_rows).to_csv(args.outputs_dir / "als_tuning_results.csv", index=False)
        print(f"Wrote {args.outputs_dir / 'als_tuning_results.csv'} ({len(tuning_rows)} rows)")

    if "bpr" in models:
        tuning_rows = []
        candidate_runs = []
        for params in _parse_bpr_grid(args.bpr_grid):
            model = BPRRecommender(
                **params,
                samples_per_epoch=args.bpr_samples_per_epoch,
            )
            model.fit(
                train,
                user_col=schema.HOUSEHOLD_ID,
                item_col=schema.PRODUCT_ID,
                weight_col=weight_col,
            )
            candidates = model.recommend(users, k=args.k)
            row = _evaluate("bpr", candidates, eval_frame, train, metric_features)
            row.update(params)
            tuning_rows.append(row)
            candidate_runs.append(candidates)

        best_idx = _select_best(tuning_rows)
        best_candidates = candidate_runs[best_idx]
        _write_candidates(best_candidates, "bpr", args.outputs_dir / "candidates_bpr.csv")
        comparison_rows.append({"model_name": "bpr", **tuning_rows[best_idx]})
        pd.DataFrame(tuning_rows).to_csv(args.outputs_dir / "bpr_tuning_results.csv", index=False)
        print(f"Wrote {args.outputs_dir / 'bpr_tuning_results.csv'} ({len(tuning_rows)} rows)")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.outputs_dir / "model_comparison.csv", index=False)
    print(f"Wrote {args.outputs_dir / 'model_comparison.csv'} ({len(comparison)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
