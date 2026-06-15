"""Train a GPU neural coupon-response ranker.

This script extends the heuristic coupon-response ranker with a supervised
pairwise model. It uses earlier campaigns for training, validation campaigns for
early stopping, and held-out test campaigns for the final comparison.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in [str(SCRIPT_ROOT), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_coupon_response_ranker as base  # noqa: E402
from promomind.data import schema  # noqa: E402

MODEL_NAME = "coupon_response_neural_ranker"
FEATURE_COLUMNS = [
    "base_signal",
    "repeat_signal",
    "cadence_signal",
    "category_signal",
    "global_signal",
    "discount_signal",
    "user_product_count_log",
    "days_since_last_log",
    "median_interval_log",
    "has_prior_product",
    "has_base_candidate",
    "campaign_type_a",
    "campaign_type_b",
    "campaign_type_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPU neural coupon-response ranker.")
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pairs-per-positive", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_all_events(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    campaigns = sources["campaigns"].copy()
    raw_start = pd.Timestamp(sources["transactions"]["transaction_timestamp"].min())
    val_start = pd.Timestamp(sources["val"]["transaction_timestamp"].min())
    test_start = pd.Timestamp(sources["test"]["transaction_timestamp"].min())
    max_observed = pd.Timestamp(sources["transactions"]["transaction_timestamp"].max())
    campaigns["success_window_end"] = campaigns["start_date"] + pd.Timedelta(days=5)

    train_mask = (
        (campaigns["start_date"] >= raw_start)
        & (campaigns["success_window_end"] < val_start)
        & (campaigns["success_window_end"] <= max_observed)
    )
    val_mask = (campaigns["start_date"] >= val_start) & (campaigns["start_date"] < test_start)
    test_mask = (campaigns["start_date"] >= test_start) & (campaigns["success_window_end"] <= max_observed)
    events = pd.concat(
        [
            campaigns.loc[train_mask].assign(split="train"),
            campaigns.loc[val_mask].assign(split="validation"),
            campaigns.loc[test_mask].assign(split="test"),
        ],
        ignore_index=True,
    )
    events[base.EVENT_COL] = (
        events[schema.HOUSEHOLD_ID].astype(str)
        + "_"
        + events["campaign_id"].astype(str)
        + "_"
        + events["start_date"].dt.strftime("%Y%m%d")
    )
    return events[
        [
            base.EVENT_COL,
            "split",
            schema.HOUSEHOLD_ID,
            "campaign_id",
            "campaign_type",
            "start_date",
            "success_window_end",
        ]
    ].drop_duplicates(base.EVENT_COL)


def add_model_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for column in ["user_product_count", "days_since_last", "median_interval_days"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["user_product_count_log"] = np.log1p(out["user_product_count"].fillna(0.0).clip(lower=0.0))
    out["days_since_last_log"] = np.log1p(out["days_since_last"].fillna(365.0).clip(lower=0.0))
    out["median_interval_log"] = np.log1p(out["median_interval_days"].fillna(365.0).clip(lower=0.0))
    out["has_prior_product"] = (out["user_product_count"].fillna(0.0) > 0).astype(float)
    out["has_base_candidate"] = (pd.to_numeric(out["base_signal"], errors="coerce").fillna(0.0) > 0).astype(float)
    campaign_type = out["campaign_type"].fillna("").astype(str).str.upper()
    out["campaign_type_a"] = campaign_type.str.contains("A").astype(float)
    out["campaign_type_b"] = campaign_type.str.contains("B").astype(float)
    out["campaign_type_c"] = campaign_type.str.contains("C").astype(float)
    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def attach_labels(features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    labels = truth[[base.EVENT_COL, schema.PRODUCT_ID]].drop_duplicates().copy()
    labels["label"] = 1.0
    out = features.merge(labels, on=[base.EVENT_COL, schema.PRODUCT_ID], how="left")
    out["label"] = out["label"].fillna(0.0)
    return out


def normalize_features(train: pd.DataFrame, frames: list[pd.DataFrame]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    matrix = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = [((frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32) for frame in frames]
    return normalized, mean, std


def build_pair_indices(train: pd.DataFrame, pairs_per_positive: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_indices: list[int] = []
    neg_indices: list[int] = []
    for _, group in train.groupby(base.EVENT_COL, sort=False):
        positives = group.index[group["label"].to_numpy(dtype=float) > 0].to_numpy(dtype=np.int64)
        negatives = group.index[group["label"].to_numpy(dtype=float) <= 0].to_numpy(dtype=np.int64)
        if len(positives) == 0 or len(negatives) == 0:
            continue
        for positive in positives:
            sampled = rng.choice(negatives, size=min(pairs_per_positive, len(negatives)), replace=len(negatives) < pairs_per_positive)
            pos_indices.extend([int(positive)] * len(sampled))
            neg_indices.extend([int(item) for item in sampled])
    return np.asarray(pos_indices, dtype=np.int64), np.asarray(neg_indices, dtype=np.int64)


def resolve_device(requested: str) -> str:
    return base.resolve_device(requested)


def build_model(input_dim: int, hidden_dim: int, dropout: float):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, hidden_dim // 2),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim // 2, 1),
    )


def score_model(model: Any, matrix: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            batch = torch.as_tensor(matrix[start : start + batch_size], device=device)
            scores.append(model(batch).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32) if scores else np.array([], dtype=np.float32)


def rank_by_scores(features: pd.DataFrame, scores: np.ndarray, k: int) -> pd.DataFrame:
    ranked = features.copy()
    ranked["final_score"] = scores
    ranked = ranked.sort_values([base.EVENT_COL, "final_score", schema.PRODUCT_ID], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby(base.EVENT_COL).cumcount() + 1
    return ranked[ranked["rank"] <= k].copy()


def train_neural_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    val_events: pd.DataFrame,
    val_truth: pd.DataFrame,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    args: argparse.Namespace,
    device: str,
    eval_ks: list[int],
) -> tuple[Any, dict[str, float], list[dict[str, float]]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_reset = train.reset_index(drop=True)
    pos_idx, neg_idx = build_pair_indices(train_reset, args.pairs_per_positive, args.seed)
    if len(pos_idx) == 0:
        raise RuntimeError("No positive/negative training pairs were generated.")

    model = build_model(train_matrix.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    x_train = torch.as_tensor(train_matrix, device=device)
    pos_idx_t = torch.as_tensor(pos_idx, device=device)
    neg_idx_t = torch.as_tensor(neg_idx, device=device)

    best_metric = -math.inf
    best_state = None
    best_row: dict[str, float] = {}
    stale_epochs = 0
    history_rows: list[dict[str, float]] = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(pos_idx))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_order = torch.as_tensor(order[start : start + args.batch_size], device=device)
            batch_pos = pos_idx_t[batch_order]
            batch_neg = neg_idx_t[batch_order]
            pos_score = model(x_train[batch_pos]).squeeze(-1)
            neg_score = model(x_train[batch_neg]).squeeze(-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = score_model(model, val_matrix, device=device, batch_size=args.batch_size)
        val_ranked = rank_by_scores(validation, val_scores, k=max(eval_ks))
        metrics = base.evaluate_ranked(val_ranked, val_truth, val_events, eval_ks)
        row = {
            "epoch": float(epoch),
            "train_pair_loss": float(np.mean(losses)) if losses else 0.0,
            **metrics,
        }
        history_rows.append(row)
        metric = float(metrics.get("ndcg_at_10", 0.0))
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_row = row
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_row, history_rows


def load_or_build_features(args: argparse.Namespace, sources: dict[str, pd.DataFrame], events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = args.outputs_dir / "coupon_response_all_features.csv"
    truth_path = args.outputs_dir / "coupon_response_all_truth.csv"
    if args.reuse_features and feature_path.exists() and truth_path.exists():
        return base._read_csv(feature_path), base._read_csv(truth_path)
    truth = base.build_truth(events, sources["transactions"], sources["coupons"])
    features = base.build_feature_table(
        sources,
        events,
        max_global_products=args.max_global_products,
        max_candidates_per_event=args.max_candidates_per_event,
        top_categories=args.top_categories,
        category_products=args.category_products,
    )
    features.to_csv(feature_path, index=False)
    truth.to_csv(truth_path, index=False)
    return features, truth


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    eval_ks = sorted(set(args.eval_k))
    device = resolve_device(args.device)
    print(f"Neural scoring/training device: {device}")

    sources = base.load_sources(args)
    events = make_all_events(sources)
    features, truth = load_or_build_features(args, sources, events)
    features = attach_labels(add_model_features(features), truth)

    train = features[features["split"] == "train"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    test = features[features["split"] == "test"].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Train/validation/test features are required for neural ranking.")

    (train_matrix, val_matrix, test_matrix), mean, std = normalize_features(train, [train, validation, test])
    val_events = events[events["split"] == "validation"].copy()
    test_events = events[events["split"] == "test"].copy()
    val_truth = truth[truth[base.EVENT_COL].isin(set(val_events[base.EVENT_COL]))].copy()
    test_truth = truth[truth[base.EVENT_COL].isin(set(test_events[base.EVENT_COL]))].copy()

    model, best_val_row, history_rows = train_neural_ranker(
        train,
        validation,
        val_events,
        val_truth,
        train_matrix,
        val_matrix,
        args,
        device,
        eval_ks,
    )

    val_scores = score_model(model, val_matrix, device=device, batch_size=args.batch_size)
    test_scores = score_model(model, test_matrix, device=device, batch_size=args.batch_size)
    val_ranked = rank_by_scores(validation, val_scores, k=max(eval_ks))
    test_ranked = rank_by_scores(test, test_scores, k=max(eval_ks))

    comparison_rows = []
    for split, ranked, split_events, split_truth in [
        ("validation", val_ranked, val_events, val_truth),
        ("test", test_ranked, test_events, test_truth),
    ]:
        row = {
            "model_name": MODEL_NAME,
            "split": split,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "pairs_per_positive": args.pairs_per_positive,
            "learning_rate": args.learning_rate,
        }
        row.update(base.evaluate_ranked(ranked, split_truth, split_events, eval_ks))
        comparison_rows.append(row)

    ranked_all = pd.concat([val_ranked, test_ranked], ignore_index=True)
    ranked_out = base.attach_product_metadata(ranked_all, sources["products"], truth)
    ranked_out["model_name"] = MODEL_NAME

    ranked_out.to_csv(args.outputs_dir / "candidates_coupon_response_neural_ranker.csv", index=False)
    ranked_out.to_csv(args.outputs_dir / "reranked_recommendations.csv", index=False)
    pd.DataFrame(history_rows).to_csv(args.outputs_dir / "coupon_response_neural_training_history.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.outputs_dir / "coupon_response_neural_model_comparison.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLUMNS, "mean": mean, "std": std}).to_csv(
        args.outputs_dir / "coupon_response_neural_feature_scaler.csv",
        index=False,
    )

    print(f"Wrote {args.outputs_dir / 'candidates_coupon_response_neural_ranker.csv'} ({len(ranked_out)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_neural_training_history.csv'} ({len(history_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_neural_model_comparison.csv'} ({len(comparison_rows)} rows)")
    print("Best validation row:")
    print(pd.DataFrame([best_val_row]).to_string(index=False))
    print("Final comparison:")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
