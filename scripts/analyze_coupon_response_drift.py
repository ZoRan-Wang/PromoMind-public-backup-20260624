"""Summarize split drift for the coupon-response ranking task."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze coupon-response split drift.")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_path = args.outputs_dir / "coupon_response_all_features.csv"
    truth_path = args.outputs_dir / "coupon_response_all_truth.csv"
    if not feature_path.exists() or not truth_path.exists():
        raise FileNotFoundError(
            "Run scripts/run_coupon_response_xgboost_ranker.py once before drift analysis."
        )

    features = pd.read_csv(feature_path, usecols=["split", "event_id"])
    truth = pd.read_csv(truth_path, usecols=["event_id", "product_id"])
    event_split = features[["split", "event_id"]].drop_duplicates()
    truth_counts = truth.groupby("event_id")["product_id"].nunique()
    event_split["truth_items"] = event_split["event_id"].map(truth_counts).fillna(0).astype(int)
    event_split["is_positive_event"] = event_split["truth_items"] > 0

    candidate_rows = features.groupby("split").agg(
        candidate_rows=("event_id", "size"),
        events=("event_id", "nunique"),
    )
    event_rows = event_split.groupby("split").agg(
        positive_events=("is_positive_event", "sum"),
        positive_event_rate=("is_positive_event", "mean"),
        avg_truth_items_per_event=("truth_items", "mean"),
        avg_truth_items_per_positive_event=("truth_items", lambda values: values[values > 0].mean()),
        max_truth_items=("truth_items", "max"),
    )
    summary = candidate_rows.join(event_rows).reset_index()
    summary.to_csv(args.outputs_dir / "coupon_response_split_drift.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.outputs_dir / 'coupon_response_split_drift.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
