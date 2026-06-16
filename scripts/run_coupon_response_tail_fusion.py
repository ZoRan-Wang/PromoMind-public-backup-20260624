"""Fuse two coupon-response rankers while preserving the primary top ranks.

The default use case keeps the current best XGBoost ranker's top-10 unchanged
and fills the tail of the top-20 list from a higher-recall secondary ranker.
The keep point is selected on validation campaigns only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in [str(SCRIPT_ROOT), str(SRC_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_coupon_response_ranker as base  # noqa: E402
from promomind.data import schema  # noqa: E402

MODEL_NAME = "coupon_response_tail_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse primary and secondary coupon-response rankings.")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--primary-candidates",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidates_coupon_response_xgboost_ranker_pf_interval_best.csv",
    )
    parser.add_argument(
        "--secondary-candidates",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv",
    )
    parser.add_argument("--truth", type=Path, default=REPO_ROOT / "outputs" / "coupon_response_all_truth.csv")
    parser.add_argument("--eval-k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--preserve-min-rank", type=int, default=10)
    parser.add_argument("--preserve-max-rank", type=int, default=15)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument("--primary-metric", default="recall_at_20")
    return parser.parse_args()


def _read_ranked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    frame = pd.read_csv(path)
    if "rank" not in frame.columns:
        if "final_rank" not in frame.columns:
            raise ValueError(f"{path} must contain rank or final_rank.")
        frame["rank"] = frame["final_rank"]
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce").astype(int)
    frame[schema.PRODUCT_ID] = pd.to_numeric(frame[schema.PRODUCT_ID], errors="coerce").astype(int)
    return frame


def fuse_rankings(primary: pd.DataFrame, secondary: pd.DataFrame, keep_primary_top: int, output_k: int) -> pd.DataFrame:
    """Keep primary ranks through ``keep_primary_top`` and fill the tail from secondary."""

    if keep_primary_top < 1:
        raise ValueError("keep_primary_top must be >= 1.")
    if output_k < keep_primary_top:
        raise ValueError("output_k must be >= keep_primary_top.")

    secondary_by_event = {event_id: group.sort_values("rank") for event_id, group in secondary.groupby(base.EVENT_COL)}
    rows: list[dict[str, object]] = []
    for event_id, primary_group in primary.groupby(base.EVENT_COL, sort=False):
        primary_sorted = primary_group.sort_values("rank")
        selected: list[dict[str, object]] = []
        used_products: set[int] = set()

        for _, row in primary_sorted[primary_sorted["rank"] <= keep_primary_top].iterrows():
            product_id = int(row[schema.PRODUCT_ID])
            item = row.to_dict()
            item["fusion_source"] = "primary_head"
            selected.append(item)
            used_products.add(product_id)

        secondary_group = secondary_by_event.get(event_id)
        if secondary_group is not None:
            for _, row in secondary_group.iterrows():
                if len(selected) >= output_k:
                    break
                product_id = int(row[schema.PRODUCT_ID])
                if product_id in used_products:
                    continue
                item = row.to_dict()
                item["fusion_source"] = "secondary_tail"
                selected.append(item)
                used_products.add(product_id)

        for _, row in primary_sorted.iterrows():
            if len(selected) >= output_k:
                break
            product_id = int(row[schema.PRODUCT_ID])
            if product_id in used_products:
                continue
            item = row.to_dict()
            item["fusion_source"] = "primary_backfill"
            selected.append(item)
            used_products.add(product_id)

        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
            item["final_rank"] = rank
            item["fusion_keep_primary_top"] = keep_primary_top
            rows.append(item)

    fused = pd.DataFrame(rows)
    if "model_name" in fused.columns:
        fused["base_model_name"] = fused["model_name"]
    fused["model_name"] = MODEL_NAME
    return fused


def _events_from_ranked(ranked: pd.DataFrame, split: str) -> pd.DataFrame:
    return ranked[ranked["split"].eq(split)][[base.EVENT_COL]].drop_duplicates().copy()


def _evaluate_split(ranked: pd.DataFrame, truth: pd.DataFrame, split: str, eval_ks: list[int]) -> dict[str, float | str]:
    split_ranked = ranked[ranked["split"].eq(split)].copy()
    events = _events_from_ranked(ranked, split)
    split_truth = truth[truth[base.EVENT_COL].isin(set(events[base.EVENT_COL]))].copy()
    metrics = base.evaluate_ranked(split_ranked, split_truth, events, eval_ks)
    metrics["split"] = split
    return metrics


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.preserve_min_rank > args.preserve_max_rank:
        raise ValueError("--preserve-min-rank must be <= --preserve-max-rank.")
    if args.preserve_max_rank > args.output_k:
        raise ValueError("--preserve-max-rank must be <= --output-k.")

    primary = _read_ranked(args.primary_candidates)
    secondary = _read_ranked(args.secondary_candidates)
    truth = pd.read_csv(args.truth)
    eval_ks = sorted(set(args.eval_k))

    search_rows = []
    fused_by_keep: dict[int, pd.DataFrame] = {}
    for keep in range(args.preserve_min_rank, args.preserve_max_rank + 1):
        fused = fuse_rankings(primary, secondary, keep, args.output_k)
        fused_by_keep[keep] = fused
        for split in ["validation", "test"]:
            row = _evaluate_split(fused, truth, split, eval_ks)
            row["keep_primary_top"] = keep
            row["output_k"] = args.output_k
            row["primary_metric"] = args.primary_metric
            search_rows.append(row)

    search = pd.DataFrame(search_rows)
    validation = search[search["split"].eq("validation")].copy()
    best_validation = validation.sort_values(
        [args.primary_metric, "ndcg_at_20", "keep_primary_top"],
        ascending=[False, False, True],
    ).iloc[0]
    best_keep = int(best_validation["keep_primary_top"])
    selected = fused_by_keep[best_keep]

    comparison_rows = []
    for split in ["validation", "test"]:
        row = _evaluate_split(selected, truth, split, eval_ks)
        row.update(
            {
                "model_name": MODEL_NAME,
                "primary_candidates": str(args.primary_candidates),
                "secondary_candidates": str(args.secondary_candidates),
                "model_selection": "validation",
                "primary_metric": args.primary_metric,
                "keep_primary_top": best_keep,
                "output_k": args.output_k,
            }
        )
        comparison_rows.append(row)

    search.to_csv(args.outputs_dir / "coupon_response_tail_fusion_search.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.outputs_dir / "coupon_response_tail_fusion_model_comparison.csv", index=False)
    selected.to_csv(args.outputs_dir / "candidates_coupon_response_tail_fusion.csv", index=False)
    selected.to_csv(args.outputs_dir / "reranked_recommendations.csv", index=False)

    comparison_parts = []
    final_path = args.outputs_dir / "coupon_response_final_model_comparison.csv"
    if final_path.exists():
        comparison_parts.append(pd.read_csv(final_path))
    comparison_parts.append(comparison)
    pd.concat(comparison_parts, ignore_index=True, sort=False).to_csv(final_path, index=False)

    print(f"Selected keep_primary_top={best_keep} by validation {args.primary_metric}")
    print(f"Wrote {args.outputs_dir / 'candidates_coupon_response_tail_fusion.csv'} ({len(selected)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_tail_fusion_search.csv'} ({len(search)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_tail_fusion_model_comparison.csv'} ({len(comparison)} rows)")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
