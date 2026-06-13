"""Search a rank ensemble for the strongest next-basket candidate source.

The default ensemble combines:

* official Cornac TIFUKNN, which gives the strongest ranking quality
* PromoMind hybrid_strong, which gives the strongest recall coverage

It writes:

* outputs/candidates_sota_ensemble.csv
* outputs/sota_ensemble_weight_search.csv
* outputs/sota_ensemble_model_comparison.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402
from promomind.models.candidates import ITEM_COL, RANK_COL, SCORE_COL, USER_COL, sort_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PromoMind SOTA-style rank ensemble search.")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--eval-file", default="valid_interactions.csv")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--primary-metric", default="ndcg_at_10")
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument(
        "--cornac-file",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidates_cornac_tifuknn.csv",
    )
    parser.add_argument(
        "--hybrid-file",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidates_hybrid_strong.csv",
    )
    return parser.parse_args()


def _read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate file: {path}")
    frame = pd.read_csv(path)
    required = [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "base_rank"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = frame[required].rename(
        columns={
            schema.HOUSEHOLD_ID: USER_COL,
            schema.PRODUCT_ID: ITEM_COL,
            "base_rank": RANK_COL,
        }
    )
    out[USER_COL] = out[USER_COL].astype(str)
    out[ITEM_COL] = out[ITEM_COL].astype(str)
    out[RANK_COL] = pd.to_numeric(out[RANK_COL], errors="coerce")
    out = out.dropna(subset=[USER_COL, ITEM_COL, RANK_COL])
    return out


def _read_truth(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    frame = pd.read_csv(path, usecols=[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]).dropna().drop_duplicates()
    frame[schema.HOUSEHOLD_ID] = frame[schema.HOUSEHOLD_ID].astype(str)
    frame[schema.PRODUCT_ID] = frame[schema.PRODUCT_ID].astype(str)
    return frame.groupby(schema.HOUSEHOLD_ID)[schema.PRODUCT_ID].apply(set).to_dict()


def _rank_ensemble(
    cornac: pd.DataFrame,
    hybrid: pd.DataFrame,
    cornac_weight: float,
    k: int,
) -> pd.DataFrame:
    hybrid_weight = 1.0 - cornac_weight
    parts = []
    for frame, weight in [(cornac, cornac_weight), (hybrid, hybrid_weight)]:
        if weight <= 0:
            continue
        work = frame[[USER_COL, ITEM_COL, RANK_COL]].copy()
        work[SCORE_COL] = weight / work[RANK_COL].astype(float)
        parts.append(work[[USER_COL, ITEM_COL, SCORE_COL]])

    combined = (
        pd.concat(parts, ignore_index=True)
        .groupby([USER_COL, ITEM_COL], as_index=False, sort=False)[SCORE_COL]
        .sum()
    )
    return sort_candidates(combined, k=k)


def _evaluate_fast(candidates: pd.DataFrame, truth: dict[str, set[str]], eval_ks: list[int]) -> dict[str, float]:
    max_k = max(eval_ks)
    top = candidates[candidates[RANK_COL] <= max_k].copy()
    by_user = {str(user): group for user, group in top.groupby(USER_COL, sort=False)}
    row: dict[str, float] = {"n_candidates": float(len(candidates))}
    recalls = {k: [] for k in eval_ks}
    ndcgs = {k: [] for k in eval_ks}

    for user, relevant in truth.items():
        group = by_user.get(user)
        if group is None:
            for k in eval_ks:
                recalls[k].append(0.0)
                ndcgs[k].append(0.0)
            continue

        hits_by_k = {k: 0 for k in eval_ks}
        dcg_by_k = {k: 0.0 for k in eval_ks}
        for rank, item in zip(group[RANK_COL].to_numpy(), group[ITEM_COL].to_numpy(), strict=False):
            if item not in relevant:
                continue
            for k in eval_ks:
                if rank <= k:
                    hits_by_k[k] += 1
                    dcg_by_k[k] += 1.0 / math.log2(rank + 1)

        for k in eval_ks:
            recalls[k].append(hits_by_k[k] / len(relevant))
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs[k].append(dcg_by_k[k] / idcg if idcg else 0.0)

    for k in eval_ks:
        row[f"recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        row[f"ndcg_at_{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
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
    out = out[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "base_score", "model_name", "base_rank"]]
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.weight_step <= 0 or args.weight_step > 1:
        raise ValueError("--weight-step must be in (0, 1]")

    eval_ks = sorted(set(args.eval_k))
    if not eval_ks or min(eval_ks) <= 0:
        raise ValueError("--eval-k values must be positive")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    cornac = _read_candidates(args.cornac_file)
    hybrid = _read_candidates(args.hybrid_file)
    truth = _read_truth(args.processed_dir / args.eval_file)

    weights = np.arange(0.0, 1.0 + args.weight_step / 2, args.weight_step)
    search_rows: list[dict[str, float]] = []
    candidate_runs: list[pd.DataFrame] = []
    for cornac_weight in weights:
        cornac_weight = round(float(cornac_weight), 6)
        candidates = _rank_ensemble(cornac, hybrid, cornac_weight=cornac_weight, k=args.k)
        row = {
            "model_name": "sota_ensemble",
            "cornac_weight": cornac_weight,
            "hybrid_weight": round(1.0 - cornac_weight, 6),
        }
        row.update(_evaluate_fast(candidates, truth, eval_ks))
        search_rows.append(row)
        candidate_runs.append(candidates)

    if args.primary_metric not in search_rows[0]:
        raise ValueError(f"--primary-metric must be one of: {sorted(search_rows[0])}")

    best_idx = max(
        range(len(search_rows)),
        key=lambda idx: (
            float(search_rows[idx][args.primary_metric]),
            float(search_rows[idx].get("recall_at_10", 0.0)),
            float(search_rows[idx].get("ndcg_at_20", 0.0)),
        ),
    )
    best_candidates = candidate_runs[best_idx]
    best_row = search_rows[best_idx]

    _write_candidates(best_candidates, "sota_ensemble", args.outputs_dir / "candidates_sota_ensemble.csv")
    pd.DataFrame(search_rows).to_csv(args.outputs_dir / "sota_ensemble_weight_search.csv", index=False)
    pd.DataFrame([best_row]).to_csv(args.outputs_dir / "sota_ensemble_model_comparison.csv", index=False)
    print(f"Wrote {args.outputs_dir / 'sota_ensemble_weight_search.csv'} ({len(search_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'sota_ensemble_model_comparison.csv'} (1 rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
