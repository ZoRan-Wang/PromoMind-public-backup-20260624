"""Run official Cornac next-basket models for SOTA-level validation.

This script is intentionally separate from ``run_candidate_models.py`` because
Cornac is an optional research dependency. It emits the same candidate schema:

  household_id, product_id, base_score, model_name, base_rank

Current default: Cornac TIFUKNN, which is a community-standard next-basket
model and outperforms the local TIFU-style approximation on our validation
split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402
from promomind.evaluation.ranking import ndcg_at_k, recall_at_k  # noqa: E402
from promomind.models.candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Cornac NBR models.")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--train-file", default="train_interactions.csv")
    parser.add_argument("--eval-file", default="valid_interactions.csv")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--tifuknn-grid",
        default="300:0.9:0.7:0.7:7",
        help=(
            "Grid as neighbors:within_decay:group_decay:alpha:n_groups. "
            "Default matches Cornac/TIFUKNN paper-style settings."
        ),
    )
    return parser.parse_args()


def _require_cornac():
    try:
        from cornac.data import BasketDataset
        from cornac.models import TIFUKNN
    except ImportError as exc:
        raise ImportError(
            "This script requires optional Cornac dependencies. Install with: "
            "python -m pip install cornac==2.2.2 --no-deps"
        ) from exc
    return BasketDataset, TIFUKNN


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Generate processed files first.")
    return pd.read_csv(path)


def _string_ids(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def _parse_tifuknn_grid(value: str) -> list[dict[str, float | int]]:
    params = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        n_neighbors, within_decay_rate, group_decay_rate, alpha, n_groups = entry.split(":")
        params.append(
            {
                "n_neighbors": int(n_neighbors),
                "within_decay_rate": float(within_decay_rate),
                "group_decay_rate": float(group_decay_rate),
                "alpha": float(alpha),
                "n_groups": int(n_groups),
            }
        )
    return params


def _build_cornac_dataset(train: pd.DataFrame):
    BasketDataset, _ = _require_cornac()
    required = [schema.HOUSEHOLD_ID, schema.BASKET_ID, schema.PRODUCT_ID, schema.WEEK]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"train interactions missing required columns for Cornac NBR: {missing}")

    work = _string_ids(train[required].dropna(), [schema.HOUSEHOLD_ID, schema.BASKET_ID, schema.PRODUCT_ID])
    work[schema.WEEK] = pd.to_numeric(work[schema.WEEK], errors="coerce").fillna(0).astype(int)
    data = list(work[required].itertuples(index=False, name=None))
    return BasketDataset.build(data, fmt="UBIT")


def _history_baskets(dataset) -> dict[int, list[list[int]]]:
    histories: dict[int, list[list[int]]] = {}
    for batch_users, _, batch_items in dataset.ubi_iter(batch_size=1, shuffle=False):
        histories[int(batch_users[0])] = [list(map(int, basket)) for basket in batch_items[0]]
    return histories


def _score_tifuknn(dataset, model, users: list[str], k: int) -> pd.DataFrame:
    histories = _history_baskets(dataset)
    index_item = {idx: item_id for item_id, idx in dataset.iid_map.items()}
    records: list[dict[str, object]] = []

    for user in users:
        if user not in dataset.uid_map:
            continue
        user_idx = dataset.uid_map[user]
        scores = model.score(user_idx, histories[user_idx])
        if len(scores) > k:
            selected = np.argpartition(-scores, k - 1)[:k]
        else:
            selected = np.arange(len(scores))
        selected = selected[np.lexsort((selected, -scores[selected]))][:k]
        for item_idx in selected:
            score = float(scores[item_idx])
            if score <= 0:
                continue
            records.append(
                {
                    USER_COL: user,
                    ITEM_COL: index_item[int(item_idx)],
                    SCORE_COL: score,
                }
            )

    return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)


def _write_candidates(candidates: pd.DataFrame, model_name: str, path: Path) -> None:
    out = candidates.rename(
        columns={
            USER_COL: schema.HOUSEHOLD_ID,
            ITEM_COL: schema.PRODUCT_ID,
            SCORE_COL: "base_score",
            "rank": "base_rank",
        }
    ).copy()
    out["model_name"] = model_name
    out = out[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "base_score", "model_name", "base_rank"]]
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


def _evaluate(candidates: pd.DataFrame, truth: pd.DataFrame, eval_ks: list[int]) -> dict[str, float]:
    row: dict[str, float] = {"n_candidates": float(len(candidates))}
    for k in eval_ks:
        row[f"recall_at_{k}"] = recall_at_k(candidates, truth, k)
        row[f"ndcg_at_{k}"] = ndcg_at_k(candidates, truth, k)
    return row


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    eval_ks = sorted(set(args.eval_k))
    if not eval_ks or min(eval_ks) <= 0:
        raise ValueError("--eval-k values must be positive")

    _, TIFUKNN = _require_cornac()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    train = _read_required(args.processed_dir / args.train_file)
    eval_frame = _read_required(args.processed_dir / args.eval_file)
    train = _string_ids(train, [schema.HOUSEHOLD_ID, schema.BASKET_ID, schema.PRODUCT_ID])
    eval_frame = _string_ids(eval_frame, [schema.HOUSEHOLD_ID, schema.PRODUCT_ID])

    users = sorted(
        set(train[schema.HOUSEHOLD_ID].dropna().unique())
        & set(eval_frame[schema.HOUSEHOLD_ID].dropna().unique()),
        key=str,
    )
    truth = (
        eval_frame[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]]
        .dropna()
        .drop_duplicates()
        .rename(columns={schema.HOUSEHOLD_ID: USER_COL, schema.PRODUCT_ID: ITEM_COL})
    )

    dataset = _build_cornac_dataset(train)
    tuning_rows: list[dict[str, object]] = []
    candidate_runs: list[pd.DataFrame] = []
    for params in _parse_tifuknn_grid(args.tifuknn_grid):
        model = TIFUKNN(**params, verbose=False)
        model.fit(dataset)
        candidates = _score_tifuknn(dataset, model, users=users, k=args.k)
        row: dict[str, object] = {"model_name": "cornac_tifuknn", **params}
        row.update(_evaluate(candidates, truth, eval_ks))
        tuning_rows.append(row)
        candidate_runs.append(candidates)

    primary_k = eval_ks[0]
    best_idx = max(
        range(len(tuning_rows)),
        key=lambda idx: (
            float(tuning_rows[idx].get(f"ndcg_at_{primary_k}", 0.0)),
            float(tuning_rows[idx].get(f"recall_at_{primary_k}", 0.0)),
        ),
    )
    best_candidates = candidate_runs[best_idx]
    _write_candidates(
        best_candidates,
        "cornac_tifuknn",
        args.outputs_dir / "candidates_cornac_tifuknn.csv",
    )
    pd.DataFrame(tuning_rows).to_csv(args.outputs_dir / "cornac_tifuknn_tuning_results.csv", index=False)
    pd.DataFrame([tuning_rows[best_idx]]).to_csv(
        args.outputs_dir / "cornac_model_comparison.csv",
        index=False,
    )
    print(f"Wrote {args.outputs_dir / 'cornac_tifuknn_tuning_results.csv'} ({len(tuning_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'cornac_model_comparison.csv'} (1 rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
