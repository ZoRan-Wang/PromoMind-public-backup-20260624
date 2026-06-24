"""Promotion-aware reranking for coupon-response candidates.

Applies the five-term formula from the proposal:

final_score = alpha * base
            + beta  * promotion
            + gamma * coupon
            - lam   * discount_cost
            + rho   * diversity

Builds on the final tail-fusion ranker

Column mapping from the tail-fusion candidate file:
    base:          final_score      (tail-fusion fused ranker score)
    promotion:     global_signal    (normalised per event)
    coupon:        coupon_eligible  (binary)
    discount_cost: discount_signal  (historical discount proxy, [0,1])
    diversity:     computed here    (1 - same_category_count / event_candidate_count)

Outputs
-------
outputs/promo_reranked_recommendations.csv        Owner C deliverable: feature scores + final_score + final_rank + recommend_coupon
outputs/reranking_metrics.csv Evaluation table: Recall, NDCG, Hit, Coverage, ILD, Novelty, BU at K=5,10,20

Usage
-----
python scripts/run_coupon_response_reranking.py
python scripts/run_coupon_response_reranking.py --lam 0.05 --gamma 0.1 --rho 0.1 --primary-metric ndcg_at_10
python scripts/run_coupon_response_reranking.py --grid-search --primary-metric ndcg_at_10
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import product as iproduct
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))

from promomind.evaluation.business import business_utility_at_k as _bu_lib

MODEL_NAME = "reranking_promo_aware"

# Default weights from grid search winner (NB3, optimising NDCG@10): gamma=0.1, lam=0.05, rho=0.1
DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 0.0
DEFAULT_GAMMA = 0.1
DEFAULT_LAM   = 0.05
DEFAULT_RHO   = 0.1

GRID = {
    "beta":  [0.0, 0.05, 0.1, 0.2],
    "gamma": [0.0, 0.1],
    "lam":   [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "rho":   [0.0, 0.05, 0.1, 0.2],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _minmax_per_event(df: pd.DataFrame, col: str) -> pd.Series:
    def _scale(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)
    return df.groupby("event_id")[col].transform(_scale)


def _prepare(cands: pd.DataFrame) -> pd.DataFrame:
    df = cands.copy()
    df = df.rename(columns={"final_score": "base_score", "final_rank": "base_rank"})
    df["base_score_norm"]    = _minmax_per_event(df, "base_score")
    df["promotion_score"]    = _minmax_per_event(df, "global_signal")
    df["coupon_score"]       = df["coupon_eligible"].astype(float)
    # diversity: category rarity within each event's candidate set
    cat_cnt   = df.groupby(["event_id", "product_category"])["product_id"].transform("count")
    event_cnt = df.groupby("event_id")["product_id"].transform("count")
    df["diversity_score"] = 1.0 - (cat_cnt / event_cnt)
    return df


def _rerank(df: pd.DataFrame, alpha: float, beta: float, gamma: float,
            lam: float, rho: float) -> pd.DataFrame:
    out = df.copy()
    out["final_score"] = (
          alpha * out["base_score_norm"]
        + beta  * out["promotion_score"]
        + gamma * out["coupon_score"]
        - lam   * out["discount_signal"]
        + rho   * out["diversity_score"]
    )
    out["final_rank"] = (
        out.groupby("event_id")["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


# ── metrics ───────────────────────────────────────────────────────────────────

def _truth_lookup(truth: pd.DataFrame) -> dict[str, set]:
    return truth.groupby("event_id")["product_id"].apply(set).to_dict()


def _positive_event_hit_at_k(df: pd.DataFrame, truth_lookup: dict, k: int,
                              split: str) -> float:
    sub   = df[(df["split"] == split) & (df["final_rank"] <= k)]
    eids  = set(truth_lookup.keys()) & set(df[df["split"] == split]["event_id"])
    hits  = []
    for eid in eids:
        top = set(sub.loc[sub["event_id"] == eid, "product_id"])
        hits.append(1 if top & truth_lookup[eid] else 0)
    return float(np.mean(hits)) if hits else 0.0


def _coverage_at_k(df: pd.DataFrame, k: int, split: str) -> float:
    sub   = df[df["split"] == split]
    total = sub["product_id"].nunique()
    rec   = sub[sub["final_rank"] <= k]["product_id"].nunique()
    return rec / total if total else 0.0


def _ild_at_k(df: pd.DataFrame, k: int, split: str) -> float:
    sub  = df[(df["split"] == split) & (df["final_rank"] <= k)]
    ilds = []
    for _, grp in sub.groupby("event_id"):
        cats = grp["product_category"].values
        if len(cats) < 2:
            ilds.append(0.0)
            continue
        n_pairs = n_dissim = 0
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                n_pairs  += 1
                n_dissim += int(cats[i] != cats[j])
        ilds.append(n_dissim / n_pairs)
    return float(np.mean(ilds)) if ilds else 0.0


def _novelty_at_k(df: pd.DataFrame, k: int, split: str) -> float:
    sub      = df[df["split"] == split]
    n_events = sub["event_id"].nunique()
    pop      = (sub.groupby("product_id")["event_id"].nunique() / n_events).rename("pop")
    top_k    = sub[sub["final_rank"] <= k].merge(pop, on="product_id", how="left")
    top_k["nov"] = -np.log2(top_k["pop"].clip(lower=1e-9))
    return float(top_k.groupby("event_id")["nov"].mean().mean())


def _bu_at_k(df: pd.DataFrame, truth: pd.DataFrame, k: int, split: str) -> float:
    sub = df[df["split"] == split].copy()
    sub = sub.rename(columns={"final_rank": "rank"})
    sub["expected_revenue"] = 1.0
    sub["discount_cost"] = sub["discount_signal"] * sub["recommend_coupon"].astype(float)
    return _bu_lib(
        sub,
        ground_truth=truth,
        k=k,
        user_col="event_id",
        item_col="product_id",
        revenue_col="expected_revenue",
        discount_col="discount_cost",
    )


def _recall_at_k(df: pd.DataFrame, truth_lookup: dict, k: int, split: str) -> float:
    sub    = df[(df["split"] == split) & (df["final_rank"] <= k)]
    recalls = []
    for eid, relevant in truth_lookup.items():
        if eid not in set(df[df["split"] == split]["event_id"]):
            continue
        if not relevant:
            continue
        recommended = set(sub.loc[sub["event_id"] == eid, "product_id"])
        recalls.append(len(recommended & relevant) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def _ndcg_at_k(df: pd.DataFrame, truth_lookup: dict, k: int, split: str) -> float:
    sub    = df[(df["split"] == split) & (df["final_rank"] <= k)]
    scores = []
    for eid, relevant in truth_lookup.items():
        if eid not in set(df[df["split"] == split]["event_id"]):
            continue
        if not relevant:
            continue
        ranked = (sub[sub["event_id"] == eid]
                  .sort_values("final_rank")["product_id"]
                  .tolist())
        dcg  = sum(1.0 / math.log2(r + 1) for r, item in enumerate(ranked, 1) if item in relevant)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(relevant)) + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _evaluate(df: pd.DataFrame, truth: pd.DataFrame, k_values: list[int],
              split: str) -> dict[str, float]:
    split_events  = set(df[df["split"] == split]["event_id"])
    truth_lookup  = {eid: prods for eid, prods in _truth_lookup(truth).items()
                     if eid in split_events}

    results: dict[str, float] = {}
    for k in k_values:
        results[f"recall_at_{k}"]   = _recall_at_k(df, truth_lookup, k, split)
        results[f"ndcg_at_{k}"]     = _ndcg_at_k(df, truth_lookup, k, split)
        results[f"hit_at_{k}"]      = _positive_event_hit_at_k(df, truth_lookup, k, split)
        results[f"coverage_at_{k}"] = _coverage_at_k(df, k, split)
        results[f"ild_at_{k}"]      = _ild_at_k(df, k, split)
        results[f"novelty_at_{k}"]  = _novelty_at_k(df, k, split)
        results[f"bu_at_{k}"]       = _bu_at_k(df, truth, k, split)
    return results


# ── printing ──────────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict[str, float], k_values: list[int], split: str,
                   params: dict) -> None:
    p = params
    print(f"\n{'─'*60}")
    print(f"Split: {split}  |  α={p['alpha']} β={p['beta']} γ={p['gamma']} "
          f"λ={p['lam']} ρ={p['rho']}")
    print(f"{'─'*60}")
    header = f"{'Metric':<22}" + "".join(f"  K={k:<6}" for k in k_values)
    print(header)
    print("─" * len(header))
    for metric in ["recall", "ndcg", "hit", "coverage", "ild", "novelty", "bu"]:
        label = {
            "recall": "Recall@K", "ndcg": "NDCG@K", "hit": "Pos. Event Hit@K",
            "coverage": "Coverage@K", "ild": "ILD@K",
            "novelty": "Novelty@K", "bu": "BU@K",
        }[metric]
        row = f"{label:<22}" + "".join(
            f"  {metrics.get(f'{metric}_at_{k}', float('nan')):.4f}" for k in k_values
        )
        print(row)
    print(f"{'─'*60}")


# ── grid search ───────────────────────────────────────────────────────────────

def _grid_search(df: pd.DataFrame, truth: pd.DataFrame, prepared: pd.DataFrame,
                 k_values: list[int], split: str, primary_metric: str) -> dict:
    keys   = list(GRID.keys())
    combos = list(iproduct(*GRID.values()))
    print(f"\nGrid search: {len(combos)} combinations, optimising {primary_metric} on {split} …")

    best_score = -float("inf")
    best_params: dict = {}
    rows = []

    for combo in combos:
        params = dict(zip(keys, combo))
        reranked = _rerank(prepared, alpha=DEFAULT_ALPHA, **params)
        metrics  = _evaluate(reranked, truth, k_values, split)
        row = {"alpha": DEFAULT_ALPHA, **params, **metrics}
        rows.append(row)
        score = metrics.get(primary_metric, -float("inf"))
        if score > best_score:
            best_score = score
            best_params = {"alpha": DEFAULT_ALPHA, **params}

    search_df = pd.DataFrame(rows).sort_values(primary_metric, ascending=False)
    print(f"\nTop 10 by {primary_metric}:")
    display_cols = ["alpha", "beta", "gamma", "lam", "rho",
                    "recall_at_10", "ndcg_at_10", "hit_at_10", "bu_at_10",
                    "recall_at_20", "ndcg_at_20"]
    print(search_df[display_cols].head(10).to_string(index=False, float_format="{:.4f}".format))
    return best_params, search_df


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promotion-aware reranking — Owner C deliverable.")
    p.add_argument("--candidates", type=Path,
                   default=REPO_ROOT / "outputs" /
                   "candidates_coupon_response_tail_fusion.csv")
    p.add_argument("--truth", type=Path,
                   default=REPO_ROOT / "outputs" / "coupon_response_all_truth.csv")
    p.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--output-name", default="promo_reranked_recommendations.csv")
    p.add_argument("--eval-split", default="test", choices=["validation", "test", "both"])
    p.add_argument("--eval-k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--beta",  type=float, default=DEFAULT_BETA,
                   help="Promotion signal weight (default 0 — does not improve NDCG@10 on this dataset)")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                   help="Coupon boost weight (default 0.1 — grid search winner)")
    p.add_argument("--lam",   type=float, default=DEFAULT_LAM,
                   help="Discount cost penalty weight (default 0.05 — grid search winner)")
    p.add_argument("--rho",   type=float, default=DEFAULT_RHO,
                   help="Diversity weight (default 0.1 — grid search winner)")
    p.add_argument("--grid-search", action="store_true",
                   help="Run grid search over beta/gamma/lam/rho and pick best params")
    p.add_argument("--primary-metric", default="ndcg_at_10",
                   choices=["ndcg_at_10", "recall_at_10", "hit_at_10", "bu_at_10",
                            "ndcg_at_20", "recall_at_20"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    cands = pd.read_csv(args.candidates)
    # The tail-fusion file ships with its own `rank` column (== final_rank). Drop it so it
    # cannot collide with the `rank` that `_bu_at_k` derives from this layer's final_rank.
    if "rank" in cands.columns:
        cands = cands.drop(columns=["rank"])
    truth = pd.read_csv(args.truth)
    print(f"  Candidates: {len(cands):,} rows | splits: {sorted(cands['split'].unique())}")
    print(f"  Truth:      {len(truth):,} rows")

    print("Preparing features …")
    prepared = _prepare(cands)

    splits = ["validation", "test"] if args.eval_split == "both" else [args.eval_split]

    if args.grid_search:
        best_params, search_df = _grid_search(
            cands, truth, prepared, args.eval_k, splits[0],
            args.primary_metric
        )
        search_path = args.outputs_dir / "grid_search_results.csv"
        search_df.to_csv(search_path, index=False)
        print(f"\nFull grid search saved to {search_path}")
        alpha = best_params["alpha"]
        beta  = best_params["beta"]
        gamma = best_params["gamma"]
        lam   = best_params["lam"]
        rho   = best_params["rho"]
    else:
        alpha, beta, gamma, lam, rho = args.alpha, args.beta, args.gamma, args.lam, args.rho

    print(f"\nApplying reranking: α={alpha} β={beta} γ={gamma} λ={lam} ρ={rho} …")
    reranked = _rerank(prepared, alpha=alpha, beta=beta, gamma=gamma, lam=lam, rho=rho)
    reranked["model_name"]    = MODEL_NAME
    reranked["rerank_alpha"]  = alpha
    reranked["rerank_beta"]   = beta
    reranked["rerank_gamma"]  = gamma
    reranked["rerank_lambda"] = lam
    reranked["rerank_rho"]    = rho

    # ── evaluate ──────────────────────────────────────────────────────────────
    metric_rows = []
    for split in splits:
        metrics = _evaluate(reranked, truth, args.eval_k, split)
        _print_metrics(metrics, args.eval_k, split,
                       {"alpha": alpha, "beta": beta, "gamma": gamma, "lam": lam, "rho": rho})
        row = {"model_name": MODEL_NAME, "split": split,
               "alpha": alpha, "beta": beta, "gamma": gamma, "lam": lam, "rho": rho,
               **metrics}
        metric_rows.append(row)

    metrics_df  = pd.DataFrame(metric_rows)
    metrics_path = args.outputs_dir / "reranking_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # ── save output CSV (Owner C interface) ───────────────────────────────────
    # Required columns: feature scores, final_score, final_rank, recommend_coupon.
    # Also expose `rank` (== final_rank) so the evaluation notebooks, which key on
    # `rank`, can consume this file directly.
    reranked["rank"] = reranked["final_rank"]
    out_path = args.outputs_dir / args.output_name
    reranked.to_csv(out_path, index=False)
    print(f"Reranked candidates saved to {out_path}  ({len(reranked):,} rows)")
    print(f"  Columns: {reranked.columns.tolist()}")


if __name__ == "__main__":
    main()
