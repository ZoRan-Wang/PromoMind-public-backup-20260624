# Reranking — Owner C Guide

This document covers everything Owner C needs to reproduce the promotion-aware reranking results: what files are needed, how to run the script, how to use the notebooks, and what outputs are produced.

## Prerequisites

Activate the project virtual environment before running anything:

```bash
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

The following files must exist in `outputs/` before running. If any are missing, download the **Final outputs package** from Google Drive and extract the zip so each CSV sits at `outputs/<filename>.csv`:

**[Final outputs package](https://drive.google.com/file/d/1K-Doro51f55lpWCWSSSEo60aqkhWNU9h/view?usp=drivesdk)** — contains all required `outputs/` files including the candidates CSV, truth CSV, and pre-generated reranked output.

| File | Source | Used by |
|---|---|---|
| `outputs/candidates_coupon_response_tail_fusion.csv` | Owner B — final tail-fusion ranker | reranking **input** (notebook 1 / script) |
| `outputs/coupon_response_all_truth.csv` | Owner B | all three notebooks / script |
| `outputs/promo_reranked_recommendations.csv` | **produced** by notebook 1 / the script | read by notebooks 2 and 3 |

The reranking builds on the **final tail-fusion ranker** (pipeline: XGBoost primary → tail fusion → promotion-aware reranking). `promo_reranked_recommendations.csv` is generated, not an input — notebook 1 (or the script) writes it, and notebooks 2 and 3 read it.

---

## Quick start — one command

Run the reranking script to get all metrics and the output CSV in one shot:

```bash
python scripts/run_coupon_response_reranking.py
```

This uses the default tail-fusion candidates (`outputs/candidates_coupon_response_tail_fusion.csv`) and the grid-search-winner defaults (γ=0.1, λ=0.05, ρ=0.1, all others 0). It prints a full metrics table and writes two files:

- `outputs/promo_reranked_recommendations.csv` — Owner C deliverable for Owner D
- `outputs/reranking_metrics.csv` — Recall, NDCG, Hit, Coverage, ILD, Novelty, BU at K=5/10/20

---

## Script usage

```bash
python scripts/run_coupon_response_reranking.py [OPTIONS]
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--lam` | 0.05 | Discount cost penalty weight λ — grid search winner |
| `--gamma` | 0.1 | Coupon boost weight γ — grid search winner |
| `--rho` | 0.1 | Diversity weight ρ — grid search winner |
| `--beta` | 0.0 | Promotion signal weight β (set to 0: `global_signal` does not improve NDCG@10 in the ablation) |
| `--alpha` | 1.0 | Base score weight α (keep at 1.0) |
| `--eval-split` | test | Split to evaluate on: `test`, `validation`, or `both` |
| `--eval-k` | 5 10 20 | Cutoff values for evaluation |
| `--grid-search` | off | Run grid search over β/γ/λ/ρ to find best weights |
| `--primary-metric` | ndcg_at_10 | Metric to optimise during grid search |
| `--output-name` | promo_reranked_recommendations.csv | Output filename inside `outputs/` |

### Examples

```bash
# Default run (γ=0.1, λ=0.05, ρ=0.1)
python scripts/run_coupon_response_reranking.py \
  --candidates outputs/candidates_coupon_response_tail_fusion.csv \
  --truth outputs/coupon_response_all_truth.csv

# Try a stronger discount penalty
python scripts/run_coupon_response_reranking.py \
  --candidates outputs/candidates_coupon_response_tail_fusion.csv \
  --truth outputs/coupon_response_all_truth.csv --lam 0.2

# Evaluate on both splits
python scripts/run_coupon_response_reranking.py --eval-split both

# Grid search optimising NDCG@10
python scripts/run_coupon_response_reranking.py --grid-search --primary-metric ndcg_at_10

# Grid search optimising Business Utility@10
python scripts/run_coupon_response_reranking.py --grid-search --primary-metric bu_at_10
```

### Reranking formula

```
final_score = α × base_score
            + β × promotion_score
            + γ × coupon_score
            − λ × discount_cost
            + ρ × diversity_score
```

| Term | Source column | Notes |
|---|---|---|
| `base_score` | `final_score` (tail fusion) | Normalised per event to [0, 1] |
| `promotion_score` | `global_signal` | Normalised per event |
| `coupon_score` | `coupon_eligible` | Binary — all candidates are 1 in this pool |
| `discount_cost` | `discount_signal` | Historical discount proxy, already [0, 1] |
| `diversity_score` | Computed | `1 − (same-category items / total event candidates)` |

---

## Notebooks

The three notebooks in `notebooks/` provide detailed analysis. **Run them in number order** — notebook 1 writes `promo_reranked_recommendations.csv`, which notebooks 2 and 3 read.

### `notebooks/01_discount_penalty_ablation.ipynb` (run first)

Implements and ablates the full five-term formula on top of the tail-fusion ranker. Runs an additive ablation (one term at a time), a λ sensitivity sweep, and a grid search, then **writes `outputs/promo_reranked_recommendations.csv`** using the grid-search-winner weights.

Key finding: a small discount penalty (λ ≈ 0.05–0.1) is a win-win — it lifts NDCG@10 and BU@10 slightly above the tail-fusion baseline by breaking ties in the (mostly tied) fused score toward cheaper, equally-relevant products. β (promotion) is mixed and γ (coupon) is inert (all candidates are already coupon-eligible).

Outputs: `outputs/discount_penalty_ablation.csv`, `outputs/discount_penalty_lambda_sweep.csv`, `outputs/grid_search_results.csv`, `outputs/best_reranking_params.json`, `outputs/promo_reranked_recommendations.csv`

### `notebooks/02_diversity_coverage_novelty.ipynb`

Computes Coverage@K, Intra-List Diversity (ILD@K), and Novelty@K. Compares the promotion-aware reranking against the tail-fusion baseline.

Key finding: the reranking's diversity term (ρ=0.1) raises ILD over the tail-fusion baseline at K=5/10 (e.g. ILD@10 0.6808 → 0.7141) at negligible coverage cost and only a small novelty dip. All metrics are identical at K=20 because each event has exactly 20 candidates (top-20 is the full pool).

Outputs: `outputs/diversity_coverage_novelty.csv`

### `notebooks/03_business_utility_evaluation.ipynb`

Computes Business Utility@K across the test split, with a discount-cost-weight sensitivity sweep and a per-campaign-type breakdown.

Key finding: among events that land a hit, BU stays comfortably positive across the full sensitivity range; the business reranking has near-identical BU to the tail-fusion baseline (BU@10 1.2638 vs 1.2688). The portfolio view (all events) is negative, driven by the low ~9.5% 5-day hit rate, not by the discount penalty.

Outputs: none saved by default — this notebook is analysis-only and renders its plots inline. (Each notebook keeps a commented `plt.savefig(...)` line you can uncomment to export a PNG such as `bu_sensitivity.png` if you want figures for slides.)

---

## Output file schema

`outputs/promo_reranked_recommendations.csv` contains all original feature columns from the tail-fusion candidate file plus the following reranking columns:

| Column | Description |
|---|---|
| `base_score` | Original tail-fusion fused score (preserved for reference) |
| `base_rank` | Original tail-fusion rank (preserved for reference) |
| `base_score_norm` | `base_score` normalised to [0,1] per event |
| `promotion_score` | `global_signal` normalised to [0,1] per event |
| `coupon_score` | `coupon_eligible` cast to float |
| `diversity_score` | Category rarity within the event's candidate set |
| `final_score` | Adjusted score from the reranking formula |
| `final_rank` | New rank after reranking (1 = best) |
| `recommend_coupon` | Coupon recommendation flag (carried through from the candidate file) |
| `rerank_alpha/beta/gamma/lambda/rho` | Weights used — for reproducibility |

---

## Ablation results summary

From `notebooks/01_discount_penalty_ablation.ipynb` (test split, reranking on the tail-fusion base):

| Variant | Recall@10 | NDCG@10 | Hit@10 | BU@10 |
|---|---|---|---|---|
| Base only (α=1) = tail fusion | 0.4080 | 0.3259 | 0.5321 | 1.2586 |
| + Promotion (β=0.1) | 0.4117 | 0.3239 | 0.5229 | 1.2813 |
| + Coupon (γ=0.1) | 0.4080 | 0.3259 | 0.5321 | 1.2586 |
| + Discount penalty (λ=0.5) | 0.4055 | 0.3091 | 0.5138 | 1.2587 |
| + Diversity (ρ=0.1) | 0.4108 | 0.3247 | 0.5321 | 1.2586 |
| Full formula | 0.4117 | 0.3110 | 0.5229 | 1.2813 |

**Recommendation:** use `--lam 0.05 --gamma 0.1 --rho 0.1` as the default operating point (grid-search winner from notebook 1, NDCG@10=0.3286). This is the configuration saved in `outputs/promo_reranked_recommendations.csv`. The λ=0.5 row above sits past the sweet spot — at λ≈0.05–0.1 the discount penalty lifts both NDCG@10 and BU@10 above the tail-fusion baseline (see the λ sweep).

**Note on BU numbers:** BU@10 here uses the **events-with-hits denominator** (`business_utility_at_k`) — the average of `(n_hits − discount_cost)` over events that landed at least one hit — so values sit above 1.0. This is a different denominator from the portfolio (all-events) view in notebook 3. BU is also sensitive to the `recommend_coupon` distribution in the candidate file (`discount_cost = discount_signal × recommend_coupon`), so a different candidate CSV will shift the numbers. To reproduce the table exactly, run on `outputs/candidates_coupon_response_tail_fusion.csv` from the [Final outputs package](https://drive.google.com/file/d/1K-Doro51f55lpWCWSSSEo60aqkhWNU9h/view?usp=drivesdk) on Google Drive.
