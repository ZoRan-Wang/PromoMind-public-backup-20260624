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

| File | Source |
|---|---|
| `outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv` | Owner B / XGBoost ranker |
| `outputs/coupon_response_all_truth.csv` | Owner B / XGBoost ranker |
| `outputs/reranked_recommendations.csv` | Tail fusion script (for notebooks 1 and 2) |
| `outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv` | Owner B / XGBoost ranker (for tail fusion) |

---

## Quick start — one command

Run the reranking script to get all metrics and the output CSV in one shot:

```bash
python scripts/run_coupon_response_reranking.py \
  --candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv \
  --truth outputs/coupon_response_all_truth.csv
```

This uses the grid-search-winner defaults (γ=0.1, λ=0.05, ρ=0.1, all others 0). It prints a full metrics table and writes two files:

- `outputs/reranked_recommendations.csv` — Owner C deliverable for Owner D
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
| `--beta` | 0.0 | Promotion signal weight β (set to 0: `global_signal` hurt ranking in ablation) |
| `--alpha` | 1.0 | Base score weight α (keep at 1.0) |
| `--eval-split` | test | Split to evaluate on: `test`, `validation`, or `both` |
| `--eval-k` | 5 10 20 | Cutoff values for evaluation |
| `--grid-search` | off | Run grid search over β/γ/λ/ρ to find best weights |
| `--primary-metric` | ndcg_at_10 | Metric to optimise during grid search |
| `--output-name` | reranked_recommendations.csv | Output filename inside `outputs/` |

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
| `base_score` | `final_score` (XGBoost) | Normalised per event to [0, 1] |
| `promotion_score` | `global_signal` | Normalised per event |
| `coupon_score` | `coupon_eligible` | Binary — all candidates are 1 in this pool |
| `discount_cost` | `discount_signal` | Historical discount proxy, already [0, 1] |
| `diversity_score` | Computed | `1 − (same-category items / total event candidates)` |

---

## Notebooks

The three notebooks in `notebooks/` provide detailed analysis. Run them in order with the project venv as the kernel.

### `notebooks/01_business_utility_evaluation.ipynb`

Computes Business Utility@K across the full test split and verifies against the saved model comparison checkpoint.

Key finding: BU crosses zero at λ ≈ 0.13. Use λ = 0.05–0.10 to keep BU positive while reducing discount cost.

Outputs: `outputs/bu_sensitivity.png`

### `notebooks/02_diversity_coverage_novelty.ipynb`

Computes Coverage@K, Intra-List Diversity (ILD@K), and Novelty@K. Compares tail fusion vs XGBoost primary.

Key finding: tail fusion does not improve ILD or novelty vs XGBoost primary on the test split. ILD at K=10 is ~0.68 for both models, meaning ~68% of product-pairs within each list belong to different categories.

Outputs: `outputs/diversity_coverage_novelty.csv`, `outputs/diversity_coverage_novelty_plot.png`

### `notebooks/03_discount_penalty_ablation.ipynb`

Implements and ablates the full five-term formula. Runs an additive ablation (one term at a time), a λ sensitivity sweep, and a grid search.

Key finding: only λ (discount penalty) has a real business effect. β and γ are ineffective on this dataset. Use the grid search section to pick your operating point.

After running, update `BEST_LAMBDA` (and other weights) in section 9 to produce your final `outputs/reranked_recommendations.csv`.

Outputs: `outputs/discount_penalty_ablation.csv`, `outputs/discount_penalty_lambda_sweep.csv`, `outputs/grid_search_results.csv`, `outputs/discount_penalty_tradeoff.png`, `outputs/reranked_recommendations.csv`

---

## Output file schema

`outputs/reranked_recommendations.csv` contains all original feature columns from the XGBoost candidate file plus the following reranking columns:

| Column | Description |
|---|---|
| `xgb_score` | Original XGBoost LTR score (preserved for reference) |
| `xgb_rank` | Original XGBoost rank (preserved for reference) |
| `base_score_norm` | `xgb_score` normalised to [0,1] per event |
| `promotion_score` | `global_signal` normalised to [0,1] per event |
| `coupon_score` | `coupon_eligible` cast to float |
| `diversity_score` | Category rarity within the event's candidate set |
| `final_score` | Adjusted score from the reranking formula |
| `final_rank` | New rank after reranking (1 = best) |
| `recommend_coupon` | Coupon recommendation flag (from XGBoost output) |
| `rerank_alpha/beta/gamma/lambda/rho` | Weights used — for reproducibility |

---

## Ablation results summary

From `notebooks/03_discount_penalty_ablation.ipynb` (test split):

| Variant | Recall@10 | NDCG@10 | Hit@10 | BU@10 |
|---|---|---|---|---|
| Base only (λ=0) | 0.4154 | 0.3291 | 0.5321 | 0.2936 |
| + Promotion (β=0.1) | 0.4028 | 0.3222 | 0.5321 | 0.2972 |
| + Coupon (γ=0.1) | 0.4154 | 0.3291 | 0.5321 | 0.2936 |
| + Discount penalty (λ=0.5) | 0.4150 | 0.3084 | 0.5138 | **0.4066** |
| + Diversity (ρ=0.1) | 0.4177 | 0.3265 | 0.5229 | 0.2634 |
| Full formula | 0.4045 | 0.3110 | 0.5229 | 0.3794 |

**Recommendation:** use `--lam 0.05 --gamma 0.1 --rho 0.1` as the default operating point (grid search winner from notebook 03, NDCG@10=0.3296). This is the configuration saved in `outputs/reranked_recommendations.csv`. BU@10 values above reflect the notebook 03 formula (positive-events denominator); re-run notebook 03 after any formula changes to refresh these numbers.

**Note on BU numeric reproduction:** BU values are sensitive to the `recommend_coupon` distribution in the candidate file — the `discount_cost` term is `discount_signal × recommend_coupon`, so different candidate CSVs (e.g., tail-fused output vs. the raw XGBoost output) produce numerically different BU scores even with identical weights and formula. To reproduce the table above exactly, use `outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv` from the [Final outputs package](https://drive.google.com/file/d/1K-Doro51f55lpWCWSSSEo60aqkhWNU9h/view?usp=drivesdk) on Google Drive.
