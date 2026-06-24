# PromoMind Part-C Notebooks (Business Reranking & Evaluation)

These notebooks cover the **promotion-aware business reranking** layer and its evaluation. They sit on top of the modelling pipeline (`XGBoost primary → tail fusion`) and consume its committed candidate files from `outputs/`.

## Inputs (produced by the modelling pipeline, see `docs/next_flow_handoff.md`)

- `outputs/candidates_coupon_response_tail_fusion.csv` — the final tail-fusion ranker (the base the reranking builds on).
- `outputs/coupon_response_all_truth.csv` — ground-truth purchases per event.

If these are missing after a fresh clone, regenerate them with the reproduction commands in `docs/next_flow_handoff.md`.

## Run order

Run the notebooks **in number order** — notebook 1 produces the artifact the other two read.

1. **`01_discount_penalty_ablation.ipynb`**
   - Applies the five-term reranking formula (`α·base + β·promotion + γ·coupon − λ·discount_cost + ρ·diversity`) on top of the tail-fusion candidates.
   - Runs the additive ablation, λ sweep, and grid search.
   - **Writes `outputs/promo_reranked_recommendations.csv`** (the promotion-aware reranked list) plus `best_reranking_params.json`, `discount_penalty_ablation.csv`, `grid_search_results.csv`.

2. **`02_diversity_coverage_novelty.ipynb`**
   - Reads `promo_reranked_recommendations.csv` and compares it against the tail-fusion baseline.
   - Computes Coverage@K, Intra-List Diversity@K, and Novelty@K (RQ4).
   - Writes `outputs/diversity_coverage_novelty.csv`.

3. **`03_business_utility_evaluation.ipynb`**
   - Reads `promo_reranked_recommendations.csv` and computes Business Utility@K (RQ2/RQ3), with a discount-cost sensitivity sweep and a per-campaign-type breakdown.

## Script equivalent

`scripts/run_coupon_response_reranking.py` reproduces notebook 1's reranking from the command line (same default tail-fusion input, same grid-search winner, writing `promo_reranked_recommendations.csv` and `reranking_metrics.csv`):

```bash
python scripts/run_coupon_response_reranking.py                       # default params (grid-search winner)
python scripts/run_coupon_response_reranking.py --grid-search --primary-metric ndcg_at_10
```

## Notes

- `outputs/` is git-ignored; regenerate locally as needed.
- Each event has exactly 20 candidates, so K=20 metrics are set-based (identical across rankings of the same pool).
