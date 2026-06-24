# Zixun Cleaning Pipeline Integration

## Status

- Date: 2026-06-24
- Branch: `codex/zixun-cleaning-retrain`
- Source branch: `origin/Zixun`
- Rollback snapshot: `external/rollback_before_zixun_20260624_140852`

## What Changed

Zixun's cleaning flow adds a reproducible RDS/RDA-to-processed pipeline in `scripts/clean_completejourney.py`. The flow reads the committed Complete Journey files under `data/raw/completejourney/`, keeps raw files unchanged, and writes generated CSV artifacts under ignored `data/processed/`.

The integrated flow uses train-only product selection and train-only behavioral features. This prevents validation/test leakage in product catalog selection, product statistics, household profiles, discount-cost estimates, and coupon/promotion feature construction.

## Regeneration Command

```powershell
python scripts/clean_completejourney.py --top-products 10000
```

For the full browser demo, run the end-to-end artifact builder:

```powershell
python scripts/build_web_demo_artifacts.py
```

This command exports ignored `data/raw/*.csv` from the committed RDS/RDA source files, rebuilds `data/processed/`, reruns the final modeling chain, restores `outputs/reranked_recommendations.csv`, and writes the timing-demo files required by `app/web_demo/server.py`.

## Generated Local Tables

| File | Rows |
| --- | ---: |
| `transactions_clean.csv` | 1,161,272 |
| `train_interactions.csv` | 886,225 |
| `valid_interactions.csv` | 127,664 |
| `test_interactions.csv` | 147,383 |
| `products_clean.csv` | 92,331 |
| `demographics_clean.csv` | 801 |
| `coupons_clean.csv` | 111,332 |
| `coupon_redemptions_clean.csv` | 2,102 |
| `campaigns_clean.csv` | 6,589 |
| `campaign_descriptions_clean.csv` | 27 |
| `product_features.csv` | 10,000 |
| `household_features.csv` | 2,428 |
| `product_week_promotion_features.csv` | 240,639 |
| `household_product_coupon_features.csv` | 1,606,893 |
| `discount_cost_features.csv` | 10,000 |

`data/processed/cleaning_audit.json` records the split policy, filtering policy, row counts, missing-value counts, and generated-table checks.

## Rebuilt Model Chain

```powershell
python scripts/run_candidate_models.py --models all --als-backend auto
python scripts/run_cornac_nbr_models.py
python scripts/run_sota_ensemble.py
python scripts/run_coupon_response_xgboost_ranker.py --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv -Force
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv -Force
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --primary-metric recall_at_20 --selection-profile tail_recall --preserve-min-rank 7 --preserve-max-rank 12
python scripts/run_coupon_response_reranking.py --candidates outputs/candidates_coupon_response_tail_fusion.csv --truth outputs/coupon_response_all_truth.csv --output-name reranked_C.csv --eval-split both
```

## Current Held-Out Test Result

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate-only coupon baseline | 0.1479 | 0.1399 | 0.1835 | 0.1616 | 0.1434 |
| Primary XGBoost LTR | 0.4006 | 0.3165 | 0.5138 | 0.5188 | 0.3518 |
| Category-Embedding XGBoost LTR | 0.4099 | 0.3212 | 0.5321 | 0.5238 | 0.3535 |
| Final Tail Fusion | 0.4138 | 0.3225 | 0.5321 | 0.5184 | 0.3520 |

## Verification

```powershell
python -m pytest -q
```

Verified result:

```text
43 passed
```
