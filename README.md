# PromoMind

PromoMind is a project skeleton, proposal package, and execution workspace for a promotion-aware grocery basket and coupon recommender system.

The project predicts products each household is likely to buy next, then re-ranks recommendations using promotion exposure, coupon availability, discount-cost proxy, diversity, and business-utility signals.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `PromoMind_bilingual_proposal.docx` | Full bilingual proposal in Word format |
| `PromoMind_bilingual_proposal.md` | Full bilingual proposal in Markdown |
| `PromoMind_5_second_meeting_message.md` | Short meeting pitch |
| `PromoMind_four_person_work_split.md` | Four-person work allocation |
| `PromoMind_PM_team_execution_pack.md` | Detailed PM execution plan |
| `PromoMind_task_board.csv` | Task board for tracking execution |
| `PromoMind_RACI_matrix.csv` | RACI ownership matrix |
| `docs/next_flow_handoff.md` | Current handoff for the next teammate |
| `docs/final_guideline_compliance.md` | Course guideline-to-evidence checklist |
| `docs/coupon_response_improvement.md` | Final modeling report and ablation evidence |
| `deliverables/final_solution_2026_06_24/` | Final solution summary, metrics snapshot, recommendation sample, and presentation outline |
| `docs/` | Project plan, data dictionary, experiment protocol, and demo spec |
| `src/promomind/` | Importable Python package for data preparation, models, reranking, and evaluation |
| `scripts/` | Dataset export, preprocessing, and synthetic sample-data utilities |
| `app/` | Streamlit demo shell |
| `data/` | Complete Journey raw RDS/RDA files plus ignored generated processed outputs |
| `rendered/` | Rendered DOCX page images used for layout QA |

## Current Handoff

Use `docs/next_flow_handoff.md` as the operational entry point for the next teammate. It lists the verified commit, final model, expected metrics, reproduction commands, demo command, and role-by-role next actions.

Use `deliverables/final_solution_2026_06_24/` for final-presentation materials:

- `final_results_summary.md`
- `final_presentation_outline.md`
- `final_metrics.csv`
- `top10_recommendation_sample.csv`

## Google Drive Artifacts

Generated outputs and large CSV caches are stored on Google Drive instead of GitHub:

- [PromoMind Final Artifacts folder](https://drive.google.com/drive/folders/12K-x6t3J-1JQWSQqNrdeRbbRqedNpAHd)
- [Final outputs package](https://drive.google.com/file/d/1K-Doro51f55lpWCWSSSEo60aqkhWNU9h/view?usp=drivesdk): final `outputs/reranked_recommendations.csv`, model-comparison CSVs, search CSVs, final docs, and presentation-ready snapshots.
- [Raw CSV cache package](https://drive.google.com/file/d/1BAqzSp6x6-V8QYk-e5eC-_btKuQ3Vgy8/view?usp=drivesdk): local `data/raw/*.csv` exports from The Complete Journey.
- [Processed cache package](https://drive.google.com/file/d/1_fJivrUgHJ-bq8CQEld8VeX17S_5XMd2/view?usp=drivesdk): local `data/processed/*.csv` files.

The Drive folder is shared read-only to anyone with the link. Use the final outputs package when you want to run the Streamlit demo without regenerating all model artifacts. Use the raw/processed cache packages only when you want to skip local R export or preprocessing.

## Proposed Project

**Title:** PromoMind: A Promotion-aware Grocery Basket Recommender for Retail Marketing Optimization

**Dataset:** The Complete Journey

**Core task:** Given a household's historical grocery transactions, recommend the Top-K products it is most likely to purchase in the next week.

**X-factor:** Promotion-aware re-ranking that combines:

- base recommendation score
- promotion exposure
- coupon availability
- estimated discount cost
- recommendation-list diversity
- business utility proxy

## Setup

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Install optional recommender backends:

```bash
python -m pip install -e ".[dev,recommenders]"
```

Generate a tiny synthetic dataset for smoke tests:

```bash
python scripts/make_sample_data.py
```

Clean the committed RDS/RDA data into local processed files:

```bash
python scripts/clean_completejourney.py --top-products 10000
```

For optional CSV exports in `data/raw/`, use:

```bash
python scripts/prepare_dataset.py --raw-dir data/raw --processed-dir data/processed
```

Run first-stage candidate generation models:

```bash
python scripts/run_candidate_models.py --models popularity,personal_topfreq,category,itemknn,tifu_knn,upcf,hybrid_strong --k 50
```

Run all implemented Member B models, including ALS and BPR comparisons:

```bash
python scripts/run_candidate_models.py --models all --k 50
```

Run the official Cornac TIFUKNN next-basket benchmark:

```bash
python scripts/run_cornac_nbr_models.py --k 50
```

Run the final protocol-best rank ensemble:

```bash
python scripts/run_sota_ensemble.py --weight-step 0.01 --primary-metric ndcg_at_10
```

Build the coupon timing demo artifacts with external TBP/TARS next-basket code:

```bash
python scripts/build_coupon_timing_demo.py --demo-events 3 --prediction-length 5 --max-label-rows 5000
```

Run the upgraded coupon-response ranker with CUDA scoring when available:

```bash
python scripts/run_coupon_response_ranker.py --device auto --primary-metric ndcg_at_10
```

Run the final supervised XGBoost learning-to-rank coupon-response model with CUDA when available:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
```

Run optional robustness diagnostics and ablations:

```bash
python scripts/analyze_coupon_response_drift.py
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme expected_lead_timing --expected-lead-min-days 1 --expected-lead-max-days 2 --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --ensemble-top-n 2
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-text-embedding-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-text-match-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-event-category-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --final-train-scope train
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --primary-metric recall_at_20 --selection-profile tail_recall --preserve-min-rank 7 --preserve-max-rank 12
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --wide-search --use-value-features --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-score-blend --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-rank-fusion --primary-metric ndcg_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-coupon-family-features --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-redemption-features --primary-metric recall_at_20
```

Run the PyTorch pairwise neural ranker:

```bash
python scripts/run_coupon_response_neural_ranker.py --reuse-features --device auto
```

After `outputs/coupon_response_features.csv` has been generated once, rerun the weight search quickly with:

```bash
python scripts/run_coupon_response_ranker.py --reuse-features --device auto --primary-metric ndcg_at_10
```

For grocery next-basket prediction, repeat purchases are allowed by default. Use `--exclude-seen` only for a discovery-only experiment.

Run tests:

```bash
pytest
```

Run the demo:

```bash
streamlit run app/streamlit_app.py
```

If `outputs/reranked_recommendations.csv` or `outputs/demo_time_name_recommendations.csv` exists, the Streamlit page shows the coupon timing demo: historical basket input on the left when available and predicted time-product pairs on the right.

## Coupon-Response Upgrade

The upgraded coupon-response ranker evaluates household-campaign exposures with this label:

```text
success = the exposed household bought the ranked campaign coupon product within 5 days after campaign start
```

Current held-out test result:

- `Positive Event Hit@10`: 53.21% vs. 18.35% for the candidate-only coupon baseline
- `NDCG@10`: 0.3225 vs. 0.1399
- `Recall@10`: 0.4138 vs. 0.1479
- Final artifact: `outputs/reranked_recommendations.csv` generated by validation-selected tail fusion

Details are in `docs/coupon_response_improvement.md`.

Optional exploratory switches `--wide-search`, `--search-score-blend`, `--search-rank-fusion`, `--use-value-features`, `--use-coupon-family-features`, `--use-redemption-features`, `--use-content-features`, `--use-response-priors`, `--use-text-embedding-features`, `--use-text-match-features`, `--use-category-embedding-features`, and `--use-derived-features` are available for ablation. The final reported artifact is the tail-fused ranker, not the standalone default XGBoost output.

The original Complete Journey raw files are committed under `data/raw/completejourney/` because each file is below GitHub's 100MB limit in RDS/RDA format. Generated CSV exports and processed outputs remain ignored by Git.

## Included Raw Data

The repository includes these original Complete Journey artifacts:

- `data/raw/completejourney/transactions.rds`
- `data/raw/completejourney/promotions.rds`
- `data/raw/completejourney/products.rda`
- `data/raw/completejourney/coupons.rda`
- `data/raw/completejourney/coupon_redemptions.rda`
- `data/raw/completejourney/campaigns.rda`
- `data/raw/completejourney/campaign_descriptions.rda`
- `data/raw/completejourney/demographics.rda`
- `data/raw/completejourney/transactions_sample.rda`
- `data/raw/completejourney/promotions_sample.rda`

## Expected Raw CSVs

The preprocessing scripts expect CSV exports with these names:

- `transactions.csv`
- `products.csv`
- `demographics.csv`
- `promotions.csv`
- `coupons.csv`
- `coupon_redemptions.csv`

Use `scripts/download_completejourney.R` as the R-side reference for exporting tables from the committed/package RDS/RDA data into local CSVs when needed.

## Team Workstreams

- **A: Data Lead** - data cleaning, time split, EDA, feature tables.
- **B: Model Lead** - popularity baselines, ItemKNN, ALS, BPR.
- **C: Business/Reranking Lead** - promotion/coupon features, re-ranking, Business Utility@K.
- **D: Integration/Demo Lead** - LightGCN attempt, Streamlit demo, final integration, PPT/report.

## Minimum Delivery Line

The minimum complete project is:

1. Data cleaning and time-based split.
2. Popularity baseline and ALS recommendation model.
3. Promotion-aware re-ranking.
4. Business Utility@K evaluation.
5. Streamlit demo and final presentation materials.
