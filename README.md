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
| `docs/` | Project plan, data dictionary, experiment protocol, and demo spec |
| `src/promomind/` | Importable Python package for data preparation, models, reranking, and evaluation |
| `scripts/` | Dataset export, preprocessing, and synthetic sample-data utilities |
| `app/` | Streamlit demo shell |
| `data/` | Complete Journey raw RDS/RDA files plus ignored generated processed outputs |
| `rendered/` | Rendered DOCX page images used for layout QA |

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

Prepare processed files from CSV exports in `data/raw/`:

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
python scripts/run_cornac_nbr_models.py --k 50 --tifuknn-grid 300:0.9:0.7:0.7:7
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

Run the supervised XGBoost learning-to-rank coupon-response model with CUDA when available:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --primary-metric recall_at_20
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

- `Positive Event Hit@10`: 53.21% vs. 19.27% for the SOTA-candidate-only coupon baseline
- `NDCG@10`: 0.3248 vs. 0.1489
- `Recall@10`: 0.4105 vs. 0.1570

Details are in `docs/coupon_response_improvement.md`.

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
