# PromoMind Local Web Demo

## Version

- Date: 2026-06-24
- Version: v0.12
- Scope: local HTML frontend plus Python backend for the final presentation demo

## Visual Thesis

An operational retail-marketing workspace with calm grocery colors, dense ranking evidence, and immediate rolling household coupon-portfolio controls.

## Content Plan

- Control surface: choose household and coupon window.
- Evidence surface: show final Recall@10, NDCG@10, and Positive Hit@10.
- Recommendation surface: show Top-10 time-bounded household coupon offers with coupon flags, observed hit evidence, campaign source, and reason text.
- Context surface: show recent household purchase history before the selected coupon window.
- Outcome-evidence surface: show 5-day hit SKUs, no-hit aggregate late SKU/category rates, and concrete follow-up purchase cases for the selected household-window.
- Model comparison surface: show held-out test Positive Hit@10 across baseline, XGBoost LTR, and final tail fusion.

## Interaction Thesis

- Changing the selected household-window updates all context, history, recommendations, and KPIs.
- Moving the coupon-window slider updates the active offer pool for the same household.
- Recommendation rows reveal product, category, score, observed outcome, and concise explanation in one scan.
- Window shortcuts switch directly between high-hit and low-hit coupon windows.
- Window shortcuts switch directly between high-hit, late exact product, late same-category, and low-hit coupon windows.
- Motion highlights page entry, ranked rows, and observed hit labels.
- Household history shows the real purchase sequence before the coupon window.

## Files

- `server.py`: standard-library HTTP backend and API server.
- `static/index.html`: browser UI shell.
- `static/style.css`: layout and visual design.
- `static/app.js`: API calls and DOM rendering.

## Data Inputs

- `outputs/reranked_recommendations.csv`
- `outputs/demo_time_name_recommendations.csv`
- `outputs/coupon_response_tail_fusion_model_comparison.csv`
- `outputs/coupon_response_final_model_comparison.csv`
- `outputs/coupon_response_heuristic_model_comparison_zixun.csv`
- `data/processed/transactions_clean.csv`
- `data/processed/product_features.csv`

## Run

From the repository root:

```powershell
python app/web_demo/server.py --port 8766
```

Open:

```text
http://127.0.0.1:8766/
```

## Fresh Clone Setup

The UI files are tracked in GitHub:

- `app/web_demo/server.py`
- `app/web_demo/static/index.html`
- `app/web_demo/static/style.css`
- `app/web_demo/static/app.js`

The data files are intentionally ignored by Git. Before starting the server, restore or regenerate the local artifacts listed in the Data Inputs section.

One-command rebuild from committed source files:

```powershell
python scripts/build_web_demo_artifacts.py
```

This command starts from `data/raw/completejourney/*.rds/.rda`, exports ignored raw CSVs, rebuilds processed data, reruns models, restores final tail-fusion recommendations, and writes the timing demo files.

Optional verification after generation:

```powershell
python scripts/build_web_demo_artifacts.py --run-checks
```

Fast path:

```powershell
python scripts/restore_local_artifacts.py --clean
python app/web_demo/server.py --port 8766
```

Manual rebuild path:

```powershell
python scripts/clean_completejourney.py --top-products 10000
python scripts/run_candidate_models.py --models all --als-backend auto
python scripts/run_cornac_nbr_models.py
python scripts/run_sota_ensemble.py
python scripts/run_coupon_response_xgboost_ranker.py --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv -Force
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv -Force
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --primary-metric recall_at_20 --selection-profile tail_recall --preserve-min-rank 7 --preserve-max-rank 12
python scripts/run_coupon_response_ranker.py --reuse-features --device auto --primary-metric ndcg_at_10
Copy-Item outputs/coupon_response_model_comparison.csv outputs/coupon_response_heuristic_model_comparison_zixun.csv -Force
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --primary-metric recall_at_20 --selection-profile tail_recall --preserve-min-rank 7 --preserve-max-rank 12
python scripts/build_coupon_timing_demo.py --demo-events 3 --prediction-length 5 --max-label-rows 5000 --no-bootstrap
```

The second tail-fusion command restores `outputs/reranked_recommendations.csv` after the heuristic baseline script writes its own ranking output.

## API

- `GET /api/bootstrap`
  - returns final metrics and selectable rolling household portfolios
- `GET /api/recommendations?portfolio_id=<household_id_yyyymmdd>&coupon_slots=<n>`
  - returns selected household-window context, KPIs, history rows, Top-10 coupon offers, and late purchase evidence

## Test Log

- v0.1: implemented local backend, static frontend, final test event browser, and aligned showcase event using demo timing data.
- v0.1 verification: `python -m compileall app/web_demo/server.py` passed.
- v0.1 verification: `/api/bootstrap` returned 718 events.
- v0.1 verification: `/api/recommendations` returned the showcase event with 5 recommendation rows and 24 history rows.
- v0.1 verification: Playwright loaded `http://127.0.0.1:8766`, captured a screenshot, and reported zero console errors or warnings.
- v0.2: sorted held-out test events with observed Top-10 hits first and labeled dropdown options as `HIT`, `NO HIT`, or `HISTORY` for presentation control.
- v0.3: added presentation preset buttons, event storyline strip, animated recommendation rows, hit pulses, and two reasonable no-hit samples.
- v0.3 verification: `python -m compileall app/web_demo/server.py` and `node --check app/web_demo/static/app.js` passed.
- v0.3 verification: `/api/bootstrap` returned 718 events and 6 presentation presets.
- v0.3 verification: browser clicks passed for `955_19_20171115`, `1216_20_20171127`, and `1142_22_20171206`; fresh browser error and warning logs were empty.
- v0.4: connected real household purchase history from `transactions_clean.csv` and `product_features.csv` for every selected event.
- v0.5: regenerated demo artifacts after merging Zixun's cleaning pipeline; final tail fusion now reports Recall@10 0.4138, NDCG@10 0.3225, and Positive Hit@10 53.21%.
- v0.6: main demo output is household-level coupon portfolios. The backend aggregates test coupon candidates by household and deduplicates products.
- v0.7: main demo output is rolling household coupon portfolios keyed by `household_id + coupon_start_date`. The backend keeps each basket aligned with currently active coupon offers for the selected coupon window.
- v0.8: default presentation presets prioritize high-confidence household-window examples with observed hits.
- v0.9: coupon offers are fixed to Top-10 products from the full active coupon-eligible pool for the selected household and coupon window. Controls are household selection and coupon-window slider only.
- v0.10: sidebar shortcuts are grouped as high-hit windows and low-hit windows for direct presentation navigation.
- v0.11: added no-hit late evidence metrics, concrete later exact-product and same-category purchase cases, and late-evidence shortcut groups.
- v0.11 verification: `python -m compileall app\web_demo\server.py`, `node --check app\web_demo\static\app.js`, and `python -m pytest -q` passed.
- v0.11 verification: `/api/bootstrap` returned 12 shortcuts across high-hit, late exact product, late same category, and low-hit groups.
- v0.11 verification: browser check on `http://127.0.0.1:8766/` showed No-hit late SKU 24.96%, No-hit late category 68.19%, concrete cases for `1703_20171115`, and zero console errors or warnings.
- v0.12: moved outcome evidence directly under the Top-10 recommendation table and added concrete 5-day SKU hit cases for hit windows.
- v0.12 verification: `python -m compileall app\web_demo\server.py`, `node --check app\web_demo\static\app.js`, and `python -m pytest -q` passed.
- v0.12 verification: API checks showed `955_20171115` has four observed Top-10 hits and returns visible hit SKU cases; browser DOM check confirmed the evidence block is inside the recommendation area and console logs are clean.
