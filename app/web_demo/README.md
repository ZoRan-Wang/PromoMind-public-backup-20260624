# PromoMind Local Web Demo

## Version

- Date: 2026-06-24
- Version: v0.3
- Scope: local HTML frontend plus Python backend for the final presentation demo

## Visual Thesis

An operational retail-marketing workspace with calm grocery colors, dense ranking evidence, and immediate household-campaign controls.

## Content Plan

- Control surface: choose household-campaign event and coupon slots.
- Evidence surface: show final Recall@10, NDCG@10, and Positive Hit@10.
- Recommendation surface: show Top-20 coupon product ranking with coupon flags, observed hit evidence, and reason text.
- Context surface: show recent household purchase history for the selected event.
- Model comparison surface: show held-out test Positive Hit@10 across baseline, XGBoost LTR, and final tail fusion.

## Interaction Thesis

- Changing the selected event updates all context, history, recommendations, and KPIs.
- Moving the coupon slot slider updates coupon flags immediately.
- Recommendation rows reveal product, category, score, observed outcome, and concise explanation in one scan.
- Presentation presets switch directly between high-hit events and reasonable no-hit events.
- Motion highlights page entry, ranked rows, and observed hit labels.
- Household history shows the real purchase sequence before coupon start for the selected event.

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

1. Download the Drive output and processed packages from the root README.
2. Put extracted output files under `outputs/`.
3. Put extracted processed files under `data/processed/`.
4. Run `python app/web_demo/server.py --port 8766`.

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
  - returns final metrics and selectable test events
- `GET /api/recommendations?event_id=<event_id>&coupon_slots=<n>`
  - returns selected event context, KPIs, history rows, and Top-20 recommendations

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
