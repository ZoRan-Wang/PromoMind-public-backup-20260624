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

## Files

- `server.py`: standard-library HTTP backend and API server.
- `static/index.html`: browser UI shell.
- `static/style.css`: layout and visual design.
- `static/app.js`: API calls and DOM rendering.

## Data Inputs

- `outputs/reranked_recommendations.csv`
- `outputs/demo_history_input.csv`
- `outputs/demo_time_name_recommendations.csv`
- `outputs/coupon_response_tail_fusion_model_comparison.csv`
- `outputs/coupon_response_final_model_comparison.csv`

## Run

From the repository root:

```powershell
python app/web_demo/server.py --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

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
