# PromoMind Next-Flow Handoff

Verification target: `codex/zixun-cleaning-retrain` before merge back to `main`. The model/result state summarized here was regenerated on 2026-06-24 after merging Zixun's cleaning pipeline.

This file is the current handoff for the next teammate. Older PM files and task boards are retained as planning history; use this document as the current operational source of truth.

## Current State

The project now has a complete coupon-response recommendation pipeline:

- Dataset: The Complete Journey grocery retail data.
- Core task: for each household-campaign exposure, rank campaign coupon products.
- Response label: a product is positive if the exposed household buys it within five days after campaign start.
- Final model: pull-forward interval XGBoost learning-to-rank plus validation-selected tail fusion.
- Final local artifact: `outputs/reranked_recommendations.csv`.
- Committed result snapshot: `deliverables/final_solution_2026_06_24/final_metrics.csv`.

Final held-out test result:

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary XGBoost LTR | 0.4006 | 0.3165 | 0.5138 | 0.5188 | 0.3518 |
| Category-Embedding XGBoost LTR | 0.4099 | 0.3212 | 0.5321 | 0.5238 | 0.3535 |
| Final Tail Fusion | 0.4138 | 0.3225 | 0.5321 | 0.5184 | 0.3520 |

Do not claim universal SOTA. The correct claim is:

> Under our Complete Journey coupon-response protocol, the final time-aware XGBoost and tail-fusion ranker substantially improves coupon-specific hit rate over the SOTA-candidate-only coupon baseline.

## What Is In GitHub

Committed and safe to use:

- Code: `scripts/`, `src/`, `tests/`, `app/`
- Proposal material: `deliverables/proposal_2026_06_03/`
- Final solution snapshot: `deliverables/final_solution_2026_06_24/`
- Current method report: `docs/coupon_response_improvement.md`
- Requirement mapping: `docs/final_guideline_compliance.md`
- Raw compressed Complete Journey files: `data/raw/completejourney/`

Not committed:

- `outputs/`
- `data/processed/`
- local raw CSV exports under `data/raw/*.csv`

Those folders are intentionally ignored because some files are large. Restore them from GitHub-committed cache packages when needed:

```powershell
python scripts/restore_local_artifacts.py --clean
```

The ZIP packages live under `artifacts/local_cache/` and are split to stay below GitHub's 100MB file limit.

## Setup

```powershell
cd "D:\SMU\608 recommendation\PromoMind_proposal"
python -m pip install -r requirements.txt
```

If raw CSV files are missing, regenerate them from the committed RDS/RDA files:

```powershell
Rscript scripts/download_completejourney.R
```

Then build processed data from the committed RDS/RDA files using Zixun's train-only cleaning flow:

```powershell
python scripts/clean_completejourney.py --top-products 10000
```

## Reproduce The Final Model

Run the primary XGBoost ranker:

```powershell
python scripts/run_coupon_response_xgboost_ranker.py --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv -Force
```

Run the category-embedding secondary ranker:

```powershell
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv -Force
```

Run final tail fusion. Under the regenerated Zixun-cleaned data, the category-embedding ranker is the stronger head model and validation selects `keep_primary_top=8`.

```powershell
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --primary-metric recall_at_20 --selection-profile tail_recall --preserve-min-rank 7 --preserve-max-rank 12
```

Expected final test metrics:

```text
Recall@10                  0.4138
NDCG@10                    0.3225
Positive Event Hit@10      0.5321
Recall@20                  0.5184
NDCG@20                    0.3520
```

## Run Checks

```powershell
python -m pytest -q
python -m compileall scripts src tests
```

Expected current result:

```text
43 passed
compileall passes
```

## Run Presentation Web Demo

The committed browser demo is under `app/web_demo/`. This is the presentation UI used at `http://127.0.0.1:8766/`.

```powershell
python app/web_demo/server.py --port 8766
```

Open:

```text
http://127.0.0.1:8766/
```

This server requires local generated artifacts under `outputs/` and `data/processed/`. Those folders are ignored by Git. Use `docs/zixun_cleaning_pipeline.md` to regenerate them, or restore them from the committed cache packages.

Fast restore from committed cache packages:

```powershell
python scripts/restore_local_artifacts.py --clean
```

One-command rebuild from committed source files:

```powershell
python scripts/build_web_demo_artifacts.py
```

## Run Legacy Streamlit Demo

```powershell
streamlit run app/streamlit_app.py
```

The Streamlit page reads `outputs/reranked_recommendations.csv` when present. If that file is missing after a fresh clone, run the reproduction commands above first.

## What Each Next Teammate Should Do

Data/report owner:

- Use `docs/data_dictionary.md` and `docs/final_guideline_compliance.md`.
- Explain data tables, time split, leakage controls, and coupon/demographic sparsity.

Model owner:

- Use `docs/member_b_modeling.md`, `docs/member_b_sota_positioning.md`, and `docs/coupon_response_improvement.md`.
- Explain why time-aware repeat/cadence features dominate this grocery task.

Business/reranking owner:

- Use `docs/coupon_response_improvement.md`.
- Explain pull-forward interval labels, coupon-response ranking, tail fusion, and Business Utility proxy limitations.

Demo/presentation owner:

- Use `deliverables/final_solution_2026_06_24/final_presentation_outline.md`.
- Verify Streamlit opens locally before the meeting.
- Keep `deliverables/final_solution_2026_06_24/top10_recommendation_sample.csv` as backup if live demo fails.

## Known Boundaries

- Coupon redemption is too sparse to be the main supervised target.
- Demographics are partial and should be used for display or subgroup discussion only.
- `outputs/` is local and ignored by Git.
- NLP product-text analysis was implemented, but it did not beat the final model because Complete Journey product text is mostly structured metadata rather than rich item text.
- RedNote/NoteLLM-style multimodal recommendation is future work, not a current-result claim.
