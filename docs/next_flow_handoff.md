# PromoMind Next-Flow Handoff

Verification target: latest `main` after this handoff commit. The model/result state summarized here was produced from commit `33df97f6f9d4fe1b03ae2679db604f5fe2ebc7ef` and then packaged into the final handoff files.

This file is the current handoff for the next teammate. Older PM files and task boards are retained as planning history; use this document as the current operational source of truth.

## Current State

The project now has a complete coupon-response recommendation pipeline:

- Dataset: The Complete Journey grocery retail data.
- Core task: for each household-campaign exposure, rank campaign coupon products.
- Response label: a product is positive if the exposed household buys it within five days after campaign start.
- Final model: pull-forward interval XGBoost learning-to-rank plus top-10-profile tail fusion.
- Final local artifact: `outputs/reranked_recommendations.csv`.
- Committed result snapshot: `deliverables/final_solution_2026_06_24/final_metrics.csv`.

Final held-out test result:

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary XGBoost LTR | 0.4154 | 0.3291 | 0.5321 | 0.5058 | 0.3557 |
| Final Tail Fusion | 0.4187 | 0.3304 | 0.5413 | 0.5207 | 0.3594 |

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

Those folders are intentionally ignored because some files are large. Regenerate them locally when needed.

## Setup

```powershell
cd "D:\SMU\608 recommendation\PromoMind_proposal"
python -m pip install -r requirements.txt
```

If raw CSV files are missing, regenerate them from the committed RDS/RDA files:

```powershell
Rscript scripts/download_completejourney.R
```

Then build processed data:

```powershell
python scripts/prepare_dataset.py
```

## Reproduce The Final Model

Run the primary XGBoost ranker:

```powershell
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv -Force
```

Run the category-embedding secondary ranker:

```powershell
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv -Force
```

Restore the primary result, then run final tail fusion:

```powershell
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --primary-metric recall_at_20 --selection-profile top10_ndcg --preserve-min-rank 7 --preserve-max-rank 12
```

Expected final test metrics:

```text
Recall@10                  0.4187
NDCG@10                    0.3304
Positive Event Hit@10      0.5413
Recall@20                  0.5207
NDCG@20                    0.3594
```

## Run Checks

```powershell
python -m pytest -q
python -m compileall scripts src tests
```

Expected current result:

```text
35 passed
compileall passes
```

## Run Demo

```powershell
streamlit run app/streamlit_app.py
```

The demo reads `outputs/reranked_recommendations.csv` when present. If that file is missing after a fresh clone, run the reproduction commands above first.

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
