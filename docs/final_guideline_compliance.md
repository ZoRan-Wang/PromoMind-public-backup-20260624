# Final Project Guideline Compliance

This file maps the course final-solution requirements to the current PromoMind repository evidence.

## Requirement Checklist

| Course requirement | Current evidence | Status |
| --- | --- | --- |
| Finalized dataset | `data/raw/completejourney/`, `scripts/download_completejourney.R`, `scripts/prepare_dataset.py`, `docs/data_dictionary.md` | Complete |
| Finalized recommendation problem | `docs/coupon_response_improvement.md`, `deliverables/final_solution_2026_06_24/final_results_summary.md` | Complete |
| Algorithms | `scripts/run_candidate_models.py`, `scripts/run_cornac_nbr_models.py`, `scripts/run_coupon_response_ranker.py`, `scripts/run_coupon_response_xgboost_ranker.py`, `scripts/run_coupon_response_neural_ranker.py`, `scripts/run_coupon_response_tail_fusion.py` | Complete |
| Experimental results | `docs/coupon_response_improvement.md`, `deliverables/final_solution_2026_06_24/final_metrics.csv` | Complete |
| Applicability and significance | `docs/coupon_response_improvement.md`, `deliverables/final_solution_2026_06_24/final_results_summary.md` | Complete |
| Proposed or attempted extensions with evidence | XGBoost ablations, neural ranker, text embedding, text match, category embedding, rank fusion, tail fusion in `docs/coupon_response_improvement.md` | Complete |
| Working recommender demo | `app/streamlit_app.py`; run with `streamlit run app/streamlit_app.py` after generating `outputs/reranked_recommendations.csv` | Complete locally |
| Future work | `docs/coupon_response_improvement.md`, final section; RedNote/NoteLLM, causal uplift, richer product text/images | Complete |

## Proposal Requirement Coverage

| Proposal requirement | Current evidence |
| --- | --- |
| Dataset and collection methodology | `deliverables/proposal_2026_06_03/proposal_en.md`, `deliverables/proposal_2026_06_03/proposal_zh.md`, `scripts/download_completejourney.R` |
| Expected data size | proposal files and `docs/data_dictionary.md` |
| Recommendation problem and significance | proposal files and final summary |
| Recommender algorithms and library acknowledgement | proposal files, `README.md`, `requirements.txt`, model scripts |
| Experiments to test models | `docs/experiment_protocol.md`, `docs/coupon_response_improvement.md` |
| X-factor | promotion-aware coupon-response ranking, Business Utility proxy, Streamlit demo |

## Final Presentation Claims To Use

Use these claims:

- We use a real grocery retail dataset with transaction, product, campaign, coupon, promotion, redemption, and demographic tables.
- We reformulate the second stage as household-campaign coupon-response ranking.
- The final model combines time-aware XGBoost learning-to-rank with validation-selected top-10-profile tail fusion.
- Held-out test Positive Event Hit@10 improves from 19.27% for the SOTA-candidate-only coupon baseline to 54.13%.
- The final tail-fusion model reaches Recall@10 0.4187 and NDCG@10 0.3304.
- NLP/product-text features were attempted and documented, but timing and repeat behavior are stronger for this dataset.

Avoid these claims:

- Do not claim universal SOTA across all grocery recommender tasks.
- Do not claim causal coupon uplift; the dataset does not provide randomized treatment-control evidence.
- Do not claim true profit optimization; use revenue-minus-discount or Business Utility proxy language.
- Do not claim demographics cover all households.
- Do not claim coupon redemption is the main label.

## Verification Commands

```powershell
python -m pytest -q
python -m compileall scripts src tests
```

Current verified result:

```text
35 passed
compileall passes
```
