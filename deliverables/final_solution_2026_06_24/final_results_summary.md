# PromoMind Final Results Summary

## Problem

PromoMind is a promotion-aware grocery recommender. The final task is:

```text
For each household-campaign exposure, rank active campaign coupon products.
Success means the household buys the ranked coupon product within five days after campaign start.
```

This is more business-specific than ordinary next-basket recommendation because it evaluates which products are worth coupon or promotion attention.

## Dataset

The project uses The Complete Journey, a real household-level grocery retail dataset with:

- item-level transactions
- product metadata
- promotion exposure
- campaign membership
- coupon-product mappings
- coupon redemption history
- partial household demographics

Coupon redemption and demographics are sparse, so they are supporting signals and analysis fields rather than the main supervised target.

## Final Method

The final method has two stages:

1. Candidate and response scoring with time-aware XGBoost learning-to-rank.
2. Top-10-profile tail fusion, where validation selects `keep_primary_top=7`.

The XGBoost ranker uses:

- base candidate signal
- household-product repeat signal
- cadence and interval features
- global product response signal
- campaign type
- graded pull-forward interval labels

The tail-fusion step keeps the strongest primary ranks and fills the remaining Top-20 list from the category co-occurrence embedding variant.

## Held-Out Test Results

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary XGBoost LTR | 0.4154 | 0.3291 | 0.5321 | 0.5058 | 0.3557 |
| Final Tail Fusion | 0.4187 | 0.3304 | 0.5413 | 0.5207 | 0.3594 |

The main improvement against the earlier SOTA-candidate-only coupon baseline is:

```text
Positive Event Hit@10: 19.27% -> 54.13%
NDCG@10:               0.1489 -> 0.3304
Recall@10:             0.1570 -> 0.4187
```

## Extension Evidence

Attempted extensions include:

- PyTorch pairwise neural ranker
- expected coupon-lead relevance labels
- pull-forward interval relevance labels
- XGBoost wide search
- score blending and rank fusion
- value, coupon-family, redemption, and response-prior features
- TF-IDF/SVD product-text profile
- direct TF-IDF product-text match
- category co-occurrence embedding
- final tail fusion

The final model is not chosen because it is the most complex. It is chosen because it has the strongest held-out top-10 coupon-response result.

## Demo

Run locally:

```powershell
streamlit run app/streamlit_app.py
```

The demo uses `outputs/reranked_recommendations.csv`. If that file is missing, regenerate the final pipeline using `docs/next_flow_handoff.md`.

## Correct Final Claim

Under our Complete Journey coupon-response protocol, PromoMind improves coupon-specific hit rate substantially over the SOTA-candidate-only coupon baseline and provides a practical promotion-aware recommendation workflow.

Do not claim universal SOTA or causal coupon uplift.
