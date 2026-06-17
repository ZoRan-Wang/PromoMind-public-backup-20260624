# Final Presentation Outline

Target: June 24 final project solution presentation.

## Slide 1: Title And Business Question

Title: PromoMind: Promotion-Aware Grocery Basket And Coupon Recommendation

Key message:

Retailers should not only ask what a household may buy next. They should ask which likely purchases deserve coupon or promotion support under limited marketing budget.

## Slide 2: Dataset

Use The Complete Journey:

- 2,469 households
- item-level grocery transactions
- product metadata
- campaigns, coupons, promotions, coupon redemptions
- partial demographics

Mention:

- coupon redemption is sparse
- demographics are incomplete
- main label is future purchase response, not redemption alone

## Slide 3: Final Problem Definition

For each household-campaign exposure:

```text
Rank campaign coupon products.
Positive label = product bought within five days after campaign start.
```

Why this matters:

- improves coupon targeting
- reduces irrelevant coupon pushes
- supports business-aware recommendation rather than pure next-basket prediction

## Slide 4: System Architecture

Show two stages:

1. Candidate generation and response scoring
2. Promotion-aware ranking and tail fusion

Final model:

- XGBoost learning-to-rank
- pull-forward interval relevance label
- category-embedding secondary ranker
- validation-selected top-10-profile tail fusion

## Slide 5: Model And Feature Signals

Main signals:

- base candidate relevance
- household-product repeat behavior
- cadence and median interval
- product global response
- campaign type
- category co-occurrence tail source

Mention attempted but non-final signals:

- value and discount features
- redemption features
- NLP product-text features
- neural ranker

## Slide 6: Results

Use this table:

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary XGBoost LTR | 0.4154 | 0.3291 | 0.5321 | 0.5058 | 0.3557 |
| Final Tail Fusion | 0.4187 | 0.3304 | 0.5413 | 0.5207 | 0.3594 |

Main line:

Positive Event Hit@10 improves from 19.27% to 54.13% against the SOTA-candidate-only coupon baseline.

## Slide 7: Extension Evidence

Say that we did not stop at one model:

- heuristic time-aware ranker
- PyTorch pairwise neural ranker
- XGBoost LTR
- expected-lead and pull-forward timing labels
- text embedding and direct text-match NLP ablations
- category co-occurrence embedding
- tail fusion

Conclusion:

Timing and repeat behavior generalize better than product-text metadata for this dataset.

## Slide 8: Demo

Show Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

Demo fallback:

- use `deliverables/final_solution_2026_06_24/top10_recommendation_sample.csv`
- show one household-campaign event and its Top-10 coupon product recommendations

## Slide 9: Applicability And Limits

Applicability:

- grocery retailers
- membership retail
- e-commerce coupon targeting
- campaign product prioritization

Limits:

- no causal uplift claim
- no true profit because product cost is unavailable
- demographics are partial
- product text is structured metadata, not rich content

## Slide 10: Future Work

Future work:

- causal uplift model with treatment-control data
- true margin-aware business utility if cost data is available
- richer product text/image features
- RedNote/NoteLLM-style multimodal retrieval after adding content data
- real-time budget optimization

## Speaker Split

Speaker A:

- dataset
- split
- data risks

Speaker B:

- candidate generation
- XGBoost ranker
- results

Speaker C:

- coupon-response formulation
- tail fusion
- business interpretation

Speaker D:

- demo
- limitations
- future work
