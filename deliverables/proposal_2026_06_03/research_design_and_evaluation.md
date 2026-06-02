# Research Design And Evaluation Plan

This project should be presented as a recommender systems research project with a working-system component, not only as a product idea.

## Research Objective

Study whether promotion and coupon signals can improve grocery basket recommendation decisions when the base task is next-period household-product prediction under implicit feedback.

The research contribution is a two-stage evaluation:

1. Compare candidate-generation recommenders for next-period grocery purchase prediction.
2. Test whether promotion-aware reranking changes the trade-off between ranking accuracy, catalog/list health, and estimated business utility.

## Research Context And Gap

The project draws on implicit-feedback collaborative filtering, chronological next-basket evaluation, auxiliary product and marketing metadata, and multi-objective reranking. The gap is that grocery recommendation should not only rank likely purchases; it should also evaluate whether marketing context changes the trade-off among relevance, coupon/promotion cost, and diversity when true product margin and causal treatment labels are unavailable.

## Research Questions

| ID | Research question | Why it matters |
| --- | --- | --- |
| RQ1 | Which candidate-generation approach performs best for household-level grocery basket prediction? | Grocery data has strong repeat-purchase and popularity effects, so complex models must beat simple baselines. |
| RQ2 | Does adding promotion and coupon information improve recommendation quality or business utility? | Retail recommendations are useful only if they support better marketing allocation. |
| RQ3 | Does discount-cost-aware reranking reduce wasteful coupon assignment while preserving acceptable Recall@K/NDCG@K? | Coupons have cost; optimizing only hit rate may over-discount likely purchases. |
| RQ4 | Does diversity control improve recommendation-list health without unacceptable accuracy loss? | A grocery list dominated by one category is less useful for basket expansion. |
| RQ5 | Are sparse coupon redemptions and partial demographics still useful as auxiliary signals or analysis dimensions? | The dataset has marketing context, but not all marketing fields are dense enough to be direct targets. |

## Hypotheses

| Hypothesis | Expected evidence |
| --- | --- |
| H1: ALS/BPR-style implicit-feedback models will outperform global popularity on NDCG@K, but popularity will remain a strong baseline. | Model comparison table on validation/test. |
| H2: Category-aware baselines will improve personalization over global popularity for households with stable category preferences. | Category Popularity vs Global Popularity metrics and examples. |
| H3: Promotion/coupon reranking can improve Business Utility@K even if Recall@K changes modestly. | Reranking ablation table with accuracy and utility side by side. |
| H4: Discount-cost penalties will reduce estimated coupon waste compared with promotion/coupon-only reranking. | Discount-aware variant vs promotion+coupon variant. |
| H5: Diversity control will increase department/category variety with a measurable but bounded ranking-quality trade-off. | Diversity metric and Recall/NDCG guardrail. |

## Experimental Design

### Data Split

Use a chronological split:

| Split | Weeks | Use |
| --- | --- | --- |
| Train | 1-40 | Fit recommenders and train-window historical features. |
| Validation | 41-46 | Tune hyperparameters and reranking weights. |
| Test | 47-53 | Report final held-out performance. |

If time allows, evaluate rolling next-week prediction inside validation/test and aggregate. If runtime is limited, evaluate pooled future holdout periods and state this limitation.

### Protocol Rules

- Select product-frequency filters using train data only.
- Compute household/product historical features using train-window behavior only.
- Use validation for hyperparameters and reranking weights.
- Report test results once after the validation choice is fixed.
- Average ranking metrics at the household level.
- If time permits, use paired bootstrap confidence intervals across households.
- Label whether repeat purchases are allowed. If excluding previously purchased items, call that a discovery setting.

### Candidate Models

Required:

- Global Popularity.
- Category Popularity.
- Implicit ALS.

Standard:

- ItemKNN.
- BPR if runtime permits.

Bonus:

- LightGCN through RecBole, using the same candidate-output schema.

### Reranking Variants

| Variant | Purpose |
| --- | --- |
| Base only | Measures candidate model without business-aware changes. |
| Base + promotion | Tests product-week promotion signal. |
| Base + coupon | Tests coupon/campaign eligibility signal. |
| Base + promotion + coupon | Tests combined marketing lift. |
| Base + promotion + coupon - discount cost | Tests cost-aware recommendation. |
| Full reranking + diversity | Tests final business-aware recommendation list. |

### Signal Construction

| Signal | Construction rule |
| --- | --- |
| Promotion score | Product-week promotion exposure assumed available at recommendation planning time. |
| Coupon score | Coupon/campaign eligibility from mappings and household campaign exposure, not future redemption. |
| Discount cost proxy | Train-period retail/coupon discount history aggregated by product/category. |
| Diversity score | Department/category penalty or reward inside Top-K. |

## Metrics

### Ranking Accuracy

- Recall@10 and Recall@20.
- NDCG@10 and NDCG@20.

### Catalog And List Health

- Coverage: how much of the catalog is recommended.
- Diversity: department/category variety within Top-K.
- Novelty: whether the model avoids only recommending globally popular products.

### Business Proxy

```text
Business Utility@K =
  sum(estimated sales value for hit products in Top-K)
  - lambda_cost * sum(estimated discount cost for hit products in Top-K)
```

This is not profit. It is a revenue-minus-discount proxy because product cost and margin are unavailable.

## Validity Threats

| Threat | Control |
| --- | --- |
| Temporal leakage | Use chronological split; compute historical features from train windows only. |
| Coupon redemption sparsity | Do not use redemption as the only target; use it as auxiliary feature/analysis. |
| Partial demographics | Do not drop households without demographics; use demographics only for optional analysis/demo. |
| Product catalog size | Use train-period product-frequency filtering and report the rule. |
| Promotion causality | Do not claim causal uplift; report association-based reranking and proxy utility. |
| Campaign seasonality | Report chronological split clearly and avoid random split leakage. |
| Popularity bias | Include popularity/category baselines and novelty/coverage metrics. |

## Proposal-Stage Language

Use:

- "We will test..."
- "We plan to evaluate..."
- "We expect..."
- "If runtime permits..."

Avoid:

- "We have proven..."
- "The model improves..."
- "Profit increases..."
- "Coupon uplift is learned..."
