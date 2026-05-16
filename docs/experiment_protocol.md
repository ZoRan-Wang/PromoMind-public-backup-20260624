# Experiment Protocol

## Objective

Evaluate whether PromoMind can recommend next-week grocery products and whether promotion-aware reranking improves a business-utility proxy without unacceptable ranking-quality loss.

The experiment has two stages:

1. Candidate generation predicts likely next-week products from household purchase history.
2. Reranking adjusts candidate order using promotion, coupon, discount-cost, and diversity signals.

## Time Split

Use a chronological split by `week`:

| Split | Weeks | Purpose |
| --- | --- | --- |
| Train | 1-40 | Fit candidate models and train-window features |
| Validation | 41-46 | Tune ALS/BPR and reranking weights |
| Test | 47-53 | Final held-out reporting |

Implementation rules:

- Build each household's ground truth as products purchased in the target split.
- Do not include future weeks in model fitting or historical features.
- If evaluating "next-week" specifically, run rolling target weeks inside validation/test and aggregate metrics. If time is limited, evaluate each split as a pooled held-out future period and document that choice.
- Product and household filters must be decided using train data and then applied consistently.

## Candidate Generation

Each candidate model must emit the same schema:

| Column | Required |
| --- | --- |
| `household_id` | Yes |
| `product_id` | Yes |
| `base_score` | Yes |
| `model_name` | Yes |
| `base_rank` | Yes |

Recommended candidate depth:

- Generate Top-100 per household when runtime allows.
- Reranking and final evaluation should report Top-10 and Top-20.
- If Top-100 is too slow, use Top-50 and state the candidate depth in every result table.

### Required Models

| Model | Status | Notes |
| --- | --- | --- |
| Global popularity | Required | Same high-frequency products for all households; sanity baseline |
| Category popularity | Required | Recommend popular products from household's train-period top categories; fallback to global popularity |
| ALS | Required | Main implicit-feedback model; tune on validation |

### Standard/Bonus Models

| Model | Status | Notes |
| --- | --- | --- |
| ItemKNN | Standard | Similar-product baseline from household-product matrix |
| BPR | Standard if time allows | Pairwise ranking matrix factorization |
| LightGCN | Bonus | Must use same output schema as other candidate models |

## Reranking Variants

Rerank candidates using:

```text
final_score =
  alpha * normalized_base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

Normalize scores before combining when scales differ across models.

Required ablation variants:

| Variant | Formula behavior |
| --- | --- |
| Base only | `alpha * normalized_base_score` |
| Promotion | Base + `promotion_score` |
| Coupon | Base + `coupon_score` |
| Promotion + Coupon | Base + both marketing signals |
| Discount-aware | Base + promotion + coupon - discount cost |
| Full reranking | Discount-aware + diversity |

Validation tuning:

- Start with `alpha` dominant so ranking quality remains anchored to candidate relevance.
- Tune `beta`, `gamma`, `lambda`, and `rho` on validation.
- Record every tested weight set in `outputs/reranking_tuning_results.csv` if implemented.
- Select the final weight set using Business Utility@K with a guardrail on Recall@K/NDCG@K.

## Diversity Control

At least one diversity method should be implemented in the standard delivery:

- Department cap: no more than `N` products from the same department in Top-K.
- Category penalty: subtract a small penalty for repeated `product_category`.
- Coverage reward: add a small score when a product adds a new department/category to the current list.

Report the diversity method and its effect separately from promotion/coupon effects.

## Metrics

### Ranking Quality

| Metric | Required | Description |
| --- | --- | --- |
| `Recall@10` | Yes | Fraction of held-out products captured in Top-10 |
| `Recall@20` | Yes | Fraction of held-out products captured in Top-20 |
| `NDCG@10` | Yes | Rank-sensitive relevance for Top-10 |
| `NDCG@20` | Yes | Rank-sensitive relevance for Top-20 |

### Catalog and List Health

| Metric | Required | Description |
| --- | --- | --- |
| Coverage | Standard | Share/count of unique products recommended |
| Diversity | Standard | Department/category variety in Top-K |
| Novelty | Standard | Penalizes recommending only globally popular products |

### Business Utility

Use a proxy, not true profit:

```text
Business Utility@K =
  sum(estimated sales value for hit products in Top-K)
  - lambda_cost * sum(estimated discount cost for hit products in Top-K)
```

Rules:

- Count value only for hit products.
- Estimate discount cost from train-period discount history or known coupon/promotion proxy fields.
- Report `lambda_cost` and run sensitivity if time allows.
- Avoid calling the metric "profit" unless true margin data exists.

## Acceptance Criteria

Minimum acceptance:

- Time split exists and passes leakage checks.
- Popularity, category popularity, and ALS produce candidate files.
- Reranked ALS output exists with promotion/coupon signals and final ranks.
- Final result table includes Recall@10, Recall@20, NDCG@10, NDCG@20, and Business Utility@10.
- Demo can display at least one household's Top-10 recommendations.

Standard acceptance:

- ItemKNN and either BPR or documented BPR attempt are included.
- Reranking ablation table includes all required variants.
- Diversity and coverage metrics are reported.
- Budget slider behavior is connected to coupon assignment or cost threshold.
- Validation-selected weights are documented and final test results are reported once.

Bonus acceptance:

- LightGCN result is generated using the shared candidate schema.
- Lambda/budget sensitivity chart is included.
- Demo has stable screenshots or a backup recording.

## Reporting Format

Use this result table shape:

| model_variant | recall_at_10 | recall_at_20 | ndcg_at_10 | ndcg_at_20 | coverage | diversity | novelty | business_utility_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| popularity |  |  |  |  |  |  |  |  |
| category_popularity |  |  |  |  |  |  |  |  |
| als |  |  |  |  |  |  |  |  |
| als_full_reranking |  |  |  |  |  |  |  |  |

Every table should include:

- Split used: validation or test.
- Candidate depth: Top-50 or Top-100.
- Final K: 10 or 20.
- Data version or run date.
- Any product/household filtering rule.
