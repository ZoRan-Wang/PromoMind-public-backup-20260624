---
name: Reranking task
about: Promotion-aware reranking, coupon features, discount proxy, diversity, or business metrics
title: "[Reranking] "
labels: ["reranking"]
assignees: ""
---

## Goal

Describe the promotion/coupon/reranking/business-metric work.

## Inputs

- Candidate file:
- Promotion/coupon/discount feature files:
- Split used:

## Expected Outputs

- [ ] Reranked recommendation file:
- [ ] Ablation or metric table:
- [ ] Notes on weights or formula:

## Reranked Schema Checklist

- [ ] `household_id`
- [ ] `product_id`
- [ ] `base_score`
- [ ] `promo_score`
- [ ] `coupon_score`
- [ ] `discount_cost_proxy`
- [ ] `diversity_score`
- [ ] `final_score`
- [ ] `final_rank`
- [ ] `recommend_coupon`

## Leakage and Business Wording

- [ ] Promotion/coupon signals are available at recommendation time.
- [ ] Future coupon redemptions are not used for ranking.
- [ ] Business Utility is described as a proxy, not true profit.

## Acceptance Criteria

- [ ] Reranking before/after metrics are reported.
- [ ] At least one ablation comparison is included.
- [ ] Output can be consumed by the Streamlit demo.
