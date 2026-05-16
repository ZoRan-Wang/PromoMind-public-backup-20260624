---
name: Model task
about: Candidate-generation models and ranking metrics
title: "[Model] "
labels: ["model"]
assignees: ""
---

## Goal

Describe the candidate-generation model, baseline, or evaluation update.

## Inputs

- Interaction split:
- Product/household features:
- Existing candidate files:

## Expected Outputs

- [ ] Candidate file:
- [ ] Metrics table:
- [ ] Parameter/tuning notes:

## Candidate Schema Checklist

- [ ] `household_id`
- [ ] `product_id`
- [ ] `base_score`
- [ ] `model_name`
- [ ] `base_rank`

## Acceptance Criteria

- [ ] Runs on the agreed split.
- [ ] Excludes or clearly labels train-period repeat purchases.
- [ ] Reports Recall@10/20 and NDCG@10/20 where applicable.
- [ ] Output can be consumed by reranking without schema edits.
