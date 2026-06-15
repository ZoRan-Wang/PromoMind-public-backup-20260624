# Coupon-Response Ranking Upgrade

## Why We Changed The Method

The ordinary next-basket models predict likely grocery purchases, but they do not specifically optimize for campaign coupon products. In the earlier timing demo, the model could predict some repurchases within five days, but coupon-eligible hits were weak.

The upgraded runner reframes the second stage as a coupon-response ranking task:

```text
For each household-campaign exposure, rank active campaign coupon products.
Success = the household buys the ranked coupon product within 5 days after campaign start.
```

## Method

Script:

```bash
python scripts/run_coupon_response_ranker.py --device auto --primary-metric ndcg_at_10
```

Supervised GPU learning-to-rank variant:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --primary-metric recall_at_20
```

PyTorch pairwise neural variant:

```bash
python scripts/run_coupon_response_neural_ranker.py --reuse-features --device auto
```

Fast rerun after features are generated:

```bash
python scripts/run_coupon_response_ranker.py --reuse-features --device auto --primary-metric ndcg_at_10
```

The scorer uses:

- base signal from `outputs/candidates_sota_ensemble.csv`
- household-product repeat frequency
- time-aware cadence signal based on last purchase and median interval
- household category affinity
- campaign coupon-product global popularity
- historical discount signal

Weights are searched on validation campaigns only, then fixed for test campaigns.

## GPU

The local machine has an NVIDIA RTX GPU and PyTorch CUDA is available. The heuristic runner uses CUDA for vectorized score calculation when `--device auto` resolves to `cuda`.

The PyTorch pairwise ranker trains on CUDA. The XGBoost ranker uses `device=cuda` and `tree_method=hist` when CUDA is available.

Feature engineering and event-level ranking still use pandas on CPU because cuDF/CuPy are not installed locally.

## Current Results

Generated output:

- `outputs/coupon_response_features.csv`
- `outputs/coupon_response_truth.csv`
- `outputs/candidates_coupon_response_ranker.csv`
- `outputs/reranked_recommendations.csv`
- `outputs/coupon_response_weight_search.csv`
- `outputs/coupon_response_model_comparison.csv`
- `outputs/coupon_response_final_model_comparison.csv`
- `outputs/coupon_response_xgboost_search.csv`

Validation has 1,523 household-campaign events and 697 positive events.
Test has 715 household-campaign events and 109 positive events.

Test comparison after the first heuristic upgrade:

| Model | Recall@10 | NDCG@10 | Positive Event Hit@10 | All Event Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| Coupon base intersection | 0.1570 | 0.1489 | 0.1927 | 0.0294 |
| Coupon global popularity | 0.1884 | 0.1045 | 0.2569 | 0.0392 |
| Coupon repeat-cadence | 0.4005 | 0.2979 | 0.5046 | 0.0769 |
| Coupon-response ranker | 0.3945 | 0.3145 | 0.5046 | 0.0769 |

Additional supervised model exploration:

| Model | Device | Recall@10 | NDCG@10 | Positive Event Hit@10 | All Event Hit@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch pairwise neural ranker | CUDA | 0.3983 | 0.3132 | 0.4954 | 0.0755 |
| XGBoost ranker, validation-selected | CUDA | 0.4105 | 0.3255 | 0.5321 | 0.0811 |

The main gain is against the previous SOTA-candidate-only coupon baseline:

```text
Positive Event Hit@10: 19.27% -> 53.21%
NDCG@10:               0.1489 -> 0.3255
Recall@10:             0.1570 -> 0.4105
```

The XGBoost configuration above is selected only from validation campaigns using a conservative tolerance rule: choose the simplest configuration within `0.001` of the best validation primary metric.

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --primary-metric recall_at_20
```

The selected configuration is then retrained on train plus validation events before test scoring.

An optional historical response-prior feature set is available through:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-response-priors
```

It is not the default because validation-selected response priors were less stable under the held-out campaign split.

Content-affinity features inspired by product text/multimodal recommendation are also available:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-content-features
```

These features use department, brand, product category, and product type affinity. They improved validation in one run but reduced held-out NDCG@10, so they are kept as an optional exploration rather than the default final model.

Event-relative rank and interaction features are available through:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-derived-features
```

They include repeat-by-cadence, base-by-repeat, interval-error, and per-event percentile ranks. They also showed validation gains but weaker held-out NDCG@10, so they remain optional rather than default.

## Candidate-Pool Ceiling

The held-out test candidate pool is not the main bottleneck:

```text
test truth-item coverage in generated candidate pool: 99.56%
test positive-event any coverage:                 100.00%
```

This means almost every purchased coupon product is already available to the ranker. Further improvement mainly depends on better cross-campaign ranking generalization, not simply adding more candidates.

## External Research Positioning

The change is aligned with recent next-basket findings:

- TIFU-KNN and repeat-frequency baselines remain strong for grocery next-basket recommendation.
- Time-aware item weighting methods such as TAIW motivate using real elapsed time instead of only basket order.
- RedNote/NoteLLM-style multimodal LLM recommendation is promising for content-rich items, but Complete Journey lacks product images, reviews, and social notes. It is better positioned as future work for cold-start or similar-product retrieval after adding product text/image data.

Recommended wording:

```text
We do not claim a universal new SOTA. Under our Complete Journey coupon-response protocol, the upgraded time-aware and supervised coupon-response rankers substantially improve coupon-specific hit rate over the SOTA-candidate-only coupon baseline.
```
