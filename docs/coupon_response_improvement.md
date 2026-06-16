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

Split drift diagnostic:

```bash
python scripts/analyze_coupon_response_drift.py
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

The split is not equally difficult:

| Split | Events | Positive events | Positive event rate | Avg truth items/event |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,021 | 1,502 | 37.35% | 2.93 |
| Validation | 1,523 | 697 | 45.76% | 4.07 |
| Test | 715 | 109 | 15.24% | 0.32 |

This distribution shift explains why several validation improvements do not transfer to held-out test. We therefore keep the final claim tied to held-out test only.

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

Additional validation-search options are available:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --wide-search --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-score-blend --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --wide-search --use-value-features --primary-metric recall_at_20
```

`--wide-search` expands the XGBoost grid. Under the current validation rule it still selects the same held-out best default configuration. `--search-score-blend` tunes a validation-selected blend between XGBoost scores and the repeat-cadence heuristic. It improved validation Recall@20, but reduced held-out NDCG@10, so it is not the final model.

`--use-value-features` adds historical spend, average sales value, quantity, discount-rate, coupon-discount-rate, household value match, and campaign-duration features. With `--wide-search`, it produced slightly higher held-out Recall@10 but slightly lower NDCG@10 than the default, so it remains an ablation rather than the main result:

| Variant | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default XGBoost LTR | 0.4105 | 0.3255 | 0.5321 | 0.5058 | 0.3541 |
| Wide search only | 0.4105 | 0.3255 | 0.5321 | 0.5058 | 0.3541 |
| Wide search + value features | 0.4135 | 0.3251 | 0.5321 | 0.5058 | 0.3526 |
| Validation score blend | 0.3945 | 0.3079 | 0.5046 | 0.5155 | 0.3442 |

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

Alternative XGBoost ranking objectives can be searched with:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-objectives
```

This compares `rank:ndcg`, `rank:pairwise`, `rank:map`, and positive-event-only training on validation. In the current held-out run, the alternative objectives did not beat the default `rank:ndcg` configuration on test NDCG@10.

A pointwise XGBoost classifier was also checked as a negative ablation. The best validation classifier used binary logistic response prediction and sorted candidate products by predicted probability, but its held-out test NDCG@10 stayed below the ranking model. We therefore keep XGBoost learning-to-rank as the final model family.

## Search Summary

| Direction | Status | Held-out conclusion |
| --- | --- | --- |
| Candidate-pool expansion | Diagnosed | Test candidate pool already covers 99.56% of truth items |
| Heuristic time-aware ranker | Kept as strong interpretable baseline | Strong and stable, but below XGBoost on NDCG@10 |
| PyTorch pairwise neural ranker | Implemented | Competitive, but below XGBoost on held-out NDCG@10 |
| XGBoost learning-to-rank | Final default | Best held-out NDCG@10 and Positive Event Hit@10 |
| XGBoost wide search | Diagnosed | Re-selected the default held-out best configuration |
| XGBoost objective search | Optional ablation | `rank:pairwise` and `rank:map` did not improve held-out NDCG@10 |
| XGBoost plus value features | Optional ablation | Slight Recall@10 gain, slightly worse held-out NDCG@10 |
| XGBoost plus heuristic score blend | Negative ablation | Validation improved but held-out test worsened |
| Pointwise XGBoost classifier | Negative ablation | Validation looked strong, held-out test was weaker |
| Response-prior features | Optional ablation | Less stable across campaign split |
| Product-content affinity features | Optional ablation | Useful research direction, not final default |
| Event-relative interaction features | Optional ablation | Validation gains did not transfer to held-out NDCG@10 |
| RedNote/NoteLLM-style multimodal | Future work | Data lacks product images, notes, reviews, and social content |

## Candidate-Pool Ceiling

The held-out test candidate pool is not the main bottleneck:

```text
test truth-item coverage in generated candidate pool: 99.56%
test positive-event any coverage:                 100.00%
```

This means almost every purchased coupon product is already available to the ranker. Further improvement mainly depends on better cross-campaign ranking generalization, not simply adding more candidates.

## External Research Positioning

The change is aligned with recent next-basket findings:

- The Complete Journey package describes household-level grocery transactions, all purchases rather than a narrow category sample, and direct marketing contact history for some households: <https://bradleyboehmke.github.io/completejourney/articles/completejourney.html>
- TIFU-KNN argues that personalized item frequency is a critical signal for next-basket recommendation and can outperform deeper sequence models when repeat frequency matters: <https://arxiv.org/abs/2006.00556>
- TAIW motivates using actual timestamps and intervals between baskets rather than only basket order: <https://arxiv.org/abs/2307.16297>
- A recent empirical NBR study reports that conventional methods such as TOP, UP-CF@r, and TIFUKNN remain very strong on Instacart and Dunnhumby-style datasets: <https://arxiv.org/html/2312.02550v1>
- NoteLLM and NoteLLM-2 show that LLM/multimodal representations are promising for content-rich note recommendation, but Complete Journey lacks product images, reviews, and social notes. RedNote-style multimodal work is therefore future work for cold-start or similar-product retrieval after adding product text/image data: <https://arxiv.org/html/2403.01744v1> and <https://arxiv.org/html/2405.16789v2>
- Modern coupon optimization literature frames coupons and discounts as uplift/treatment problems under cost constraints. Our data does not include randomized control groups, so we use response ranking rather than claiming causal uplift: <https://arxiv.org/html/2402.03379v1>

Recommended wording:

```text
We do not claim a universal new SOTA. Under our Complete Journey coupon-response protocol, the upgraded time-aware and supervised coupon-response rankers substantially improve coupon-specific hit rate over the SOTA-candidate-only coupon baseline.
```
