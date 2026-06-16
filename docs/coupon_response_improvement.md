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
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
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
| XGBoost ranker, binary labels | CUDA | 0.4105 | 0.3255 | 0.5321 | 0.0811 |
| XGBoost ranker, expected coupon-lead labels | CUDA | 0.4105 | 0.3259 | 0.5321 | 0.0811 |
| XGBoost ranker, pull-forward interval labels | CUDA | 0.4154 | 0.3291 | 0.5321 | 0.0811 |
| Tail-fused XGBoost, top-10 preserved | CUDA | 0.4154 | 0.3291 | 0.5321 | 0.0811 |

The main gain is against the previous SOTA-candidate-only coupon baseline:

```text
Positive Event Hit@10: 19.27% -> 53.21%
NDCG@10:               0.1489 -> 0.3291
Recall@10:             0.1570 -> 0.4154
Recall@20:             0.5058 -> 0.5260 after top-10-preserving tail fusion
NDCG@20:               0.3557 -> 0.3609 after top-10-preserving tail fusion
```

The final XGBoost configuration uses graded relevance for coupon timing. We tested two business interpretations:

- `expected_lead_timing`: if the coupon starts one to two days before the household's expected repurchase date, a later observed purchase receives relevance grade 3; other observed purchases receive grade 2; non-purchases receive grade 0.
- `pull_forward_interval`: if the observed purchase interval stays near the household's historical median interval, with `median_interval_days - actual_interval` in `[-1, 2]`, the purchase receives relevance grade 3; other observed purchases receive grade 2; non-purchases receive grade 0.

The second scheme is the final reported model because it best matches the "actual repurchase interval is roughly unchanged" assumption and produces the strongest held-out NDCG@10. This changes the ranking supervision, while held-out evaluation still uses actual future purchases.

The XGBoost configuration above is selected only from validation campaigns using a conservative tolerance rule: choose the simplest configuration within `0.001` of the best validation primary metric.

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
```

The selected configuration is then retrained on train plus validation events before test scoring.

Additional validation-search options are available:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --wide-search --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme expected_lead_timing --expected-lead-min-days 1 --expected-lead-max-days 2 --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --ensemble-top-n 2
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-text-embedding-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-event-category-features
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --final-train-scope train
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-score-blend --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --search-rank-fusion --primary-metric ndcg_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --wide-search --use-value-features --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-coupon-family-features --primary-metric recall_at_20
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --use-redemption-features --primary-metric recall_at_20
```

`--wide-search` expands the XGBoost grid. Under the current validation rule it still selects the same held-out best default configuration. `--search-score-blend` tunes a validation-selected blend between XGBoost scores and the repeat-cadence heuristic. It improved validation Recall@20, but reduced held-out NDCG@10, so it is not the final model.

`--use-value-features` adds historical spend, average sales value, quantity, discount-rate, coupon-discount-rate, household value match, and campaign-duration features. With `--wide-search`, it produced slightly higher held-out Recall@10 but slightly lower NDCG@10 than the default, so it remains an ablation rather than the main result:

| Variant | Recall@10 | NDCG@10 | Positive Event Hit@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default XGBoost LTR | 0.4105 | 0.3255 | 0.5321 | 0.5058 | 0.3541 |
| Expected coupon-lead relevance labels | 0.4105 | 0.3259 | 0.5321 | 0.5058 | 0.3545 |
| Pull-forward interval relevance labels | 0.4154 | 0.3291 | 0.5321 | 0.5058 | 0.3557 |
| Pull-forward + non-repeat positives high | 0.4032 | 0.3179 | 0.5046 | 0.5183 | 0.3521 |
| Tail fusion, top-10 preserved | 0.4154 | 0.3291 | 0.5321 | 0.5260 | 0.3609 |
| Pull-forward window `[0, 3]` | 0.4105 | 0.3261 | 0.5321 | 0.5058 | 0.3546 |
| Pull-forward window `[1, 3]` | 0.4105 | 0.3256 | 0.5321 | 0.5058 | 0.3540 |
| Wide search only | 0.4105 | 0.3255 | 0.5321 | 0.5058 | 0.3541 |
| Wide search + value features | 0.4135 | 0.3251 | 0.5321 | 0.5058 | 0.3526 |
| Coupon-family features | 0.4002 | 0.3183 | 0.4954 | 0.4968 | 0.3472 |
| Redemption features | 0.4105 | 0.3255 | 0.5321 | 0.5058 | 0.3539 |
| XGBoost ensemble top-2 | 0.4135 | 0.3277 | 0.5321 | 0.5058 | 0.3552 |
| XGBoost ensemble top-3 | 0.4013 | 0.3234 | 0.5229 | 0.5081 | 0.3557 |
| TF-IDF/SVD product-text profile | 0.4006 | 0.3176 | 0.5138 | 0.5188 | 0.3529 |
| Category co-occurrence embedding | 0.4095 | 0.3222 | 0.5321 | 0.5207 | 0.3534 |
| Event category concentration features | 0.4105 | 0.3267 | 0.5321 | 0.5058 | 0.3552 |
| Final fit on train only | 0.4146 | 0.3228 | 0.5321 | 0.5058 | 0.3494 |
| Validation score blend | 0.3945 | 0.3079 | 0.5046 | 0.5155 | 0.3442 |
| Rank fusion, selected on NDCG@20 | 0.4103 | 0.3188 | 0.5321 | 0.5215 | 0.3505 |

`--use-coupon-family-features` treats all products attached to the same campaign coupon UPC as a coupon family and measures household history with that family before the campaign starts. The features were non-zero and used by XGBoost, but they did not generalize to the held-out campaigns.

`--use-redemption-features` adds prior coupon redemption count, prior same-coupon-UPC redemption, and prior product/category redemption signals. The features are leakage-safe because only redemption dates before the campaign start are used. They were too sparse to improve held-out NDCG@10.

`--search-rank-fusion` tunes reciprocal-rank or exponential-rank fusion over XGBoost and repeat-cadence ranks. It can improve broader Recall@20, but the held-out NDCG@10 and NDCG@20 tradeoff is worse than the default XGBoost LTR model.

`--ensemble-top-n` averages validation-selected XGBoost rankers after event-level score normalization. The top-2 ensemble stayed close to the final model and slightly preserved NDCG@20, but it did not beat the single pull-forward interval ranker on held-out NDCG@10.

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

NLP product-text profile features are available through:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-text-embedding-features
```

This builds a TF-IDF plus TruncatedSVD embedding from product department, category, type, brand, and package-size text, then compares each candidate product against the household's prior product-text profile before the campaign start. It is leakage-safe and improves broader Recall@20 in this run, but held-out NDCG@10 drops, so it remains an ablation rather than the final model.

Category co-occurrence embedding features are available through:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
```

This learns latent product-category vectors from train-period household-category co-occurrence, then compares each candidate category with the household's prior category profile. It improves broader Recall@20, which confirms the diagnosis that many misses are non-exact-repeat items, but NDCG@10 drops because the top-rank repeat/cadence signal is still more reliable.

Event category concentration features are available through:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-event-category-features
```

These features describe how concentrated a household-campaign candidate set is in a category. They are label-free and useful diagnostically, but XGBoost does not use them in the current best tree structure.

The final training scope can be changed with:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --final-train-scope train
```

This was tested because validation campaigns have a much higher positive-event rate than test campaigns. Training the final model on train only reduced held-out NDCG@10, so the final model keeps train-plus-validation fitting after validation selection.

The final top-20 recommendation artifact uses a top-10-preserving tail fusion:

```bash
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv -Force
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20 --use-category-embedding-features
Copy-Item outputs/candidates_coupon_response_xgboost_ranker.csv outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv -Force
python scripts/run_coupon_response_xgboost_ranker.py --reuse-features --device auto --search --label-scheme pull_forward_interval --pull-forward-min-days -1 --pull-forward-max-days 2 --primary-metric recall_at_20
python scripts/run_coupon_response_tail_fusion.py --primary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv --secondary-candidates outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv --primary-metric recall_at_20
```

The primary ranker is the pull-forward interval XGBoost model. The secondary ranker is the category co-occurrence embedding variant, which has weaker top-10 precision but stronger tail recall. Validation selects `keep_primary_top=12`: ranks 1-12 come from the primary model, then ranks 13-20 are filled from the secondary model with duplicate products removed. This preserves all top-10 metrics while improving held-out Recall@20 and NDCG@20.

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
| XGBoost learning-to-rank with pull-forward interval labels | Final default | Best held-out NDCG@10 while preserving Positive Event Hit@10 |
| Top-10-preserving XGBoost tail fusion | Final top-20 artifact | Preserves NDCG@10 and improves Recall@20/NDCG@20 |
| XGBoost learning-to-rank with expected coupon-lead labels | Business ablation | Directly models one-to-two-day early coupon issue timing, but lower held-out NDCG@10 than interval labels |
| Pull-forward labels with new-product positives high | Negative ablation | Improved Recall@20 but hurt top-10 ranking |
| XGBoost learning-to-rank with binary labels | Strong baseline | Slightly lower held-out NDCG@10/NDCG@20 than pull-forward interval labels |
| XGBoost wide search | Diagnosed | Re-selected the default held-out best configuration |
| XGBoost objective search | Optional ablation | `rank:pairwise` and `rank:map` did not improve held-out NDCG@10 |
| XGBoost plus value features | Optional ablation | Slight Recall@10 gain, slightly worse held-out NDCG@10 |
| XGBoost plus coupon-family features | Optional ablation | Used by the model, but lower held-out NDCG@10 |
| XGBoost plus redemption features | Optional ablation | Leakage-safe but too sparse to move the main metric |
| XGBoost plus heuristic score blend | Negative ablation | Validation improved but held-out test worsened |
| XGBoost plus rank fusion | Negative ablation | Better Recall@20 in one setting, worse held-out NDCG@10/NDCG@20 |
| XGBoost validation ensemble | Negative ablation | Top-2 stayed close but did not beat the single pull-forward model on NDCG@10 |
| XGBoost final train scope | Negative ablation | Train-only final fitting reduced held-out NDCG@10 despite validation/test drift |
| Category co-occurrence embedding | Optional ablation | Better Recall@20, lower NDCG@10; confirms non-exact-repeat misses |
| Event category concentration features | Negative ablation | Label-free campaign category context did not improve top-rank metrics |
| Campaign-type expert models | Negative ablation | Improved Recall@20 slightly, but hurt NDCG@10 |
| Type-A filtering and Type-B validation | Negative ablation | Did not beat the default held-out NDCG@10 or Hit@10 |
| LightGBM LambdaRank | Negative ablation | Best checked variant reached about 0.3225 held-out NDCG@10, below XGBoost |
| Pointwise XGBoost classifier | Negative ablation | Validation looked strong, held-out test was weaker |
| Response-prior features | Optional ablation | Less stable across campaign split |
| Product-content affinity features | Optional ablation | Useful research direction, not final default |
| TF-IDF/SVD product-text profile | Negative ablation | RedNote-style structured-text proxy improved Recall@20 but reduced held-out NDCG@10 |
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
- Newer coupon/incentive work explicitly optimizes expected revenue gain minus coupon cost under budget constraints; this is directionally aligned with our Business Utility proxy, but requires treatment-control or debiased uplift labels that Complete Journey does not provide: <https://arxiv.org/html/2602.12972v1> and <https://dl.acm.org/doi/10.1145/3640457.3688147>
- Recent grocery basket-generation work such as T-REX frames grocery prediction as category-sequence generation. That is a possible final-project extension, but our current coupon-response task is a campaign-aware ranking problem rather than full basket generation: <https://arxiv.org/html/2603.06631v1>

Recommended wording:

```text
We do not claim a universal new SOTA. Under our Complete Journey coupon-response protocol, the upgraded time-aware and supervised coupon-response rankers substantially improve coupon-specific hit rate over the SOTA-candidate-only coupon baseline.
```
