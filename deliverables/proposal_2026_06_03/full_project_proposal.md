# PromoMind: Studying Promotion-aware Reranking For Grocery Basket Recommendation

## 1. Project Motivation

Grocery retailers have rich household purchase histories but often use promotion and coupon decisions in a broad, inefficient way. A normal recommender system asks which products a household is likely to buy. PromoMind extends that question: given limited promotion or coupon budget, which likely purchases should be promoted, which should be left alone, and how should the recommendation list balance relevance, discount cost, and product diversity?

The research objective is to study whether promotion and coupon signals can improve multi-objective grocery recommendation under sparse, observational retail data. The working system and Streamlit demo are supporting artifacts. The research core is the comparison of candidate-generation models and the controlled evaluation of promotion-aware reranking.

## 2. Research Context And Gap

This project connects several recommender systems topics from the course:

- Implicit feedback: purchases are positive interactions, but non-purchases are not explicit dislikes.
- Matrix factorization and pairwise ranking: ALS and BPR are natural candidates for sparse household-product data.
- Evaluation: chronological ranking evaluation is needed because grocery recommendation is a future-prediction task.
- Auxiliary information: product categories, promotions, coupons, campaigns, discounts, and demographics can support reranking and explanation.
- Deep recommendation: LightGCN can be tested as a graph-based extension if time permits.

The gap we study is not simply "can we predict grocery baskets." The more specific research gap is whether marketing context can improve the trade-off among relevance, coupon/promotion cost, and list diversity without using unavailable true margin labels or making unsupported causal uplift claims.

## 3. Dataset And Collection Method

The project will use The Complete Journey dataset, released through the public `completejourney` project. The dataset is suitable because it contains household-level grocery transactions, product metadata, promotions, campaigns, coupons, coupon redemptions, and partial household demographics.

Collection method:

- We will use the public `completejourney` package/source repository and its original RDS/RDA data artifacts.
- The raw data artifacts are already stored in this repository under `data/raw/completejourney/`.
- The script `scripts/download_completejourney.R` documents the reproducible extraction/export path from the package/raw artifacts.
- We will not scrape private websites or collect new personal data.
- Generated processed CSV files will remain reproducible outputs from the raw files and scripts.

Expected data size:

- 2,469 households.
- About 1,469,307 transaction rows.
- About 92,331 products.
- About 116,204 coupon-product-campaign mappings.
- About 6,589 campaign-household exposure records.
- About 2,102 coupon redemption records.
- About 801 households with demographics.

The sparse coupon redemption and partial demographics coverage are treated as project risks, not blockers. The main supervised task remains next-week product purchase prediction.

Coupon redemption will be analyzed cautiously because it is sparse and post-treatment. It will not be the only supervised target, and target-period redemption events will not be used as ranking features.

## 4. Recommendation Problem

For each household `h`, given all transactions before target week `t`, predict the Top-K products that the household is likely to purchase in the next week or future holdout period.

The core label is implicit feedback:

```text
y_hjt = 1 if household h buys product j in the target period
```

The model will use purchase interactions rather than explicit ratings. Purchase counts, spend, and recency can be used as confidence or feature signals.

Planned chronological split:

| Split | Weeks | Purpose |
| --- | --- | --- |
| Train | 1-40 | Fit recommenders and historical features |
| Validation | 41-46 | Tune ALS/BPR and reranking weights |
| Test | 47-53 | Final held-out evaluation |

This time-based split is important because real recommenders use past behavior to predict future baskets.

## 5. Research Questions And Hypotheses

This project is framed as a recommender systems research project with a working demo. The working system is the artifact; the research value comes from comparing models and testing how marketing-aware reranking changes recommendation trade-offs.

| RQ | Hypothesis | Main comparison |
| --- | --- | --- |
| RQ1: How well do implicit-feedback recommenders predict next-period grocery baskets under a chronological split? | H1: Personalized models such as ALS/BPR will outperform global and category popularity on Recall@K and NDCG@K. | Popularity, Category Popularity, ItemKNN, ALS, BPR |
| RQ2: Does promotion-aware reranking improve business-oriented utility without unacceptable ranking loss? | H2: Adding promotion/coupon signals will improve Business Utility@K, with at most a small decrease in Recall@K/NDCG@K. | ALS vs ALS + promotion/coupon reranking |
| RQ3: How does discount-cost penalization change the recommendation trade-off? | H3: Adding a discount-cost penalty will reduce wasteful coupon allocation and improve revenue-minus-discount proxy utility. | Reranking with vs without cost penalty |
| RQ4: Can diversity control improve list health in grocery recommendation? | H4: Diversity reranking will increase category diversity and coverage while keeping Recall@K within an acceptable drop threshold. | Full reranking with vs without diversity |
| RQ5: Are more complex models worth it for this sparse retail setting? | H5: LightGCN may improve ranking quality, but ALS/BPR may offer a better accuracy-runtime trade-off. | ALS/BPR vs LightGCN, if completed |

## 6. Methodology

PromoMind is designed as a two-stage recommender.

### Stage 1: Candidate Generation

The first stage produces likely products for each household.

Required models:

- Global Popularity baseline.
- Category Popularity baseline.
- Implicit ALS matrix factorization.

Standard models if time allows:

- ItemKNN.
- BPR matrix factorization.

Bonus extension:

- LightGCN on the household-product interaction graph.

Library acknowledgements:

- `implicit` may be used for ALS/BPR-style implicit-feedback recommendation.
- RecBole may be used for LightGCN.
- Cornac may be considered for course-aligned recommendation experiments if needed.

Every model must output the same candidate schema:

| Column | Meaning |
| --- | --- |
| `household_id` | Raw household id |
| `product_id` | Raw product id |
| `base_score` | First-stage recommendation score |
| `model_name` | Candidate generator name |
| `base_rank` | Rank before reranking |

### Stage 2: Promotion-aware Reranking

The second stage converts candidate recommendations into marketing-aware recommendations.

Planned scoring formula:

```text
final_score =
  alpha * normalized_base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

Operational signal definitions:

- `promotion_score`: product-week promotion exposure assumed available at recommendation planning time.
- `coupon_score`: household-product or household-campaign-product eligibility derived from coupon/campaign mappings, not from target-period redemption.
- `discount_cost_proxy`: estimated discount cost from train-period retail and coupon discount history.
- `diversity_score`: department/category penalty or reward to prevent the Top-K list from collapsing into one category.

Leakage prevention:

- Target-period purchases are labels only.
- Target-period sales values are used only after ranking to score hit products.
- Target-period coupon redemption is not used as a ranking feature.
- Product and household historical features are computed from train-window behavior.

The system will not claim to optimize true profit because product cost/margin is unavailable. Business value will be reported as an estimated revenue-minus-discount proxy.

## 7. Experiment Plan

Protocol:

- Train candidate models only on Weeks 1-40.
- Tune model hyperparameters and reranking weights on Weeks 41-46.
- Report final held-out results on Weeks 47-53.
- Decide product-frequency filtering from the train period and apply it consistently.
- Report household-level averaged metrics; if time permits, add paired bootstrap confidence intervals across households.
- Keep repeat-purchase and discovery settings explicit. Grocery recommendation may include repeat products unless the experiment is labeled as new-item discovery.

RQ-to-experiment mapping:

| Research question | Planned experiment | Evidence |
| --- | --- | --- |
| RQ1 | Compare Popularity, Category Popularity, ItemKNN, ALS, BPR, and optional LightGCN | Recall@K, NDCG@K, runtime notes |
| RQ2 | Compare base ALS with promotion/coupon reranking variants | Recall/NDCG plus Business Utility@K |
| RQ3 | Compare reranking with and without discount-cost penalty | Utility, discount cost, coupon assignment count |
| RQ4 | Compare reranking with and without diversity control | Coverage, diversity, Recall/NDCG guardrail |
| RQ5 | Compare ALS/BPR with LightGCN if completed | Accuracy-runtime trade-off and implementation feasibility |

Ranking metrics:

- Recall@10 and Recall@20.
- NDCG@10 and NDCG@20.

Catalog/list metrics:

- Coverage.
- Diversity.
- Novelty.

Business metric:

```text
Business Utility@K =
  sum(estimated sales value for hit products in Top-K)
  - lambda_cost * sum(estimated discount cost for hit products in Top-K)
```

Main model comparison:

- Popularity.
- Category Popularity.
- ItemKNN.
- ALS.
- BPR if completed.
- LightGCN if completed.

Reranking ablation:

- ALS only.
- ALS + promotion.
- ALS + coupon.
- ALS + promotion + coupon.
- ALS + promotion + coupon - discount cost.
- ALS + full reranking with diversity.

Sensitivity analyses if time permits:

- Vary reranking weights `beta`, `gamma`, `lambda`, and `rho`.
- Treat the demo marketing budget slider as a core visualization of coupon allocation; sensitivity curves are a bonus analysis.
- Plot Recall@K versus Business Utility@K.
- Plot Business Utility@K versus coupon budget or discount-cost threshold.
- Plot diversity versus Recall@K.

Validity controls:

- Use chronological split to reduce temporal leakage.
- Compute household/product historical features from train windows only.
- Treat coupon redemption as auxiliary because it is sparse.
- Treat demographics as optional because coverage is partial.
- Report Business Utility as a proxy, not true profit.
- Do not make causal promotion uplift claims from observational exposure data.

## 8. Expected Research Contribution And Demonstration

The expected research contribution is a controlled study of promotion-aware reranking for grocery basket recommendation. The X-factor is not simply a richer demo; it is the second-stage decision layer that studies how relevance, coupon/promotion signals, discount-cost proxy, and diversity interact.

PromoMind will generate:

- Top-K products each household is likely to buy.
- Which recommended products should receive a coupon or promotion push.
- Estimated discount-cost impact.
- A budget-aware view showing how coupon decisions change when marketing budget changes.

The proposed Streamlit demo will let the user select a household, view purchase profile summaries, inspect Top-10 recommendations, see coupon flags and reasons, and adjust a marketing budget slider.

The demo visualizes the research trade-offs. It should not be presented as evidence of final performance until the offline experiments are actually run.

## 9. Risks, Validity Threats, And Mitigations

| Risk or validity threat | Mitigation |
| --- | --- |
| Coupon redemptions are sparse | Use coupon redemption as auxiliary analysis and reranking signal, not main target |
| Demographics cover only part of households | Use demographics only for optional profile/demo or subgroup analysis |
| Product catalog is large | Filter by train-period product frequency, such as Top 10,000 products or purchase count >= 20 |
| No true product margin | Report Business Utility as revenue-minus-discount proxy, not profit |
| Promotion exposure is observational | Do not claim causal uplift; frame reranking as association-based decision support |
| Campaign seasonality may affect results | Use chronological validation/test and report split dates/weeks clearly |
| Popularity bias may dominate grocery recommendations | Include strong popularity/category baselines and novelty/coverage metrics |
| LightGCN may be too slow | Treat LightGCN as bonus; keep ALS/BPR plus reranking as the stable main path |

## 10. Expected Final Deliverables

- Reproducible data preparation pipeline.
- Candidate generation models and shared output schema.
- Promotion-aware reranking implementation.
- Offline evaluation tables.
- Streamlit demo.
- Final report and presentation with a clear discussion of applicability, limitations, and future work.
