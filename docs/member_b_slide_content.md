# Member B PPT Content: Candidate Generation Models

## Slide 1: Candidate Generation Role

**Title:** First-Stage Candidate Generation

**Main message:** Before deciding which products deserve coupons or promotion support, PromoMind first predicts which products each household is likely to buy next.

**Bullets:**

- Input: household-product interactions from the training weeks.
- Output: Top-K product candidates for each household.
- Standard output: `household_id`, `product_id`, `base_score`, `model_name`, `base_rank`.
- These candidates are passed to the promotion-aware reranking stage.

**Speaker note:** This stage answers the pure recommendation question: "What is this household likely to purchase next?" The business-aware layer comes after this, so we keep base recommendation scores separate from promotion and coupon signals.

## Slide 2: Baselines

**Title:** Baselines: Popularity and Category Popularity

**Main message:** Strong baselines are necessary because grocery purchases have repeated and mass-market patterns.

**Bullets:**

- Popularity baseline ranks products by global purchase volume.
- Category popularity adds household-level category preference.
- Seen products are filtered out when generating candidates.
- These baselines set the minimum performance bar for ALS and BPR.

**Speaker note:** A grocery recommender can look good by repeatedly recommending common staples. That is why we compare advanced models against both global popularity and a category-aware version.

## Slide 3: Collaborative Filtering Models

**Title:** ItemKNN and Matrix Factorization

**Main message:** Collaborative filtering captures product co-purchase and household-specific preference patterns beyond global popularity.

**Bullets:**

- ItemKNN scores products similar to a household's previous purchases.
- Implicit ALS learns household and product latent factors from purchase behavior.
- BPR directly optimizes pairwise ranking between purchased and unpurchased items.
- All models emit the same Top-K candidate schema.

**Speaker note:** ALS is our main model because supermarket data has implicit feedback. A purchase is positive feedback, but a missing purchase is not necessarily a negative rating.

## Slide 4: Validation and Output

**Title:** Model Selection and Evaluation

**Main message:** Candidate models are selected using validation-week ranking metrics before reranking is applied.

**Bullets:**

- Metrics: Recall@10/20, NDCG@10/20, Coverage@20, Diversity@20, Novelty@20.
- ALS tuning grid tests factor size, regularization, iterations, and alpha.
- BPR tuning grid tests factor size, learning rate, regularization, and epochs.
- Final files: `candidates_als.csv`, `model_comparison.csv`, and tuning result tables.

**Speaker note:** We do not tune candidate generation using discount or coupon outcomes. Promotion-aware business utility is evaluated after Member C's reranking stage.
