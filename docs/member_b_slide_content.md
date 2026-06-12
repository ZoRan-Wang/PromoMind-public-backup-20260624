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

## Slide 2: Strong Grocery Baselines

**Title:** Repeat-aware Grocery Baselines

**Main message:** Grocery next-basket recommendation is repeat-heavy, so strong baselines must model household-level repeat frequency.

**Bullets:**

- Popularity baseline ranks products by global purchase volume.
- Category popularity adds household-level category preference.
- Personal Top Frequency ranks each household's own repeated products.
- Repeat purchases are allowed in the main experiment because grocery baskets contain recurring staples.
- These baselines set the real performance bar for ALS, BPR, and ItemKNN.

**Speaker note:** A grocery recommender should not automatically remove previously purchased products. In supermarkets, repeat purchases are part of the target behavior, so we separately monitor novelty and coverage instead of forcing every recommendation to be new.

## Slide 3: Community-level Next-basket Model

**Title:** TIFU-KNN Style Candidate Generation

**Main message:** The strongest single model is a TIFU-KNN style next-basket recommender: time-decayed household frequency plus neighbor household preference.

**Bullets:**

- Build each household's time-decayed product-frequency vector.
- Find similar households using cosine similarity.
- Combine personal repeat preference with neighbor preference.
- Output the same Top-K candidate schema for reranking.

**Speaker note:** This follows the next-basket recommendation literature, where TIFU-KNN and recency-aware user collaborative filtering are strong methods on grocery-style datasets.

## Slide 4: Validation and Output

**Title:** Model Selection and Evaluation

**Main message:** The final Member B handoff uses the strongest validation candidate source, not the most complicated model.

**Bullets:**

- Metrics: Recall@10/20, NDCG@10/20, Coverage@20, Diversity@20, Novelty@20.
- TIFU-KNN has the best Recall@10 and Recall@20.
- The strong hybrid has the best NDCG@10 and NDCG@20.
- Final handoff files: `candidates_tifu_knn.csv` and `candidates_hybrid_strong.csv`.

**Speaker note:** ALS and BPR remain useful comparison models, but they are not the final accuracy leaders on this grocery split. Promotion-aware business utility is evaluated after Member C's reranking stage.

## Slide 5: Current Result Table

**Title:** Full Validation Results

**Main message:** Repeat-aware next-basket models are much stronger than generic collaborative filtering on this task.

**Bullets:**

- Personal Top Frequency: Recall@10 0.0984, NDCG@10 0.3790.
- TIFU-KNN style: Recall@10 0.1011, NDCG@10 0.3851.
- Strong Hybrid: Recall@10 0.1001, NDCG@10 0.3888.
- Previous generic models are lower: ItemKNN NDCG@10 0.1980, ALS NDCG@10 0.0743.

**Speaker note:** The key takeaway is methodological: on grocery data, the community-level answer is not "use a deeper model first"; it is "model repeat consumption and temporal frequency correctly."
