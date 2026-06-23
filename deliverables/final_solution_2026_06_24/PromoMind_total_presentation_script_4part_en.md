# PromoMind Full Presentation Script (4-Part Speaker Version)

## Usage Notes

- Matching deck: `deliverables/final_solution_2026_06_24/PromoMind_final_presentation_zh.pptx`
- Recommended total duration: 10-12 minutes
- Speaker structure: 4 parts, about 2.5-3 minutes per person
- Required coverage: finalized dataset, problem, algorithms, experimental results, applicability, significance, extensions with evidence, working recommender demo, future work

## Part 1: Dataset, Problem Definition, Course Requirement Coverage

Corresponding slides: 1-4.

### Slide 1: Title

Good morning everyone. Our group project is PromoMind, a promotion-aware grocery basket and coupon recommendation system for retail marketing optimization.

The core question is: under a limited marketing budget, which campaign coupon products should be recommended to a specific household. The project moves from next-purchase prediction to coupon product response priority ranking inside a concrete marketing campaign.

On the held-out campaign test split, our final model reaches Recall@10 of 0.4187, NDCG@10 of 0.3304, and Positive Event Hit@10 of 54.13%. These results show that the system can place more truly responsive products into the Top-10 recommendation list and improve the probability that a positive response event is covered.

### Slide 2: Course Requirements Coverage

This slide shows how our final delivery covers the course requirements.

First, the finalized dataset and problem are clearly defined. We use The Complete Journey dataset and define the task as household-campaign coupon-product ranking.

Second, the algorithm section covers multiple levels of recommender methods, including Popularity, ItemKNN, ALS, BPR, TIFU-KNN, Cornac TIFUKNN, and the final XGBoost Learning-to-Rank model.

Third, the experimental results use a held-out campaign test split and report Recall, NDCG, and Positive Event Hit.

Fourth, the extension section includes a neural ranker, text features, category embeddings, and tail fusion. Each direction has experimental evidence.

Fifth, we built a working recommender demo with Streamlit. The demo shows household selection, campaign selection, Top-K recommendations, coupon flags, and a fallback sample.

### Slide 3: Dataset

Our dataset is The Complete Journey, a real grocery retail dataset. It covers 2,469 households and contains about 1.47 million product-level transaction records.

The tables include transactions, products, promotions, campaigns, coupons, coupon redemptions, and demographics. Transactions provide purchase history and future response labels. Products provide department, category, and brand information. Campaigns and coupons define the exposure window and coupon product pool. Promotions provide display and mailer context.

One important design choice is the supervised target. We use coupon-product purchase response. Coupon redemption is very sparse in this dataset, so using redemption as the primary label would make the learning signal unstable. We define a positive response as purchasing the corresponding coupon product within 5 days after the campaign start. This target is more suitable for coupon-product response ranking.

### Slide 4: Problem Definition

The final recommendation problem is defined as ranking coupon products inside each household-campaign exposure.

If a household buys a coupon product within 5 days after the campaign start, that product is labeled as a positive response.

This definition is closer to coupon targeting than ordinary next-basket prediction. The ranked items are limited to the coupon product pool already provided by the campaign. The model goal is to rank more responsive coupon products near the top.

In our evaluation protocol, we use 715 test events, including 109 positive events. All features only use information available before the campaign start to avoid future information leakage.

Part 1 ends here. Part 2 will introduce the system architecture, candidate generation, and first-stage model results.

## Part 2: System Architecture, Candidate Generation, First-Stage Results

Corresponding slides: 5-6.

### Slide 5: System Architecture

PromoMind uses a two-stage recommendation architecture.

The first step is raw retail tables, including transactions, products, promotions, campaigns, coupons, and related source tables.

The second step is cleaning and chronological time split. We split train, validation, and test data by time order, so the experimental protocol matches the prediction setting of a real deployed recommender system.

The third step is candidate generation. The goal of this stage is coverage. It first finds products that are likely to be relevant to each household.

The fourth step is XGBoost Learning-to-Rank. This stage focuses the candidate space on campaign coupon products and learns response priority at the household-campaign-product level.

The fifth step is tail fusion and demo. The final artifact is `outputs/reranked_recommendations.csv`. The demo can read this file and display Top-K coupon product recommendations.

The key architecture idea is simple: Stage 1 handles coverage, and Stage 2 handles marketing-aware ranking.

### Slide 6: Candidate Generation Results

In the first stage, we implemented several candidate generation methods, including Personal Top Frequency, TIFU-KNN style, Strong Hybrid, Official Cornac TIFUKNN, and Final rank ensemble.

The results show that Personal Top Frequency reaches NDCG@10 of 0.3790. This means a household's own repeat-purchase habit is very strong in the grocery setting. TIFU-KNN style reaches NDCG@10 of 0.3851, showing that time-aware preference modeling is useful. Strong Hybrid reaches 0.3935. Official Cornac TIFUKNN reaches 0.4210. The final rank ensemble reaches the highest NDCG@10 of 0.4278.

The conclusion from this stage is that repeat frequency, temporal decay, and similar-household preference are more important than generic matrix factorization in the grocery setting. Grocery shopping has strong periodicity and repeat-purchase behavior, so the model needs to capture that stable rhythm.

The first-stage candidate pool covers 99.56% of the test truth items in the later analysis. This shifts the main bottleneck from finding relevant products to ranking the correct products near the top.

Part 2 ends here. Part 3 will introduce coupon-response ranking, final test results, and extension experiments.

## Part 3: Coupon-Response Ranking, Experimental Results, Extensions With Evidence

Corresponding slides: 7-9.

### Slide 7: Coupon-Response Learning-to-Rank

The second stage uses XGBoost `rank:ndcg` for Learning-to-Rank.

The training unit is household-campaign-product. Each household, each campaign, and each coupon product form one ranking sample.

The label uses pull-forward interval relevance. This label considers whether the product was purchased, when it was purchased, and how that purchase timing relates to the ideal early coupon delivery window. The model therefore learns response priority.

The main features include repeat signal, cadence signal, global signal, and category embedding. Repeat signal shows whether the household bought this product before. Cadence signal shows whether the product matches the household's historical repurchase rhythm. Global signal shows whether the campaign product has broad response evidence. Category embedding supplements tail candidates that are not exact repeat purchases.

The business interpretation of this stage is important. Supervised LTR learns response priority. It does not claim that the coupon caused the purchase.

### Slide 8: Final Held-Out Test Results

The final held-out test results show that Final tail fusion reaches Recall@10 of 0.4187, NDCG@10 of 0.3304, Positive Event Hit@10 of 0.5413, Recall@20 of 0.5207, and NDCG@20 of 0.3594.

Compared with the candidate-only coupon baseline, Recall@10 increases from 0.1570 to 0.4187, NDCG@10 increases from 0.1489 to 0.3304, and Positive Event Hit@10 increases from 19.27% to 54.13%.

These results show that coupon baseline ranking based only on candidate models is not enough. After adding time-aware features, repurchase rhythm, and campaign-level response features, the model ranks positive-response coupon products into Top-10 more accurately.

The Final tail fusion strategy keeps the top 7 positions from the main XGBoost ranker and fills later positions with category co-occurrence embedding candidates. This design preserves XGBoost's strength in Top-10 precision and uses category embedding to support Top-20 recall.

### Slide 9: Extensions With Evidence

The project includes several extension experiments.

The PyTorch pairwise neural ranker is competitive, with held-out NDCG@10 below XGBoost. After comparing expected-lead labels and pull-forward labels, we selected pull-forward interval labels. TF-IDF and SVD product text features help part of Recall@20, with weaker Top-10 ranking than the final model. Category co-occurrence embedding is useful as a tail source. Rank fusion and score fusion produce validation gains, with less stable transfer to the held-out test split.

These experiments support the following research conclusion: product text in The Complete Journey is relatively structured, and text features can supplement part of the coverage. Timing and repeat behavior are more stable for Top-10 ranking. More complex models do not automatically produce better generalization.

Part 3 ends here. Part 4 will introduce business interpretation, demo, applicability, limitations, and future work.

## Part 4: Applicability, Significance, Demo, Limitations, Future Work

Corresponding slides: 10-12.

### Slide 10: Promotion-Aware Business Interpretation

PromoMind output should be interpreted as coupon product response priority for already exposed households.

In business use, it can help a marketing team decide display order and coupon flags within a campaign product pool. The Business Utility proxy can explain marketing constraints, such as coupon slots or discount-cost sensitivity.

The project does not claim causal coupon uplift. It does not claim true profit maximization. It does not claim universal applicability to every grocery recommendation setting. The dataset has no randomized treatment-control design and no real product margin.

This boundary makes the project conclusion reliable. We can show that the model is effective under the current observation-based response ranking protocol. Future causal uplift work would require a new experimental design or propensity correction.

### Slide 11: Working Recommender Demo

The demo starts with Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

The demo supports household and campaign selection. It shows purchase history, coupon start time, prediction time, and Top-K coupon product recommendations.

The page also shows coupon flags, metadata, and key ranking fields. The formal output file is `outputs/reranked_recommendations.csv`. If the live environment cannot load the full result file, `top10_recommendation_sample.csv` can be used as the fallback sample.

The demo turns offline experimental results into an interactive recommender prototype. Users can see which coupon products are recommended to a household under a campaign. Users can also control the number of displayed items through coupon slots or marketing budget.

### Slide 12: Applicability, Limitations, and Future Work

PromoMind applies to supermarket loyalty coupon recommendation, e-commerce campaign product prioritization, household-personalized ranking inside a fixed campaign product pool, and coupon slot allocation under marketing budget constraints.

The project significance is that it separates "what the user may buy" from "which products deserve marketing support." Ordinary recommender systems focus on relevance. PromoMind further incorporates campaign context, coupon context, response timing, and business constraints into the ranking process.

The limitations include four points. There is no treatment-control design, so causal uplift cannot be estimated. There is no real product margin, so true profit maximization cannot be performed. Coupon redemption and demographics are sparse. The demo is still an offline prototype.

Future work has four directions. First, introduce randomized treatment-control or propensity correction to build a causal uplift model. Second, use real margin data for cost-aware optimization. Third, add richer text and image signals to explore multimodal retrieval. Fourth, apply time-aware calibration for campaign drift and convert the offline CSV pipeline into a monitorable and retrainable production service.

Final summary: PromoMind completes an end-to-end recommender system workflow from real retail data, time-safe data processing, and multi-model candidate generation to campaign-aware coupon-response ranking, tail fusion, business interpretation, and Streamlit demo. The final results show that stable repeat-purchase behavior and timing rhythm are core signals for this grocery coupon-response task.

## 4-Speaker Assignment Overview

| Part | Speaker | Slides | Core Content |
| --- | --- | --- | --- |
| Part 1 | Speaker A | 1-4 | Topic, course requirement coverage, dataset, final problem definition |
| Part 2 | Speaker B | 5-6 | System architecture, candidate generation, first-stage model results |
| Part 3 | Speaker C | 7-9 | XGBoost LTR, final experimental results, extension experiments |
| Part 4 | Speaker D | 10-12 | Business interpretation, demo, applicability, limitations, future work |

## One-Minute Closing Version

PromoMind's final task is response priority ranking for coupon products within each household-campaign exposure. The project uses The Complete Journey real retail dataset and follows a two-stage architecture: the first stage uses multiple next-basket candidate models for coverage, and the second stage uses time-aware XGBoost LTR plus tail fusion for coupon-response ranking. The final model reaches Recall@10 of 0.4187, NDCG@10 of 0.3304, and Positive Event Hit@10 of 54.13% on the held-out test split, clearly above the candidate-only coupon baseline. The project also provides a Streamlit working demo, final result files, and a complete report. Its main value is helping marketing teams choose and rank coupon products more reasonably under a limited budget. Future work can extend it to causal uplift, real margin optimization, multimodal product understanding, and production-grade recommendation service.
