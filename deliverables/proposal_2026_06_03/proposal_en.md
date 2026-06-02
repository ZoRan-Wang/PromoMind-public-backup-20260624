# PromoMind: A Promotion-aware Grocery Basket and Coupon Recommender

## 1. Project Motivation

Grocery and fresh-food retailers usually have large amounts of member purchase history, but promotion and coupon delivery is often broad and inefficient. A standard recommender mainly answers: "What is this household likely to buy next?" PromoMind studies a more realistic retail decision problem: when promotion budget and coupon cost are limited, which recommended products deserve promotion or coupon support, and which products should simply remain normal recommendations?

The research objective is to evaluate whether promotion, coupon, discount-cost, and product-diversity signals can improve next-basket grocery recommendation on real retail data. The system first predicts each household's likely Top-K products for the next period, then applies promotion-aware reranking to produce a recommendation list that is more useful for marketing decisions.

## 2. Dataset And Collection Method

We will use the public dataset The Complete Journey. The dataset is available through the `completejourney` project and records household-level shopping behavior from a real retailer. It includes not only transaction records, but also product metadata, promotions, campaigns, coupons, coupon redemptions, and partial household demographics. Therefore, it is more suitable for studying "recommendation plus promotion decision" than datasets that only contain orders and product categories.

Data collection and usage method:

- Use the public `completejourney` data source and original RDS/RDA files.
- The raw data files are already stored in `data/raw/completejourney/`.
- `scripts/download_completejourney.R` documents the reproducible data export workflow.
- We will not scrape private data or collect additional personal data.
- Future CSV files or feature tables will be reproducible outputs from the raw files and scripts.

Expected data size:

| Data component | Size |
| --- | --- |
| households | 2,469 households |
| transactions | about 1,469,307 transaction rows |
| products | about 92,331 products |
| coupon-product-campaign mapping | about 116,204 rows |
| campaign-household exposure | about 6,589 rows |
| coupon redemptions | about 2,102 rows |
| demographics | about 801 households with profile information |

What the dataset looks like:

- The transaction table is close to item-level receipt data. Each row usually corresponds to one product purchased by one household in one basket.
- Important fields include `household_id`, `basket_id`, `product_id`, `week`, `quantity`, `sales_value`, `retail_disc`, `coupon_disc`, and `coupon_match_disc`.
- The product table provides department, product category, brand, and product description.
- Promotion and coupon-related tables provide product-week promotion exposure, coupon-product-campaign mappings, household campaign exposure, and redemption history.
- The demographics table provides partial household profile information such as income range, age range, and household size.

Two main data risks will be handled explicitly. First, coupon redemption has only about 2,102 rows and is sparse, so "whether a coupon is redeemed" will not be used as the only supervised target. Second, demographics only cover part of the households, so they will be used as optional analysis and demo information, not as required inputs for the main model.

## 3. Recommendation Problem Definition

The project task is: given household `h`'s full purchase history before target week `t`, recommend the Top-K products that the household is most likely to purchase in the future target period.

Because grocery data has no ratings, purchase behavior is treated as implicit feedback. The core label can be written as:

```text
y_hjt = 1 if household h buys product j in the target period
```

We plan to use a chronological split instead of a random split:

| Split | Weeks | Purpose |
| --- | --- | --- |
| Train | 1-40 | Train recommender models and historical features |
| Validation | 41-46 | Tune hyperparameters and choose reranking weights |
| Test | 47-53 | Final offline evaluation |

This better matches a real recommender system: the system can only use past behavior to predict future behavior, and it should not leak information from future shopping records.

## 4. Model Methodology

PromoMind uses a two-stage recommendation architecture.

The first stage is candidate generation. It uses recommendation models to generate candidate products that each household may buy. Planned models include:

- Global Popularity baseline.
- Category Popularity baseline.
- ItemKNN.
- Implicit ALS matrix factorization.
- BPR matrix factorization, if time allows.
- LightGCN, if time allows, as a deep recommendation or graph recommendation extension.

External libraries will be clearly acknowledged in the presentation:

- `implicit` may be used for ALS/BPR-style implicit-feedback recommenders.
- RecBole may be used for LightGCN.
- Cornac may be considered as a course-aligned experiment framework.

The second stage is promotion-aware reranking. After the first-stage model outputs a base relevance score, the system reranks candidate products using promotion, coupon, discount-cost, and diversity signals:

```text
final_score =
  alpha * normalized_base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

Definitions:

- `normalized_base_score` represents the purchase likelihood from the recommender model.
- `promotion_score` represents whether the product has promotion exposure in the recommendation target period.
- `coupon_score` represents whether a household-product or household-campaign-product coupon match exists.
- `discount_cost_proxy` is built from historical retail discount, coupon discount, and related discount fields.
- `diversity_score` prevents the recommendation list from concentrating on only one product category.

The key design point is that we do not only optimize Recall@K. We study the trade-off among recommendation hits, promotion availability, coupon cost, and product diversity.

## 5. Experiment Design

The project will be evaluated as a recommender systems research project, not only as a demo.

Main research questions:

| Research Question | Planned experiment |
| --- | --- |
| RQ1: Which candidate generation model works better for grocery next-basket recommendation? | Compare Popularity, Category Popularity, ItemKNN, ALS, BPR, and LightGCN |
| RQ2: Do promotion and coupon signals improve the business utility proxy? | Compare base ALS with ALS plus promotion/coupon reranking |
| RQ3: Does adding a discount-cost penalty reduce ineffective coupon allocation? | Compare reranking with and without discount-cost penalty |
| RQ4: Does diversity control improve recommendation list health? | Compare full reranking with and without the diversity term |
| RQ5: Is LightGCN worth the additional complexity? | If completed, compare ALS/BPR with LightGCN on quality and runtime |

Recommendation accuracy metrics:

- Recall@10 and Recall@20.
- NDCG@10 and NDCG@20.

List and catalog metrics:

- Coverage.
- Diversity.
- Novelty.

Business proxy metric:

```text
Business Utility@K =
  estimated sales value of hit products
  - lambda_cost * estimated discount cost of hit products
```

Important note: Business Utility@K is not profit. The dataset does not include product cost or true margin, so we will describe it rigorously as a revenue-minus-discount proxy.

Reranking ablation experiments:

- Base model only.
- Base + promotion.
- Base + coupon.
- Base + promotion + coupon.
- Base + promotion + coupon - discount cost.
- Full reranking with diversity.

## 6. X-factor And Demo

The X-factor of this project is promotion-aware reranking and budget-aware coupon decision-making. The system does not only output Top-K products; it also decides which products in the recommendation list are suitable for coupon or promotion support.

The planned demo will use Streamlit:

- Select a household.
- Show the household's recent purchase categories, average basket value, frequent brands, and available profile information.
- Output Top-10 recommended products.
- For each product, show department, category, brand, base recommendation score, coupon recommendation, and explanation.
- Add a marketing budget slider to show how coupon allocation, estimated sales value, estimated discount cost, and Business Utility@K change when the budget changes.

Recommendation explanations will be rule-based, for example:

- The household recently purchased this category frequently.
- The product co-occurs highly with products in the household's historical baskets.
- The product currently has promotion exposure.
- The product belongs to the household's frequent department.

The demo is used to explain the model and business trade-offs. The final evidence will still come from offline evaluation metrics.

## 7. Project Boundary And Risk Control

- Coupon redemption is sparse: the main task remains next-period product purchase prediction, and coupon data is used as an auxiliary reranking signal.
- Demographics coverage is incomplete: demographics are optional analysis/demo signals, not required main-model inputs.
- True cost and margin are unavailable: we will not claim to optimize real profit, only a revenue-minus-discount proxy.
- Promotion exposure is observational data: we will not claim causal uplift.
- LightGCN is a bonus: even if LightGCN is not completed, Popularity/ALS/BPR plus promotion-aware reranking and demo still form a complete project.

## 8. Expected Contribution

PromoMind contributes by extending grocery next-basket recommendation from pure purchase prediction to promotion-aware retail decision-making. It can demonstrate several course topics at the same time: implicit feedback, matrix factorization, ranking metrics, auxiliary information, deep recommender extension, and business-aware evaluation. The final system is closer to the real question that retailers care about: under limited marketing budget, how should products be recommended, how should coupons be allocated, and how can ineffective discounts be reduced?
