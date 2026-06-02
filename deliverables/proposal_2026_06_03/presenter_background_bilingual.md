# Presenter Background Notes

Purpose: help the two presenters understand the proposal without reading every project file.

## 1. Five-second Version

English:

> PromoMind is a promotion-aware grocery recommender. We use The Complete Journey dataset to predict each household's next-basket products, then rerank recommendations using promotion, coupon, discount-cost, and diversity signals.

中文：

> PromoMind 是一个促销感知型超市购物篮推荐系统。我们用 The Complete Journey 预测每个家庭下次可能买的商品，再结合促销、优惠券、折扣成本和多样性做二次排序。

## 2. What The Teacher Needs To Hear

English:

- Dataset: The Complete Journey.
- Collection method: public `completejourney` data artifacts, no private scraping.
- Expected size: 2,469 households, about 1.47M transaction rows, 92k products, coupon/promotion/campaign/demographic tables.
- Problem: next-period household-product Top-K recommendation using implicit feedback.
- Methods: Popularity, Category Popularity, ItemKNN, Implicit ALS, BPR if time allows, LightGCN as bonus.
- Experiments: chronological split, Recall@K, NDCG@K, coverage, diversity, novelty, Business Utility@K, reranking ablations.
- X-factor: promotion-aware reranking and marketing budget/coupon decision demo.

中文：

- 数据集：The Complete Journey。
- 采集方法：公开 `completejourney` 数据文件，不爬取私人数据。
- 预计规模：2,469 个家庭，约 147 万交易明细，约 9.2 万商品，以及优惠券、促销、营销活动、用户画像等表。
- 推荐问题：基于 implicit feedback 的下一阶段 household-product Top-K 推荐。
- 方法：Popularity、Category Popularity、ItemKNN、Implicit ALS，时间允许再做 BPR，LightGCN 作为 bonus。
- 实验：时间切分、Recall@K、NDCG@K、coverage、diversity、novelty、Business Utility@K、重排序消融。
- 亮点：promotion-aware reranking 和营销预算/优惠券决策 demo。

## 3. Presenter 1 Notes: Dataset, Problem, Setup

Presenter 1 owns Slides 1-5.

### Dataset Choice

English:

The Complete Journey is better than a plain basket dataset because it combines transaction behavior with product metadata and marketing context. That lets us study recommendation and promotion decision together.

中文：

The Complete Journey 比普通购物篮数据更适合这个项目，因为它不仅有购买记录，还有商品信息、促销、优惠券、营销活动和部分用户画像。因此我们可以同时研究“推荐什么”和“哪些推荐值得促销推送”。

### Dataset Features

English:

The transaction table is item-level receipt data. Important fields include household id, basket id, product id, week, quantity, sales value, retail discount, coupon discount, and coupon match discount. Product metadata adds department, category, brand, and product description. Marketing tables add promotion exposure, coupon mappings, campaign exposure, and redemption history.

中文：

交易表接近“小票商品明细”。关键字段包括 household id、basket id、product id、week、quantity、sales value、retail discount、coupon discount、coupon match discount。商品表补充 department、category、brand、product description。营销相关表补充 promotion exposure、coupon mapping、campaign exposure 和 redemption history。

### Split

English:

Use Weeks 1-40 for training, Weeks 41-46 for validation, and Weeks 47-53 for testing. The key reason is temporal realism: recommenders use past behavior to predict future baskets.

中文：

用 Week 1-40 训练，Week 41-46 验证，Week 47-53 测试。核心理由是时间真实性：真实推荐系统只能用过去预测未来。

### Presenter 1 Must Avoid

English:

- Do not claim we already have final experimental results.
- Do not say demographics cover every household.
- Do not say coupon redemption is the main supervised target.
- Do not spend too much time on algorithm details.

中文：

- 不要说我们已经有最终实验结果。
- 不要说 demographics 覆盖所有 household。
- 不要说 coupon redemption 是主监督目标。
- 不要展开讲模型细节，把模型交给 Presenter 2。

## 4. Presenter 2 Notes: Models, Reranking, Experiments, Demo

Presenter 2 owns Slides 6-10.

### Model Ladder

English:

Start with baselines because grocery data has strong popularity and repeat-purchase effects. Then compare stronger implicit-feedback recommenders. LightGCN is only a bonus extension, not the minimum delivery line.

中文：

先做 baseline，因为超市数据有很强的热门商品和复购效应。然后比较更强的 implicit-feedback 推荐模型。LightGCN 只是加分项，不是最低完成线。

### Reranking Formula

```text
final_score =
  alpha * base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

English meaning:

- `base_score`: relevance from the recommender model.
- `promotion_score`: whether the product has promotion exposure.
- `coupon_score`: whether coupon eligibility/mapping exists.
- `discount_cost_proxy`: estimated discount cost.
- `diversity_score`: prevents one-category recommendation lists.

中文含义：

- `base_score`：推荐模型给出的相关性。
- `promotion_score`：商品是否有促销曝光。
- `coupon_score`：是否有优惠券资格或映射。
- `discount_cost_proxy`：估计折扣成本。
- `diversity_score`：避免推荐列表全部集中在一个类别。

### Business Utility

English:

Business Utility@K is not profit. It is a revenue-minus-discount proxy because true product cost and margin are unavailable.

中文：

Business Utility@K 不是 profit。由于数据没有真实成本和利润率，它只是 revenue-minus-discount proxy。

### Demo

English:

The Streamlit demo should show a household selector, household profile summary, Top-10 recommendations, coupon decision, rule-based explanation, and a marketing budget slider.

中文：

Streamlit demo 应展示 household selector、家庭画像摘要、Top-10 推荐、coupon 决策、规则解释和 marketing budget slider。

### Presenter 2 Must Avoid

English:

- Do not claim causal uplift.
- Do not claim true profit optimization.
- Do not claim LightGCN is required.
- Do not say the demo replaces offline evaluation.

中文：

- 不要说这是 causal uplift。
- 不要说优化真实 profit。
- 不要说 LightGCN 必须完成。
- 不要说 demo 可以替代离线实验评估。
