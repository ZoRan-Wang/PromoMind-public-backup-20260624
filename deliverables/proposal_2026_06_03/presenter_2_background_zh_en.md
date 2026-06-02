# Presenter 2 Background: Methods, Experiments, Reranking, Demo

Presenter 2 owns Slides 6-10.

## 1. One-Sentence Method Summary

English:

PromoMind first generates likely products with recommender models, then reranks them with promotion, coupon, discount-cost, and diversity signals.

Chinese:

PromoMind 先用推荐模型生成候选商品，再用促销、优惠券、折扣成本和多样性信号做二次排序。

## 2. Why Baselines Matter

English:

Grocery data has strong popularity and repeat-purchase effects. If ALS, BPR, or LightGCN cannot beat simple popularity baselines, the complex model is not convincing.

Chinese:

超市数据有很强的热门商品和复购效应。如果 ALS、BPR 或 LightGCN 打不过简单 popularity baseline，复杂模型就没有说服力。

## 3. Candidate Models

English:

- Popularity: recommend globally popular products.
- Category Popularity: recommend popular products from a household's frequent categories.
- ItemKNN: recommend products similar to what the household bought before.
- Implicit ALS: matrix factorization for implicit feedback.
- BPR: pairwise ranking model, if time allows.
- LightGCN: graph-based model on the household-product interaction graph, bonus only.

Chinese:

- Popularity：推荐全局最热门商品。
- Category Popularity：根据 household 高频购买类别推荐该类别热门商品。
- ItemKNN：推荐和历史购买商品相似的商品。
- Implicit ALS：适合 implicit feedback 的矩阵分解模型。
- BPR：pairwise ranking 模型，时间允许再做。
- LightGCN：基于 household-product 二部图的图模型，只作为加分项。

## 4. Library Acknowledgement

English:

We may use `implicit` for ALS/BPR-style implicit recommenders and RecBole for LightGCN. If Cornac is used, it will be acknowledged as an experiment framework.

Chinese:

我们可能使用 `implicit` 做 ALS/BPR 类型的隐式反馈推荐，使用 RecBole 做 LightGCN。如果使用 Cornac，会明确说明它是实验框架。

## 5. Promotion-aware Reranking

English:

The reranking formula is:

```text
final_score =
  alpha * base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

Meaning:

- `base_score`: relevance from the recommender.
- `promotion_score`: whether the product has planned promotion exposure.
- `coupon_score`: whether household/product coupon eligibility exists.
- `discount_cost_proxy`: estimated discount cost.
- `diversity_score`: avoids lists with only one category.

Chinese:

重排序公式是：

```text
final_score =
  alpha * base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

含义：

- `base_score`：推荐模型给出的相关性。
- `promotion_score`：商品是否有计划中的促销曝光。
- `coupon_score`：household/product 是否有优惠券资格或映射。
- `discount_cost_proxy`：估计折扣成本。
- `diversity_score`：避免推荐列表全是同一类商品。

## 6. Business Utility Is Not Profit

English:

Business Utility@K is a revenue-minus-discount proxy:

```text
Business Utility@K =
  estimated sales value of hit products
  - lambda_cost * estimated discount cost of hit products
```

It is not profit because the dataset does not include product margin or cost.

Chinese:

Business Utility@K 是 revenue-minus-discount 的代理指标：

```text
Business Utility@K =
  命中商品的估计销售额
  - lambda_cost * 命中商品的估计折扣成本
```

它不是 profit，因为数据集没有商品成本和利润率。

## 7. Experiment Plan

English:

We will use chronological split and compare:

- Popularity.
- Category Popularity.
- ItemKNN.
- ALS.
- BPR if completed.
- LightGCN if completed.

Metrics:

- Recall@10/20.
- NDCG@10/20.
- Coverage.
- Diversity.
- Novelty.
- Business Utility@K proxy.

Reranking ablations:

- Base only.
- Base + promotion.
- Base + coupon.
- Base + promotion + coupon.
- Discount-aware reranking.
- Full reranking with diversity.

Chinese:

我们会用时间切分，并比较：

- Popularity。
- Category Popularity。
- ItemKNN。
- ALS。
- 如果完成，加入 BPR。
- 如果完成，加入 LightGCN。

指标：

- Recall@10/20。
- NDCG@10/20。
- Coverage。
- Diversity。
- Novelty。
- Business Utility@K 代理指标。

重排序消融：

- 只用 base score。
- base + promotion。
- base + coupon。
- base + promotion + coupon。
- 加入 discount cost 的版本。
- 加入 diversity 的 full reranking。

## 8. Demo Concept

English:

The demo is a Streamlit app. It will let the user choose a household, inspect the household profile, view Top-10 products, see coupon decisions and reasons, and adjust a marketing budget slider.

Chinese:

demo 是一个 Streamlit 应用。用户可以选择 household，查看 household profile，看 Top-10 推荐商品、优惠券决策和推荐理由，并通过 marketing budget slider 调整优惠券分配。

## 9. Presenter 2 Must Avoid

English:

- Do not say this is causal uplift.
- Do not say Business Utility is real profit.
- Do not say LightGCN is required.
- Do not say the demo replaces offline evaluation.

Chinese:

- 不要说这是 causal uplift。
- 不要说 Business Utility 是真实利润。
- 不要说 LightGCN 是必做。
- 不要说 demo 可以代替离线评估。

