# Presenter 1 Background: Dataset, Features, Problem

Presenter 1 owns Slides 1-5.

## 1. One-Sentence Project Idea

English:

PromoMind studies whether grocery basket recommendation can be improved by using promotion and coupon context, not only historical purchases.

Chinese:

PromoMind 研究的是：超市购物篮推荐能不能不仅看历史购买，还结合促销和优惠券信息来做得更好。

## 2. Dataset Choice

English:

We choose The Complete Journey because it is not only a transaction dataset. It includes household-level grocery purchases plus product metadata, promotions, campaigns, coupons, coupon redemptions, and partial demographics.

Chinese:

我们选择 The Complete Journey，因为它不只是交易数据。它同时有家庭级购物记录、商品属性、促销、营销活动、优惠券、优惠券兑换和部分用户画像。

## 3. Collection Method

English:

The data comes from the public `completejourney` project. We use the original RDS/RDA artifacts and keep them in `data/raw/completejourney/`. The script `scripts/download_completejourney.R` documents the reproducible extraction/export path. We do not scrape private data.

Chinese:

数据来自公开的 `completejourney` 项目。我们使用原始 RDS/RDA 数据文件，并放在 `data/raw/completejourney/`。`scripts/download_completejourney.R` 记录了可复现的数据导出路径。我们不爬取私人数据。

## 4. Expected Data Size

English:

- 2,469 households.
- About 1.47 million transaction rows.
- About 92k products.
- Promotion, coupon, campaign, redemption, and demographic tables.
- Coupon redemption is sparse, about 2,102 rows.
- Demographics cover only part of households, about 801 households.

Chinese:

- 2,469 个 household。
- 大约 147 万条交易明细。
- 大约 9.2 万个商品。
- 包含促销、优惠券、营销活动、兑换记录和用户画像表。
- 优惠券兑换很稀疏，大约 2,102 条。
- demographics 只覆盖部分 household，大约 801 个。

## 5. What The Dataset Looks Like

English:

The main table is like item-level receipt data. Each row is close to one product purchased in one basket. Important fields include:

- `household_id`: which household.
- `basket_id`: which shopping basket.
- `product_id`: which product.
- `week`: when it happened.
- `quantity`: purchase quantity.
- `sales_value`: transaction value.
- `retail_disc`, `coupon_disc`, `coupon_match_disc`: discount signals.

Product metadata adds:

- department.
- product category.
- brand.
- product description.

Promotion/coupon tables add:

- product-week promotion exposure.
- coupon/product/campaign mapping.
- household campaign exposure.
- redemption history, used cautiously.

Chinese:

主表类似小票级别的商品明细，每一行接近某个 basket 里买了一个商品。关键字段包括：

- `household_id`：哪个家庭。
- `basket_id`：哪一次购物篮。
- `product_id`：哪个商品。
- `week`：发生在哪一周。
- `quantity`：购买数量。
- `sales_value`：销售金额。
- `retail_disc`、`coupon_disc`、`coupon_match_disc`：折扣相关信号。

商品表补充：

- department。
- product category。
- brand。
- product description。

促销和优惠券表补充：

- product-week 级别促销曝光。
- coupon/product/campaign 映射。
- household campaign exposure。
- redemption history，但只能谨慎用作辅助分析。

## 6. Recommendation Problem

English:

The task is Top-K next-period product recommendation. Given a household's purchase history before a target week, recommend products it is likely to buy in the future period. Because there are no ratings, purchases are implicit positive feedback.

Chinese:

任务是下一阶段 Top-K 商品推荐。给定某个 household 在目标周之前的购物历史，推荐它未来最可能购买的商品。因为没有评分，所以购买行为本身就是 implicit positive feedback。

## 7. Why Chronological Split

English:

We use past weeks to predict future weeks: train on Weeks 1-40, validate on Weeks 41-46, and test on Weeks 47-53. This avoids the unrealistic leakage caused by random splits.

Chinese:

我们用过去预测未来：Week 1-40 训练，Week 41-46 验证，Week 47-53 测试。这样比随机切分更符合真实推荐系统，也能避免未来信息泄露。

## 8. Presenter 1 Must Avoid

English:

- Do not say we have final results.
- Do not say demographics cover every household.
- Do not say coupon redemption is the main target.
- Do not over-explain model details; hand that to Presenter 2.

Chinese:

- 不要说我们已经有最终实验结果。
- 不要说 demographics 覆盖所有 household。
- 不要说 coupon redemption 是主监督目标。
- 不要展开讲模型细节，把模型部分交给 Presenter 2。

