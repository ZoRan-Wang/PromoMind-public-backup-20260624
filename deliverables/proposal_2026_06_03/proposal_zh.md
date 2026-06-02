# PromoMind：促销感知型超市购物篮与优惠券推荐系统

## 1. 项目动机

超市和生鲜零售商通常拥有大量会员购买记录，但促销和优惠券投放往往比较粗放。普通推荐系统主要回答“这个家庭下次可能买什么”。PromoMind 进一步研究一个更接近真实零售决策的问题：在促销预算和优惠券成本有限的情况下，哪些推荐商品值得配合促销或优惠券推送，哪些商品只需要正常推荐即可。

本项目的研究目标是：在真实超市零售数据上，评估促销、优惠券、折扣成本和商品多样性等信息是否能改进下一次购物篮推荐。系统会先预测家庭下一阶段可能购买的 Top-K 商品，再通过促销感知重排序生成更适合营销决策的推荐列表。

## 2. 数据集与采集方法

我们使用公开数据集 The Complete Journey。该数据集通过 `completejourney` 项目公开，记录了真实零售商的家庭级购物行为。它不仅包含交易记录，还包含商品信息、促销、营销活动、优惠券、优惠券兑换和部分家庭人口统计信息，因此比只包含订单和商品类别的数据集更适合研究“推荐 + 促销决策”问题。

数据采集和使用方式：

- 使用公开的 `completejourney` 数据源和原始 RDS/RDA 文件。
- 原始数据文件已经保存在仓库的 `data/raw/completejourney/`。
- `scripts/download_completejourney.R` 记录了可复现的数据导出流程。
- 不爬取私人数据，也不新增个人隐私数据。
- 后续 CSV 或特征表都从原始文件和脚本可复现生成。

预计数据规模：

| 数据部分 | 规模 |
| --- | --- |
| households | 2,469 个家庭 |
| transactions | 约 1,469,307 条交易明细 |
| products | 约 92,331 个商品 |
| coupon-product-campaign mapping | 约 116,204 条 |
| campaign-household exposure | 约 6,589 条 |
| coupon redemptions | 约 2,102 条 |
| demographics | 约 801 个家庭有画像信息 |

数据长什么样：

- 交易表接近“小票商品明细”，一行通常对应一个家庭在某个 basket 中购买的某个商品。
- 关键字段包括 `household_id`、`basket_id`、`product_id`、`week`、`quantity`、`sales_value`、`retail_disc`、`coupon_disc`、`coupon_match_disc`。
- 商品表提供 department、product category、brand、product description。
- 促销和优惠券相关表提供 product-week 促销曝光、coupon-product-campaign 映射、household campaign exposure 和 redemption history。
- demographics 表提供收入段、年龄段、家庭规模等部分用户画像。

两个主要数据风险也会在项目中明确处理。第一，coupon redemption 只有约 2,102 条，比较稀疏，所以不会把“是否兑换优惠券”作为唯一监督目标。第二，demographics 只覆盖部分家庭，因此只作为可选分析和 demo 信息，不作为主模型必需输入。

## 3. 推荐问题定义

项目任务定义为：给定家庭 `h` 在目标周 `t` 之前的所有历史购物行为，推荐该家庭在未来目标周期最可能购买的 Top-K 商品。

由于超市购物数据没有评分，购买行为被视为 implicit feedback。核心标签可以写成：

```text
y_hjt = 1 if household h buys product j in the target period
```

我们计划采用按时间切分的数据集，而不是随机切分：

| Split | Weeks | 用途 |
| --- | --- | --- |
| Train | 1-40 | 训练推荐模型和历史特征 |
| Validation | 41-46 | 调参和选择重排序权重 |
| Test | 47-53 | 最终离线评估 |

这样更符合真实推荐系统：系统只能用过去预测未来，不能从未来购物记录中泄漏信息。

## 4. 模型方法

PromoMind 采用两阶段推荐架构。

第一阶段是 candidate generation，用推荐模型生成每个家庭可能购买的候选商品。计划比较：

- Global Popularity baseline。
- Category Popularity baseline。
- ItemKNN。
- Implicit ALS matrix factorization。
- BPR matrix factorization，如果时间允许。
- LightGCN，如果时间允许，作为深度推荐/图推荐扩展。

外部库会在汇报中明确说明：

- `implicit` 可用于 ALS/BPR 类型的隐式反馈推荐模型。
- RecBole 可用于 LightGCN。
- Cornac 可作为课程相关的实验框架候选。

第二阶段是 promotion-aware reranking。第一阶段模型输出基础相关性分数后，系统再结合促销、优惠券、折扣成本和多样性进行重排序：

```text
final_score =
  alpha * normalized_base_score
  + beta * promotion_score
  + gamma * coupon_score
  - lambda * discount_cost_proxy
  + rho * diversity_score
```

其中：

- `normalized_base_score` 表示推荐模型给出的购买可能性。
- `promotion_score` 表示商品在推荐目标周期是否有促销曝光。
- `coupon_score` 表示 household-product 或 household-campaign-product 是否有优惠券匹配。
- `discount_cost_proxy` 用历史 retail discount、coupon discount 等构造折扣成本代理变量。
- `diversity_score` 用来避免推荐列表全部集中在同一商品类别。

这个设计的关键点是：我们不只追求 Recall@K，也研究在推荐命中率、促销可用性、优惠券成本和商品多样性之间如何权衡。

## 5. 实验设计

项目会按照 recommender systems research 的方式评估，而不是只展示 demo。

主要研究问题：

| Research Question | 计划实验 |
| --- | --- |
| RQ1: 哪种 candidate generation 模型更适合 grocery next-basket recommendation？ | 比较 Popularity、Category Popularity、ItemKNN、ALS、BPR、LightGCN |
| RQ2: 促销和优惠券信号是否提升商业效用代理指标？ | 比较 base ALS 与 ALS + promotion/coupon reranking |
| RQ3: 加入 discount-cost penalty 是否减少无效发券？ | 比较有无折扣成本惩罚的 reranking |
| RQ4: diversity 控制是否改善推荐列表健康度？ | 比较有无多样性项的 full reranking |
| RQ5: LightGCN 是否值得额外复杂度？ | 如果完成，比较 ALS/BPR 与 LightGCN 的效果和运行成本 |

推荐准确率指标：

- Recall@10、Recall@20。
- NDCG@10、NDCG@20。

列表和目录指标：

- Coverage。
- Diversity。
- Novelty。

商业代理指标：

```text
Business Utility@K =
  estimated sales value of hit products
  - lambda_cost * estimated discount cost of hit products
```

注意：Business Utility@K 不是 profit。数据集没有商品成本和真实利润率，所以我们只能严谨地称它为 revenue-minus-discount proxy。

重排序消融实验：

- Base model only。
- Base + promotion。
- Base + coupon。
- Base + promotion + coupon。
- Base + promotion + coupon - discount cost。
- Full reranking with diversity。

## 6. X-factor 与 Demo

项目的 X-factor 是 promotion-aware reranking 和预算感知优惠券决策。系统不只是输出 Top-K 商品，还会判断推荐列表中哪些商品适合配优惠券或促销推送。

计划 demo 使用 Streamlit：

- 选择一个 household。
- 展示该家庭最近购买类别、平均 basket value、常买品牌和可用画像。
- 输出 Top-10 推荐商品。
- 每个商品展示 department、category、brand、基础推荐分数、是否建议 coupon、推荐理由。
- 加入 marketing budget slider，展示预算变化时 coupon 分配、estimated sales value、estimated discount cost 和 Business Utility@K 的变化。

推荐理由用规则生成，例如：

- 该家庭近期频繁购买同类商品。
- 商品与历史 basket 中商品共现较高。
- 商品当前有促销曝光。
- 商品属于该家庭高频 department。

Demo 的作用是解释模型和商业权衡，最终证据仍然来自离线实验指标。

## 7. 项目边界与风险控制

- Coupon redemption 稀疏：主任务保持为下一阶段商品购买预测，coupon 作为辅助重排序信号。
- Demographics 覆盖不完整：只作为可选分析，不作为主模型依赖。
- 没有真实成本和利润率：不声称优化真实 profit，只报告 revenue-minus-discount proxy。
- 促销曝光是 observational data：不声称 causal uplift。
- LightGCN 作为 bonus：即使 LightGCN 未完成，Popularity/ALS/BPR + promotion-aware reranking + demo 仍然构成完整项目。

## 8. 预期贡献

PromoMind 的贡献在于把 grocery next-basket recommendation 从单纯预测购买扩展到促销感知的零售决策问题。它能同时展示课程中的 implicit feedback、matrix factorization、ranking metrics、auxiliary information、deep recommender extension 和 business-aware evaluation。最终系统将更接近真实零售商会关心的问题：在有限营销预算下，如何推荐商品、如何分配优惠券、如何减少无效折扣。
