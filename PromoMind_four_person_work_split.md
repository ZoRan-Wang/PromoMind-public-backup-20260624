# PromoMind 四人分工方案

## 分工原则

四个人尽量平均分，每个人都负责一条完整工作线：数据/模型/实验/展示材料都要有自己的交付物。不要让某个人只负责写 PPT，也不要让某个人独自承担全部建模。最终每个人都能在 presentation 中讲清楚自己负责的模块。

建议把四个人暂时命名为 A、B、C、D，之后换成真实姓名即可。

---

## A：数据工程与实验底座负责人

### 核心职责

A 负责把 The Complete Journey 数据集处理成所有模型都能直接使用的训练、验证、测试数据，并保证后续实验口径统一。

### 具体任务

1. 数据下载与读取
   - 安装或获取 `completejourney` 数据。
   - 整理核心表：transactions、products、promotions、campaigns、coupons、coupon_redemptions、demographics。
   - 明确每张表的主键和连接键，例如 `household_id`、`product_id`、`basket_id`、`week`。

2. 数据清洗
   - 处理缺失值、异常 quantity、异常 sales value。
   - 统一字段命名。
   - 过滤过低频商品，例如购买次数少于 20 的商品，或只保留 Top 10,000 高频商品。
   - 生成 household-product 交互表。

3. 时间切分
   - 按 week 做时间划分。
   - 建议：
     - Week 1-40：训练集
     - Week 41-46：验证集
     - Week 47-53：测试集
   - 输出统一的 train/valid/test 文件，供 B、C、D 使用。

4. EDA 与数据描述
   - 统计家庭数、商品数、交易行数、basket 数。
   - 统计每周交易量趋势。
   - 统计热门 department、热门 product_category、热门品牌。
   - 统计每个 household 的购买频率分布。
   - 分析 coupon redemption 和 demographics 的稀疏性。

5. 实验工具函数
   - 写统一的数据加载函数。
   - 写统一的正样本 ground truth 构造函数。
   - 写统一的 candidate 商品过滤逻辑。

### 交付物

- `data_processed/` 文件夹
  - `train_interactions.csv`
  - `valid_interactions.csv`
  - `test_interactions.csv`
  - `product_features.csv`
  - `household_features.csv`
  - `promotion_coupon_features.csv`
- 一份 EDA notebook。
- 一页 presentation：Dataset and Problem Setup。
- 一小段报告文字：数据集介绍、清洗规则、时间切分方式。

### 会议中负责讲

“我们为什么选 The Complete Journey、数据有哪些表、怎么切分训练验证测试、为什么 coupon 和 demographics 不能作为唯一主任务。”

---

## B：基础推荐模型与主模型负责人

### 核心职责

B 负责实现第一阶段 candidate generation，也就是预测家庭下一周可能买什么。这个部分是项目的模型主体。

### 具体任务

1. Popularity Baseline
   - 根据训练集统计全局最热门商品。
   - 对所有 household 推荐相同的 Top-K 商品。
   - 作为最低基线。

2. Category Popularity Baseline
   - 根据 household 历史购买的高频类别，推荐该类别中的热门商品。
   - 比全局 popularity 更个性化一点。
   - 用来证明 grocery 推荐不能只靠全局热门商品。

3. ItemKNN
   - 基于 household-product 交互矩阵构造 item-item 相似度。
   - 推荐与用户历史购买商品相似的商品。
   - 作为传统协同过滤基线。

4. Implicit ALS
   - 使用 implicit feedback 构造稀疏矩阵。
   - 尝试不同参数：
     - factors: 32, 64, 128
     - regularization: 0.01, 0.05, 0.1
     - iterations: 20, 50
     - alpha: 10, 20, 40
   - 在验证集上选择最佳参数。

5. BPR Matrix Factorization
   - 用 pairwise ranking 思路训练。
   - 和 ALS 对比。
   - 如果时间紧，BPR 可以作为第二优先级，ALS 必须完成。

6. 模型输出接口
   - 每个模型输出统一格式：
     - `household_id`
     - `product_id`
     - `base_score`
     - `model_name`
     - `rank`
   - 输出 Top-50 或 Top-100 candidate，供 C 做促销重排序。

### 交付物

- `models/baselines.py`
- `models/implicit_als.py`
- `models/bpr.py`
- `outputs/candidates_als.csv`
- `outputs/candidates_bpr.csv`
- 一张模型比较表：Popularity、Category Popularity、ItemKNN、ALS、BPR。
- 一页 presentation：Candidate Generation Models。
- 一小段报告文字：模型原理、训练方式、主要参数。

### 会议中负责讲

“第一阶段模型怎么生成候选商品，为什么 ALS/BPR 适合 implicit feedback，模型相对于 popularity baseline 有没有提升。”

---

## C：促销感知重排序与商业指标负责人

### 核心职责

C 负责项目的 X-factor：把普通推荐结果升级成 promotion-aware recommender，也就是引入促销、优惠券、折扣成本和商业效用。

### 具体任务

1. 促销特征构造
   - 从 promotions 表中构造 product-week 级别特征。
   - 判断某个商品在目标 week 是否有促销曝光。
   - 区分不同促销类型，例如 display、mailer、in-store 等，如果字段支持。

2. 优惠券特征构造
   - 从 coupons、campaigns、coupon_redemptions 中构造 household-product 或 household-campaign-product 级别特征。
   - 判断 household 是否收到过相关 campaign。
   - 判断商品是否有 coupon 映射。
   - coupon redemption 只作为辅助信号，不作为主监督目标。

3. 折扣成本代理变量
   - 用 `retail_disc`、`coupon_disc`、`coupon_match_disc` 构造 discount cost proxy。
   - 计算商品平均折扣、类别平均折扣、家庭历史折扣敏感度。
   - 注意报告中不能说是真实利润，只能说 revenue-minus-discount proxy。

4. Promotion-aware Re-ranking
   - 接收 B 输出的候选列表。
   - 实现重排序公式：
     - final_score = alpha * base_score
     - plus beta * promotion_score
     - plus gamma * coupon_score
     - minus lambda * discount_cost
     - plus rho * diversity_score
   - 在验证集调参：
     - alpha, beta, gamma, lambda, rho。

5. Diversity 控制
   - 避免 Top-K 全是同一 department 或 product_category。
   - 可以实现简单规则：
     - 每个 department 最多出现 N 个商品。
     - 或者同类商品分数略微惩罚。

6. 商业指标
   - 实现 Business Utility@K。
   - 计算 estimated sales value。
   - 计算 estimated discount cost。
   - 比较纯 ALS/BPR 和 reranked ALS/BPR 的结果。

### 交付物

- `features/promotion_features.py`
- `features/coupon_features.py`
- `rerank/promotion_reranker.py`
- `metrics/business_metrics.py`
- `outputs/reranked_recommendations.csv`
- 一张消融实验表：
  - ALS only
  - ALS + promotion
  - ALS + coupon
  - ALS + promotion + coupon + discount cost
  - ALS + full reranking + diversity
- 一页 presentation：Promotion-aware Re-ranking and Business Utility。
- 一小段报告文字：X-factor、重排序公式、商业指标解释。

### 会议中负责讲

“我们项目和普通推荐系统的区别在哪里，促销和优惠券怎么进入排序，为什么 Business Utility 只能叫 proxy 而不能叫 profit。”

---

## D：LightGCN、Demo 与最终整合负责人

### 核心职责

D 负责项目的进阶展示和最终产品化表达：LightGCN 作为增强模型，Streamlit demo 作为课堂展示亮点，同时把所有结果整合成最终展示材料。

### 具体任务

1. LightGCN 数据格式准备
   - 根据 A 的 train/valid/test 数据，转换成 RecBole 或自定义 LightGCN 需要的格式。
   - 建立 household 和 product 的连续 id mapping。
   - 确保训练集、验证集、测试集不穿越时间。

2. LightGCN 训练
   - 使用 RecBole 的 LightGCN 实现。
   - 尝试基础参数：
     - embedding_size: 64
     - n_layers: 2 或 3
     - learning_rate: 0.001
     - train_batch_size: 2048 或 4096
   - 输出和 B 一样格式的 Top-K candidate。
   - 如果时间不够，LightGCN 作为 bonus，不影响主项目完整性。

3. 结果整合
   - 收集 B 的模型结果和 C 的 reranking 结果。
   - 做统一结果表：
     - Recall@10
     - Recall@20
     - NDCG@10
     - NDCG@20
     - Coverage
     - Diversity
     - Novelty
     - Business Utility@K

4. Streamlit Demo
   - 实现一个简单但完整的交互页面。
   - 页面包括：
     - household_id 选择器
     - household 历史购买类别
     - 平均 basket value
     - 常买品牌
     - Top-10 推荐商品
     - 是否建议 coupon
     - 推荐理由
     - marketing budget slider
   - budget slider 影响可发 coupon 的商品数量或 discount cost 阈值。

5. 推荐解释
   - 写规则型 explanation。
   - 示例：
     - “该家庭最近多次购买同类商品。”
     - “该商品与历史 basket 中的商品高度共现。”
     - “该商品当前有促销曝光。”
     - “该商品属于该家庭高频 department。”

6. 最终展示材料整合
   - 整理所有人的图表和结果。
   - 统一图表风格。
   - 制作最终 presentation 的方法架构图和 demo 截图。

### 交付物

- `models/lightgcn_recbole/`
- `app/streamlit_app.py`
- `outputs/final_results_table.csv`
- `outputs/demo_recommendations.csv`
- 一页 presentation：Demo and System Architecture。
- 一页 presentation：Final Results and Takeaways。
- 一小段报告文字：LightGCN、demo 设计、系统架构。

### 会议中负责讲

“系统最终长什么样、demo 怎么操作、LightGCN 是否带来提升、最终结果说明了什么。”

---

## 四人协作接口

为了避免互相等太久，建议一开始就规定统一文件格式。

### A 输出给 B/C/D

`train_interactions.csv`

| column | meaning |
| --- | --- |
| household_id | household id |
| product_id | product id |
| week | purchase week |
| quantity | purchase quantity |
| sales_value | sales amount |
| retail_disc | retail discount |
| coupon_disc | coupon discount |
| coupon_match_disc | coupon match discount |

`product_features.csv`

| column | meaning |
| --- | --- |
| product_id | product id |
| department | department |
| product_category | product category |
| brand | brand |
| product_description | product name or description |

### B 输出给 C/D

`candidates_MODEL.csv`

| column | meaning |
| --- | --- |
| household_id | household id |
| product_id | candidate product |
| base_score | model predicted score |
| model_name | ALS, BPR, ItemKNN, etc. |
| base_rank | original rank before reranking |

### C 输出给 D

`reranked_recommendations.csv`

| column | meaning |
| --- | --- |
| household_id | household id |
| product_id | recommended product |
| base_score | first-stage recommendation score |
| promo_score | promotion feature score |
| coupon_score | coupon feature score |
| discount_cost_proxy | estimated discount cost |
| diversity_score | diversity adjustment |
| final_score | reranked score |
| final_rank | final rank |
| recommend_coupon | yes/no |

### D 输出给全组

`final_results_table.csv`

| column | meaning |
| --- | --- |
| model | model or reranking variant |
| recall_at_10 | Recall@10 |
| recall_at_20 | Recall@20 |
| ndcg_at_10 | NDCG@10 |
| ndcg_at_20 | NDCG@20 |
| coverage | coverage |
| diversity | diversity |
| novelty | novelty |
| business_utility_at_10 | business utility proxy |

---

## 推荐时间线

### Day 1-2：数据和接口定下来

- A 完成数据读取、清洗和时间切分初版。
- B 用 A 的初版数据先跑 Popularity。
- C 先研究 promotions/coupons/campaigns/coupon_redemptions 表怎么连接。
- D 搭建 Streamlit 空壳和 LightGCN 数据格式草稿。

### Day 3-4：第一批模型结果

- A 完成 EDA 图和数据说明。
- B 完成 Popularity、Category Popularity、ItemKNN、ALS。
- C 完成 promotion/coupon 特征初版。
- D 完成 demo 页面框架和结果表模板。

### Day 5-6：重排序和商业指标

- B 完成 BPR 或 ALS 调参。
- C 完成 promotion-aware reranking 和 Business Utility@K。
- D 接入 reranked recommendations 到 demo。
- A 帮忙检查数据口径和结果是否有 leakage。

### Day 7：整合和展示

- 四个人各自写自己负责部分的报告文字。
- D 整合 PPT 和 demo。
- A 检查数据页。
- B 检查模型页。
- C 检查 reranking 和 business utility 页。
- 全组一起 rehearsal。

---

## 每个人的最终 presentation 分钟数

如果总展示时间是 12 分钟，可以这样分：

- A：2.5 分钟，讲问题定义、数据集、时间切分。
- B：3 分钟，讲 baseline、ALS/BPR、主模型结果。
- C：3 分钟，讲 promotion-aware reranking、商业指标、消融实验。
- D：3.5 分钟，讲 LightGCN、demo、最终结论。

如果总展示时间只有 8 分钟：

- A：1.5 分钟。
- B：2 分钟。
- C：2 分钟。
- D：2.5 分钟。

---

## 工作量平衡说明

- A 的工作量主要在数据清洗、EDA 和统一接口，前期最重。
- B 的工作量主要在传统推荐模型，项目中期最重。
- C 的工作量主要在 X-factor、商业指标和消融实验，项目后期最关键。
- D 的工作量主要在 LightGCN、demo 和最终整合，展示前最重。

整体上四个人工作量接近：

- A：25%，数据和实验基础。
- B：25%，第一阶段推荐模型。
- C：25%，促销重排序和商业评估。
- D：25%，进阶模型、demo 和最终整合。

如果时间不够，优先级如下：

1. 必须完成：A 的数据切分、B 的 Popularity + ALS、C 的 promotion-aware reranking、D 的最终结果表和 demo 简版。
2. 尽量完成：BPR、ItemKNN、完整 diversity/novelty 指标。
3. 加分完成：LightGCN、营销预算滑杆、漂亮 demo 页面。

