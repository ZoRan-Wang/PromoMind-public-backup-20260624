# PromoMind 团队执行包

角色占位：A、B、C、D。正式使用时把 A/B/C/D 替换成真实姓名。

项目定位：PromoMind 是一个面向超市零售的促销感知购物篮和优惠券推荐系统。主任务是预测每个 household 下一周最可能购买的 Top-K 商品；项目亮点是把促销、优惠券、折扣成本和多样性加入二阶段重排序，形成商业效用导向的推荐系统。

---

## 1. 项目最终要交什么

### 1.1 最终成品清单

全组最终要交付以下材料：

1. 可运行代码
   - 数据处理代码。
   - Baseline 和 ALS/BPR 推荐模型代码。
   - Promotion-aware reranking 代码。
   - Evaluation metrics 代码。
   - Streamlit demo 代码。
   - LightGCN 代码，如果时间允许。

2. 实验结果
   - 主模型对比表。
   - 消融实验表。
   - Business Utility@K 表。
   - Coverage、Diversity、Novelty 表。
   - 至少 3-5 张可放进 PPT 的图。

3. 最终报告
   - Problem definition。
   - Dataset description。
   - Methodology。
   - Experiments and evaluation。
   - Promotion-aware reranking。
   - Demo and business implication。
   - Limitations and future work。

4. 最终展示 PPT
   - 题目和商业背景。
   - 数据集和任务定义。
   - 系统架构图。
   - 模型方法。
   - 实验结果。
   - X-factor：促销感知重排序。
   - Demo 截图或现场演示。
   - 结论和分工。

5. Demo
   - Streamlit 页面。
   - 可选择 household_id。
   - 显示 household 历史行为。
   - 显示 Top-10 推荐商品。
   - 显示是否建议 coupon。
   - 显示推荐理由。
   - 最好有 marketing budget slider。

---

## 2. 团队分工总览

| 成员 | 角色 | 主要责任 | 项目阶段压力点 | 展示时负责讲 |
| --- | --- | --- | --- | --- |
| A | Data Lead | 数据读取、清洗、时间切分、EDA、数据接口 | 前期最重 | 数据集、任务定义、时间切分 |
| B | Model Lead | Popularity、Category Popularity、ItemKNN、ALS、BPR | 中期最重 | 候选生成模型和主实验 |
| C | Business/Reranking Lead | 促销特征、优惠券特征、重排序、商业指标、消融实验 | 中后期最重 | X-factor 和 Business Utility |
| D | Integration/Demo Lead | LightGCN、结果整合、Streamlit demo、PPT 汇总 | 后期最重 | Demo、系统架构、最终结论 |

工作量目标：

- A：25%。负责项目地基，保证所有人数据口径一致。
- B：25%。负责核心推荐模型，保证系统能预测下一周购买。
- C：25%。负责项目亮点，保证不是普通推荐系统。
- D：25%。负责进阶模型、demo 和最终整合，保证项目能展示出来。

---

## 3. A 的详细任务：Data Lead

### 3.1 A 的一句话职责

A 要保证所有人用的是同一套干净、无数据泄漏、可复现实验的数据。A 是全组的数据接口负责人。

### 3.2 A 必须完成的任务

#### A1. 数据获取和目录搭建

具体要做：

- 下载或读取 The Complete Journey 数据。
- 确认可以访问以下表：
  - transactions
  - products
  - promotions
  - campaigns
  - campaign_descriptions
  - coupons
  - coupon_redemptions
  - demographics
- 建立项目目录：
  - `data/raw/`
  - `data/processed/`
  - `notebooks/`
  - `src/`
  - `outputs/`
  - `app/`
  - `slides_assets/`

交付物：

- 原始数据读取说明。
- `data/processed/schema_notes.md`
- 项目文件夹结构。

验收标准：

- 全组成员能在自己电脑上知道每张表在哪里。
- A 能解释每张表的主键、外键和用途。

#### A2. 表结构和数据字典

具体要做：

- 为每张核心表列出字段。
- 标注每个字段的含义。
- 标注哪些字段会进入模型。
- 标注哪些字段只用于分析或展示。
- 标注哪些字段不能用于训练，以避免 leakage。

重点字段：

- `household_id`
- `product_id`
- `basket_id`
- `week`
- `quantity`
- `sales_value`
- `retail_disc`
- `coupon_disc`
- `coupon_match_disc`
- `department`
- `product_category`
- `brand`

交付物：

- `data/processed/data_dictionary.md`

验收标准：

- B、C、D 不需要重新猜字段含义。
- 每个后续模型都能引用这个数据字典。

#### A3. 数据清洗

具体要做：

- 统一字段名。
- 删除或标注无效 transaction。
- 处理缺失值。
- 检查 `quantity <= 0` 的记录。
- 检查 `sales_value < 0` 的记录。
- 检查同一个 `basket_id` 里是否存在重复 product。
- 过滤极低频商品。

推荐过滤规则：

- 先保留购买次数 Top 10,000 的 product。
- 或者保留购买次数 >= 20 的 product。
- 两个版本都可以先跑，最后选一个稳定版本。

交付物：

- `data/processed/transactions_clean.csv`
- `data/processed/products_clean.csv`
- `notebooks/01_data_cleaning.ipynb`

验收标准：

- 输出清洗前后行数对比。
- 说明删除或过滤了多少商品和交易。
- B 能直接用清洗后数据构造 sparse matrix。

#### A4. 时间切分

具体要做：

- 使用 week 做时间切分。
- 推荐切分：
  - Week 1-40：train。
  - Week 41-46：valid。
  - Week 47-53：test。
- 每个 split 输出 household-product 购买记录。
- valid/test 的 ground truth 必须是目标周真实购买商品。

交付物：

- `data/processed/train_interactions.csv`
- `data/processed/valid_interactions.csv`
- `data/processed/test_interactions.csv`
- `data/processed/split_summary.csv`

验收标准：

- 没有使用未来周数据训练模型。
- `train.max_week < valid.min_week <= valid.max_week < test.min_week`。
- 输出每个 split 的 household 数、product 数、interaction 数。

#### A5. 特征表输出

具体要做：

- 输出商品特征表。
- 输出 household 特征表。
- 输出促销和优惠券初始表，供 C 深加工。

交付物：

`product_features.csv` 至少包含：

| 字段 | 含义 |
| --- | --- |
| product_id | 商品 id |
| department | 大类 |
| product_category | 商品类别 |
| brand | 品牌 |
| product_description | 商品描述 |

`household_features.csv` 至少包含：

| 字段 | 含义 |
| --- | --- |
| household_id | 家庭 id |
| total_baskets_train | 训练期 basket 数 |
| total_spend_train | 训练期总消费 |
| avg_basket_value_train | 平均 basket 金额 |
| top_department_train | 最常买 department |
| top_category_train | 最常买 category |
| demographics_available | 是否有人口统计信息 |

交付物：

- `data/processed/product_features.csv`
- `data/processed/household_features.csv`
- `data/processed/promotion_base.csv`
- `data/processed/coupon_base.csv`

验收标准：

- B 能用 product_features 做 category baseline。
- C 能用 promotion_base 和 coupon_base 做重排序特征。
- D 能用 household_features 做 demo 左侧信息。

#### A6. EDA 图表

具体要做：

至少输出以下图：

- 每周交易量趋势。
- 商品购买频次长尾分布。
- household 活跃度分布。
- Top 10 department。
- Top 10 product_category。
- coupon redemption 稀疏性分析。
- demographics 覆盖率。

交付物：

- `outputs/figures/weekly_transaction_trend.png`
- `outputs/figures/product_frequency_distribution.png`
- `outputs/figures/household_activity_distribution.png`
- `outputs/figures/top_departments.png`
- `outputs/figures/coupon_demographics_sparsity.png`

验收标准：

- 每张图都能直接放进 PPT。
- 图标题、坐标轴、单位清楚。

### 3.3 A 的最终展示材料

A 负责 PPT 中这些页：

1. Dataset Overview
2. Problem Definition
3. Time-based Split
4. Data Risks and Mitigation

A 负责报告中这些小节：

- Dataset。
- Data preprocessing。
- Train/validation/test split。
- Feature construction overview。

---

## 4. B 的详细任务：Model Lead

### 4.1 B 的一句话职责

B 要做出项目第一阶段的推荐模型，证明系统可以根据历史购买预测下一周商品。B 是候选商品生成负责人。

### 4.2 B 必须完成的任务

#### B1. 统一模型输入

具体要做：

- 接收 A 的 `train_interactions.csv`。
- 构建 household-product sparse matrix。
- 建立 id mapping：
  - raw `household_id` -> matrix row id。
  - raw `product_id` -> matrix column id。
- 保存 mapping，方便 C/D 回查原始 id。

交付物：

- `src/models/data_loader.py`
- `outputs/mappings/household_id_map.csv`
- `outputs/mappings/product_id_map.csv`

验收标准：

- 每个模型都复用同一套 mapping。
- 模型输出能转换回原始 household_id 和 product_id。

#### B2. Popularity Baseline

具体要做：

- 统计 train 中每个 product 的购买次数。
- 对所有 household 推荐同一组 Top-K 商品。
- K 至少支持 10、20、50。

交付物：

- `src/models/popularity.py`
- `outputs/candidates_popularity.csv`

验收标准：

- 输出格式和后续模型一致。
- 能计算 Recall@10/20 和 NDCG@10/20。

#### B3. Category Popularity Baseline

具体要做：

- 对每个 household 统计训练期高频 department/product_category。
- 在该 household 高频类别中推荐热门商品。
- 如果 household 历史太少，则 fallback 到 global popularity。

交付物：

- `src/models/category_popularity.py`
- `outputs/candidates_category_popularity.csv`

验收标准：

- 结果比 global popularity 更个性化。
- 能解释推荐来自 household 高频类别。

#### B4. ItemKNN

具体要做：

- 用 household-product matrix 计算 item-item 相似度。
- 根据 household 历史购买商品加权召回相似商品。
- 避免推荐训练期已经买过的商品，或者单独标记 repeat recommendation。

交付物：

- `src/models/itemknn.py`
- `outputs/candidates_itemknn.csv`

验收标准：

- 能输出每个 household 的 Top-50 candidate。
- 能与 Popularity、ALS 做同口径评估。

#### B5. Implicit ALS

具体要做：

- 使用 implicit feedback 训练 ALS。
- 输入矩阵可以用购买次数，也可以用 log-scaled count。
- 尝试参数：
  - factors: 32, 64, 128
  - regularization: 0.01, 0.05, 0.1
  - iterations: 20, 50
  - alpha: 10, 20, 40
- 用 valid set 选最佳参数。

交付物：

- `src/models/implicit_als.py`
- `outputs/candidates_als.csv`
- `outputs/model_tuning/als_tuning_results.csv`

验收标准：

- ALS 至少要跑出一版稳定结果。
- ALS 必须进入最终结果表。
- 能说明最佳参数是怎么选的。

#### B6. BPR Matrix Factorization

具体要做：

- 训练 BPR 模型。
- 与 ALS 做对比。
- 如果时间不足，BPR 可以降级为加分项，但要优先保证 ALS。

交付物：

- `src/models/bpr.py`
- `outputs/candidates_bpr.csv`
- `outputs/model_tuning/bpr_tuning_results.csv`

验收标准：

- 如果完成，BPR 进入最终模型对比表。
- 如果未完成，要在报告 limitation 里说明。

#### B7. 统一 candidate 输出

B 输出给 C 的所有 candidate 文件必须统一格式：

| 字段 | 含义 |
| --- | --- |
| household_id | 原始 household id |
| product_id | 原始 product id |
| base_score | 模型预测分数 |
| model_name | 模型名 |
| base_rank | 原始排序 |

每个 household 推荐 Top-50 或 Top-100 candidate。

交付物：

- `outputs/candidates_als.csv`
- `outputs/candidates_bpr.csv`
- `outputs/candidates_itemknn.csv`
- `outputs/candidates_popularity.csv`
- `outputs/candidates_category_popularity.csv`

验收标准：

- C 可以不改代码直接读取这些文件做 reranking。
- D 可以读取这些文件做 demo。

### 4.3 B 的最终展示材料

B 负责 PPT 中这些页：

1. Candidate Generation
2. Baselines
3. ALS/BPR Model
4. Model Performance Comparison

B 负责报告中这些小节：

- Baseline methods。
- Matrix factorization methods。
- Hyperparameter tuning。
- Candidate generation results。

---

## 5. C 的详细任务：Business/Reranking Lead

### 5.1 C 的一句话职责

C 要把普通推荐系统变成“促销感知型推荐系统”。C 是项目创新点和商业指标负责人。

### 5.2 C 必须完成的任务

#### C1. Promotion 特征

具体要做：

- 从 promotions 表构造 product-week 特征。
- 判断 product 在目标 week 是否有促销。
- 如果字段支持，区分：
  - display。
  - mailer。
  - in-store placement。
  - temporary price reduction。
- 输出每个 product-week 的 promotion_score。

交付物：

- `src/features/promotion_features.py`
- `data/processed/product_week_promotion_features.csv`

验收标准：

- 任意一个 product_id + week 能查到是否有 promotion。
- 能解释 promotion_score 怎么计算。

#### C2. Coupon 特征

具体要做：

- 研究 coupons、campaigns、coupon_redemptions 的连接方式。
- 构造 household-product 或 household-campaign-product 级别特征。
- 标记 household 是否接触过相关 campaign。
- 标记 product 是否属于 coupon 覆盖范围。
- coupon redemption 不作为主监督目标，只作为辅助分析。

交付物：

- `src/features/coupon_features.py`
- `data/processed/household_product_coupon_features.csv`

验收标准：

- 能说明 coupon redemption 为什么稀疏。
- 能说明为什么 coupon 只做 reranking signal。

#### C3. Discount Cost Proxy

具体要做：

- 用以下字段构造折扣成本代理：
  - `retail_disc`
  - `coupon_disc`
  - `coupon_match_disc`
- 计算商品历史平均 discount。
- 计算类别平均 discount。
- 计算 household 折扣敏感度。

建议特征：

- `avg_product_discount`
- `avg_category_discount`
- `household_discount_ratio`
- `estimated_discount_cost`

交付物：

- `src/features/discount_features.py`
- `data/processed/discount_cost_features.csv`

验收标准：

- 明确写在报告里：这是 proxy，不是真实利润。
- 能被 Business Utility@K 使用。

#### C4. Promotion-aware Reranking

具体要做：

接收 B 的 candidate 文件，计算最终分数：

`final_score = alpha * base_score + beta * promotion_score + gamma * coupon_score - lambda * discount_cost_proxy + rho * diversity_score`

需要实现：

- baseline reranking：只用 base_score。
- promotion reranking：base_score + promotion。
- coupon reranking：base_score + coupon。
- discount-aware reranking：base_score + promotion + coupon - discount cost。
- full reranking：加 diversity。

交付物：

- `src/rerank/promotion_reranker.py`
- `outputs/reranked_als.csv`
- `outputs/reranked_bpr.csv`

验收标准：

- 每个 household 输出 Top-10/20 最终推荐。
- 能比较 reranking 前后指标变化。

#### C5. Diversity 控制

具体要做：

- 避免 Top-K 全是同一个 department。
- 实现至少一种 diversity 方法：
  - 每个 department 最多 N 个。
  - 重复 department 降分。
  - product_category 覆盖奖励。

交付物：

- `src/rerank/diversity.py`
- `outputs/diversity_ablation.csv`

验收标准：

- Top-K 的 category/department 分布比纯 ALS 更分散。
- 不严重牺牲 Recall@K。

#### C6. Business Utility@K

具体要做：

实现：

`Business Utility@K = sum estimated sales value of hit products - lambda * sum estimated discount cost of hit products`

注意：

- 只对命中的推荐商品计算 sales value。
- discount cost 用 proxy。
- lambda 可以做敏感性分析。

交付物：

- `src/metrics/business_metrics.py`
- `outputs/business_utility_results.csv`
- `outputs/lambda_sensitivity.csv`

验收标准：

- 能回答：reranking 是否牺牲一点 Recall，但换来更高 Business Utility。
- 图表可以直接放进 PPT。

#### C7. 消融实验

必须输出以下变体：

| Variant | 含义 |
| --- | --- |
| ALS only | 只用 ALS base_score |
| ALS + Promotion | 加促销特征 |
| ALS + Coupon | 加优惠券特征 |
| ALS + Promotion + Coupon | 加促销和优惠券 |
| ALS + Promotion + Coupon - Discount | 加折扣成本惩罚 |
| ALS + Full Reranking | 加 diversity |

交付物：

- `outputs/reranking_ablation_results.csv`

验收标准：

- 表中至少包含 Recall@10、NDCG@10、Diversity、Business Utility@10。

### 5.3 C 的最终展示材料

C 负责 PPT 中这些页：

1. Why Promotion-aware?
2. Reranking Formula
3. Business Utility@K
4. Ablation Study

C 负责报告中这些小节：

- Promotion and coupon feature construction。
- Reranking method。
- Business utility proxy。
- Ablation study。

---

## 6. D 的详细任务：Integration/Demo Lead

### 6.1 D 的一句话职责

D 要把模型结果变成可以展示的系统。D 是最终集成、demo 和 presentation 负责人。

### 6.2 D 必须完成的任务

#### D1. 统一评估表整合

具体要做：

- 收集 B 的模型结果。
- 收集 C 的 reranking 结果。
- 整合成一张最终结果表。

最终结果表字段：

| 字段 | 含义 |
| --- | --- |
| model_variant | 模型或重排序版本 |
| recall_at_10 | Recall@10 |
| recall_at_20 | Recall@20 |
| ndcg_at_10 | NDCG@10 |
| ndcg_at_20 | NDCG@20 |
| coverage | 覆盖率 |
| diversity | 多样性 |
| novelty | 新颖性 |
| business_utility_at_10 | 商业效用代理 |

交付物：

- `outputs/final_results_table.csv`
- `outputs/final_results_table_for_ppt.png`

验收标准：

- 最终 PPT 只引用这一张结果总表。
- 数值和 B/C 的原始输出一致。

#### D2. LightGCN

具体要做：

- 把 A 的 train/valid/test 转成 RecBole 格式。
- 建立 household/product 连续 id。
- 训练 LightGCN。
- 输出和 B 一样格式的 candidate。

推荐优先级：

- 如果时间够：完成 LightGCN 并进入最终结果。
- 如果时间紧：只做 LightGCN 尝试，作为 limitation/future work。
- 不允许因为 LightGCN 影响 ALS + reranking 主线。

交付物：

- `src/models/lightgcn_recbole/`
- `outputs/candidates_lightgcn.csv`
- `outputs/lightgcn_results.csv`

验收标准：

- 如果 LightGCN 完成，能进入统一结果表。
- 如果未完成，PPT 中不强行吹，只放 future work。

#### D3. Streamlit Demo

具体要做：

页面组件：

- household_id selector。
- household profile panel。
- historical top departments。
- average basket value。
- frequent brands。
- Top-10 recommendations table。
- coupon recommendation flag。
- explanation column。
- marketing budget slider。
- KPI cards：
  - estimated sales value。
  - estimated discount cost。
  - business utility proxy。
  - category diversity。

交付物：

- `app/streamlit_app.py`
- `outputs/demo_recommendations.csv`
- `slides_assets/demo_screenshot.png`

验收标准：

- 老师现场能看到系统不是只输出 product_id。
- 每个推荐商品有类别、品牌、推荐分数和推荐理由。
- 即使真实模型很简单，demo 也能讲清楚业务逻辑。

#### D4. 推荐解释

具体要做：

为每个推荐生成 rule-based explanation：

- 该家庭最近购买过同类商品。
- 该商品属于该家庭高频 department。
- 该商品与历史 basket 中商品高度共现。
- 该商品当前有促销曝光。
- 该商品适合 coupon 推送。

交付物：

- `src/explain/explanations.py`
- demo 表格中的 `reason` 字段。

验收标准：

- 每条推荐至少有一个原因。
- 原因不能是空泛的“模型认为你会喜欢”。

#### D5. 系统架构图

具体要做：

画一张系统架构图：

1. Raw Data
2. Data Cleaning and Time Split
3. Candidate Generation
4. Promotion/Coupon Feature Layer
5. Business-aware Reranking
6. Evaluation
7. Streamlit Demo

交付物：

- `slides_assets/system_architecture.png`

验收标准：

- 架构图能一眼看出两阶段推荐系统。
- 能体现项目 X-factor 在第二阶段。

#### D6. PPT 和最终报告整合

具体要做：

- 收集团队每个人的文字和图。
- 统一 PPT 风格。
- 删除重复内容。
- 确保每个图有标题和结论。
- 确保每个人都有展示页。

交付物：

- `final/PromoMind_presentation.pptx`
- `final/PromoMind_report.docx` 或 `final/PromoMind_report.pdf`

验收标准：

- PPT 不是四个人材料硬拼。
- 每页只讲一个核心点。
- 结果表和 demo 截图清楚。

### 6.3 D 的最终展示材料

D 负责 PPT 中这些页：

1. System Architecture
2. LightGCN, if completed
3. Streamlit Demo
4. Final Takeaways

D 负责报告中这些小节：

- Demo design。
- System integration。
- Deep model extension。
- Final conclusion。

---

## 7. 统一文件接口

全组必须遵守统一接口，否则后面会大量浪费时间。

### 7.1 A 给全组的数据文件

`data/processed/train_interactions.csv`

| 字段 | 必须有 |
| --- | --- |
| household_id | yes |
| product_id | yes |
| week | yes |
| quantity | yes |
| sales_value | yes |
| retail_disc | yes |
| coupon_disc | yes |
| coupon_match_disc | yes |

`data/processed/product_features.csv`

| 字段 | 必须有 |
| --- | --- |
| product_id | yes |
| department | yes |
| product_category | yes |
| brand | yes |
| product_description | optional but preferred |

`data/processed/household_features.csv`

| 字段 | 必须有 |
| --- | --- |
| household_id | yes |
| total_baskets_train | yes |
| total_spend_train | yes |
| avg_basket_value_train | yes |
| top_department_train | yes |
| top_category_train | yes |
| demographics_available | yes |

### 7.2 B 给 C/D 的 candidate 文件

`outputs/candidates_MODEL.csv`

| 字段 | 必须有 |
| --- | --- |
| household_id | yes |
| product_id | yes |
| base_score | yes |
| model_name | yes |
| base_rank | yes |

### 7.3 C 给 D 的 reranked 文件

`outputs/reranked_MODEL.csv`

| 字段 | 必须有 |
| --- | --- |
| household_id | yes |
| product_id | yes |
| base_score | yes |
| promo_score | yes |
| coupon_score | yes |
| discount_cost_proxy | yes |
| diversity_score | yes |
| final_score | yes |
| final_rank | yes |
| recommend_coupon | yes |
| reason_signal | optional but preferred |

### 7.4 D 给全组的最终输出

`outputs/final_results_table.csv`

| 字段 | 必须有 |
| --- | --- |
| model_variant | yes |
| recall_at_10 | yes |
| recall_at_20 | yes |
| ndcg_at_10 | yes |
| ndcg_at_20 | yes |
| coverage | yes |
| diversity | yes |
| novelty | yes |
| business_utility_at_10 | yes |

---

## 8. 任务依赖关系

关键依赖：

- B 依赖 A 的 `train_interactions.csv` 和 `product_features.csv`。
- C 依赖 A 的 promotion/coupon 原始整理，也依赖 B 的 candidate 输出。
- D 依赖 A 的 household/product features、B 的 candidate、C 的 reranked recommendations。
- PPT 和报告依赖所有人提供图表和文字。

不能等待到最后才整合。建议 Day 2 就先使用小样本跑通全链路：

1. A 输出 100 个 household 的 sample split。
2. B 用 sample 跑 Popularity。
3. C 对 Popularity candidate 做简单 reranking。
4. D 把 sample result 接进 demo。

这样即使后面模型变复杂，系统接口已经通了。

---

## 9. 推荐时间线

下面按 7 天项目冲刺设计。如果实际时间更长，可以拉长每个阶段；如果更短，直接跳到最低完成线。

### Day 0：开工会议

目标：定题、定接口、定每个人责任。

A：

- 确认数据是否能下载和读取。
- 确认字段命名。

B：

- 确认使用 `implicit` 还是 Cornac/其他库。
- 准备 sparse matrix 代码结构。

C：

- 确认 promotions/coupons/campaigns 连接方式。
- 初步定义 reranking 公式。

D：

- 建 demo 空壳。
- 建 PPT 大纲。

全组输出：

- 确认本执行包。
- 确认统一文件接口。

### Day 1：数据初版和最小链路

A：

- 输出 cleaned sample。
- 输出 train/valid/test sample。

B：

- 用 sample 跑 Popularity baseline。
- 输出 `candidates_popularity_sample.csv`。

C：

- 写最简单 reranking 函数。
- 用假 promotion_score 跑通。

D：

- Streamlit 读取 sample recommendation。
- 页面能显示 household 和 Top-10 商品。

验收：

- 小样本全链路跑通：data -> model -> rerank -> demo。

### Day 2：完整数据和 baseline

A：

- 完成完整数据清洗。
- 完成时间切分。
- 完成 product_features 和 household_features。

B：

- 完成 Popularity。
- 完成 Category Popularity。

C：

- 完成 promotion feature 初版。
- 完成 coupon feature 初版。

D：

- 完成 demo 页面布局。
- 准备系统架构图初版。

验收：

- 全组可以使用完整 train/valid/test。
- 有第一版 baseline 指标。

### Day 3：主模型第一版

A：

- 完成 EDA 图表。
- 检查数据 leakage。

B：

- 完成 ItemKNN。
- 完成 ALS 第一版。
- 输出 Top-100 candidates。

C：

- 接入 B 的 ALS candidate。
- 完成 promotion-aware reranking 第一版。

D：

- demo 接入 ALS recommendation。
- 汇总第一版结果表。

验收：

- ALS candidate 可以被 C/D 使用。
- Reranking 前后至少有一版结果。

### Day 4：调参与商业指标

A：

- 补充数据说明文字。
- 帮 C 检查 promotion/coupon join 是否合理。

B：

- ALS 调参。
- 尝试 BPR。

C：

- 完成 Business Utility@K。
- 完成 discount cost proxy。
- 完成 diversity 控制。

D：

- demo 增加 recommendation reason。
- demo 增加 KPI cards。

验收：

- 有主模型结果表。
- 有商业指标结果表。

### Day 5：消融实验和图表

A：

- 输出最终 EDA 图。
- 写 Dataset PPT 页。

B：

- 输出最终 candidate generation 结果。
- 写 Model PPT 页。

C：

- 输出消融实验表。
- 输出 lambda sensitivity 或 budget trade-off 图。
- 写 Reranking PPT 页。

D：

- 尝试 LightGCN。
- 整合 final_results_table。
- 做 demo 截图。

验收：

- 每个人至少交 1-2 页 PPT 内容。
- 所有核心结果表完成。

### Day 6：报告和 PPT 整合

A：

- 完成数据部分报告文字。

B：

- 完成模型部分报告文字。

C：

- 完成 X-factor 和商业指标报告文字。

D：

- 整合 PPT。
- 整合报告。
- 检查 demo 是否能现场运行。

验收：

- 第一版完整 PPT。
- 第一版完整报告。
- demo 可运行。

### Day 7：彩排和修正

全组：

- 每个人按自己的展示时间讲一遍。
- 记录卡顿和问题。
- 修正 PPT、报告、demo。

A：

- 检查数据口径问题。

B：

- 检查模型结果是否可解释。

C：

- 检查 business utility 说法是否严谨。

D：

- 检查 demo 现场稳定性。

验收：

- 最终提交材料完成。
- 每个人知道自己讲哪几页。

---

## 10. 每个人每天要汇报什么

每天 standup 每个人只回答三件事：

1. 昨天完成了什么？
2. 今天要交付什么？
3. 有没有 blocker？

### A 每天汇报模板

- 数据处理进度：
- 当前输出文件：
- 数据问题：
- 是否影响 B/C/D：

### B 每天汇报模板

- 已完成模型：
- 当前最好指标：
- 已输出 candidate 文件：
- 是否需要 A 改数据：

### C 每天汇报模板

- 已完成特征：
- reranking 版本：
- Business Utility 结果：
- 是否需要 B 新 candidate：

### D 每天汇报模板

- demo 当前功能：
- 已整合结果：
- PPT/报告进度：
- 是否需要其他人补图或补文字：

---

## 11. 最终 PPT 分页分配

建议 12-14 页。

| 页码 | 标题 | 负责人 | 内容 |
| --- | --- | --- | --- |
| 1 | Title | D | 项目名称、成员、课程 |
| 2 | Business Motivation | A/C | 零售发券痛点、项目目标 |
| 3 | Dataset Overview | A | The Complete Journey 表结构和规模 |
| 4 | Task Definition | A | 下一周 Top-K 推荐、implicit feedback、时间切分 |
| 5 | System Architecture | D | 两阶段架构：candidate generation + reranking |
| 6 | Baselines | B | Popularity、Category Popularity、ItemKNN |
| 7 | Matrix Factorization | B | ALS/BPR 方法和参数 |
| 8 | Promotion-aware Reranking | C | 重排序公式和特征 |
| 9 | Evaluation Metrics | C | Recall/NDCG/Coverage/Diversity/Business Utility |
| 10 | Main Results | B/D | 模型对比表 |
| 11 | Ablation Study | C | reranking 消融实验 |
| 12 | Demo | D | Streamlit 截图或现场演示 |
| 13 | Limitations | A/C/D | coupon 稀疏、demo 限制、profit proxy |
| 14 | Conclusion | D | 最终 takeaways 和未来工作 |

---

## 12. 最终报告分工

| 报告部分 | 负责人 | 字数建议 |
| --- | --- | --- |
| Abstract | D 汇总，全组确认 | 150-200 words |
| Introduction | A + C | 500-700 words |
| Dataset | A | 600-800 words |
| Problem Definition | A | 300-500 words |
| Methodology: Baselines | B | 500-700 words |
| Methodology: ALS/BPR | B | 600-800 words |
| Methodology: Promotion-aware Reranking | C | 700-900 words |
| Evaluation Metrics | C | 400-600 words |
| Experiments and Results | B + C + D | 800-1200 words |
| Demo/System Design | D | 500-700 words |
| Limitations | A + C + D | 400-600 words |
| Conclusion | D 汇总，全组确认 | 250-400 words |

---

## 13. RACI 矩阵说明

RACI 含义：

- R = Responsible，实际干活的人。
- A = Accountable，最终负责结果的人。
- C = Consulted，需要咨询的人。
- I = Informed，只需要同步进度的人。

| 工作项 | A | B | C | D |
| --- | --- | --- | --- | --- |
| 数据下载和读取 | A/R | I | I | I |
| 数据清洗 | A/R | C | C | I |
| 时间切分 | A/R | C | C | I |
| Product features | A/R | C | C | C |
| Household features | A/R | I | I | C |
| Popularity baseline | I | B/R | I | I |
| Category baseline | C | B/R | I | I |
| ItemKNN | I | B/R | I | I |
| ALS | I | B/R | I | I |
| BPR | I | B/R | I | I |
| Promotion features | C | I | C/R | I |
| Coupon features | C | I | C/R | I |
| Discount proxy | C | I | C/R | I |
| Reranking formula | I | C | C/R | C |
| Business Utility@K | I | C | C/R | C |
| Final metrics table | I | C | C | D/R |
| LightGCN | I | C | I | D/R |
| Streamlit demo | C | C | C | D/R |
| PPT integration | I | C | C | D/R |
| Final report integration | C | C | C | D/R |

---

## 14. 风险清单

| 风险 | 影响 | 负责人 | 预防方案 | 备选方案 |
| --- | --- | --- | --- | --- |
| 数据下载或读取失败 | 全组无法开始 | A | Day 0 先验证数据读取 | 用 sample 或公开 CSV 版本先跑 |
| 商品太多导致模型慢 | ALS/BPR 训练慢 | A/B | 过滤 Top 10,000 商品 | 先 Top 5,000 商品跑通 |
| coupon redemption 太稀疏 | 无法做 coupon 监督学习 | C | 主任务定义为购买预测 | coupon 只作为 reranking signal |
| demographics 覆盖少 | 用户画像不完整 | A/D | 只做可选展示和分组分析 | demo 对缺失 household 显示 unknown |
| ALS 指标不如 popularity | 模型说服力弱 | B | 调参、过滤商品、处理 repeat purchase | 强调 grocery baseline 很强，并用 reranking 做亮点 |
| Reranking 降低 Recall 太多 | X-factor 被质疑 | C | 调 alpha 保留 base_score 权重 | 展示 trade-off，而不是只追求 Recall |
| LightGCN 跑不出来 | 深度模型缺失 | D | 作为 bonus，不阻塞主线 | 用 ALS/BPR + reranking 完成主项目 |
| Demo 现场不稳定 | 展示风险 | D | 提前录屏或截图 | 用截图讲 demo 流程 |
| 四个人文件格式不一致 | 整合困难 | D/A | 强制统一接口 | Day 2 做接口检查 |

---

## 15. 最低完成线和加分线

### 15.1 最低完成线

如果时间很紧，必须完成：

- A：数据清洗、时间切分、product_features、household_features。
- B：Popularity、Category Popularity、ALS。
- C：Promotion-aware reranking、Business Utility@K。
- D：最终结果表、Streamlit 简版 demo、PPT 整合。

最低完成线已经足够形成完整项目：

数据 -> ALS 推荐 -> 促销重排序 -> 商业指标 -> demo。

### 15.2 标准完成线

在最低完成线基础上增加：

- B：ItemKNN、BPR。
- C：完整消融实验、diversity、novelty。
- D：demo budget slider、推荐解释。

### 15.3 加分线

在标准完成线基础上增加：

- D：LightGCN。
- C：lambda sensitivity curve。
- D：更完整 demo UI。
- 全组：更漂亮的系统架构图和 ER 图。

---

## 16. PM 对每个人的明确要求

### 对 A 的要求

你不是只做 EDA。你要保证全组的数据口径一致。B、C、D 后面所有代码都依赖你的 processed files，所以你必须最先交出 sample version，再交 full version。

必须按时交：

- Day 1：sample split。
- Day 2：full split。
- Day 3：EDA 图。
- Day 5：数据页 PPT 和报告文字。

### 对 B 的要求

你不是只跑一个模型。你要做出从简单到复杂的推荐模型梯度，让结果表有比较价值。ALS 是必须完成项，BPR 是尽量完成项。

必须按时交：

- Day 2：Popularity 和 Category Popularity。
- Day 3：ALS 第一版。
- Day 4：ALS 调参结果。
- Day 5：模型页 PPT 和报告文字。

### 对 C 的要求

你负责项目最有新意的地方。重点不是做一个很复杂的 coupon uplift model，而是把促销、优惠券、折扣成本合理进入推荐排序，并严谨解释 Business Utility 是 proxy。

必须按时交：

- Day 2：promotion/coupon 特征初版。
- Day 3：reranking 第一版。
- Day 4：Business Utility@K。
- Day 5：消融实验和 X-factor PPT 页。

### 对 D 的要求

你负责让项目看起来像一个完整系统，而不是四份 notebook。LightGCN 是加分项，但 demo、最终结果表、PPT 整合是必须完成项。

必须按时交：

- Day 1：demo 空壳。
- Day 3：demo 接入 ALS candidate。
- Day 5：demo 接入 reranking result。
- Day 6：完整 PPT 初版。
- Day 7：最终 demo 和提交材料。

---

## 17. 可以直接发群里的版本

大家这个项目我建议按四条线并行做，避免一个人全扛模型、一个人只写 PPT。

A 负责数据线：The Complete Journey 数据读取、清洗、时间切分、EDA、product/household 特征表。A 的输出是全组地基，Day 1 先给 sample split，Day 2 给 full split。

B 负责推荐模型线：Popularity、Category Popularity、ItemKNN、Implicit ALS、BPR。ALS 是必须完成，BPR 尽量完成。B 要输出统一 candidate 文件给 C 和 D。

C 负责项目 X-factor：promotion/coupon 特征、discount cost proxy、promotion-aware reranking、Business Utility@K、消融实验。C 要证明我们的系统不只是普通推荐，而是促销和优惠券决策系统。

D 负责整合展示线：LightGCN 尝试、Streamlit demo、最终结果表、PPT 和报告整合。LightGCN 是加分项，但 demo 和最终材料是必须完成。

最低完成线是：A 数据切分 + B Popularity/ALS + C promotion-aware reranking/business utility + D demo/PPT。加分项是 BPR、LightGCN、budget slider 和更完整的消融实验。

每天每个人同步三件事：昨天完成了什么、今天交付什么、有没有 blocker。所有文件必须按统一接口输出，不然后面整合会很痛苦。

