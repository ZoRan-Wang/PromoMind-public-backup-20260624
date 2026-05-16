# PromoMind: Promotion-aware Grocery Basket and Coupon Recommender

中文标题：PromoMind：面向超市零售的个性化商品与优惠券推荐系统

## 5 秒会议版 / 5-second meeting note

**中文：** 我的提议是 PromoMind：一个面向超市零售的促销感知购物篮和优惠券推荐系统，用家庭历史购物、商品信息、促销和优惠券数据预测下一次最可能购买的商品，并做商业效用重排序。完整版在这里：[粘贴共享文档链接]。线上会议里如果大家对这个提议有疑问，可以直接问问题，我就不再重复总结一遍。

**English:** My proposal is PromoMind: a promotion-aware grocery basket and coupon recommender for supermarket retail. It predicts each household's next likely purchases and re-ranks recommendations using promotion, coupon, discount cost, and business utility signals. Full proposal: [paste shared document link]. In the online meeting, please just ask questions about the proposal; I will not repeat the full summary.

## One-page Snapshot

- **Project:** PromoMind: promotion-aware grocery basket and coupon recommender
- **Dataset:** The Complete Journey
- **Main task:** Next-week household-product Top-K recommendation
- **Core models:** Popularity, Category Popularity, ItemKNN, Implicit ALS, BPR
- **Advanced model:** LightGCN via RecBole
- **X-factor:** Promotion-aware re-ranking with coupon, discount-cost, and diversity signals
- **Metrics:** Recall@K, NDCG@K, Coverage, Diversity, Novelty, Business Utility@K
- **Demo:** Streamlit dashboard with household selector and marketing budget slider

## 中文完整版

### 1. 项目标题

- PromoMind: A Promotion-aware Grocery Basket Recommender for Retail Marketing Optimization
- 中文标题：PromoMind：面向超市零售的个性化商品与优惠券推荐系统
- 一句话定位：这不是普通的 Top-K 商品推荐，而是一个把“预测下一篮购买”和“优惠券/促销预算分配”结合起来的零售增长决策系统。

### 2. 背景与商业问题

- 超市、电商、生鲜平台和会员制零售商拥有大量会员交易记录，但促销和发券往往仍然粗放：很多优惠券发给了本来就会购买的人，也有很多促销曝光给到了低转化家庭。
- 传统推荐系统主要回答“用户可能喜欢什么”或“用户下一次可能买什么”。PromoMind 进一步回答一个更接近真实业务的问题：在预算有限的情况下，哪些商品值得配合优惠券或促销曝光推送给哪些家庭。
- 项目目标是同时提高下一次购买命中率、减少无效发券、提升客单价，并用可解释的推荐理由支持营销决策。

### 3. 数据集选择：The Complete Journey

- 本项目使用 The Complete Journey 数据集。该数据集通过 completejourney 公开包提供，覆盖 2,469 个家庭在一年内的超市购物交易，记录的是家庭层面的全部购物，而不是只截取少数商品类别。
- 它包含交易、商品、促销、优惠券、营销活动和部分家庭画像，天然适合做“推荐 + 促销决策”的项目。官方说明列出的核心表包括 campaigns、campaign_descriptions、coupons、coupon_redemptions、demographics、products、promotions_sample 和 transactions_sample。
- 数据规模足够支撑课程项目：完整 transactions 表有 1,469,307 行和 11 个变量；完整 promotions 表有 20,940,529 行和 5 个变量；demographics 表有 801 行和 8 个变量。交易字段包含 household_id、store_id、basket_id、product_id、quantity、sales_value、retail_disc、coupon_disc、coupon_match_disc、week 和 transaction_timestamp。
- 相比 Instacart，The Complete Journey 更适合本题的商业叙事。Instacart 更适合下一篮复购预测；The Complete Journey 多了促销、优惠券、营销活动和家庭画像，可以把系统包装成真实零售环境下的营销优化引擎。

### 4. 推荐任务定义

- 核心任务：给定家庭 h 在第 t 周之前的所有购物行为，推荐该家庭在第 t+1 周最可能购买的 Top-K 商品，并对推荐列表进行促销感知重排序。
- 反馈形式：使用 implicit feedback。超市购物没有显式评分，因此购买行为本身就是正反馈；购买次数、购买金额和最近购买时间可以作为置信度或特征。
- 目标变量：y(h, j, t) = 1 if household h buys product j in week t+1; otherwise 0.
- 时间划分建议：week 1-40 作为训练集，week 41-46 作为验证集，week 47-53 作为测试集。这样用过去预测未来，比随机划分更符合真实推荐系统部署场景。

### 5. 方法框架

- 第一层是 candidate generation，用协同过滤模型生成候选商品。基础模型包括 Popularity Baseline、Category Popularity Baseline 和 ItemKNN；主模型使用 Implicit ALS 或 BPR Matrix Factorization；进阶模型使用 LightGCN。
- Popularity 和 Category Popularity 是必要基线，因为 grocery 数据存在强复购和强大众偏好。如果复杂模型不能明显超过这些基线，模型就缺乏说服力。
- Implicit ALS / BPR 可以用 Python implicit 库实现。官方文档支持 ALS、BPR 和 item-item models，并提供多线程训练；ALS 和 BPR 还支持 CUDA kernel，因此实现风险较低。
- LightGCN 将 household-product 交互看成二部图，通过邻居聚合学习用户和商品 embedding。RecBole 已经实现 LightGCN，并采用 pairwise training mode，因此可以作为深度推荐模型部分，而不需要从零实现图神经网络。
- 如果希望贴近课程生态，也可以用 Cornac 做 BPR、PMF、评价指标和实验管理。Cornac 的定位是用于构建、评估、比较推荐模型，并强调利用文本、图像、社交网络等辅助信息。

### 6. X-factor：促销感知重排序

- PromoMind 的关键创新是二阶段架构。第一阶段模型输出家庭 h 对商品 j 的基础偏好分数 p_hat(h, j)。第二阶段把促销、优惠券、折扣成本、类别多样性和用户历史偏好加入最终排序。
- 建议重排序公式：Score*(h, j) = alpha * p_hat(h, j) + beta * Promo(j, t) + gamma * Coupon(h, j) - lambda * DiscountCost(h, j) + rho * Diversity(h, j).
- Promo(j, t) 表示商品在目标周是否有促销展示或 mailer/in-store placement；Coupon(h, j) 表示该家庭是否收到或可匹配该商品相关 coupon；DiscountCost 可以用 retail_disc、coupon_disc、coupon_match_disc 构造折扣成本代理变量；Diversity 用来避免 Top-K 推荐全是同一类商品。
- 这个设计使项目不只优化 Recall@K 或 NDCG@K，还能展示推荐准确率和营销成本之间的权衡。

### 7. 评估指标

- 准确性指标：Recall@10、Recall@20、NDCG@10、NDCG@20。
- 系统性质指标：Coverage 衡量系统能覆盖多少家庭和商品；Diversity 衡量推荐列表是否过度集中在少数类别；Novelty 衡量系统是否只推荐高频爆款。
- 商业指标：Business Utility@K = sum estimated sales value of hits - lambda * sum estimated discount cost of hits.
- 需要严谨说明：数据中没有真实进货成本，因此该指标不能声称是真实 profit，而是 revenue-minus-discount proxy，即销售收入减折扣成本的商业代理指标。

### 8. 实验设计

- 主实验：比较 Popularity、Category Popularity、ItemKNN、Implicit ALS、BPR 和 LightGCN 在 Recall@K、NDCG@K、Coverage、Diversity、Novelty 和 Business Utility@K 上的表现。
- 消融实验：比较纯协同过滤、协同过滤 + 商品类别特征、协同过滤 + 促销特征、协同过滤 + 促销感知重排序。
- 关键研究问题包括：促销信息是否提升下一周购买预测；优惠券重排序是否会牺牲一部分准确率但提升商业效用；LightGCN 是否比 ALS/BPR 更适合 household-product 图结构；加入商品类别和家庭画像是否帮助低活跃家庭或冷启动场景。

### 9. Demo 设计

- 建议用 Streamlit 做一个可交互 demo。用户选择 household_id 后，页面左侧展示该家庭过去几周的主要购买类别、平均 basket value、常买品牌、是否有孩子、收入段等信息；右侧展示 Top-10 推荐商品。
- 每个推荐商品显示 department、product_category、brand、预估购买概率、是否建议配 coupon、推荐理由。推荐理由可用规则生成，例如“该家庭近期连续购买同类商品”“该商品与历史 basket 商品高度共现”“该商品当前有促销曝光”“该商品属于该家庭高频 department”。
- 加分设计是营销预算滑杆。预算低时，只给最可能转化的商品附 coupon；预算高时，覆盖更多推荐商品。滑杆变化后，展示 Recall@K、estimated sales value、estimated discount cost 和 Business Utility@K 的变化。

### 10. 可行性与风险控制

- 可行性：家庭数只有几千级，交易行数约 147 万，pandas、scipy sparse matrix、implicit ALS/BPR 和 RecBole 都能处理。商品数较多，可以先过滤到高频商品，例如购买次数超过 20 或 Top 10,000 商品。
- 风险一：coupon redemption 较稀疏，不适合作为唯一监督目标。解决方案是把主任务定义为下一周购买预测，coupon redemption 只作为辅助分析和重排序信号。
- 风险二：demographics 只覆盖部分家庭。解决方案是把人口统计信息作为可选特征和分组分析，不作为主模型必需输入。
- 风险三：真实利润不可得。解决方案是使用 estimated business utility 或 revenue-minus-discount proxy，不声称优化真实利润。
- 风险四：LightGCN 可能来不及。解决方案是先完成 Popularity、ALS/BPR、Recall/NDCG 和 promotion-aware reranking；即使 LightGCN 未完成，项目仍然完整。

### 11. 最小可交付路线

- MVP 1：完成数据清洗、时间切分、Popularity 和 Category Popularity baseline。
- MVP 2：完成 Implicit ALS / BPR 训练和 Recall@K、NDCG@K 评估。
- MVP 3：加入商品类别、促销和优惠券映射特征，完成 promotion-aware reranking。
- MVP 4：加入 Business Utility@K、Coverage、Diversity、Novelty，并形成结果表。
- MVP 5：完成 Streamlit demo 和会议展示材料。LightGCN 作为增强项，如果时间允许再加入。

### 12. 会议使用方式

- 建议会议中不要重新口头复述整份提案。先把共享文档链接发给组员，然后只说 5 秒版本。
- 如果大家对题目、数据、模型或工作量有疑问，可以直接进入问答。这样节省会议时间，也能让讨论集中在是否采用这个项目、如何分工、第一周先做什么。

## English Full Proposal

### 1. Project Title

- PromoMind: A Promotion-aware Grocery Basket Recommender for Retail Marketing Optimization
- Core positioning: PromoMind is not only a Top-K product recommender. It combines next-basket prediction with coupon and promotion-aware business re-ranking.

### 2. Background and Business Problem

- Grocery retailers, e-commerce platforms, fresh food delivery services, and membership-based retailers own rich household-level transaction data, but coupon distribution is often broad and inefficient.
- A standard recommender asks what a user is likely to buy. PromoMind asks a more business-oriented question: under a limited marketing budget, which products should be promoted to which households with coupons or promotional exposure?
- The goal is to improve next-purchase hit rate, reduce wasted coupons, increase basket value, and provide explainable recommendation reasons for retail marketing decisions.

### 3. Dataset: The Complete Journey

- This project uses The Complete Journey dataset, available through the completejourney public package. The dataset contains one year of grocery shopping transactions for 2,469 households and records all purchases made by those households, rather than only a limited group of product categories.
- The dataset is highly suitable for a recommendation-plus-promotion project because it includes transactions, product metadata, promotions, coupons, campaign exposure, coupon redemptions, and partial demographic information.
- The full transactions table contains 1,469,307 rows and 11 variables. The full promotions table contains 20,940,529 rows and 5 variables. The demographics table contains 801 rows and 8 variables. Transaction fields include household_id, store_id, basket_id, product_id, quantity, sales_value, retail_disc, coupon_disc, coupon_match_disc, week, and transaction_timestamp.
- Compared with Instacart, The Complete Journey supports a stronger business narrative. Instacart is excellent for next-basket prediction, while The Complete Journey also supports promotion, coupon, campaign, and household-profile analysis.

### 4. Recommendation Task

- Given all shopping behavior of household h before week t, recommend the Top-K products that household h is most likely to buy in week t+1, then re-rank the list using promotion-aware business signals.
- The feedback is implicit. Grocery shopping has no explicit star ratings, so purchases are treated as positive feedback. Purchase frequency, sales value, and recency can be used as confidence weights or features.
- Target definition: y(h, j, t) = 1 if household h buys product j in week t+1; otherwise 0.
- A time-based split is recommended: weeks 1-40 for training, weeks 41-46 for validation, and weeks 47-53 for testing. This is more realistic than random splitting because real recommenders use the past to predict the future.

### 5. Modeling Framework

- The first layer is candidate generation. Baselines include Popularity, Category Popularity, and ItemKNN. Main collaborative filtering models include Implicit ALS and BPR Matrix Factorization. The advanced model is LightGCN.
- Popularity and Category Popularity are important baselines because grocery purchases have strong repeat-purchase and mass-preference patterns. A complex model is not convincing if it cannot beat these baselines.
- Implicit ALS and BPR can be implemented using the Python implicit library, which supports ALS, BPR, item-item models, multi-threaded training, and CUDA support for ALS/BPR.
- LightGCN treats household-product interactions as a bipartite graph and learns household and product embeddings through neighborhood aggregation. RecBole already provides a LightGCN implementation with pairwise training mode.
- Cornac can be used if the project wants to align with a course-friendly recommender framework. It supports recommendation model training, evaluation, comparison, and auxiliary information.

### 6. X-factor: Promotion-aware Re-ranking

- PromoMind's main novelty is a two-stage architecture. The first-stage model estimates a household-product preference score p_hat(h, j). The second stage re-ranks candidates using promotion, coupon, discount cost, diversity, and household preference signals.
- Proposed scoring function: Score*(h, j) = alpha * p_hat(h, j) + beta * Promo(j, t) + gamma * Coupon(h, j) - lambda * DiscountCost(h, j) + rho * Diversity(h, j).
- Promo(j, t) captures whether product j has target-week mailer or in-store promotion exposure. Coupon(h, j) captures whether the household received or can be matched to a relevant coupon. DiscountCost can be proxied using retail_disc, coupon_disc, and coupon_match_disc. Diversity prevents the final Top-K list from being concentrated in one category.
- This turns the project from an accuracy-only recommender into a retail decision tool that balances prediction performance and marketing cost.

### 7. Evaluation Metrics

- Accuracy metrics: Recall@10, Recall@20, NDCG@10, and NDCG@20.
- System behavior metrics: Coverage measures how many households and products the system can serve; Diversity measures whether recommendations are too concentrated in a few categories; Novelty measures whether the model only recommends popular products.
- Business metric: Business Utility@K = sum estimated sales value of hits - lambda * sum estimated discount cost of hits.
- Important limitation: because true product cost is unavailable, this should not be described as real profit. It is a revenue-minus-discount proxy.

### 8. Experiment Design

- Main comparison: Popularity, Category Popularity, ItemKNN, Implicit ALS, BPR, and LightGCN on Recall@K, NDCG@K, Coverage, Diversity, Novelty, and Business Utility@K.
- Ablation study: pure collaborative filtering, collaborative filtering plus product category features, collaborative filtering plus promotion features, and collaborative filtering plus promotion-aware re-ranking.
- Key research questions: Does promotion information improve next-week purchase prediction? Does coupon-aware re-ranking trade off a small amount of accuracy for higher business utility? Is LightGCN better than ALS/BPR for household-product graph structure? Do product category and household profile features help low-activity or cold-start households?

### 9. Demo Plan

- Build a Streamlit demo. The user selects a household_id. The left panel shows recent top purchase categories, average basket value, common brands, kids indicator, income segment, and other available household features. The right panel shows Top-10 recommended products.
- Each recommendation displays department, product_category, brand, predicted purchase score, coupon recommendation flag, and a simple explanation.
- Explanation rules can be deterministic: the household recently bought similar products, the product often co-occurs with items in the household's baskets, the product has current promotion exposure, or the product belongs to a high-frequency department for this household.
- The strongest demo feature is a marketing budget slider. With a low budget, only the highest-conversion products receive coupons; with a higher budget, more products are coupon-supported. The dashboard updates estimated sales value, estimated discount cost, Recall@K, and Business Utility@K.

### 10. Feasibility and Risk Control

- Feasibility is strong. There are only a few thousand households and around 1.47 million transaction rows. pandas, scipy sparse matrices, implicit ALS/BPR, and RecBole can handle this scale. Product count can be controlled by filtering to frequent items, such as products with more than 20 purchases or the Top 10,000 products.
- Risk 1: coupon redemptions are sparse. Mitigation: make next-week product purchase prediction the main task and use coupon redemption only as an auxiliary analysis and re-ranking signal.
- Risk 2: demographics only cover some households. Mitigation: use demographics as optional features and subgroup analysis, not as required inputs for the main model.
- Risk 3: true profit is unavailable. Mitigation: use estimated business utility or revenue-minus-discount proxy and avoid claiming true profit optimization.
- Risk 4: LightGCN may take extra time. Mitigation: finish baselines, ALS/BPR, ranking metrics, and promotion-aware re-ranking first. LightGCN is an enhancement, not a dependency for project completeness.

### 11. Minimum Delivery Roadmap

- MVP 1: data cleaning, time split, Popularity and Category Popularity baselines.
- MVP 2: Implicit ALS / BPR training plus Recall@K and NDCG@K evaluation.
- MVP 3: product category features, promotion/coupon mapping, and promotion-aware re-ranking.
- MVP 4: Business Utility@K, Coverage, Diversity, Novelty, and final result tables.
- MVP 5: Streamlit demo and presentation materials. LightGCN is added if time allows.

### 12. How to Use This Proposal in the Meeting

- Do not repeat the full proposal verbally. Send the shared document link, then say the 5-second version.
- If teammates have questions about the topic, dataset, models, or workload, move directly into Q&A. This keeps the online meeting focused on adoption, division of work, and first-week implementation tasks.

## References / 参考资料

1. [The Complete Journey User Guide](https://cran.r-project.org/web/packages/completejourney/vignettes/completejourney.html) - Dataset overview, one-year household transactions, table descriptions, full transaction and promotion sizes.
2. [completejourney package reference manual](https://cran.r-project.org/web/packages/completejourney/refman/completejourney.html) - Official row counts and field descriptions for package data tables.
3. [completejourney GitHub repository](https://github.com/bradleyboehmke/completejourney) - Repository overview and access instructions for full transactions and promotions.
4. [implicit documentation](https://benfred.github.io/implicit/) - ALS, BPR, item-item models, multi-threaded training, and CUDA support for ALS/BPR.
5. [RecBole LightGCN documentation](https://recbole.io/docs/recbole/recbole.model.general_recommender.lightgcn.html) - LightGCN neighborhood aggregation over the user-item graph and pairwise training mode.
6. [Cornac JMLR paper](https://www.jmlr.org/papers/v21/19-805.html) - Cornac as a framework for multimodal recommendation with auxiliary information.
7. [Cornac quickstart](https://cornac.readthedocs.io/en/v2.2.1/user/quickstart.html) - Cornac experiment workflow for datasets, models, metrics, and model comparison.