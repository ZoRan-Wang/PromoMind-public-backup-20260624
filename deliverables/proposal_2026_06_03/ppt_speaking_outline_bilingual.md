# PromoMind PPT Speaking Outline

Target: max 10 minutes. Rehearsal target: 9:30 or less.

Two-presenter split:

- Presenter 1: Slides 1-5, about 3:50.
- Presenter 2: Slides 6-10, about 4:55.
- Total scripted time: about 8:50.

## Slide-by-slide Outline

| Slide | Owner | Time | English key point | 中文要点 |
| --- | --- | --- | --- | --- |
| 1. Title | Presenter 1 | 0:25 | PromoMind predicts next-basket products and decides which recommendations deserve promotion or coupon support. | PromoMind 先预测家庭下次可能买什么，再判断哪些推荐商品值得配促销或优惠券。 |
| 2. Research Problem | Presenter 1 | 0:50 | The research question is whether marketing context improves recommendation trade-offs. | 研究问题是促销和优惠券信息能否改善推荐准确率、商业效用和多样性之间的权衡。 |
| 3. Dataset | Presenter 1 | 1:05 | We use The Complete Journey, with transactions, products, promotions, coupons, campaigns, redemptions, and demographics. | 数据集是 The Complete Journey，重点讲清楚它有哪些表、规模多大、字段长什么样。 |
| 4. Task | Presenter 1 | 0:50 | The task is Top-K future product recommendation under implicit feedback with chronological split. | 任务是基于 implicit feedback 做下一阶段 Top-K 商品推荐，用时间切分避免未来信息泄漏。 |
| 5. Architecture | Presenter 1 | 0:45 | Stage 1 generates candidates; Stage 2 reranks them with promotion/coupon/cost/diversity signals. | 系统两阶段：先生成候选商品，再用促销、优惠券、折扣成本和多样性重排序。 |
| 6. Algorithms | Presenter 2 | 1:00 | Compare popularity baselines, ItemKNN, ALS, BPR if time allows, and LightGCN as bonus. | 比较 Popularity、Category Popularity、ItemKNN、ALS，时间允许再做 BPR，LightGCN 是加分项。 |
| 7. Reranking | Presenter 2 | 1:05 | X-factor: reranking balances base relevance, promotion, coupon, discount-cost proxy, and diversity. | 亮点是 promotion-aware reranking，不只是推荐商品，还决定哪些商品值得推优惠。 |
| 8. Experiments | Presenter 2 | 1:00 | Evaluate Recall@K, NDCG@K, coverage, diversity, novelty, Business Utility@K, and ablations. | 实验要比较准确率、覆盖率、多样性、新颖性、商业效用代理指标和重排序消融。 |
| 9. Demo | Presenter 2 | 0:55 | Streamlit demo visualizes household recommendations, coupon flags, explanations, and budget slider. | Demo 展示 household、Top-10 推荐、coupon 决策、推荐理由和预算滑杆。 |
| 10. Feasibility/Risks | Presenter 2 | 0:55 | The project has a stable core path, clear extensions, and controlled limitations for the final solution presentation. | 项目有稳定主线、清晰扩展方向，并且对最终方案汇报中的限制和风险有明确控制。 |

## Transition

English:

> Now that the dataset, prediction task, and system pipeline are clear, Presenter 2 will explain the models, reranking design, and evaluation plan.

中文：

> 数据集、任务定义和系统流程讲清楚之后，下面由第二位同学介绍模型、重排序设计和实验评估计划。

## Q&A Ownership

| Question type | Main responder | Core answer |
| --- | --- | --- |
| Why this dataset? | Presenter 1 | It has purchases plus promotion, coupon, campaign, product metadata, and partial demographics. |
| What features does it have? | Presenter 1 | household, basket, product, week, quantity, sales value, discounts, department, category, brand, promotion/coupon/campaign signals. |
| Why chronological split? | Presenter 1 | A real recommender predicts future baskets from past behavior; random split can leak future information. |
| Why ALS/BPR? | Presenter 2 | Grocery purchases are implicit feedback, so matrix factorization and pairwise ranking are suitable. |
| What is the X-factor? | Presenter 2 | Promotion-aware reranking and budget-aware coupon decision. |
| Is Business Utility profit? | Presenter 2 | No. It is revenue-minus-discount proxy because true cost/margin is unavailable. |
| Is this causal coupon uplift? | Presenter 2 | No. The data is observational; we use promotion/coupon signals for reranking, not causal treatment effect estimation. |
| What if LightGCN is unfinished? | Presenter 2 | It is bonus only. The core project remains complete with baselines, ALS/BPR, reranking, metrics, and demo. |
