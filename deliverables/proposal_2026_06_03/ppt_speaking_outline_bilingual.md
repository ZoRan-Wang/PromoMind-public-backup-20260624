# PromoMind Detailed PPT Speaking Script

Use this file for rehearsal. Do not read both languages in the actual presentation. Pick the language required by the class, and use the other language only to check meaning.

Target time: 8:50 to 9:20. Hard limit: 10:00.

Recommended split:

| Presenter | Slides | Target time | Role |
| --- | --- | --- | --- |
| Presenter 1 | 1-5 | about 3:50 | dataset, problem definition, system setup |
| Presenter 2 | 6-10 | about 4:55 | algorithms, reranking, experiments, demo, risks |

## Narrative Flow

| Part | Slides | What the audience should understand |
| --- | --- | --- |
| Why this matters | 1-2 | This is not only a grocery recommendation project; it studies recommendation under retail promotion decisions. |
| What data we use | 3 | The Complete Journey has transactions plus promotion, coupon, campaign, product, and demographic context. |
| What task we solve | 4-5 | We predict future household-product purchases with a chronological split, then rerank candidates with marketing signals. |
| How we test it | 6-8 | We compare recommender baselines and implicit-feedback models, then test promotion-aware reranking with offline metrics. |
| What makes it useful | 9-10 | The demo visualizes recommendation trade-offs, while the final evidence comes from experiments and risk-controlled evaluation. |

## Slide 1: Title

Time: 0:25
Owner: Presenter 1
Goal: introduce the project in one clean sentence.

English script:

> Our project is PromoMind, a promotion-aware grocery basket and coupon recommender. The core idea is simple: first, predict which products a household is likely to buy next; second, decide which of those recommendations are actually worth supporting with promotion or coupons. This makes the project different from a standard Top-K recommender, because the final output is closer to a retail marketing decision.

中文讲稿：

> 我们的项目叫 PromoMind，是一个促销感知型超市购物篮与优惠券推荐系统。核心想法很简单：第一步预测某个家庭下次最可能购买哪些商品；第二步判断这些推荐商品里，哪些值得配合促销或优惠券去推。这样它就不只是普通的 Top-K 推荐，而是更接近真实零售场景里的营销决策。

Transition:

> Next, we explain the research problem behind this design.

> 接下来我们说明为什么这个问题值得研究。

## Slide 2: Research Problem

Time: 0:50
Owner: Presenter 1
Goal: explain why ordinary recommendation is not enough.

English script:

> In grocery retail, retailers already have rich purchase histories, but coupons are often sent broadly. A normal recommender asks, "what is this household likely to buy?" Our project asks a second question: if marketing budget is limited, which recommended products should receive promotion or coupon support? The research question is whether promotion and coupon context can improve the trade-off among recommendation relevance, estimated business utility, and list diversity under sparse observational retail data.

中文讲稿：

> 在超市零售里，商家通常已经有很多会员购买记录，但优惠券发放经常比较粗放。普通推荐系统问的是“这个家庭可能会买什么”。我们的项目多问一步：如果营销预算有限，哪些推荐商品才值得配合促销或优惠券？所以研究问题是：在稀疏的观察性零售数据里，促销和优惠券上下文能不能改善推荐相关性、商业效用代理指标和推荐列表多样性之间的权衡。

Must not say:

- Do not say this is a causal coupon uplift study.
- Do not say we already have final results.

## Slide 3: Dataset And Collection

Time: 1:05
Owner: Presenter 1
Goal: make the dataset concrete and show why it fits the proposal.

English script:

> We use The Complete Journey dataset from the public `completejourney` project. It is suitable because it is not only a basket dataset. It contains household-level transactions, product metadata, promotions, campaigns, coupons, coupon redemptions, and partial demographics. The expected scale is about 2,469 households, 1.47 million transaction rows, 92 thousand products, and additional promotion and coupon tables. The raw RDS/RDA files are already in our repository, and the R script documents the reproducible export path. We are not scraping private data or collecting new personal information.

中文讲稿：

> 我们使用公开 `completejourney` 项目里的 The Complete Journey 数据集。它适合这个项目，因为它不只是一个购物篮数据集。它包含家庭级交易记录、商品信息、促销、营销活动、优惠券、优惠券兑换以及部分人口统计信息。预计规模包括 2,469 个家庭、约 147 万条交易明细、约 9.2 万个商品，以及额外的促销和优惠券相关表。原始 RDS/RDA 文件已经放在我们的仓库里，R 脚本记录了可复现的数据导出路径。我们不会爬取私人数据，也不会收集新的个人隐私数据。

Extra detail if asked:

- Transaction rows are close to receipt-line records.
- Important fields include household id, basket id, product id, week, quantity, sales value, retail/coupon discounts, department, category, brand, promotion exposure, coupon mapping, and partial demographic fields.

## Slide 4: Recommendation Task

Time: 0:50
Owner: Presenter 1
Goal: define the ML/recommender task and time split.

English script:

> The recommendation task is next-period Top-K product recommendation. For each household, we use purchase history before the target period to predict which products it will buy in the future target period. Since grocery data has no ratings, purchases are implicit positive feedback. We use a chronological split: weeks 1 to 40 for training, weeks 41 to 46 for validation, and weeks 47 to 53 for testing. This is important because a deployed recommender must use the past to predict the future. A random split would leak future shopping behavior into training.

中文讲稿：

> 推荐任务是下一阶段 Top-K 商品推荐。对每个家庭，我们用目标周期之前的购物历史，预测它在未来目标周期最可能购买哪些商品。因为超市数据没有评分，所以购买行为本身就是 implicit positive feedback。我们使用时间切分：第 1 到 40 周训练，第 41 到 46 周验证，第 47 到 53 周测试。这样做很重要，因为真实上线的推荐系统只能用过去预测未来。随机切分会把未来购物行为泄漏到训练里。

Must say clearly:

- The main label is product purchase, not coupon redemption.
- Coupon data is used later as auxiliary promotion/reranking information.

## Slide 5: System Architecture

Time: 0:45
Owner: Presenter 1
Goal: show the whole pipeline before model details.

English script:

> The system has two stages. Stage one is candidate generation: we clean the raw data, create a chronological split, and train recommenders such as popularity baselines, ItemKNN, ALS, BPR, or LightGCN if time allows. Stage two is promotion-aware reranking: we take the candidate products and adjust the final order using promotion, coupon, discount-cost, and diversity signals. The same pipeline then feeds offline evaluation and the Streamlit demo.

中文讲稿：

> 系统分成两个阶段。第一阶段是候选商品生成：我们清理原始数据，做时间切分，然后训练 popularity baseline、ItemKNN、ALS、BPR，时间允许的话再做 LightGCN。第二阶段是促销感知重排序：我们拿到候选商品以后，再用促销、优惠券、折扣成本和多样性信号调整最终顺序。同一条管线也会服务于离线评估和 Streamlit demo。

Transition to Presenter 2:

English:

> Now that the dataset, task, and system pipeline are clear, I will hand over to the model and evaluation part.

中文：

> 数据集、任务定义和系统流程讲清楚之后，下面进入模型和实验评估部分。

## Slide 6: Algorithms To Implement

Time: 1:00
Owner: Presenter 2
Goal: justify the model list and acknowledge libraries.

English script:

> We will compare simple baselines with stronger implicit-feedback recommenders. Popularity and Category Popularity are necessary because grocery data often has strong repeat-purchase and mass-market effects. If a complex model cannot beat these baselines, it is not convincing. Then we add ItemKNN and Implicit ALS. BPR is planned if time allows, and LightGCN through RecBole is a bonus extension. We will clearly acknowledge external libraries, especially `implicit` for ALS or BPR-style models and RecBole if LightGCN is attempted.

中文讲稿：

> 我们会比较简单 baseline 和更强的 implicit-feedback 推荐模型。Popularity 和 Category Popularity 很重要，因为超市数据通常有很强的复购效应和大众热门商品效应。如果复杂模型连这些 baseline 都打不过，就没有说服力。之后我们会加入 ItemKNN 和 Implicit ALS。BPR 在时间允许时实现，RecBole LightGCN 作为 bonus 扩展。外部库会明确说明，尤其是 `implicit` 用于 ALS 或 BPR 类型模型，RecBole 用于 LightGCN。

Must not overpromise:

- LightGCN is a bonus, not the core completion requirement.
- The core project remains complete with baselines, ALS/BPR-style models, reranking, metrics, and demo.

## Slide 7: Promotion-aware Reranking

Time: 1:05
Owner: Presenter 2
Goal: explain the X-factor.

English script:

> This slide is the main X-factor. The first-stage model gives us a base relevance score, but the second stage decides what is worth pushing. The final score combines base relevance, promotion exposure, coupon eligibility, discount-cost proxy, and diversity. Promotion and coupon can lift strategically pushable products. Discount cost is subtracted because not every likely purchase deserves a coupon. Diversity prevents the list from becoming too narrow. We will present this carefully as an observational decision proxy, not causal coupon uplift and not true profit optimization.

中文讲稿：

> 这一页是项目的主要亮点。第一阶段模型给出基础相关性分数，但第二阶段决定哪些商品值得推。最终分数结合基础相关性、促销曝光、优惠券匹配、折扣成本代理变量和多样性。促销和优惠券可以提升更适合营销推送的商品；折扣成本要被扣除，因为不是每个本来就可能购买的商品都值得发券；多样性用来避免推荐列表过于集中在一个类别。我们会严谨表达：这是观察性决策代理，不是 causal coupon uplift，也不是真实利润优化。

One-sentence formula explanation:

> Base score says "likely to buy"; promotion and coupon say "possible to push"; discount cost says "costs money"; diversity says "keep the list healthy."

> base score 表示“可能会买”，promotion 和 coupon 表示“适合推”，discount cost 表示“会消耗折扣成本”，diversity 表示“保持列表健康”。

## Slide 8: Research Questions And Experiments

Time: 1:00
Owner: Presenter 2
Goal: show this is an experiment-driven recommender project.

English script:

> We evaluate PromoMind as a recommender systems study, not only as a demo. The first question is which candidate-generation model works best for grocery basket prediction. The second is whether promotion and coupon signals improve utility without causing too much ranking loss. The third is whether discount-cost-aware reranking reduces inefficient coupon allocation. We will report Recall@10 and Recall@20, NDCG@10 and NDCG@20, plus coverage, diversity, novelty, and Business Utility@K. The ablation study compares base ranking, promotion, coupon, discount-cost penalty, and full reranking with diversity.

中文讲稿：

> 我们会把 PromoMind 当作 recommender systems study 来评估，而不是只做一个 demo。第一个问题是哪种候选生成模型最适合超市购物篮预测。第二个问题是促销和优惠券信号能否提升效用，同时不造成过大的排序损失。第三个问题是加入折扣成本的重排序能否减少低效发券。我们会报告 Recall@10、Recall@20、NDCG@10、NDCG@20，以及 coverage、diversity、novelty 和 Business Utility@K。消融实验会比较 base ranking、promotion、coupon、discount-cost penalty 和带 diversity 的 full reranking。

Important wording:

- Business Utility@K is a revenue-minus-discount proxy.
- It is not true profit because the dataset does not include product cost or margin.

## Slide 9: Research Contribution And Demo

Time: 0:55
Owner: Presenter 2
Goal: explain what the working recommender system will look like.

English script:

> The demo is a Streamlit interface that visualizes the recommendation trade-offs. A user can select a household, inspect household context, view Top-10 recommended products, and see coupon decisions and rule-based explanations. The marketing budget slider changes coupon allocation, so the audience can see how budget constraints affect discount cost and Business Utility. The demo is important because the course asks for a working recommender system, but the research evidence still comes from offline evaluation.

中文讲稿：

> Demo 会用 Streamlit 展示推荐权衡。用户可以选择一个 household，查看这个家庭的上下文信息，看到 Top-10 推荐商品，以及每个商品的 coupon 决策和规则解释。营销预算滑杆会改变优惠券分配，让观众看到预算限制如何影响折扣成本和 Business Utility。Demo 很重要，因为课程要求展示一个可运行的推荐系统，但研究证据仍然来自离线实验评估。

Do not say:

- Do not say the demo proves the model works.
- Say the demo visualizes the system; metrics prove performance.

## Slide 10: Feasibility And Risk Controls

Time: 0:55
Owner: Presenter 2
Goal: close with feasibility, not internal team management.

English script:

> The project has a stable core path and clear extensions. The minimum complete line is chronological split, popularity and ALS, promotion-aware reranking, Business Utility@K, and a Streamlit demo. The experiment package adds Recall and NDCG, coverage, diversity, novelty, reranking ablations, and budget sensitivity. Extensions include ItemKNN, BPR, LightGCN, demographic subgroup views, and future causal uplift framing. The risks are also controlled: coupon redemption is sparse, so it is auxiliary; demographics are partial, so they are optional; margin data is unavailable, so we use a proxy rather than claiming profit.

中文讲稿：

> 这个项目有稳定主线，也有清晰扩展方向。最低完整主线包括时间切分、Popularity 和 ALS、promotion-aware reranking、Business Utility@K，以及 Streamlit demo。实验包会加入 Recall 和 NDCG、coverage、diversity、novelty、重排序消融和预算敏感性分析。扩展方向包括 ItemKNN、BPR、LightGCN、demographic subgroup view，以及未来的 causal uplift framing。风险也有控制：coupon redemption 稀疏，所以只作为辅助信号；demographics 覆盖不完整，所以只作为可选信息；没有 margin data，所以只使用 proxy，不声称真实利润。

Closing sentence:

English:

> In short, PromoMind studies grocery recommendation as a retail decision problem, not only a purchase prediction task.

中文：

> 总结来说，PromoMind 把超市推荐作为零售决策问题来研究，而不只是购买预测任务。

## Q&A Preparation

### Q1. Why choose The Complete Journey?

English:

> Because it combines household transactions with product metadata and marketing context. We can model purchases, categories, promotions, coupons, campaigns, redemptions, and partial demographics in one project.

中文：

> 因为它同时有家庭购物交易、商品信息和营销上下文。我们可以在一个项目里建模购买行为、商品类别、促销、优惠券、营销活动、兑换记录和部分用户画像。

### Q2. Why not use coupon redemption as the main target?

English:

> Coupon redemption is sparse and post-treatment. We keep the main supervised task as product purchase prediction, then use coupon information as an auxiliary signal for reranking and analysis.

中文：

> coupon redemption 很稀疏，而且是 post-treatment 行为。我们把主监督任务保持为商品购买预测，再把优惠券信息作为重排序和分析的辅助信号。

### Q3. Is Business Utility real profit?

English:

> No. The dataset has sales and discount fields, but it does not include product cost or margin. So we call it a revenue-minus-discount proxy, not profit.

中文：

> 不是。数据集有销售额和折扣字段，但没有商品成本和利润率。所以我们称它为 revenue-minus-discount proxy，而不称为真实 profit。

### Q4. Is this causal coupon uplift?

English:

> No. Promotion and coupon exposure are observational in this dataset. We use them for association-based reranking, not for causal treatment-effect estimation.

中文：

> 不是。这个数据集里的促销和优惠券曝光是观察性数据。我们用它们做 association-based reranking，不做因果 treatment effect 估计。

### Q5. What if LightGCN is not completed?

English:

> LightGCN is an extension. The main project remains complete with popularity baselines, ALS or BPR-style implicit-feedback models, promotion-aware reranking, offline metrics, and the Streamlit demo.

中文：

> LightGCN 是扩展项。即使没有完成，项目主线仍然可以由 popularity baseline、ALS 或 BPR 类型隐式反馈模型、promotion-aware reranking、离线指标和 Streamlit demo 构成完整成果。

## Rehearsal Checklist

- Slide 3 must clearly answer: dataset, collection method, expected size, and features.
- Slide 4 must clearly answer: recommendation problem and chronological split.
- Slide 6 must acknowledge external libraries.
- Slide 7 must explain the X-factor without claiming causal uplift.
- Slide 8 must mention experiments and metrics.
- Slide 9 must present the demo as visualization, not evidence replacement.
- Slide 10 must close on feasibility and limitations, not team management.
