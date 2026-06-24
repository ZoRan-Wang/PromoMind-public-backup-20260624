# PromoMind 总 Presentation 演讲稿（4 Part 分工版）

## 使用说明

- 适配 PPT：`deliverables/final_solution_2026_06_24/PromoMind_final_presentation_zh.pptx`
- 建议总时长：10-12 分钟
- 分工方式：4 个 Part，每人约 2.5-3 分钟
- 汇报要求覆盖：finalized dataset、problem、algorithms、experimental results、applicability、significance、extensions with evidence、working recommender demo、future work

## Part 1：Dataset、Problem Definition、Course Requirement Coverage

对应 Slides 1-4。

### Slide 1：Title

大家好，我们小组的项目是 PromoMind，一个面向零售营销优化的促销感知购物篮与优惠券推荐系统。

这个项目的核心问题是：在营销预算有限的情况下，对于某一个家庭，哪些 campaign coupon products 更适合被推荐。项目目标从下一次购买预测推进到具体营销活动中的 coupon product response priority 排序。

最终模型在 held-out campaign test split 上取得了 Recall@10 0.4138、NDCG@10 0.3225，以及 Positive Event Hit@10 53.21%。这说明系统能够把更多真实正响应商品排进 Top-10，同时也能提升正响应事件被命中的概率。

### Slide 2：Course Requirements Coverage

这一页说明我们的最终交付如何覆盖课程要求。

第一，finalized dataset 和 problem 已经明确：我们使用 The Complete Journey 数据集，将任务定义为 household-campaign coupon-product ranking。

第二，algorithms 部分覆盖了多个层次，包括 Popularity、ItemKNN、ALS、BPR、TIFU-KNN、Cornac TIFUKNN，以及最终的 XGBoost Learning-to-Rank。

第三，experimental results 使用 held-out campaign test split，并报告 Recall、NDCG 和 Positive Event Hit。

第四，extension 部分包含 neural ranker、text features、category embedding 和 tail fusion，并且每个方向都有实验结果作为依据。

第五，我们完成了 working recommender demo，使用 Streamlit 展示 household、campaign、Top-K 推荐、coupon flags 和备用样本。

### Slide 3：Dataset

我们的数据集是 The Complete Journey，这是一个真实 grocery retail 数据集。它覆盖 2,469 个 households，包含约 147 万条商品级 transaction records。

数据表包括 transactions、products、promotions、campaigns、coupons、coupon redemptions 和 demographics。Transactions 提供购买历史与未来 response label，products 提供 department、category、brand 等商品信息，campaigns 和 coupons 定义曝光窗口与 coupon product pool，promotions 提供 display 和 mailer 等上下文。

这里有一个重要设计：监督目标采用 coupon-product purchase response。Redemption 在数据里非常稀疏，作为主标签会让学习信号不稳定。我们使用 campaign start 后 5 天内是否购买对应 coupon product 来定义 positive response，这更适合做 coupon-product response ranking。

### Slide 4：Problem Definition

最终推荐问题定义为：对于每一个 household-campaign exposure，对该 campaign 内的 coupon products 进行排序。

如果一个家庭在 campaign start 后 5 天内购买了某个 coupon product，这个商品就被记为 positive response。

这个定义比普通 next-basket prediction 更贴近 coupon targeting，因为排序对象限定在 campaign 已经提供的 coupon product pool。模型的目标是把更可能响应的 coupon products 排在前面。

在实验协议上，我们使用 715 个 test events，其中 109 个是 positive events。所有特征只使用 campaign start 之前的信息，避免未来信息泄漏。

Part 1 到这里结束。接下来 Part 2 会介绍系统架构、候选生成和第一阶段模型结果。

## Part 2：System Architecture、Candidate Generation、First-stage Results

对应 Slides 5-6。

### Slide 5：System Architecture

PromoMind 采用两阶段推荐架构。

第一步是 raw retail tables，包括交易、商品、促销、campaign、coupon 等原始表。

第二步是 cleaning 和 chronological time split。我们按照时间顺序划分训练、验证和测试，使实验协议更符合真实推荐系统上线后的预测场景。

第三步是 candidate generation。这个阶段的目标是解决候选覆盖问题，也就是先找出家庭可能相关的商品。

第四步是 XGBoost Learning-to-Rank。这个阶段把候选空间聚焦到 campaign coupon products，并学习 household-campaign-product 层面的 response priority。

第五步是 tail fusion 和 demo。最终产物是 `outputs/reranked_recommendations.csv`，Demo 可以读取这个文件并展示 Top-K coupon product recommendations。

整个架构的核心思想是：Stage 1 负责覆盖，Stage 2 负责营销排序。

### Slide 6：Candidate Generation Results

第一阶段我们实现了多种 candidate generation 方法，包括 Personal Top Frequency、TIFU-KNN style、Strong Hybrid、Official Cornac TIFUKNN，以及 Final rank ensemble。

从结果看，Personal Top Frequency 的 NDCG@10 是 0.3790，说明 grocery 场景里家庭自己的复购习惯非常强。TIFU-KNN style 的 NDCG@10 达到 0.3851，说明时间感知的用户偏好建模有效。Strong Hybrid 提升到 0.3935。Official Cornac TIFUKNN 达到 0.4210。最终 rank ensemble 达到最高的 NDCG@10 0.4278。

这个阶段的结论是：在 grocery 场景中，repeat frequency、temporal decay 和近邻家庭偏好比通用矩阵分解更关键。换句话说，用户买菜具有强烈的周期性和复购行为，模型需要捕捉这种稳定节奏。

第一阶段的 candidate pool 在后续分析中已经覆盖 99.56% 的测试 truth items，因此项目后半部分的主要瓶颈从“找不到商品”变成了“如何把正确商品排到前面”。

Part 2 到这里结束。接下来 Part 3 会介绍第二阶段的 coupon-response ranking、最终测试结果和扩展实验。

## Part 3：Coupon-response Ranking、Experimental Results、Extensions with Evidence

对应 Slides 7-9。

### Slide 7：Coupon-response Learning-to-Rank

第二阶段使用 XGBoost `rank:ndcg` 做 Learning-to-Rank。

训练粒度是 household-campaign-product，也就是每一个家庭、每一次 campaign、每一个 coupon product 构成一个排序样本。

标签使用 pull-forward interval relevance labels。这个标签同时考虑是否购买、购买时间和理想提前发券窗口之间的关系。这样模型学到的是 response priority。

主要特征包括 repeat signal、cadence signal、global signal 和 category embedding。Repeat signal 表示家庭过去是否买过这个商品。Cadence signal 表示这个商品是否符合家庭历史复购节奏。Global signal 表示 campaign product 是否有较广泛的响应证据。Category embedding 用于补充非精确复购的尾部候选。

这一阶段的业务解释很重要：监督式 LTR 学的是 response priority，不声称优惠券造成购买。

### Slide 8：Final Held-out Test Results

最终 held-out test 结果显示，Final tail fusion 达到 Recall@10 0.4138、NDCG@10 0.3225、Positive Event Hit@10 0.5321、Recall@20 0.5184 和 NDCG@20 0.3520。

相较 candidate-only coupon baseline，Recall@10 从 0.1479 提升到 0.4138，NDCG@10 从 0.1399 提升到 0.3225，Positive Event Hit@10 从 18.35% 提升到 53.21%。

这些结果说明，单纯使用候选模型做 coupon baseline 还不够。加入时间感知、复购节奏和 campaign-level response features 后，模型能够更准确地把正响应 coupon products 排进 Top-10。

Final tail fusion 的策略是保留主 XGBoost 的前 7 名，再用 category co-occurrence embedding 补充后续位置。这个设计保留了 XGBoost 在 Top-10 精度上的优势，也利用了 category embedding 对 Top-20 recall 的帮助。

### Slide 9：Extensions with Evidence

项目做了多种扩展实验。

PyTorch pairwise neural ranker 的结果具有竞争力，held-out NDCG@10 低于 XGBoost。Expected-lead labels 和 pull-forward labels 进行比较后，最终选择 pull-forward interval labels。TF-IDF/SVD product text 对部分 Recall@20 有帮助，Top-10 排序效果弱于最终模型。Category co-occurrence embedding 作为 tail source 有价值。Rank fusion 和 score fusion 在 validation 上有收益，迁移到 held-out test 时不稳定。

这些实验给出的研究结论是：The Complete Journey 的商品文本较结构化，文本信息能补充一部分 coverage，timing 和 repeat behavior 对 Top-10 排序更稳定。更复杂的模型不一定自动带来更好的泛化效果。

Part 3 到这里结束。接下来 Part 4 会介绍商业解释、Demo、适用性、局限性和未来工作。

## Part 4：Applicability、Significance、Demo、Limitations、Future Work

对应 Slides 10-12。

### Slide 10：Promotion-aware Business Interpretation

PromoMind 的输出应解释为：对已暴露家庭的 coupon product response priority。

在业务使用中，它可以帮助营销团队在 campaign 商品池内决定展示顺序和 coupon flags。Business Utility proxy 可以用来解释营销约束，例如 coupon slot 或 discount-cost sensitivity。

项目不声称 causal coupon uplift，不声称真实利润最大化，也不声称适用于所有 grocery recommendation 场景。原因是数据没有 randomized treatment-control，也没有真实商品 margin。

这个边界让项目结论更加可靠：我们能证明模型在当前 observation-based response ranking protocol 下有效，后续如果要做因果 uplift，需要新的实验设计或 propensity correction。

### Slide 11：Working Recommender Demo

Demo 使用 Streamlit 启动，命令是：

```powershell
streamlit run app/streamlit_app.py
```

Demo 支持选择 household 和 campaign，展示历史购买、coupon start、预测时间，并输出 Top-K coupon product recommendations。

页面还会显示 coupon flags、metadata 和关键排序字段。正式输出文件是 `outputs/reranked_recommendations.csv`。如果现场环境无法加载完整结果，可以使用 `top10_recommendation_sample.csv` 作为 fallback sample。

这个 Demo 的意义是把离线实验结果转成可交互的推荐系统原型。用户可以看到某个家庭在某次 campaign 下被推荐哪些 coupon products，也可以通过 coupon slots 或 marketing budget 控制展示数量。

### Slide 12：Applicability、Limitations and Future Work

PromoMind 的适用场景包括连锁超市会员优惠券推荐、电商 campaign product prioritization、已确定活动商品池内的家庭个性化排序，以及营销预算约束下的 coupon slot 分配原型。

它的项目意义在于把“用户可能购买什么”和“哪些商品值得营销支持”区分开。普通推荐系统只关注相关性，PromoMind 进一步把 campaign、coupon、response timing 和业务约束纳入排序过程。

局限性包括：没有 treatment-control，无法估计 causal uplift；没有真实 margin，无法做真实利润最大化；coupon redemption 和 demographics 较稀疏；Demo 仍是离线原型。

未来工作包括四个方向。第一，引入 randomized treatment-control 或 propensity correction，构建 causal uplift model。第二，获得真实 margin 后进行 cost-aware optimization。第三，加入 richer text 和 image，探索 multimodal retrieval。第四，对 campaign drift 做 time-aware calibration，并把离线 CSV pipeline 改造成可监控、可重训的生产服务。

最后总结一下：PromoMind 完成了从真实零售数据、时间安全的数据处理、多模型候选生成，到 campaign-aware coupon-response ranking、tail fusion、商业解释和 Streamlit demo 的完整推荐系统流程。最终结果表明，稳定的复购行为和时间节奏信号，是本 grocery coupon-response 任务的核心。

## 4 人分工速览

| Part | Speaker | Slides | 核心内容 |
| --- | --- | --- | --- |
| Part 1 | Speaker A | 1-4 | 题目、课程要求覆盖、数据集、最终问题定义 |
| Part 2 | Speaker B | 5-6 | 系统架构、候选生成、第一阶段模型结果 |
| Part 3 | Speaker C | 7-9 | XGBoost LTR、最终实验结果、扩展实验 |
| Part 4 | Speaker D | 10-12 | 商业解释、Demo、适用性、局限性、未来工作 |

## 一分钟压缩版结尾

PromoMind 的最终任务是对每个 household-campaign exposure 中的 coupon products 进行 response priority ranking。项目使用 The Complete Journey 真实零售数据，采用两阶段架构：第一阶段用多种 next-basket candidate models 保证覆盖，第二阶段用时间感知 XGBoost LTR 和 tail fusion 完成 coupon-response 排序。最终模型在 held-out test 上达到 Recall@10 0.4138、NDCG@10 0.3225、Positive Event Hit@10 53.21%，明显超过 candidate-only coupon baseline。项目还提供 Streamlit working demo、最终结果表和完整报告。它的主要价值是帮助营销团队在有限预算下更合理地选择和排序 coupon products。未来可以继续扩展到 causal uplift、真实 margin 优化、多模态商品理解和生产级推荐服务。
