# Two-Presenter Division

This file is the short operational split for the June 3 proposal presentation.

## Split

| Presenter | Slides | Theme | Target time |
| --- | --- | --- | --- |
| Presenter 1 | 1-5 | Dataset, features, recommendation problem, system setup | 3:50 |
| Presenter 2 | 6-10 | Methods, reranking, experiments, demo, risks | 4:55 |

Total scripted time: about 8:50. Rehearsal target: 9:30 or less.

## Presenter 1 Summary

English:

> We are using The Complete Journey dataset. It is a public household-level grocery retail dataset with transactions, product metadata, promotion exposure, coupons, campaigns, redemptions, and partial demographics. Our task is to predict which products a household will buy in the next period using implicit feedback, with a chronological split from past weeks to future weeks.

Chinese:

> 我们选择 The Complete Journey 数据集。它是一个公开的家庭级超市零售数据集，包含交易记录、商品信息、促销曝光、优惠券、营销活动、兑换记录和部分人口统计信息。我们的任务是基于 implicit feedback 预测每个 household 下一阶段会买哪些商品，并使用按时间切分的数据集来模拟真实推荐系统。

## Presenter 2 Summary

English:

> We will compare popularity baselines, ItemKNN, implicit ALS, BPR if time allows, and LightGCN as a bonus. The X-factor is promotion-aware reranking: after generating likely products, we rerank them using promotion, coupon, discount-cost proxy, and diversity signals. We will evaluate Recall@K, NDCG@K, coverage, diversity, novelty, and a revenue-minus-discount Business Utility@K proxy.

Chinese:

> 我们会比较 popularity baseline、ItemKNN、implicit ALS、时间允许的话做 BPR，LightGCN 作为 bonus。项目亮点是 promotion-aware reranking：先生成可能购买的商品，再用促销、优惠券、折扣成本代理变量和多样性信号进行二次排序。评估指标包括 Recall@K、NDCG@K、coverage、diversity、novelty，以及 revenue-minus-discount 的 Business Utility@K 代理指标。

## Transition Line

Presenter 1 to Presenter 2:

English:

> Now that the dataset and task are clear, Presenter 2 will explain the models, reranking design, and evaluation plan.

Chinese:

> 数据集和任务定义讲清楚之后，下面由第二位同学介绍模型、重排序设计和实验评估计划。
