# 24-Hour Proposal Execution Plan

Context: the June 3 presentation is a proposal, not a final project defense. Based on the latest team feedback, the presentation should be direct: explain which dataset we choose, what the dataset looks like and what features it has, what recommendation problem we will solve, and what methods/experiments we plan to use.

## PM Decision

Use two presenters for the oral proposal. Keep the slide deck research-oriented, but do not over-explain every future implementation detail.

| Presenter | Slides | Main responsibility | Time target |
| --- | --- | --- | --- |
| Presenter 1 | Slides 1-5 | Project motivation, dataset choice, dataset features, recommendation task, chronological split, system overview | 3:50 |
| Presenter 2 | Slides 6-10 | Algorithms, promotion-aware reranking, research questions, experiments, demo concept, risks and next steps | 4:55 |

Scripted talk time target: about 8:50. Rehearsal target: 9:30 or less, leaving at least 30 seconds before the 10-minute cap.

## What The Teacher Needs To Hear

Minimum proposal content:

1. We choose The Complete Journey dataset.
2. It contains household-level grocery transactions plus product, promotion, coupon, campaign, redemption, and demographic tables.
3. The core features include household id, product id, basket id, week, quantity, sales value, retail/coupon discounts, department, category, brand, promotion exposure, coupon/campaign mapping, and optional demographics.
4. The recommendation problem is next-period household-product Top-K recommendation under implicit feedback.
5. Planned methods are Popularity, Category Popularity, ItemKNN, Implicit ALS, BPR if time allows, and LightGCN as a bonus.
6. Planned experiments use chronological split, Recall@K/NDCG@K, coverage/diversity/novelty, Business Utility@K proxy, and reranking ablations.
7. X-factor is promotion-aware reranking and a budget-aware coupon decision demo.

## Presenter 1: Dataset, Problem, And Setup

### Slides

- Slide 1: Title and one-sentence project idea.
- Slide 2: Research problem and motivation.
- Slide 3: Dataset and collection method.
- Slide 4: Recommendation task and time split.
- Slide 5: System architecture.

### Must Say

- "We use The Complete Journey, a public grocery retail dataset from the `completejourney` project."
- "The raw RDS/RDA artifacts are already in our GitHub repository, and `scripts/download_completejourney.R` documents the extraction path."
- "The dataset has about 2,469 households, 1.47 million transaction rows, 92k products, plus promotion, coupon, campaign, redemption, and demographic tables."
- "The dataset features are useful because we can model purchases, item metadata, discounts, promotion exposure, coupon eligibility, and household profile signals."
- "We use a chronological split because a real recommender predicts future baskets from past behavior."

### Do Not Say

- Do not claim we have final experiment results.
- Do not say coupon redemption is the main label.
- Do not say demographics are available for all households.

## Presenter 2: Methods, Research Design, And Demo

### Slides

- Slide 6: Candidate generation models.
- Slide 7: Promotion-aware reranking.
- Slide 8: Research questions and experiments.
- Slide 9: Research contribution and demo.
- Slide 10: Plan, presentation split, and risks.

### Must Say

- "We start with popularity baselines because grocery recommendation has strong repeat-purchase and mass-popularity effects."
- "ALS/BPR are suitable because the data is implicit feedback, not ratings."
- "The promotion-aware reranker combines base relevance, promotion score, coupon score, discount-cost proxy, and diversity."
- "Business Utility@K is not profit. It is a revenue-minus-discount proxy because cost and margin are unavailable."
- "The demo visualizes trade-offs: a household selector, Top-10 recommendations, coupon flags, reasons, and a marketing budget slider."

### Do Not Say

- Do not claim causal coupon uplift.
- Do not claim LightGCN is required for the project to be complete.
- Do not imply the budget slider is final proof of performance; offline metrics are the evidence.

## Two-Presenter Transition

Transition from Presenter 1 to Presenter 2 after Slide 5:

English:

> "Now that the dataset, prediction task, and pipeline are defined, I will hand over to Presenter 2 to explain the recommendation models, promotion-aware reranking, and evaluation plan."

Chinese:

> "数据集、任务定义和系统流程讲清楚之后，下面交给第二位同学介绍推荐模型、促销感知重排序和实验设计。"

## Q&A Ownership

| Likely question | Main responder |
| --- | --- |
| Why this dataset? | Presenter 1 |
| What features does the dataset have? | Presenter 1 |
| Why chronological split? | Presenter 1 |
| Why ALS/BPR? | Presenter 2 |
| What is promotion-aware reranking? | Presenter 2 |
| Is Business Utility profit? | Presenter 2 |
| Is this causal coupon uplift? | Presenter 2 |
| What if LightGCN is not finished? | Presenter 2 |

## Final 24-Hour Checklist

- Presenter 1 reads `presenter_1_background_zh_en.md`.
- Presenter 2 reads `presenter_2_background_zh_en.md`.
- Both presenters read `proposal_deck_script.md`.
- One person uploads `PromoMind_project2_proposal_slides.pptx` to eLearn before class.
- The uploader posts the upload confirmation in the team chat.
- Rehearse once with a timer and cut extra details if the talk exceeds 9:30.
