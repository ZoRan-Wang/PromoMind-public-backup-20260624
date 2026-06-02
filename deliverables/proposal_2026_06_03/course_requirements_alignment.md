# Course Requirements Alignment

Source: Project 2 announcement posted by Dr Hady W. Lauw on May 11, 2026 at 8:19 PM, pasted by the team on June 2, 2026.

## Milestone Interpreted For This Submission

| Date | Required action | PromoMind output |
| --- | --- | --- |
| June 3, 2026 | Present project proposal, max 10 minutes, upload slides to eLearn before class | `PromoMind_project2_proposal_slides.pptx` plus this proposal pack |
| June 24, 2026 | Present project solution, including finalized dataset, problem, algorithms, experimental results, applicability, extensions, working demo, future work | Later final project package; not claimed as complete in this proposal |

## Required June 3 Content Mapping

| Announcement requirement | Where covered in deck | Owner | Evidence or wording discipline |
| --- | --- | --- | --- |
| Dataset to collect | Slide 3: Complete Journey dataset | A | States tables, source package, household-level grocery data |
| Methodology of collection | Slide 3 | A | Uses public `completejourney` package/source, raw RDS/RDA files already in repo, and `scripts/download_completejourney.R` as the extraction reference; no scraping |
| Expected data size | Slide 3 | A | 2,469 households, about 1.47M transaction rows, 92k products, promotion/coupon/campaign/demographic tables |
| Recommendation problem | Slide 4 | A/B | Next-week household-product Top-K recommendation under implicit feedback |
| Why important/significant | Slide 2 | A/C | Retail coupon waste, basket prediction, higher marketing relevance |
| Recommender algorithms | Slide 6 | B | Popularity, Category Popularity, ItemKNN, Implicit ALS, BPR, optional LightGCN |
| Acknowledge libraries | Slide 6 and proposal text | B/D | `implicit`, RecBole, optional Cornac are named as external libraries |
| Experiments to run | Slides 4 and 8 | B/C | Chronological split, Recall@K, NDCG@K, coverage, diversity, novelty, Business Utility@K, ablations |
| X-factor | Slides 7 and 9 | C/D | Promotion-aware reranking and marketing budget/coupon decision demo |
| Max 10 minutes | `proposal_deck_script.md` | D | 10 slides with a rehearsal target of 9:30 |

Operational note: this repository cannot prove the eLearn upload. After the real upload, the uploader should mark `submission_checklist.md` manually and post upload confirmation in the team chat.

## Research-Nature Alignment

| Research expectation | PromoMind response |
| --- | --- |
| Clear research questions | `research_design_and_evaluation.md` defines RQ1-RQ5; Slide 8 summarizes the core questions |
| Baselines before complex models | Popularity and Category Popularity are required before ALS/BPR/LightGCN |
| Controlled experiments | Chronological train/validation/test split and reranking ablations |
| Standard recommender metrics | Recall@K, NDCG@K, coverage, diversity, novelty |
| Honest limitations | Sparse coupon redemption, partial demographics, no true margin/profit, no causal uplift claim |
| Reproducible artifact | Raw Complete Journey files, scripts, docs, and proposal pack are in GitHub |

## What We Should Not Say On June 3

- Do not say we optimized true profit. We only plan a revenue-minus-discount proxy because item cost/margin is unavailable.
- Do not say coupon redemption will be the main supervised label. Redemption events are sparse, so purchase prediction remains the main task.
- Do not say demographics are required for the model. Demographics cover only a subset of households and are optional for analysis/demo.
- Do not show final results unless produced by a real run and clearly labeled as preliminary.

## Five-Second Group Message

We pushed the Project 2 proposal pack to GitHub. Please open `deliverables/proposal_2026_06_03`, read the deck script, pick your assigned section, and only ask questions in the meeting instead of re-summarizing the idea.
