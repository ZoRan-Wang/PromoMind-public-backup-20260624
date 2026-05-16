# PromoMind

PromoMind is a proposal and project-management package for a promotion-aware grocery basket and coupon recommender system.

The project idea is to predict the products each household is likely to buy next and then re-rank recommendations using promotion, coupon, discount-cost, diversity, and business-utility signals.

## Files

| File | Purpose |
| --- | --- |
| `PromoMind_bilingual_proposal.docx` | Full bilingual proposal in Word format |
| `PromoMind_bilingual_proposal.md` | Full bilingual proposal in Markdown |
| `PromoMind_5_second_meeting_message.md` | Short meeting pitch |
| `PromoMind_four_person_work_split.md` | Four-person work allocation |
| `PromoMind_PM_team_execution_pack.md` | Detailed PM execution plan |
| `PromoMind_task_board.csv` | Task board for tracking project execution |
| `PromoMind_RACI_matrix.csv` | RACI ownership matrix |
| `build_promomind_docs.py` | Script used to generate proposal artifacts |
| `rendered/` | Rendered DOCX page images used for layout QA |

## Proposed Project

**Title:** PromoMind: A Promotion-aware Grocery Basket Recommender for Retail Marketing Optimization

**Dataset:** The Complete Journey

**Core task:** Given a household's historical grocery transactions, recommend the Top-K products it is most likely to purchase in the next week.

**X-factor:** Promotion-aware re-ranking that combines:

- base recommendation score
- promotion exposure
- coupon availability
- estimated discount cost
- recommendation-list diversity
- business utility proxy

## Team Workstreams

- **A: Data Lead** — data cleaning, time split, EDA, feature tables.
- **B: Model Lead** — popularity baselines, ItemKNN, ALS, BPR.
- **C: Business/Reranking Lead** — promotion/coupon features, re-ranking, Business Utility@K.
- **D: Integration/Demo Lead** — LightGCN attempt, Streamlit demo, final integration, PPT/report.

## Minimum Delivery Line

The minimum complete project is:

1. Data cleaning and time-based split.
2. Popularity baseline and ALS recommendation model.
3. Promotion-aware re-ranking.
4. Business Utility@K evaluation.
5. Streamlit demo and final presentation materials.

