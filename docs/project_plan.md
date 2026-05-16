# PromoMind Project Plan

## Scope

PromoMind is a promotion-aware grocery basket recommender for The Complete Journey dataset. The core task is to recommend the Top-K products each household is likely to purchase in the next week, then rerank those candidate products using promotion, coupon, discount-cost, diversity, and business-utility signals.

The project should produce:

- Reproducible data processing with a time-based train/validation/test split.
- Candidate-generation models: popularity, category popularity, ItemKNN, ALS, and optionally BPR or LightGCN.
- Promotion-aware reranking variants and a Business Utility@K proxy.
- Evaluation tables for ranking quality, coverage, diversity, novelty, and business utility.
- A Streamlit demo that shows household-level recommendations, coupon decisions, reasons, and budget behavior.
- Final report and presentation materials with each workstream clearly represented.

Out of scope for the standard delivery:

- Real profit estimation. Discount-adjusted value is a proxy, not true margin.
- Causal coupon uplift modeling. Coupon signals are used for reranking and explanation unless a later bonus effort supports stronger assumptions.
- Production deployment. The demo is a classroom/product-prototype artifact.

## Workstreams

| Workstream | Owner | Responsibility | Required interfaces |
| --- | --- | --- | --- |
| Data | A | Raw data loading, cleaning, schema notes, time split, EDA, processed feature tables | `data/processed/train_interactions.csv`, `valid_interactions.csv`, `test_interactions.csv`, `product_features.csv`, `household_features.csv` |
| Models | B | Popularity, category popularity, ItemKNN, ALS, optional BPR; Top-K candidate output | `outputs/candidates_MODEL.csv` with `household_id`, `product_id`, `base_score`, `model_name`, `base_rank` |
| Reranking | C | Promotion/coupon/discount features, reranking formula, diversity, Business Utility@K, ablations | `outputs/reranked_MODEL.csv` with feature scores, `final_score`, `final_rank`, `recommend_coupon` |
| Demo/Integration | D | Streamlit demo, LightGCN if time permits, final results table, presentation/report integration | `outputs/final_results_table.csv`, demo-ready recommendation table, screenshots |

## Milestones

| Milestone | Target day | Exit criteria |
| --- | --- | --- |
| M0: Project kickoff | Day 0 | Dataset access confirmed, roles assigned, interfaces agreed, issue board created |
| M1: Sample end-to-end path | Day 1 | Small data sample flows through cleaning, popularity candidates, simple reranking, and demo shell |
| M2: Full data and baseline readiness | Day 2 | Full time split exists; product and household features exist; popularity/category baselines run |
| M3: First useful recommendations | Day 3 | ALS candidates generated; first promotion-aware reranking output created; demo can read recommendations |
| M4: Evaluation and business metrics | Day 4 | Recall/NDCG plus Business Utility@K run on validation; reranking knobs are configurable |
| M5: Final experiment tables | Day 5 | Model comparison and reranking ablation tables complete; demo reads final reranked output |
| M6: Report and deck draft | Day 6 | First full report, PPT, and demo backup screenshot/recording are ready |
| M7: Rehearsal and final QA | Day 7 | Presentation rehearsed; leakage, outputs, and demo stability checked |

## Day-by-Day Execution

### Day 0: Kickoff

- A: Confirm The Complete Journey files can be loaded; list available tables and exact column names.
- B: Pick modeling libraries and sparse-matrix format.
- C: Inspect promotion, campaign, coupon, and redemption join paths; draft reranking formula.
- D: Create Streamlit shell and final deck outline.
- All: Confirm file contracts and open GitHub issues for assigned tasks.

### Day 1: Sample Pipeline

- A: Produce sample cleaned transactions and sample train/valid/test split.
- B: Run popularity on the sample and emit `outputs/candidates_popularity_sample.csv`.
- C: Run a simple reranker using placeholder promotion/coupon scores.
- D: Display one household and Top-10 recommendations in Streamlit.
- Acceptance: `data -> model -> rerank -> demo` works on a small sample.

### Day 2: Full Data and Baselines

- A: Finish full cleaned transactions, time split, `product_features.csv`, and `household_features.csv`.
- B: Finish popularity and category popularity baselines.
- C: Produce first promotion and coupon feature tables.
- D: Finish demo layout and system architecture draft.
- Acceptance: full train/valid/test files are usable by B/C/D.

### Day 3: Main Model First Pass

- A: Finish EDA figures and run a leakage audit.
- B: Finish ItemKNN and first ALS Top-100 candidate output.
- C: Rerank ALS candidates with promotion/coupon/discount features.
- D: Connect demo to ALS or reranked ALS output.
- Acceptance: reranking-before/after comparison can be evaluated.

### Day 4: Tuning and Business Metrics

- A: Help validate promotion/coupon joins against the split.
- B: Tune ALS and attempt BPR if time allows.
- C: Implement Business Utility@K, discount-cost proxy, and diversity control.
- D: Add recommendation reasons and KPI cards to the demo.
- Acceptance: main result table has at least one baseline, ALS, and reranked ALS row.

### Day 5: Ablations and Visuals

- A: Finalize data slides/report text.
- B: Finalize candidate-generation model results.
- C: Finalize reranking ablation table and lambda/budget sensitivity output.
- D: Try LightGCN if feasible; integrate final results table; capture demo screenshots.
- Acceptance: every workstream has PPT-ready tables or figures.

### Day 6: Integration Draft

- A/B/C: Submit final report paragraphs and slide content for owned sections.
- D: Assemble the first complete deck and report; verify demo can run locally.
- Acceptance: a complete draft exists even if polish remains.

### Day 7: Rehearsal and Final Fixes

- All: Rehearse, time each section, record issues, and fix only high-impact problems.
- A: Recheck data paths and leakage language.
- B: Recheck model result interpretation.
- C: Recheck business utility wording and proxy limitations.
- D: Recheck demo stability and backup screenshots.
- Acceptance: final submission package is coherent and runnable.

## Delivery Lines

### Minimum Delivery

This line is enough for a complete project:

- Data cleaning, time-based split, `product_features.csv`, and `household_features.csv`.
- Popularity, category popularity, and ALS.
- Promotion-aware reranking with at least promotion/coupon signals.
- Business Utility@K proxy and final result table.
- Simple Streamlit demo with household selector, Top-10 recommendations, coupon flag, and reasons.
- Final report and presentation.

### Standard Delivery

This is the target line:

- Everything in the minimum line.
- ItemKNN and BPR, if runtime allows.
- Complete reranking ablations: base only, promotion, coupon, promotion+coupon, discount-aware, full reranking with diversity.
- Coverage, diversity, novelty, Recall@10/20, NDCG@10/20, and Business Utility@10/20.
- Budget slider that changes coupon assignment or discount-cost threshold.
- Clear demo backup screenshot or short recording.

### Bonus Delivery

Stretch goals after the standard line is stable:

- LightGCN experiment with the same candidate-output interface.
- Lambda sensitivity curve or budget trade-off chart.
- More polished Streamlit UI with richer household profile summaries.
- ER diagram or system architecture figure suitable for the final deck.
- Optional subgroup analysis by household features, with missing demographics handled explicitly.

## Coordination Rules

- Do not change another workstream's output schema without posting the proposed change and getting agreement.
- Every model output must preserve raw `household_id` and `product_id`; internal matrix ids must not leak into shared CSVs.
- Every experiment result must record the data split, model variant, parameters, and run date.
- Valid/test rows must not influence training features except through permitted historical windows.
- When blocked, ship a sample-compatible placeholder that keeps the downstream interface alive.
