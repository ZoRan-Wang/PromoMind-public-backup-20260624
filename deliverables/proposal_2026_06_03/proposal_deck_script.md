# PromoMind Proposal Deck Script

Target: max 10 minutes. Recommended rehearsal target: 9:30.

## Slide 1: Title

Owner: D  
Time: 0:25

Claim: PromoMind is a promotion-aware grocery recommender, not just a product recommender.

Talk track:

- "Our project is PromoMind, a promotion-aware grocery basket and coupon recommender."
- "The system predicts what a household is likely to buy next, then decides which recommendations are worth promotion or coupon support."

## Slide 2: Research Problem

Owner: A/C  
Time: 0:50

Claim: The research question is whether promotion/coupon context can improve the trade-off among relevance, utility proxy, and diversity.

Talk track:

- "In grocery retail, purchase history is rich, but coupon delivery is often broad."
- "A normal recommender answers what the household may buy. Our proposal also asks what should receive a coupon or promotion when marketing budget is limited."
- "Research-wise, we are testing whether that extra marketing context improves multi-objective recommendation under sparse observational data."

## Slide 3: Dataset And Collection

Owner: A  
Time: 1:05

Claim: The Complete Journey gives us the right combination of purchases, products, promotions, coupons, campaigns, and household profiles.

Talk track:

- "We will use The Complete Journey through the public completejourney project."
- "The repository already contains the original raw RDS/RDA artifacts, and `scripts/download_completejourney.R` documents the reproducible extraction path."
- "Expected size is about 2,469 households, 1.47 million transaction rows, 92k products, and tables for promotions, coupons, campaigns, redemptions, and demographics."
- "We are not collecting private data or scraping; this is a public dataset workflow."

## Slide 4: Recommendation Task

Owner: A  
Time: 0:50

Claim: We use historical household transactions to predict future products with a time-based split.

Talk track:

- "The core label is implicit feedback: a product is positive if a household buys it in the target period."
- "The standard split is weeks 1-40 for training, 41-46 for validation, and 47-53 for test."
- "This is more realistic than random splitting because the system must use the past to predict future baskets."

## Slide 5: System Architecture

Owner: D  
Time: 0:45

Claim: The system has two stages: candidate generation and promotion-aware reranking.

Talk track:

- "Stage one produces likely products using collaborative filtering and baselines."
- "Stage two reranks those candidates with promotion, coupon, discount-cost, and diversity signals."
- "The same pipeline then feeds evaluation and the Streamlit demo."

## Slide 6: Candidate Generation Models

Owner: B  
Time: 1:00

Claim: We will compare simple baselines with stronger implicit-feedback recommenders.

Talk track:

- "We start with Popularity and Category Popularity because grocery has strong repeat and mass-market behavior."
- "Then we add ItemKNN and Implicit ALS."
- "BPR is standard if time allows, and LightGCN is a bonus extension."
- "We will acknowledge external libraries clearly, especially `implicit` for ALS/BPR-style models and RecBole if LightGCN is used."

## Slide 7: Promotion-aware Reranking

Owner: C  
Time: 1:05

Claim: The X-factor is a second-stage ranking formula that balances relevance with marketing signals and discount cost.

Talk track:

- "The base model score is still the anchor."
- "Promotion and coupon signals can lift items that are strategically pushable."
- "Discount cost is subtracted because not every likely purchase deserves a coupon."
- "Diversity prevents the list from becoming one narrow category."
- "We will describe this as an observational decision proxy, not causal coupon uplift."

## Slide 8: Research Questions And Experiments

Owner: C/B  
Time: 1:00

Claim: We evaluate the project as a recommender systems research question, not only as a demo.

Talk track:

- "RQ1 asks which candidate-generation model works best for grocery baskets."
- "RQ2 asks whether promotion and coupon signals change utility or accuracy."
- "RQ3 asks whether discount-cost-aware reranking controls coupon waste while keeping Recall and NDCG acceptable."
- "Ranking quality will use Recall@10/20 and NDCG@10/20. List health will use coverage, diversity, and novelty."
- "Business Utility@K is estimated sales value for hit products minus an estimated discount-cost penalty."

## Slide 9: Research Contribution And Demo

Owner: D  
Time: 0:55

Claim: The demo visualizes the research trade-offs rather than replacing offline evaluation.

Talk track:

- "The Streamlit demo will select a household, show household profile signals, and display Top-10 recommendations."
- "Each row will show category, brand, predicted relevance, coupon decision, and a rule-based explanation."
- "The marketing budget slider is a core demo view: it changes how many coupon-eligible recommendations receive coupon support."
- "This gives a working recommender-system demonstration, while the research evidence still comes from the offline experiment tables."

## Slide 10: Plan, Ownership, And Risks

Owner: D plus all  
Time: 0:55

Claim: The project is feasible because the main line is stable even if bonus models slip.

Talk track:

- "Minimum complete line: data split, Popularity/ALS, promotion-aware reranking, Business Utility@K, and demo."
- "Standard line adds ItemKNN, BPR, full ablations, coverage/diversity/novelty, and sharper budget sensitivity analysis."
- "Bonus line adds LightGCN."
- "Main risks are sparse coupon redemption, partial demographics, no true margin data, and model runtime. We have mitigation for each."

## Q&A Prep

Question: Why not use coupon redemption as the target?

Answer: Redemption is sparse and post-treatment. We keep the main task as purchase prediction and use coupon data as auxiliary reranking and analysis signals.

Question: Is Business Utility the same as profit?

Answer: No. The dataset has sales value and discount fields but not product cost or margin. We report a revenue-minus-discount proxy.

Question: Why not random split?

Answer: A real recommender predicts future baskets from past behavior, so chronological split better matches deployment.

Question: What if LightGCN does not work?

Answer: LightGCN is a bonus. The required project remains complete with Popularity, Category Popularity, ALS/BPR if possible, promotion-aware reranking, metrics, and demo.

Question: What is the real X-factor?

Answer: The second-stage decision layer. It turns recommendations into promotion/coupon decisions under budget and cost constraints.

Question: Is this a causal promotion or coupon uplift model?

Answer: No. Promotion and coupon exposure are observational in this dataset. We use them for association-based reranking and clearly report Business Utility as a proxy, not causal profit or uplift.

Question: How do you avoid leakage from promotions and coupons?

Answer: We use chronological split, compute historical features from train windows, and do not use target-period redemption or purchases as ranking features. Target-period sales value is only used after ranking to score hit products.
