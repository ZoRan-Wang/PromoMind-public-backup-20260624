# Speaking Script For Assigned Slides

Source: `RS_Presentation(1).pptx` screenshots  
Scope: System architecture + Phase 1 candidate generation results  
Target time: about 2 minutes  
Style: simple English, presentation-ready

## Slide: System Architecture

This slide shows the full PromoMind pipeline.

We start from raw retail tables. These include transaction records, product information, promotion data, coupon data, and campaign data.

Next, we clean the data and split it by time. This is important because a real recommender system uses past behavior to predict future behavior.

Then we enter Stage 1: candidate generation. The goal of Stage 1 is coverage. It tries to identify products that may be relevant to each household.

After that, we enter Stage 2: marketing prioritization. Here we use XGBoost learning-to-rank. XGBoost means Extreme Gradient Boosting. Learning-to-rank means the model learns the order of coupon products for each household and campaign.

The final step is tail fusion and demo. Tail fusion keeps the strongest ranking results and adds useful category-based products near the end of the list.

The main idea is simple: Stage 1 finds possible products, and Stage 2 decides which coupon products should be shown first.

## Slide: Phase 1 Candidate Generation Results

This slide shows the result of Stage 1.

We tested several candidate generation methods.

Personal Top Frequency uses the household's own purchase history. It works well because grocery shopping has strong repeat behavior.

TIFU-KNN style means Temporal-Item-Frequency-based User K Nearest Neighbors style. In the presentation, I will say it as Tee-foo K nearest neighbors. It looks at similar households and also considers purchase timing.

Strong Hybrid combines several useful signals. Official Cornac TIFUKNN is the library implementation of the time-aware nearest-neighbor model.

The final rank ensemble gives the best result. It has Recall at 10 of 0.1051 and NDCG at 10 of 0.4278. NDCG means normalized discounted cumulative gain. It rewards putting correct products higher in the list.

The conclusion is that grocery recommendation depends strongly on repeat frequency, time decay, and neighbor household preference. These signals are more important than general matrix factorization alone.

This gives us a strong candidate pool for the next stage: coupon-response ranking.

## Pronunciation Notes

- XGBoost: X G Boost, or Extreme Gradient Boosting.
- LTR: learning-to-rank.
- TIFU-KNN style: Tee-foo K nearest neighbors style. Full name: Temporal-Item-Frequency-based User K Nearest Neighbors.
- NDCG: N D C G, normalized discounted cumulative gain.
- Recall at 10: Recall at ten.
