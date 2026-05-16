# Streamlit Demo Specification

## Goal

The PromoMind demo should make the two-stage recommendation system understandable to a non-technical audience: choose a household, inspect its purchase profile, view Top-10 recommendations, see coupon decisions, and adjust a marketing budget slider to show the trade-off between relevance and promotional cost.

## Data Inputs

Expected demo inputs:

| File | Purpose |
| --- | --- |
| `data/processed/household_features.csv` | Household profile cards and selector metadata |
| `data/processed/product_features.csv` | Product names/categories/brands for table display |
| `outputs/reranked_als.csv` or `outputs/demo_recommendations.csv` | Final recommendations, scores, coupon flags, and reasons |
| `outputs/final_results_table.csv` | KPI cards or summary results |

Minimum recommendation columns:

| Column | Meaning |
| --- | --- |
| `household_id` | Selected household |
| `product_id` | Recommended product |
| `final_rank` | Display rank |
| `final_score` | Reranked score |
| `base_score` | Candidate-generation score |
| `promo_score` | Promotion signal |
| `coupon_score` | Coupon signal |
| `discount_cost_proxy` | Estimated promotional cost |
| `recommend_coupon` | Coupon flag shown in the table |
| `reason` or `reason_signal` | Human-readable reason |

## Page Layout

### Header

- App title: `PromoMind`
- Short subtitle: promotion-aware grocery recommendations.
- Optional compact KPI row: selected model, Recall@10, NDCG@10, Business Utility@10.

### Sidebar

Controls:

- Household selector.
- Model variant selector if multiple outputs exist: `ALS`, `ALS + promotion`, `ALS + full reranking`, optional `BPR` or `LightGCN`.
- Recommendation depth selector: Top-10 or Top-20.
- Marketing budget slider.
- Optional toggle: show debug scores.

### Main Area

Recommended order:

1. Household profile summary.
2. Recommendation table.
3. Explanation panel or expandable row details.
4. Budget impact summary.
5. Optional model comparison chart.

## Household Selector

Required behavior:

- Selector lists valid `household_id` values from the recommendation file.
- Default household should have non-empty recommendations and enough profile data for a good demo.
- If a selected household has missing demographics, display `Unknown` or omit demographic fields instead of failing.
- If no recommendations exist for a household, show a clear fallback message and prompt the user to select another household.

Suggested household profile fields:

- Total train baskets.
- Average basket value.
- Top department.
- Top product category.
- Frequently purchased brand, if available.
- Demographics availability flag.

## Recommendation Table

Required columns:

| Display column | Source |
| --- | --- |
| Rank | `final_rank` |
| Product | `product_description` if available, else `product_id` |
| Department | `department` |
| Category | `product_category` |
| Brand | `brand` |
| Coupon? | `recommend_coupon` |
| Reason | `reason` or generated from `reason_signal` |

Optional debug columns behind a toggle:

- `base_score`
- `promo_score`
- `coupon_score`
- `discount_cost_proxy`
- `diversity_score`
- `final_score`

Table behavior:

- Sort by `final_rank`.
- Show Top-10 by default.
- Keep product ids available for traceability.
- Use clear coupon labels such as `Yes`, `No`, or `Budget limited`.

## Coupon Flag Behavior

`recommend_coupon` should be true only when:

- The product has coupon eligibility or campaign exposure for the selected household/week.
- The item survives the budget rule.
- The estimated discount cost does not exceed the active threshold, if threshold mode is used.

If coupon data is incomplete:

- Use `Coupon eligible` for items with coupon signal.
- Use `Recommend coupon` only for items selected under the slider budget.
- Add a small note that coupon assignment is a decision proxy, not observed future redemption.

## Recommendation Reasons

Each displayed recommendation should have at least one reason. Suggested rule order:

1. Product belongs to a household's frequent department/category.
2. Product is similar or co-occurs with prior basket items.
3. Product has current promotion exposure.
4. Product is eligible for a coupon.
5. Product keeps the recommendation list diverse.
6. Product is a high-scoring ALS candidate.

Example reason text:

- `Matches this household's frequent category.`
- `Promotion signal is active for the target week.`
- `Coupon eligible and within the current budget.`
- `Adds category variety to the basket.`
- `High candidate score from ALS.`

Avoid vague reasons such as `The model thinks this is good`.

## Budget Slider

The marketing budget slider should control coupon recommendations, not the base recommendation list itself unless explicitly labeled.

Recommended modes:

### Mode A: Coupon Count Budget

- Slider represents maximum number of coupon recommendations shown for the selected household.
- Example range: `0` to `10`.
- Behavior:
  - Rank coupon-eligible items by `coupon_score - discount_cost_proxy` or by final rank.
  - Set `recommend_coupon = Yes` for only the first `budget` eligible items.
  - Other eligible items display `Budget limited`.

### Mode B: Discount Cost Budget

- Slider represents maximum allowed total `discount_cost_proxy`.
- Example range: `0` to a high percentile of Top-10 cost.
- Behavior:
  - Traverse recommendations by final rank.
  - Recommend coupon only while cumulative estimated discount cost remains under budget.
  - Display cumulative cost and remaining budget.

Minimum implementation can use Mode A. Standard delivery should include Mode A or B and clearly label the slider.

## Budget Impact Summary

Show compact indicators under the table:

- Coupons recommended.
- Estimated discount cost.
- Estimated value among known hits, if demo is using labeled validation/test examples.
- Average promotion/coupon score in Top-K.

Do not imply live revenue or true profit. Use labels such as `estimated`, `proxy`, or `validation example` where appropriate.

## Empty and Error States

Handle:

- Missing recommendation file.
- Household with no recommendations.
- Missing product metadata.
- Missing coupon scores.
- Budget too low for coupon assignment.

Each state should fail gracefully with a short message and keep the page running.

## Demo Acceptance Criteria

Minimum:

- App loads locally with `streamlit run`.
- Household selector works.
- Top-10 table shows products, coupon flag, and reasons.
- Missing metadata does not crash the app.

Standard:

- Budget slider changes coupon flags or budget-limited labels.
- Debug score toggle exists.
- KPI cards or final result summary are visible.
- Demo is backed up by a screenshot or short recording.

Bonus:

- Model variant selector switches between base and reranked outputs.
- Budget impact chart updates with the slider.
- Household profile includes compact history/category visuals.
