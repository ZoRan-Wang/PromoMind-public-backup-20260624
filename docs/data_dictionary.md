# PromoMind Data Dictionary

This dictionary documents the expected tables and shared processed files for PromoMind. The exact column names must be confirmed against the local copy of The Complete Journey before implementation, because public/package versions may use slightly different names.

## Core Entities

| Entity | Meaning | Common key |
| --- | --- | --- |
| Household | Shopper household tracked across baskets | `household_id` |
| Basket | One shopping trip or transaction group | `basket_id` |
| Product | UPC/product item available for purchase | `product_id` |
| Week | Relative week in the dataset timeline | `week` |
| Campaign | Marketing campaign or coupon distribution event | `campaign_id` |
| Coupon | Coupon offer, often mapped to products or campaigns | `coupon_id` or coupon UPC field |
| Store | Retail store where a basket/promotion occurs | `store_id` |

## Expected Raw Tables

### `transactions`

Main implicit-feedback table. This is the source for purchase history, target labels, spend, and discount proxies.

| Column | Meaning | Use |
| --- | --- | --- |
| `household_id` | Household identifier | Required for all modeling and evaluation |
| `basket_id` | Shopping basket identifier | Basket aggregation and EDA |
| `product_id` | Purchased product identifier | Required item key |
| `week` | Dataset week of purchase | Time split and historical feature windows |
| `quantity` or `product_quantity` | Units purchased | Interaction strength and cleaning checks |
| `sales_value` | Net or recorded sales amount | Business Utility@K proxy for hits |
| `retail_disc` | Retail discount amount | Discount-cost proxy |
| `coupon_disc` | Coupon discount amount | Discount-cost proxy and coupon sensitivity |
| `coupon_match_disc` | Manufacturer or matched coupon discount | Discount-cost proxy |
| `store_id` | Store identifier | Optional joins to store-level promotion exposure |
| `purchase_time` or timestamp field | Time within basket/day if available | EDA only unless needed for basket ordering |

Cleaning expectations:

- Keep only valid household, product, basket, and week keys.
- Investigate `quantity <= 0` and `sales_value < 0`; either remove, correct, or document.
- Decide whether repeated product rows in a basket should be summed.
- Product filtering should be documented, for example Top 10,000 products or products with at least 20 purchases.

Leakage notes:

- Valid/test purchases are labels, not training interactions.
- Do not compute product popularity, household preference, discount averages, or category preference using future weeks.
- `sales_value` from the target period may be used only to score hit products during evaluation, not to rank candidates.

### `products`

Product metadata used for category baselines, diversity, demo display, and explanations.

| Column | Meaning | Use |
| --- | --- | --- |
| `product_id` | Product identifier | Join key |
| `department` | Broad merchandise department | Category baseline, diversity, demo |
| `product_category` | Product category | Category baseline, diversity, demo |
| `product_type` | More detailed type if available | Optional explanation/display |
| `brand` | Brand or private-label indicator | Demo profile and explanation |
| `manufacturer_id` | Manufacturer identifier if available | Optional analysis |
| `package_size` | Package text/size if available | Demo display only |
| `product_description` | Human-readable name if available | Demo table and report |

Leakage notes:

- Static product attributes are safe across splits.
- Product attributes should not be inferred from future sales behavior unless the feature name includes the historical window.

### `promotions`

Product-week or product-store-week promotion exposure table.

| Column | Meaning | Use |
| --- | --- | --- |
| `product_id` | Promoted product | Join to candidates |
| `week` | Promotion week | Must align with recommendation target week |
| `store_id` | Store identifier if promotion is store-specific | Optional exposure refinement |
| `display_location` | Display placement signal if available | Promotion score |
| `mailer_location` | Mailer/ad signal if available | Promotion score |
| promotion flags or media fields | Additional exposure indicators | Promotion score |

Leakage notes:

- For a recommendation targeting week `t`, promotion signals should represent information assumed available before or during campaign planning for week `t`.
- Do not use promotion outcomes from the target week other than known planned exposure fields.

### `campaigns` and `campaign_descriptions`

Campaign metadata that helps identify who may have received offers and when campaigns ran.

| Column | Meaning | Use |
| --- | --- | --- |
| `campaign_id` | Campaign identifier | Join key |
| `household_id` | Household reached by a campaign, if present | Coupon exposure feature |
| `campaign_type` | Campaign category/type | Optional feature or EDA |
| `start_week` / `end_week` | Campaign active window if available | Time-aware coupon exposure |
| description fields | Human-readable campaign information | Documentation/demo only |

Leakage notes:

- Campaign exposure can be a reranking signal only if it is known before recommendation time.
- Campaign membership should not be derived from redemption events alone.

### `coupons`

Coupon offer mapping. Depending on data version, it may map coupons to campaigns, products, or product groups.

| Column | Meaning | Use |
| --- | --- | --- |
| `coupon_id` or coupon UPC | Coupon identifier | Join key |
| `campaign_id` | Campaign containing coupon | Join to campaign exposure |
| `product_id` or product group id | Product/product group covered by coupon | Candidate coupon eligibility |
| face value or discount field | Coupon value if available | Coupon score and cost proxy |
| offer metadata | Coupon type/description | Demo display only |

Leakage notes:

- Coupon availability is a reranking signal, not the target label.
- Avoid learning "recommended coupon" from future redemption outcomes.

### `coupon_redemptions`

Observed redemption events. Useful for sparsity analysis and cautious feature engineering.

| Column | Meaning | Use |
| --- | --- | --- |
| `household_id` | Redeeming household | EDA and historical coupon sensitivity |
| `coupon_id` or coupon UPC | Redeemed coupon | Join to coupons |
| `campaign_id` | Campaign if available | Join to campaign |
| `week` | Redemption week | Historical-only feature windows |
| `product_id` | Redeemed product if available | Optional validation of coupon mapping |

Leakage notes:

- Redemption is post-treatment behavior and is usually sparse.
- Redemption in valid/test weeks must not be used to rank valid/test recommendations.
- In the report, describe redemption as an auxiliary signal or analysis table, not the main supervised target.

### `demographics`

Optional household metadata.

| Column | Meaning | Use |
| --- | --- | --- |
| `household_id` | Household identifier | Join key |
| income/age/family fields | Household profile | Optional EDA/demo segmentation |
| household size fields | Household composition | Optional explanation |
| missing/unknown categories | Unobserved demographics | Must be handled explicitly |

Leakage notes:

- Demographics are optional. Missingness should not drop households from the main recommendation task.
- Use demographics for display or subgroup analysis unless the team has time to validate model impact.

## Processed Datasets

### `data/processed/train_interactions.csv`

Historical training interactions. Expected columns:

| Column | Meaning |
| --- | --- |
| `household_id` | Raw household id |
| `product_id` | Raw product id |
| `basket_id` | Raw basket id, if retained |
| `week` | Purchase week; expected range Week 1-40 for the standard split |
| `quantity` | Cleaned quantity |
| `sales_value` | Cleaned sales value |
| `retail_disc` | Retail discount |
| `coupon_disc` | Coupon discount |
| `coupon_match_disc` | Coupon match discount |

### `data/processed/valid_interactions.csv`

Validation labels and evaluation metadata. Expected range: Week 41-46. Same core columns as train.

### `data/processed/test_interactions.csv`

Held-out test labels and evaluation metadata. Expected range: Week 47-53. Same core columns as train.

### `data/processed/product_features.csv`

Product metadata for modeling, reranking, diversity, and demo.

| Column | Required | Meaning |
| --- | --- | --- |
| `product_id` | Yes | Raw product id |
| `department` | Yes | Broad department |
| `product_category` | Yes | Category |
| `brand` | Yes | Brand/private-label field |
| `product_description` | Preferred | Demo display name |
| `product_type` | Optional | More detailed type |

### `data/processed/household_features.csv`

Household profile derived from train weeks only.

| Column | Required | Meaning |
| --- | --- | --- |
| `household_id` | Yes | Raw household id |
| `total_baskets_train` | Yes | Number of train baskets |
| `total_spend_train` | Yes | Train-period spend |
| `avg_basket_value_train` | Yes | Train-period average basket value |
| `top_department_train` | Yes | Most frequent train department |
| `top_category_train` | Yes | Most frequent train category |
| `demographics_available` | Yes | Boolean/flag for demographic row |

### `data/processed/product_week_promotion_features.csv`

Promotion features aligned to product and target week.

| Column | Meaning |
| --- | --- |
| `product_id` | Candidate product |
| `week` | Target week |
| `promotion_score` | Combined promotion exposure score |
| `has_display` | Display flag, if available |
| `has_mailer` | Mailer flag, if available |
| `promotion_source_count` | Number of active promotion signals |

### `data/processed/household_product_coupon_features.csv`

Coupon eligibility or exposure features.

| Column | Meaning |
| --- | --- |
| `household_id` | Household receiving or eligible for coupon signal |
| `product_id` | Candidate product |
| `week` | Target week |
| `coupon_score` | Combined coupon availability/exposure score |
| `campaign_id` | Campaign source if one dominant campaign applies |
| `historical_coupon_redemption_rate` | Train-only household coupon sensitivity, if available |

### `data/processed/discount_cost_features.csv`

Historical discount-cost proxy features computed from train-only windows.

| Column | Meaning |
| --- | --- |
| `product_id` | Product id |
| `avg_product_discount_train` | Average discount for product in train |
| `avg_category_discount_train` | Average category discount in train |
| `estimated_discount_cost` | Proxy cost used by reranker |
| `household_discount_ratio_train` | Optional household-level train discount sensitivity |

## Shared Model Outputs

### `outputs/candidates_MODEL.csv`

| Column | Meaning |
| --- | --- |
| `household_id` | Raw household id |
| `product_id` | Raw product id |
| `base_score` | Candidate-generation score |
| `model_name` | `popularity`, `category_popularity`, `itemknn`, `als`, `bpr`, or `lightgcn` |
| `base_rank` | Rank before reranking |

### `outputs/reranked_MODEL.csv`

| Column | Meaning |
| --- | --- |
| `household_id` | Raw household id |
| `product_id` | Raw product id |
| `base_score` | Candidate-generation score |
| `promo_score` | Promotion score |
| `coupon_score` | Coupon score |
| `discount_cost_proxy` | Estimated discount cost |
| `diversity_score` | Diversity adjustment |
| `final_score` | Reranked score |
| `final_rank` | Rank after reranking |
| `recommend_coupon` | Boolean or yes/no coupon recommendation |
| `reason_signal` | Optional compact reason field for demo |

## Required Leakage Checks

- Confirm `train.max_week < valid.min_week <= valid.max_week < test.min_week`.
- Candidate models must train only on train weeks for validation tuning.
- Hyperparameters should be selected on validation, then reported once on test.
- Household/product historical features must include a suffix such as `_train` or an explicit feature-window note.
- Coupon redemption from valid/test weeks must not affect reranking for valid/test evaluation.
- Target-week sales values can be used only after ranking, to compute hit value in evaluation.
- Demo examples should clearly label whether they are validation/test examples or static illustrative samples.
