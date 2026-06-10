# PromoMind Data Workflow

This folder is the handoff point between the raw Complete Journey tables and the processed tables used by modeling, promotion-aware reranking, evaluation, and the demo.

## Repository Data Policy

For this class project repository, commit the original public Complete Journey raw files when they are under GitHub's 100MB per-file limit. Do not commit generated train/validation/test splits or derived processed outputs.

The full `transactions.rds` and `promotions.rds` files are small enough in their original compressed RDS form to store in GitHub. CSV conversions of those same files may become much larger, so keep the committed dataset in original R serialization format.

## Directory Layout

- `data/raw/completejourney/`: Original public Complete Journey files committed to the repository.
- `data/raw/`: Optional local CSV exports for preprocessing scripts.
- `data/processed/`: Generated CSVs from `scripts/prepare_dataset.py`; ignored by Git.

## Expected Raw Files

The committed raw dataset should include:

- `transactions.rds`
- `promotions.rds`
- `campaigns.rda`
- `campaign_descriptions.rda`
- `coupons.rda`
- `coupon_redemptions.rda`
- `demographics.rda`
- `products.rda`
- `transactions_sample.rda`
- `promotions_sample.rda`

The RDS/RDA files come from the public `bradleyboehmke/completejourney` repository and CRAN package source.

## Optional CSV Workflow

If a team member exports raw tables to CSV locally, `scripts/prepare_dataset.py` expects these filenames in `data/raw/`:

- `transactions.csv` (required): household-product basket transactions.
- `products.csv` (optional but recommended): product metadata.
- `demographics.csv` (optional): household demographic attributes.
- `promotions.csv` (optional): promotion exposure or campaign metadata.
- `coupons.csv` (optional): coupon metadata.
- `coupon_redemptions.csv` (optional): household coupon redemption events.

Column names from the R `completejourney` package and common Dunnhumby exports are normalized by the preprocessing code where possible.

## Processed Outputs

Running `python scripts/clean_completejourney.py` reads the committed RDS/RDA
files directly and writes generated files under `data/processed/`, including:

- `transactions_clean.csv`
- `train_interactions.csv`
- `valid_interactions.csv`
- `test_interactions.csv`
- `products_clean.csv`
- `demographics_clean.csv`
- `coupons_clean.csv`
- `coupon_redemptions_clean.csv`
- `product_features.csv`
- `household_features.csv`
- `product_week_promotion_features.csv`
- `household_product_coupon_features.csv`
- `discount_cost_features.csv`
- `cleaning_audit.json`

These processed files are intentionally ignored by Git. They should be regenerated locally from the committed raw data or local CSV exports.

Feature definitions:

- Promotion features aggregate display and mailer exposure across stores for
  each product-week. `promotion_score` is the average display and mailer store
  exposure rate.
- Coupon features are sparse positive signals for household-product pairs seen
  during training and active campaign weeks 41-53. Missing rows imply zero
  coupon signal. Historical redemption rates use assigned campaigns and
  redemptions from weeks 1-40 only.
- Discount-cost features use transaction discounts from weeks 1-40 only.
  Products without category metadata use the train-wide average as the
  category fallback.

To create a tiny local smoke-test dataset:

```powershell
python scripts/make_sample_data.py
python scripts/prepare_dataset.py --top-products 6
```

