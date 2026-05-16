# Complete Journey Raw Data Source

These files are original public data artifacts from the `completejourney` project:

- Repository: https://github.com/bradleyboehmke/completejourney
- CRAN package: https://cran.r-project.org/package=completejourney
- License in package metadata: CC0

Committed full-data files:

- `transactions.rds`
- `promotions.rds`

Committed package data files:

- `campaigns.rda`
- `campaign_descriptions.rda`
- `coupons.rda`
- `coupon_redemptions.rda`
- `demographics.rda`
- `products.rda`
- `transactions_sample.rda`
- `promotions_sample.rda`

The full transaction and promotion tables are kept as `.rds` files because the original compressed format stays below GitHub's 100MB single-file limit. Generated CSV splits and processed outputs should remain local and are ignored by Git.

