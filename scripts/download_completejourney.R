# Export Complete Journey package tables to local CSV files.
#
# Run this script from the repository root in R or RStudio:
#   install.packages("completejourney")
#   source("scripts/download_completejourney.R")
#
# This repository commits the original RDS/RDA files under
# data/raw/completejourney/. CSV exports are optional local working files for
# Python preprocessing and are ignored by Git.

if (!requireNamespace("completejourney", quietly = TRUE)) {
  install.packages("completejourney")
}

library(completejourney)

raw_dir <- file.path("data", "raw")
raw_original_dir <- file.path(raw_dir, "completejourney")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(raw_original_dir, recursive = TRUE, showWarnings = FALSE)

write_if_exists <- function(object_name, filename) {
  if (exists(object_name, where = asNamespace("completejourney"), inherits = FALSE)) {
    table <- get(object_name, envir = asNamespace("completejourney"))
    write.csv(table, file.path(raw_dir, filename), row.names = FALSE)
    message("Wrote ", file.path(raw_dir, filename))
  } else {
    message("Skipped missing package table: ", object_name)
  }
}

# Common completejourney package table names. If your installed package version
# exposes slightly different names, inspect with:
#   data(package = "completejourney")
# and add another write_if_exists() call below.
write_if_exists("transactions_sample", "transactions.csv")
write_if_exists("products", "products.csv")
write_if_exists("demographics", "demographics.csv")
write_if_exists("promotions_sample", "promotions.csv")
write_if_exists("coupons", "coupons.csv")
write_if_exists("coupon_redemptions", "coupon_redemptions.csv")

# Optional: download the full public transactions/promotions objects through
# the package helper and save them in original RDS form. This can take time.
# Uncomment if the committed raw files need to be refreshed.
#
# transactions <- completejourney::get_transactions(verbose = TRUE)
# saveRDS(transactions, file.path(raw_original_dir, "transactions.rds"))
#
# promotions <- completejourney::get_promotions(verbose = TRUE)
# saveRDS(promotions, file.path(raw_original_dir, "promotions.rds"))
