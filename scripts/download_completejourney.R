# Export committed Complete Journey RDS/RDA files to local CSV files.
#
# Run this script from the repository root in R or RStudio:
#   install.packages("completejourney")
#   source("scripts/download_completejourney.R")
#
# This repository commits the original RDS/RDA files under
# data/raw/completejourney/. CSV exports are optional local working files for
# Python preprocessing and are ignored by Git.

raw_dir <- file.path("data", "raw")
raw_original_dir <- file.path(raw_dir, "completejourney")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(raw_original_dir, recursive = TRUE, showWarnings = FALSE)

write_csv <- function(table, filename) {
  out_path <- file.path(raw_dir, filename)
  write.csv(table, out_path, row.names = FALSE)
  message("Wrote ", out_path, " (", nrow(table), " rows)")
}

write_rds_if_exists <- function(source_filename, output_filename) {
  source_path <- file.path(raw_original_dir, source_filename)
  if (file.exists(source_path)) {
    write_csv(readRDS(source_path), output_filename)
    return(TRUE)
  }
  message("Skipped missing source file: ", source_path)
  FALSE
}

write_rda_if_exists <- function(source_filename, output_filename) {
  source_path <- file.path(raw_original_dir, source_filename)
  if (file.exists(source_path)) {
    env <- new.env(parent = emptyenv())
    loaded_names <- load(source_path, envir = env)
    if (length(loaded_names) != 1) {
      stop("Expected one object in ", source_path, " but found: ", paste(loaded_names, collapse = ", "))
    }
    write_csv(get(loaded_names[[1]], envir = env), output_filename)
    return(TRUE)
  }
  message("Skipped missing source file: ", source_path)
  FALSE
}

write_rds_if_exists("transactions.rds", "transactions.csv")
write_rds_if_exists("promotions.rds", "promotions.csv")
write_rda_if_exists("products.rda", "products.csv")
write_rda_if_exists("demographics.rda", "demographics.csv")
write_rda_if_exists("coupons.rda", "coupons.csv")
write_rda_if_exists("coupon_redemptions.rda", "coupon_redemptions.csv")
write_rda_if_exists("campaigns.rda", "campaigns.csv")
write_rda_if_exists("campaign_descriptions.rda", "campaign_descriptions.csv")

# Optional: download the full public transactions/promotions objects through
# the package helper and save them in original RDS form. This requires the
# completejourney package and can take time. Uncomment if the committed raw
# files need to be refreshed.
#
# if (!requireNamespace("completejourney", quietly = TRUE)) {
#   install.packages("completejourney")
# }
#
# transactions <- completejourney::get_transactions(verbose = TRUE)
# saveRDS(transactions, file.path(raw_original_dir, "transactions.rds"))
#
# promotions <- completejourney::get_promotions(verbose = TRUE)
# saveRDS(promotions, file.path(raw_original_dir, "promotions.rds"))
