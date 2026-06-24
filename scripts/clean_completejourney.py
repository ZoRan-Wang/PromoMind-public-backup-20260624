"""Clean committed Complete Journey RDS/RDA data for PromoMind.

Raw files remain unchanged. Generated CSV tables and audit metadata are written
under data/processed, which is ignored by Git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402
from promomind.data.preprocess import (  # noqa: E402
    add_dataset_weeks,
    apply_product_catalog,
    build_discount_cost_features,
    build_household_category_preferences,
    build_household_features,
    build_household_product_coupon_features,
    build_product_features,
    build_product_week_promotion_features,
    clean_transactions,
    normalize_columns,
    select_products_from_train,
    week_based_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "completejourney",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--min-product-purchases",
        type=int,
        default=None,
        help="Optional minimum train-period transaction rows per product.",
    )
    parser.add_argument(
        "--top-products",
        type=int,
        default=None,
        help="Optional maximum catalog size, ranked using train rows only.",
    )
    return parser.parse_args()


def read_r_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw source file: {path}")
    objects = pyreadr.read_r(str(path))
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    return normalize_columns(next(iter(objects.values())))


def clean_keyed_table(
    frame: pd.DataFrame,
    required_keys: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = normalize_columns(frame)
    before = len(cleaned)
    for key in required_keys:
        if key not in cleaned.columns:
            raise ValueError(f"Table is missing required key: {key}")
        cleaned[key] = cleaned[key].astype("string").str.strip()
        cleaned = cleaned[cleaned[key].notna() & cleaned[key].ne("")]
    invalid_keys_removed = before - len(cleaned)
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned, {
        "raw_rows": before,
        "invalid_key_rows_removed": invalid_keys_removed,
        "exact_duplicate_rows_removed": duplicates_removed,
        "clean_rows": len(cleaned),
    }


def missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.isna().sum()
    return {str(key): int(value) for key, value in counts[counts > 0].items()}


def validate_processed_contracts(outputs: dict[str, pd.DataFrame]) -> None:
    required_columns = {
        "train_interactions.csv": [
            schema.HOUSEHOLD_ID,
            schema.PRODUCT_ID,
            schema.BASKET_ID,
            schema.WEEK,
            schema.QUANTITY,
            schema.SALES_VALUE,
            schema.RETAIL_DISC,
            schema.COUPON_DISC,
            schema.COUPON_MATCH_DISC,
        ],
        "valid_interactions.csv": [
            schema.HOUSEHOLD_ID,
            schema.PRODUCT_ID,
            schema.BASKET_ID,
            schema.WEEK,
            schema.QUANTITY,
            schema.SALES_VALUE,
            schema.RETAIL_DISC,
            schema.COUPON_DISC,
            schema.COUPON_MATCH_DISC,
        ],
        "test_interactions.csv": [
            schema.HOUSEHOLD_ID,
            schema.PRODUCT_ID,
            schema.BASKET_ID,
            schema.WEEK,
            schema.QUANTITY,
            schema.SALES_VALUE,
            schema.RETAIL_DISC,
            schema.COUPON_DISC,
            schema.COUPON_MATCH_DISC,
        ],
        "product_features.csv": [
            schema.PRODUCT_ID,
            "department",
            "product_category",
            "brand",
        ],
        "household_features.csv": [
            schema.HOUSEHOLD_ID,
            "total_baskets_train",
            "total_spend_train",
            "avg_basket_value_train",
            "top_department_train",
            "top_category_train",
            "demographics_available",
        ],
        "product_week_promotion_features.csv": [
            schema.PRODUCT_ID,
            schema.WEEK,
            "promotion_score",
            "has_display",
            "has_mailer",
            "promotion_source_count",
        ],
        "household_product_coupon_features.csv": [
            schema.HOUSEHOLD_ID,
            schema.PRODUCT_ID,
            schema.WEEK,
            "coupon_score",
            "campaign_id",
            "historical_coupon_redemption_rate",
        ],
        "discount_cost_features.csv": [
            schema.PRODUCT_ID,
            "avg_product_discount_train",
            "avg_category_discount_train",
            "estimated_discount_cost",
        ],
    }
    for filename, columns in required_columns.items():
        frame = outputs[filename]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise AssertionError(f"{filename} is missing required columns: {missing}")
        if frame[columns].isna().any().any():
            null_columns = frame[columns].columns[frame[columns].isna().any()].tolist()
            raise AssertionError(
                f"{filename} has nulls in required columns: {null_columns}"
            )

    if outputs["product_week_promotion_features.csv"].duplicated(
        [schema.PRODUCT_ID, schema.WEEK]
    ).any():
        raise AssertionError("Promotion features contain duplicate product-week rows.")
    if outputs["household_product_coupon_features.csv"].duplicated(
        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, schema.WEEK]
    ).any():
        raise AssertionError(
            "Coupon features contain duplicate household-product-week rows."
        )
    if outputs["discount_cost_features.csv"][schema.PRODUCT_ID].duplicated().any():
        raise AssertionError("Discount features contain duplicate product rows.")


def main() -> int:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_transactions = read_r_table(args.raw_dir / "transactions.rds")
    raw_products = read_r_table(args.raw_dir / "products.rda")
    raw_demographics = read_r_table(args.raw_dir / "demographics.rda")
    raw_coupons = read_r_table(args.raw_dir / "coupons.rda")
    raw_redemptions = read_r_table(args.raw_dir / "coupon_redemptions.rda")
    raw_campaigns = read_r_table(args.raw_dir / "campaigns.rda")
    raw_campaign_descriptions = read_r_table(
        args.raw_dir / "campaign_descriptions.rda"
    )
    raw_promotions = read_r_table(args.raw_dir / "promotions.rds")

    transaction_audit = {
        "raw_rows": len(raw_transactions),
        "quantity_nonpositive_rows": int((raw_transactions[schema.QUANTITY] <= 0).sum()),
        "negative_sales_value_rows": int(
            (raw_transactions[schema.SALES_VALUE] < 0).sum()
        ),
        "invalid_week_rows": int(
            (
                ~raw_transactions[schema.WEEK].between(
                    1,
                    schema.DEFAULT_TEST_END_WEEK,
                    inclusive="both",
                )
            ).sum()
        ),
        "exact_duplicate_rows": int(raw_transactions.duplicated().sum()),
        "repeated_basket_product_rows": int(
            raw_transactions.duplicated(
                [schema.HOUSEHOLD_ID, schema.BASKET_ID, schema.PRODUCT_ID]
            ).sum()
        ),
    }
    transactions = clean_transactions(raw_transactions)
    transaction_audit["clean_rows_before_catalog_filter"] = len(transactions)

    train, valid, test = week_based_split(transactions)
    selected_products = select_products_from_train(
        train,
        min_product_purchases=args.min_product_purchases,
        top_products=args.top_products,
    )
    train = apply_product_catalog(train, selected_products)
    valid = apply_product_catalog(valid, selected_products)
    test = apply_product_catalog(test, selected_products)
    transactions = pd.concat([train, valid, test], ignore_index=True).sort_values(
        [schema.TRANSACTION_TIMESTAMP, schema.BASKET_ID, schema.PRODUCT_ID]
    )
    transaction_audit["clean_rows_after_catalog_filter"] = len(transactions)
    transaction_audit["selected_products_from_train"] = len(selected_products)
    transaction_audit["product_filter"] = {
        "min_product_purchases": args.min_product_purchases,
        "top_products": args.top_products,
        "selection_window": "weeks_1_40_only",
    }

    products, products_audit = clean_keyed_table(raw_products, [schema.PRODUCT_ID])
    demographics, demographics_audit = clean_keyed_table(
        raw_demographics,
        [schema.HOUSEHOLD_ID],
    )
    coupons, coupons_audit = clean_keyed_table(
        raw_coupons,
        ["coupon_upc", schema.PRODUCT_ID, "campaign_id"],
    )
    redemptions, redemptions_audit = clean_keyed_table(
        raw_redemptions,
        [schema.HOUSEHOLD_ID, "coupon_upc", "campaign_id"],
    )
    campaigns, campaigns_audit = clean_keyed_table(
        raw_campaigns,
        ["campaign_id", schema.HOUSEHOLD_ID],
    )
    campaign_descriptions, campaign_descriptions_audit = clean_keyed_table(
        raw_campaign_descriptions,
        ["campaign_id"],
    )
    redemptions = add_dataset_weeks(
        redemptions,
        "redemption_date",
        output_column=schema.WEEK,
    )
    campaign_descriptions["start_date"] = pd.to_datetime(
        campaign_descriptions["start_date"],
        errors="coerce",
    )
    campaign_descriptions["end_date"] = pd.to_datetime(
        campaign_descriptions["end_date"],
        errors="coerce",
    )
    origin = pd.Timestamp("2017-01-01")
    campaign_descriptions["start_week"] = (
        (campaign_descriptions["start_date"] - origin).dt.days // 7 + 1
    ).clip(1, schema.DEFAULT_TEST_END_WEEK)
    campaign_descriptions["end_week"] = (
        (campaign_descriptions["end_date"] - origin).dt.days // 7 + 1
    ).clip(1, schema.DEFAULT_TEST_END_WEEK)

    product_features = build_product_features(train, products)
    product_behavioral_columns = {
        "product_purchase_rows": "product_purchase_rows_train",
        "product_units": "product_units_train",
        "product_revenue": "product_revenue_train",
        "product_unique_households": "product_unique_households_train",
        "product_unique_baskets": "product_unique_baskets_train",
        "avg_units_per_row": "avg_units_per_row_train",
        "avg_sales_value": "avg_sales_value_train",
        "total_retail_disc": "total_retail_disc_train",
        "total_coupon_disc": "total_coupon_disc_train",
        "total_coupon_match_disc": "total_coupon_match_disc_train",
    }
    product_features = product_features.rename(columns=product_behavioral_columns)
    for col in ["department", "product_category", "brand"]:
        product_features[col] = product_features[col].astype("string").fillna("Unknown")

    household_features = build_household_features(train, demographics)
    household_preferences = build_household_category_preferences(train, products)
    household_features = household_features.merge(
        household_preferences,
        on=schema.HOUSEHOLD_ID,
        how="left",
    )
    household_features["demographics_available"] = household_features[
        schema.HOUSEHOLD_ID
    ].isin(demographics[schema.HOUSEHOLD_ID])
    behavioral_columns = {
        "household_purchase_rows": "household_purchase_rows_train",
        "household_units": "household_units_train",
        "household_spend": "total_spend_train",
        "household_unique_products": "household_unique_products_train",
        "household_unique_baskets": "total_baskets_train",
        "first_purchase_at": "first_purchase_at_train",
        "last_purchase_at": "last_purchase_at_train",
        "avg_basket_value": "avg_basket_value_train",
    }
    household_features = household_features.rename(columns=behavioral_columns)
    promotion_features = build_product_week_promotion_features(
        raw_promotions,
        product_ids=selected_products,
    )
    coupon_features = build_household_product_coupon_features(
        train,
        coupons,
        campaigns,
        campaign_descriptions,
        redemptions,
    )
    discount_cost_features = build_discount_cost_features(train, products)

    split_audit = {
        "train": {
            "rows": len(train),
            "min_week": int(train[schema.WEEK].min()),
            "max_week": int(train[schema.WEEK].max()),
        },
        "valid": {
            "rows": len(valid),
            "min_week": int(valid[schema.WEEK].min()),
            "max_week": int(valid[schema.WEEK].max()),
        },
        "test": {
            "rows": len(test),
            "min_week": int(test[schema.WEEK].min()),
            "max_week": int(test[schema.WEEK].max()),
        },
    }
    if not (
        split_audit["train"]["max_week"]
        < split_audit["valid"]["min_week"]
        <= split_audit["valid"]["max_week"]
        < split_audit["test"]["min_week"]
    ):
        raise AssertionError("Chronological split leakage check failed.")

    outputs = {
        "transactions_clean.csv": transactions,
        "train_interactions.csv": train,
        "valid_interactions.csv": valid,
        "test_interactions.csv": test,
        "products_clean.csv": products,
        "demographics_clean.csv": demographics,
        "coupons_clean.csv": coupons,
        "coupon_redemptions_clean.csv": redemptions,
        "campaigns_clean.csv": campaigns,
        "campaign_descriptions_clean.csv": campaign_descriptions,
        "product_features.csv": product_features,
        "household_features.csv": household_features,
        "product_week_promotion_features.csv": promotion_features,
        "household_product_coupon_features.csv": coupon_features,
        "discount_cost_features.csv": discount_cost_features,
    }
    validate_processed_contracts(outputs)
    for filename, frame in outputs.items():
        path = args.processed_dir / filename
        frame.to_csv(path, index=False)
        print(f"Wrote {path} ({len(frame):,} rows)")

    audit = {
        "requirements": {
            "raw_files_modified": False,
            "split": "train=weeks 1-40; valid=41-46; test=47-53",
            "filters_learned_from": "train_only",
            "behavioral_features_learned_from": "train_only",
            "demographics_missing_policy": "preserve_as_missing; do not drop households",
            "target_sales_value_policy": "retained for evaluation only; not a ranking feature",
            "coupon_redemption_policy": "cleaned auxiliary table only; no future redemption features built",
            "promotion_feature_policy": "planned store-level display/mailer exposure aggregated by product-week",
            "coupon_feature_policy": "sparse positive signals for train-observed household-product affinities; target weeks 41-53",
            "discount_feature_policy": "estimated only from weeks 1-40",
        },
        "transactions": transaction_audit,
        "splits": split_audit,
        "tables": {
            "products": products_audit,
            "demographics": demographics_audit,
            "coupons": coupons_audit,
            "coupon_redemptions": redemptions_audit,
            "campaigns": campaigns_audit,
            "campaign_descriptions": campaign_descriptions_audit,
            "product_week_promotion_features": {
                "rows": len(promotion_features),
                "min_week": int(promotion_features[schema.WEEK].min()),
                "max_week": int(promotion_features[schema.WEEK].max()),
            },
            "household_product_coupon_features": {
                "rows": len(coupon_features),
                "min_week": int(coupon_features[schema.WEEK].min()),
                "max_week": int(coupon_features[schema.WEEK].max()),
                "duplicate_household_product_week_rows": int(
                    coupon_features.duplicated(
                        [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, schema.WEEK]
                    ).sum()
                ),
            },
            "discount_cost_features": {
                "rows": len(discount_cost_features),
                "duplicate_product_rows": int(
                    discount_cost_features[schema.PRODUCT_ID].duplicated().sum()
                ),
            },
        },
        "missing_values": {
            "products_clean": missing_counts(products),
            "demographics_clean": missing_counts(demographics),
            "household_features": missing_counts(household_features),
            "product_week_promotion_features": missing_counts(promotion_features),
            "household_product_coupon_features": missing_counts(coupon_features),
            "discount_cost_features": missing_counts(discount_cost_features),
        },
        "referential_integrity": {
            "transaction_product_ids_missing_from_products": int(
                transactions.loc[
                    ~transactions[schema.PRODUCT_ID].isin(products[schema.PRODUCT_ID]),
                    schema.PRODUCT_ID,
                ].nunique()
            ),
            "coupon_product_ids_missing_from_products": int(
                coupons.loc[
                    ~coupons[schema.PRODUCT_ID].isin(products[schema.PRODUCT_ID]),
                    schema.PRODUCT_ID,
                ].nunique()
            ),
            "redemption_coupon_ids_missing_from_coupons": int(
                redemptions.loc[
                    ~redemptions["coupon_upc"].isin(coupons["coupon_upc"]),
                    "coupon_upc",
                ].nunique()
            ),
            "validation_households_without_train_history": int(
                valid.loc[
                    ~valid[schema.HOUSEHOLD_ID].isin(train[schema.HOUSEHOLD_ID]),
                    schema.HOUSEHOLD_ID,
                ].nunique()
            ),
            "test_households_without_train_history": int(
                test.loc[
                    ~test[schema.HOUSEHOLD_ID].isin(train[schema.HOUSEHOLD_ID]),
                    schema.HOUSEHOLD_ID,
                ].nunique()
            ),
        },
    }
    audit_path = args.processed_dir / "cleaning_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
