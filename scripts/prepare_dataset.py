"""Prepare Complete Journey CSVs for PromoMind modeling.

Expected input files in data/raw:
  - transactions.csv (required)
  - products.csv, demographics.csv, promotions.csv, coupons.csv,
    coupon_redemptions.csv (optional)

Outputs are written as CSVs under data/processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402
from promomind.data.preprocess import (  # noqa: E402
    apply_product_catalog,
    build_household_features,
    build_product_features,
    clean_transactions,
    join_promotion_placeholders,
    read_optional_csv,
    select_products_from_train,
    time_based_split,
    week_based_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PromoMind Complete Journey data.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--min-product-purchases",
        type=int,
        default=None,
        help="Keep products with at least this many transaction rows.",
    )
    parser.add_argument(
        "--top-products",
        type=int,
        default=None,
        help="Keep only the most frequently purchased N products.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["week", "fraction"],
        default="week",
        help="Use canonical week split by default, or chronological fractions for non-standard data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    tx_path = raw_dir / schema.RAW_FILENAMES["transactions"]
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Missing required {tx_path}. Run scripts/make_sample_data.py for a smoke-test dataset "
            "or export Complete Journey tables with scripts/download_completejourney.R."
        )

    transactions = pd.read_csv(tx_path)
    products = read_optional_csv(raw_dir / schema.RAW_FILENAMES["products"])
    demographics = read_optional_csv(raw_dir / schema.RAW_FILENAMES["demographics"])
    promotions = read_optional_csv(raw_dir / schema.RAW_FILENAMES["promotions"])

    transactions_clean = clean_transactions(transactions)
    if args.split_mode == "week" and schema.WEEK in transactions_clean.columns:
        train, val, test = week_based_split(transactions_clean)
    else:
        train, val, test = time_based_split(transactions_clean)

    selected_products = select_products_from_train(
        train,
        min_product_purchases=args.min_product_purchases,
        top_products=args.top_products,
    )
    train = apply_product_catalog(train, selected_products)
    val = apply_product_catalog(val, selected_products)
    test = apply_product_catalog(test, selected_products)
    transactions_clean = pd.concat([train, val, test], ignore_index=True)

    product_features = build_product_features(train, products)
    household_features = build_household_features(train, demographics)
    transactions_with_promotions = join_promotion_placeholders(
        transactions_clean,
        promotions=promotions,
        # Coupon eligibility requires campaign-aware joins. Redemptions are
        # post-treatment outcomes and must not be attached across future weeks.
        coupons=None,
        coupon_redemptions=None,
    )

    outputs = {
        "transactions_clean": transactions_clean,
        "train_interactions": train,
        "valid_interactions": val,
        "test_interactions": test,
        "transactions_train": train,
        "transactions_val": val,
        "transactions_test": test,
        "product_features": product_features,
        "household_features": household_features,
        "transactions_with_promotions": transactions_with_promotions,
    }
    for key, df in outputs.items():
        out_path = processed_dir / schema.PROCESSED_FILENAMES[key]
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
