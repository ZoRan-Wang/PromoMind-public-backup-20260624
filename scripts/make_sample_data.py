"""Generate a tiny synthetic Complete Journey-like dataset for smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    transactions = pd.DataFrame(
        [
            ["H001", "B001", "P001", 1, 1, 2, 5.98, "S01", 0.50, 0.00, 0.00],
            ["H001", "B001", "P002", 1, 1, 1, 3.49, "S01", 0.00, 0.25, 0.00],
            ["H002", "B002", "P001", 2, 1, 1, 2.99, "S02", 0.25, 0.00, 0.00],
            ["H002", "B003", "P003", 10, 2, 2, 7.58, "S02", 0.00, 0.00, 0.00],
            ["H003", "B004", "P004", 17, 3, 1, 4.25, "S03", 0.50, 0.50, 0.00],
            ["H001", "B005", "P003", 24, 4, 1, 3.79, "S01", 0.00, 0.00, 0.00],
            ["H003", "B006", "P002", 31, 5, 2, 6.98, "S03", 0.00, 0.50, 0.25],
            ["H004", "B007", "P005", 38, 6, 1, 8.99, "S04", 1.00, 0.00, 0.00],
            ["H004", "B008", "P006", 45, 7, 1, 2.49, "S04", 0.00, 0.00, 0.00],
            ["H002", "B009", "P001", 52, 8, 3, 8.97, "S02", 0.75, 0.00, 0.00],
        ],
        columns=[
            "household_id",
            "basket_id",
            "product_id",
            "day",
            "week",
            "quantity",
            "sales_value",
            "store_id",
            "retail_disc",
            "coupon_disc",
            "coupon_match_disc",
        ],
    )

    products = pd.DataFrame(
        [
            ["P001", "Dairy", "Milk", "Brand A"],
            ["P002", "Bakery", "Bread", "Brand B"],
            ["P003", "Produce", "Apples", "Brand C"],
            ["P004", "Frozen", "Pizza", "Brand D"],
            ["P005", "Household", "Detergent", "Brand E"],
            ["P006", "Snacks", "Crackers", "Brand F"],
        ],
        columns=["product_id", "department", "commodity_desc", "brand"],
    )

    demographics = pd.DataFrame(
        [
            ["H001", "35-44", "2", "60-74K"],
            ["H002", "45-54", "3", "75-99K"],
            ["H003", "25-34", "1", "35-49K"],
            ["H004", "55-64", "2", "100-124K"],
        ],
        columns=["household_id", "age_desc", "household_size_desc", "income_desc"],
    )

    promotions = pd.DataFrame(
        [["P001", 1, "display"], ["P003", 4, "feature"], ["P005", 6, "display"]],
        columns=["product_id", "week", "promotion_type"],
    )
    coupons = pd.DataFrame(
        [["H001", "P002", "C001"], ["H003", "P002", "C002"], ["H004", "P005", "C003"]],
        columns=["household_id", "product_id", "coupon_id"],
    )
    coupon_redemptions = pd.DataFrame(
        [["H003", "P002", 5, "C002"]],
        columns=["household_id", "product_id", "week", "coupon_id"],
    )

    tables = {
        "transactions.csv": transactions,
        "products.csv": products,
        "demographics.csv": demographics,
        "promotions.csv": promotions,
        "coupons.csv": coupons,
        "coupon_redemptions.csv": coupon_redemptions,
    }
    for filename, table in tables.items():
        path = RAW_DIR / filename
        table.to_csv(path, index=False)
        print(f"Wrote {path} ({len(table)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
