"""Schema constants for the PromoMind Complete Journey data workflow."""

from __future__ import annotations

HOUSEHOLD_ID = "household_id"
PRODUCT_ID = "product_id"
BASKET_ID = "basket_id"
TRANSACTION_TIMESTAMP = "transaction_timestamp"
DAY = "day"
WEEK = "week"
QUANTITY = "quantity"
SALES_VALUE = "sales_value"
STORE_ID = "store_id"
RETAIL_DISC = "retail_disc"
COUPON_DISC = "coupon_disc"
COUPON_MATCH_DISC = "coupon_match_disc"

CANONICAL_TRANSACTION_COLUMNS = [
    HOUSEHOLD_ID,
    PRODUCT_ID,
    BASKET_ID,
    TRANSACTION_TIMESTAMP,
    DAY,
    WEEK,
    QUANTITY,
    SALES_VALUE,
    STORE_ID,
    RETAIL_DISC,
    COUPON_DISC,
    COUPON_MATCH_DISC,
]

REQUIRED_TRANSACTION_COLUMNS = [
    HOUSEHOLD_ID,
    PRODUCT_ID,
    BASKET_ID,
    QUANTITY,
    SALES_VALUE,
]

TIME_COLUMN_CANDIDATES = [TRANSACTION_TIMESTAMP, DAY, WEEK]

PRODUCT_REQUIRED_COLUMNS = [PRODUCT_ID]
HOUSEHOLD_REQUIRED_COLUMNS = [HOUSEHOLD_ID]
PROMOTION_JOIN_KEYS = [PRODUCT_ID, STORE_ID, WEEK]
COUPON_JOIN_KEYS = [HOUSEHOLD_ID, PRODUCT_ID]

DEFAULT_TRAIN_END_WEEK = 40
DEFAULT_VALID_END_WEEK = 46
DEFAULT_TEST_END_WEEK = 53

DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15

RAW_FILENAMES = {
    "transactions": "transactions.csv",
    "products": "products.csv",
    "demographics": "demographics.csv",
    "promotions": "promotions.csv",
    "coupons": "coupons.csv",
    "coupon_redemptions": "coupon_redemptions.csv",
}

PROCESSED_FILENAMES = {
    "transactions_clean": "transactions_clean.csv",
    "train_interactions": "train_interactions.csv",
    "valid_interactions": "valid_interactions.csv",
    "test_interactions": "test_interactions.csv",
    "transactions_train": "transactions_train.csv",
    "transactions_val": "transactions_val.csv",
    "transactions_test": "transactions_test.csv",
    "product_features": "product_features.csv",
    "household_features": "household_features.csv",
    "transactions_with_promotions": "transactions_with_promotions.csv",
}

COLUMN_ALIASES = {
    HOUSEHOLD_ID: ["household_id", "household_key", "hshd_num"],
    PRODUCT_ID: ["product_id", "product_key", "upc", "product_num"],
    BASKET_ID: ["basket_id", "basket_key", "transaction_id"],
    TRANSACTION_TIMESTAMP: ["transaction_timestamp", "trans_time", "timestamp", "datetime"],
    DAY: ["day", "transaction_day"],
    WEEK: ["week", "week_no", "week_number"],
    QUANTITY: ["quantity", "qty"],
    SALES_VALUE: ["sales_value", "sales_amount", "amount"],
    STORE_ID: ["store_id", "store"],
    RETAIL_DISC: ["retail_disc", "retail_discount"],
    COUPON_DISC: ["coupon_disc", "coupon_discount"],
    COUPON_MATCH_DISC: ["coupon_match_disc", "coupon_match_discount"],
}
