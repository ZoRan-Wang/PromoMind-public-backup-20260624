"""Pandas-only preprocessing utilities for The Complete Journey workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import schema


def _snake_case(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower snake-case columns and known aliases renamed."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    out = df.copy()
    out.columns = [_snake_case(col) for col in out.columns]
    rename_map: dict[str, str] = {}
    for canonical, aliases in schema.COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in out.columns and canonical not in out.columns:
                rename_map[alias] = canonical
                break
    return out.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str = "dataframe") -> None:
    """Raise a clear error when expected columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize transaction data and remove rows that cannot support modeling."""
    df = normalize_columns(transactions)
    require_columns(
        df,
        [*schema.REQUIRED_TRANSACTION_COLUMNS, schema.WEEK],
        "transactions",
    )

    key_columns = [
        schema.HOUSEHOLD_ID,
        schema.PRODUCT_ID,
        schema.BASKET_ID,
    ]
    for col in key_columns:
        df[col] = df[col].astype("string").str.strip()
        df = df[df[col].notna() & (df[col] != "")]

    numeric_defaults = {
        schema.QUANTITY: None,
        schema.SALES_VALUE: None,
        schema.RETAIL_DISC: 0.0,
        schema.COUPON_DISC: 0.0,
        schema.COUPON_MATCH_DISC: 0.0,
        schema.WEEK: pd.NA,
        schema.DAY: pd.NA,
    }
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if default is not None:
                df[col] = df[col].fillna(default)

    df = df[df[schema.QUANTITY].fillna(0) > 0]
    df = df[df[schema.SALES_VALUE].fillna(-1) >= 0]
    df = df[df[schema.WEEK].between(1, schema.DEFAULT_TEST_END_WEEK, inclusive="both")]

    if schema.TRANSACTION_TIMESTAMP in df.columns:
        df[schema.TRANSACTION_TIMESTAMP] = pd.to_datetime(
            df[schema.TRANSACTION_TIMESTAMP],
            errors="coerce",
        )
    elif schema.DAY in df.columns:
        day_values = pd.to_numeric(df[schema.DAY], errors="coerce")
        df[schema.TRANSACTION_TIMESTAMP] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
            day_values.fillna(0).astype(int),
            unit="D",
        )
    elif schema.WEEK in df.columns:
        week_values = pd.to_numeric(df[schema.WEEK], errors="coerce")
        df[schema.TRANSACTION_TIMESTAMP] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
            week_values.fillna(0).astype(int) * 7,
            unit="D",
        )
    else:
        raise ValueError("transactions needs one time column: transaction_timestamp, day, or week.")

    df = df[df[schema.TRANSACTION_TIMESTAMP].notna()]
    df = df.drop_duplicates()

    basket_product_keys = [
        schema.HOUSEHOLD_ID,
        schema.STORE_ID,
        schema.BASKET_ID,
        schema.PRODUCT_ID,
        schema.WEEK,
        schema.TRANSACTION_TIMESTAMP,
    ]
    basket_product_keys = [col for col in basket_product_keys if col in df.columns]
    value_columns = [
        schema.QUANTITY,
        schema.SALES_VALUE,
        schema.RETAIL_DISC,
        schema.COUPON_DISC,
        schema.COUPON_MATCH_DISC,
    ]
    value_columns = [col for col in value_columns if col in df.columns]
    other_columns = [
        col for col in df.columns if col not in basket_product_keys and col not in value_columns
    ]
    aggregations = {col: "sum" for col in value_columns}
    aggregations.update({col: "first" for col in other_columns})
    df = df.groupby(basket_product_keys, as_index=False, dropna=False).agg(aggregations)

    sort_cols = [schema.TRANSACTION_TIMESTAMP, schema.BASKET_ID, schema.PRODUCT_ID]
    return df.sort_values(sort_cols).reset_index(drop=True)


def select_products_from_train(
    train_transactions: pd.DataFrame,
    min_product_purchases: int | None = None,
    top_products: int | None = None,
) -> pd.Index:
    """Select catalog products using training interactions only."""
    train = normalize_columns(train_transactions)
    require_columns(train, [schema.PRODUCT_ID], "train_transactions")

    if min_product_purchases is not None and min_product_purchases < 1:
        raise ValueError("min_product_purchases must be at least 1.")
    if top_products is not None and top_products < 1:
        raise ValueError("top_products must be at least 1.")

    counts = train.groupby(schema.PRODUCT_ID).size().sort_values(
        ascending=False,
        kind="stable",
    )
    keep = counts
    if min_product_purchases is not None:
        keep = keep[keep >= min_product_purchases]
    if top_products is not None:
        keep = keep.head(top_products)
    return keep.index


def apply_product_catalog(
    transactions: pd.DataFrame,
    product_ids: Iterable[str],
) -> pd.DataFrame:
    """Apply a preselected product catalog without learning from this split."""
    df = normalize_columns(transactions)
    require_columns(df, [schema.PRODUCT_ID], "transactions")
    keep = pd.Index(product_ids, dtype="string")
    return df[df[schema.PRODUCT_ID].astype("string").isin(keep)].reset_index(drop=True)


def filter_frequent_products(
    transactions: pd.DataFrame,
    min_product_purchases: int | None = None,
    top_products: int | None = None,
) -> pd.DataFrame:
    """Keep products with enough purchase rows or the top N products by frequency."""
    df = normalize_columns(transactions)
    require_columns(df, [schema.PRODUCT_ID], "transactions")

    if min_product_purchases is None and top_products is None:
        return df.reset_index(drop=True)
    if min_product_purchases is not None and min_product_purchases < 1:
        raise ValueError("min_product_purchases must be at least 1.")
    if top_products is not None and top_products < 1:
        raise ValueError("top_products must be at least 1.")

    counts = df.groupby(schema.PRODUCT_ID).size().sort_values(ascending=False)
    keep = counts.index
    if min_product_purchases is not None:
        keep = counts[counts >= min_product_purchases].index
    if top_products is not None:
        keep = counts.loc[keep].head(top_products).index
    return df[df[schema.PRODUCT_ID].isin(keep)].reset_index(drop=True)


def time_based_split(
    transactions: pd.DataFrame,
    train_fraction: float = schema.DEFAULT_TRAIN_FRACTION,
    val_fraction: float = schema.DEFAULT_VAL_FRACTION,
    test_fraction: float = schema.DEFAULT_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transactions chronologically into train, validation, and test sets."""
    total = train_fraction + val_fraction + test_fraction
    if not 0.99 <= total <= 1.01:
        raise ValueError("train, validation, and test fractions must sum to 1.0.")
    if min(train_fraction, val_fraction, test_fraction) < 0:
        raise ValueError("split fractions must be non-negative.")

    df = clean_transactions(transactions)
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    unique_times = df[schema.TRANSACTION_TIMESTAMP].drop_duplicates().sort_values().reset_index(drop=True)
    train_end_idx = max(1, int(len(unique_times) * train_fraction))
    val_end_idx = max(train_end_idx, int(len(unique_times) * (train_fraction + val_fraction)))

    train_end = unique_times.iloc[min(train_end_idx - 1, len(unique_times) - 1)]
    val_end = unique_times.iloc[min(val_end_idx - 1, len(unique_times) - 1)]

    train = df[df[schema.TRANSACTION_TIMESTAMP] <= train_end]
    val = df[(df[schema.TRANSACTION_TIMESTAMP] > train_end) & (df[schema.TRANSACTION_TIMESTAMP] <= val_end)]
    test = df[df[schema.TRANSACTION_TIMESTAMP] > val_end]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def week_based_split(
    transactions: pd.DataFrame,
    train_end_week: int = schema.DEFAULT_TRAIN_END_WEEK,
    valid_end_week: int = schema.DEFAULT_VALID_END_WEEK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transactions by canonical Complete Journey week ranges.

    Default ranges match the project protocol:
    train <= Week 40, validation Weeks 41-46, and test Week 47+.
    """
    df = clean_transactions(transactions)
    if schema.WEEK not in df.columns:
        raise ValueError("week_based_split requires a week column.")

    df[schema.WEEK] = pd.to_numeric(df[schema.WEEK], errors="coerce")
    df = df[df[schema.WEEK].notna()].copy()
    train = df[df[schema.WEEK] <= train_end_week]
    val = df[(df[schema.WEEK] > train_end_week) & (df[schema.WEEK] <= valid_end_week)]
    test = df[df[schema.WEEK] > valid_end_week]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_product_features(
    transactions: pd.DataFrame,
    products: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build product-level frequency, revenue, basket, and discount features."""
    tx = clean_transactions(transactions)
    grouped = tx.groupby(schema.PRODUCT_ID, dropna=False)
    features = grouped.agg(
        product_purchase_rows=(schema.PRODUCT_ID, "size"),
        product_units=(schema.QUANTITY, "sum"),
        product_revenue=(schema.SALES_VALUE, "sum"),
        product_unique_households=(schema.HOUSEHOLD_ID, "nunique"),
        product_unique_baskets=(schema.BASKET_ID, "nunique"),
        avg_units_per_row=(schema.QUANTITY, "mean"),
        avg_sales_value=(schema.SALES_VALUE, "mean"),
    ).reset_index()

    for col in [schema.RETAIL_DISC, schema.COUPON_DISC, schema.COUPON_MATCH_DISC]:
        if col in tx.columns:
            discount = grouped[col].sum().rename(f"total_{col}").reset_index()
            features = features.merge(discount, on=schema.PRODUCT_ID, how="left")

    if products is not None and not products.empty:
        product_meta = normalize_columns(products)
        require_columns(product_meta, [schema.PRODUCT_ID], "products")
        product_meta[schema.PRODUCT_ID] = product_meta[schema.PRODUCT_ID].astype("string")
        features = features.merge(product_meta.drop_duplicates(schema.PRODUCT_ID), on=schema.PRODUCT_ID, how="left")

    numeric_feature_columns = [
        col
        for col in features.columns
        if col != schema.PRODUCT_ID and pd.api.types.is_numeric_dtype(features[col])
    ]
    features[numeric_feature_columns] = features[numeric_feature_columns].fillna(0)
    return features


def build_household_features(
    transactions: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build household-level purchase behavior features."""
    tx = clean_transactions(transactions)
    grouped = tx.groupby(schema.HOUSEHOLD_ID, dropna=False)
    features = grouped.agg(
        household_purchase_rows=(schema.HOUSEHOLD_ID, "size"),
        household_units=(schema.QUANTITY, "sum"),
        household_spend=(schema.SALES_VALUE, "sum"),
        household_unique_products=(schema.PRODUCT_ID, "nunique"),
        household_unique_baskets=(schema.BASKET_ID, "nunique"),
        first_purchase_at=(schema.TRANSACTION_TIMESTAMP, "min"),
        last_purchase_at=(schema.TRANSACTION_TIMESTAMP, "max"),
    ).reset_index()
    features["avg_basket_value"] = features["household_spend"] / features["household_unique_baskets"].clip(lower=1)

    if demographics is not None and not demographics.empty:
        demo = normalize_columns(demographics)
        require_columns(demo, [schema.HOUSEHOLD_ID], "demographics")
        demo[schema.HOUSEHOLD_ID] = demo[schema.HOUSEHOLD_ID].astype("string")
        features = features.merge(demo.drop_duplicates(schema.HOUSEHOLD_ID), on=schema.HOUSEHOLD_ID, how="left")

    numeric_feature_columns = [
        col
        for col in features.columns
        if col != schema.HOUSEHOLD_ID and pd.api.types.is_numeric_dtype(features[col])
    ]
    features[numeric_feature_columns] = features[numeric_feature_columns].fillna(0)
    return features


def build_household_category_preferences(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Build train-window top department and category for each household."""
    tx = clean_transactions(transactions)
    product_meta = normalize_columns(products)
    require_columns(
        product_meta,
        [schema.PRODUCT_ID, "department", "product_category"],
        "products",
    )
    product_meta = product_meta[
        [schema.PRODUCT_ID, "department", "product_category"]
    ].drop_duplicates(schema.PRODUCT_ID)
    product_meta[schema.PRODUCT_ID] = product_meta[schema.PRODUCT_ID].astype("string")
    for col in ["department", "product_category"]:
        product_meta[col] = product_meta[col].astype("string").fillna("Unknown")

    joined = tx[[schema.HOUSEHOLD_ID, schema.PRODUCT_ID]].merge(
        product_meta,
        on=schema.PRODUCT_ID,
        how="left",
    )
    joined[["department", "product_category"]] = joined[
        ["department", "product_category"]
    ].fillna("Unknown")

    output = joined[[schema.HOUSEHOLD_ID]].drop_duplicates().reset_index(drop=True)
    for source_col, output_col in [
        ("department", "top_department_train"),
        ("product_category", "top_category_train"),
    ]:
        counts = (
            joined.groupby([schema.HOUSEHOLD_ID, source_col], dropna=False)
            .size()
            .rename("purchase_rows")
            .reset_index()
            .sort_values(
                [schema.HOUSEHOLD_ID, "purchase_rows", source_col],
                ascending=[True, False, True],
                kind="stable",
            )
        )
        top = counts.drop_duplicates(schema.HOUSEHOLD_ID).rename(
            columns={source_col: output_col}
        )
        output = output.merge(
            top[[schema.HOUSEHOLD_ID, output_col]],
            on=schema.HOUSEHOLD_ID,
            how="left",
        )
    return output


def join_promotion_placeholders(
    transactions: pd.DataFrame,
    promotions: pd.DataFrame | None = None,
    coupons: pd.DataFrame | None = None,
    coupon_redemptions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach lightweight promotion/coupon indicators when source tables exist."""
    tx = clean_transactions(transactions)
    out = tx.copy()
    out["has_promotion"] = False
    out["has_coupon"] = False
    out["redeemed_coupon"] = False

    if promotions is not None and not promotions.empty:
        promo = normalize_columns(promotions)
        join_keys = [key for key in schema.PROMOTION_JOIN_KEYS if key in promo.columns and key in out.columns]
        if join_keys:
            promo_flags = promo[join_keys].drop_duplicates()
            promo_flags["has_promotion"] = True
            out = out.drop(columns=["has_promotion"]).merge(promo_flags, on=join_keys, how="left")
            out["has_promotion"] = out["has_promotion"].fillna(False).astype(bool)

    if coupons is not None and not coupons.empty:
        coupon = normalize_columns(coupons)
        join_keys = [key for key in schema.COUPON_JOIN_KEYS if key in coupon.columns and key in out.columns]
        if not join_keys and schema.PRODUCT_ID in coupon.columns:
            join_keys = [schema.PRODUCT_ID]
        if join_keys:
            coupon_flags = coupon[join_keys].drop_duplicates()
            coupon_flags["has_coupon"] = True
            out = out.drop(columns=["has_coupon"]).merge(coupon_flags, on=join_keys, how="left")
            out["has_coupon"] = out["has_coupon"].fillna(False).astype(bool)

    if coupon_redemptions is not None and not coupon_redemptions.empty:
        redemptions = normalize_columns(coupon_redemptions)
        join_keys = [
            key
            for key in [schema.HOUSEHOLD_ID, schema.PRODUCT_ID, schema.WEEK]
            if key in redemptions.columns and key in out.columns
        ]
        if join_keys:
            redemption_flags = redemptions[join_keys].drop_duplicates()
            redemption_flags["redeemed_coupon"] = True
            out = out.drop(columns=["redeemed_coupon"]).merge(redemption_flags, on=join_keys, how="left")
            out["redeemed_coupon"] = out["redeemed_coupon"].fillna(False).astype(bool)

    return out


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV when it exists; return None for absent optional tables."""
    return pd.read_csv(path) if path.exists() else None
