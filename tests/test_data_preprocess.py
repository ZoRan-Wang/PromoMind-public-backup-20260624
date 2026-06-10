from __future__ import annotations

import pandas as pd

from promomind.data.preprocess import (
    apply_product_catalog,
    build_household_category_preferences,
    clean_transactions,
    select_products_from_train,
    week_based_split,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": ["h1", "h1", "h2", "h2", "h3", "h3"],
            "store_id": ["s1"] * 6,
            "basket_id": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "product_id": ["p1", "p1", "p2", "future_only", "p2", "p3"],
            "quantity": [1, 1, 1, 1, 0, 1],
            "sales_value": [1.0, 2.0, 3.0, 4.0, 5.0, -1.0],
            "week": [1, 40, 40, 41, 46, 47],
            "transaction_timestamp": pd.to_datetime(
                [
                    "2017-01-01",
                    "2017-10-01",
                    "2017-10-01",
                    "2017-10-08",
                    "2017-11-01",
                    "2017-11-08",
                ]
            ),
        }
    )


def test_clean_transactions_removes_invalid_values():
    cleaned = clean_transactions(_transactions())

    assert len(cleaned) == 4
    assert cleaned["quantity"].gt(0).all()
    assert cleaned["sales_value"].ge(0).all()
    assert cleaned["week"].between(1, 53).all()


def test_week_split_uses_required_boundaries():
    train, valid, test = week_based_split(_transactions())

    assert set(train["week"]) == {1, 40}
    assert set(valid["week"]) == {41}
    assert test.empty


def test_catalog_selection_is_learned_from_train_only():
    train, valid, _ = week_based_split(_transactions())
    selected = select_products_from_train(train, min_product_purchases=2)

    assert list(selected) == ["p1"]
    assert apply_product_catalog(valid, selected).empty
    assert "future_only" not in selected


def test_household_category_preferences_use_given_history_only():
    train, _, _ = week_based_split(_transactions())
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "future_only"],
            "department": ["grocery", "produce", "future"],
            "product_category": ["snacks", "fruit", "future"],
        }
    )

    preferences = build_household_category_preferences(train, products)
    h1 = preferences.loc[preferences["household_id"] == "h1"].iloc[0]

    assert h1["top_department_train"] == "grocery"
    assert h1["top_category_train"] == "snacks"
    assert "future" not in set(preferences["top_department_train"])
