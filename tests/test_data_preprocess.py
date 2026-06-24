from __future__ import annotations

import pandas as pd

from promomind.data.preprocess import (
    add_dataset_weeks,
    apply_product_catalog,
    build_discount_cost_features,
    build_household_category_preferences,
    build_household_product_coupon_features,
    build_product_week_promotion_features,
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


def test_dataset_week_conversion_uses_complete_journey_origin():
    dated = pd.DataFrame(
        {"redemption_date": ["2017-01-01", "2017-01-08", "2017-10-02"]}
    )

    result = add_dataset_weeks(dated, "redemption_date")

    assert result["week"].tolist() == [1, 2, 40]


def test_promotion_features_aggregate_store_exposure():
    promotions = pd.DataFrame(
        {
            "product_id": ["p1", "p1", "p2"],
            "store_id": ["s1", "s2", "s1"],
            "week": [41, 41, 41],
            "display_location": ["1", "0", "0"],
            "mailer_location": ["0", "A", "0"],
        }
    )

    result = build_product_week_promotion_features(promotions)
    p1 = result.loc[result["product_id"] == "p1"].iloc[0]

    assert bool(p1["has_display"])
    assert bool(p1["has_mailer"])
    assert p1["promotion_source_count"] == 2
    assert p1["promotion_score"] == 0.5


def test_coupon_features_use_train_affinity_and_target_weeks_only():
    train = _transactions().query("week <= 40 and quantity > 0 and sales_value >= 0")
    coupons = pd.DataFrame(
        {
            "coupon_upc": ["c1", "c2"],
            "product_id": ["p1", "future_only"],
            "campaign_id": ["campaign", "campaign"],
        }
    )
    campaigns = pd.DataFrame(
        {"campaign_id": ["campaign"], "household_id": ["h1"]}
    )
    descriptions = pd.DataFrame(
        {
            "campaign_id": ["campaign"],
            "start_date": ["2017-10-02"],
            "end_date": ["2017-10-15"],
        }
    )
    redemptions = pd.DataFrame(
        {
            "household_id": ["h1"],
            "coupon_upc": ["old"],
            "campaign_id": ["old_campaign"],
            "redemption_date": ["2017-03-01"],
        }
    )

    result = build_household_product_coupon_features(
        train,
        coupons,
        campaigns,
        descriptions,
        redemptions,
    )

    assert set(result["product_id"]) == {"p1"}
    assert set(result["week"]) == {41, 42}


def test_discount_features_are_product_level_train_aggregates():
    train = _transactions().query("week <= 40 and quantity > 0 and sales_value >= 0").copy()
    train["retail_disc"] = [1.0, 0.0, 2.0]
    train["coupon_disc"] = 0.0
    train["coupon_match_disc"] = 0.0
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category": ["snacks", "fruit"],
        }
    )

    result = build_discount_cost_features(train, products)

    assert set(result["product_id"]) == {"p1", "p2"}
    assert result["product_id"].is_unique
    assert result["estimated_discount_cost"].ge(0).all()
    assert result["avg_category_discount_train"].notna().all()
