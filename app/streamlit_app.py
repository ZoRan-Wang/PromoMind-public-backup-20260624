"""Streamlit demo shell for PromoMind.

Run from the repository root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_demo_recommendations() -> pd.DataFrame:
    for filename in [
        "demo_recommendations.csv",
        "reranked_als.csv",
        "reranked_recommendations.csv",
    ]:
        data = load_csv(OUTPUTS / filename)
        if not data.empty:
            data = data.rename(
                columns={
                    "user_id": "household_id",
                    "item_id": "product_id",
                    "score": "base_score",
                    "rank": "final_rank",
                    "discount_cost": "discount_cost_proxy",
                }
            )
            return data
    return pd.DataFrame()


def load_coupon_timing_recommendations() -> pd.DataFrame:
    for filename in ["reranked_recommendations.csv", "demo_time_name_recommendations.csv"]:
        data = load_csv(OUTPUTS / filename)
        if data.empty:
            continue
        if "final_rank" in data.columns and "rank" not in data.columns:
            data["rank"] = data["final_rank"]
        if "model_name" in data.columns and "source_model" not in data.columns:
            data["source_model"] = data["model_name"]
        if "coupon_start_date" in data.columns and "campaign_id" in data.columns:
            return data
    return pd.DataFrame()


def render_coupon_timing_demo(
    recommendations: pd.DataFrame,
    history_input: pd.DataFrame,
) -> None:
    household_ids = sorted(recommendations["household_id"].dropna().unique().tolist())
    selected_household = st.sidebar.selectbox("Household", household_ids)

    household_recs = recommendations[recommendations["household_id"] == selected_household].copy()
    campaign_ids = sorted(household_recs["campaign_id"].dropna().unique().tolist())
    selected_campaign = st.sidebar.selectbox("Campaign", campaign_ids)
    coupon_slots = st.sidebar.slider("Coupon slots", 0, 10, 3)

    event_recs = household_recs[household_recs["campaign_id"] == selected_campaign].copy()
    event_recs = event_recs.sort_values("rank")
    if event_recs.empty:
        st.warning("No recommendations found for this household and campaign.")
        return

    if {"household_id", "campaign_id"}.issubset(history_input.columns):
        event_history = history_input[
            (history_input["household_id"] == selected_household)
            & (history_input["campaign_id"] == selected_campaign)
        ].copy()
        if "purchase_time" in event_history.columns:
            event_history = event_history.sort_values("purchase_time")
    else:
        event_history = pd.DataFrame()

    coupon_start = event_recs["coupon_start_date"].iloc[0] if "coupon_start_date" in event_recs.columns else "-"
    predicted_time = (
        event_recs["predicted_purchase_time"].iloc[0]
        if "predicted_purchase_time" in event_recs.columns
        else "-"
    )
    baskets_before = (
        int(event_recs["baskets_before_coupon"].iloc[0])
        if "baskets_before_coupon" in event_recs and not event_recs.empty
        else 0
    )

    metric_left, metric_mid, metric_right = st.columns(3)
    metric_left.metric("Coupon start", str(coupon_start))
    metric_mid.metric("Prediction time", str(predicted_time))
    metric_right.metric("Prior baskets", baskets_before)

    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Input: Historical Purchases")
        history_columns = [
            column
            for column in [
                "purchase_time",
                "basket_id",
                "product_id",
                "product_name",
                "department",
                "product_category",
            ]
            if column in event_history.columns
        ]
        st.dataframe(event_history.tail(25)[history_columns], use_container_width=True)

    with right:
        st.subheader("Output: Time-Product Pairs")
        display = event_recs.copy()
        if "coupon_recommended" not in display.columns:
            display["coupon_recommended"] = False
        display.loc[display.head(coupon_slots).index, "coupon_recommended"] = True
        output_columns = [
            column
            for column in [
                "predicted_purchase_time",
                "rank",
                "product_id",
                "product_name",
                "coupon_recommended",
                "recommend_coupon",
                "coupon_eligible",
                "success_within_5d_observed",
                "observed_purchase_time",
                "source_model",
            ]
            if column in display.columns
        ]
        st.dataframe(display[output_columns], use_container_width=True)

    st.caption(
        "Training label: success = exposed household bought the campaign coupon product "
        "within 5 days after campaign start. Model output comes from the external TBP/TARS next-basket code."
    )


def main() -> None:
    st.set_page_config(page_title="PromoMind", layout="wide")
    st.title("PromoMind")
    st.caption("Promotion-aware grocery basket and coupon recommender")

    household_features = load_csv(PROCESSED / "household_features.csv")
    product_features = load_csv(PROCESSED / "product_features.csv")
    timing_recommendations = load_coupon_timing_recommendations()
    timing_history = load_csv(OUTPUTS / "demo_history_input.csv")

    if not timing_recommendations.empty:
        render_coupon_timing_demo(timing_recommendations, timing_history)
        return

    recommendations = load_demo_recommendations()

    if recommendations.empty:
        st.info(
            "No recommendation output found yet. Generate sample data and model outputs, "
            "then place reranked recommendations in outputs/demo_recommendations.csv."
        )
        return

    household_ids = sorted(recommendations["household_id"].dropna().unique().tolist())
    selected_household = st.sidebar.selectbox("Household", household_ids)
    budget = st.sidebar.slider("Marketing budget level", 0, 100, 30)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Household Profile")
        if "household_id" not in household_features.columns:
            profile = pd.DataFrame()
        else:
            profile = household_features[
                household_features["household_id"] == selected_household
            ]
        if profile.empty:
            st.write("Profile not available.")
        else:
            st.dataframe(profile.T, use_container_width=True)

    household_recs = recommendations[
        recommendations["household_id"] == selected_household
    ].copy()

    if "final_rank" in household_recs.columns:
        household_recs = household_recs.sort_values("final_rank")
    elif "rank" in household_recs.columns:
        household_recs = household_recs.sort_values("rank")

    if (
        not product_features.empty
        and "product_id" in household_recs.columns
        and "product_id" in product_features.columns
    ):
        household_recs = household_recs.merge(product_features, on="product_id", how="left")

    coupon_limit = max(1, round(budget / 10))
    if "recommend_coupon" not in household_recs.columns:
        household_recs["recommend_coupon"] = False
    household_recs.loc[household_recs.head(coupon_limit).index, "recommend_coupon"] = True

    with right:
        st.subheader("Top Recommendations")
        display_columns = [
            column
            for column in [
                "product_id",
                "department",
                "product_category",
                "brand",
                "base_score",
                "final_score",
                "recommend_coupon",
                "reason",
            ]
            if column in household_recs.columns
        ]
        st.dataframe(household_recs.head(10)[display_columns], use_container_width=True)

    kpi_left, kpi_mid, kpi_right = st.columns(3)
    estimated_sales = household_recs.head(10).get("estimated_sales_value", pd.Series(dtype=float)).sum()
    estimated_discount = household_recs.head(10).get(
        "discount_cost_proxy", pd.Series(dtype=float)
    ).sum()
    kpi_left.metric("Estimated sales", f"{estimated_sales:,.2f}")
    kpi_mid.metric("Estimated discount cost", f"{estimated_discount:,.2f}")
    kpi_right.metric("Business utility proxy", f"{estimated_sales - estimated_discount:,.2f}")


if __name__ == "__main__":
    main()
