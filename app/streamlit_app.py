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
            return data
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="PromoMind", layout="wide")
    st.title("PromoMind")
    st.caption("Promotion-aware grocery basket and coupon recommender")

    household_features = load_csv(PROCESSED / "household_features.csv")
    product_features = load_csv(PROCESSED / "product_features.csv")
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

    if not product_features.empty and "product_id" in household_recs.columns:
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

