"""Rule-based explanations for demo recommendations."""

from __future__ import annotations

import pandas as pd


def build_recommendation_reason(row: pd.Series) -> str:
    """Return a concise human-readable reason for one recommendation row."""
    reasons: list[str] = []

    if bool(row.get("is_top_household_category", False)):
        reasons.append("high-frequency household category")
    if bool(row.get("has_promotion", False)) or row.get("promo_score", 0) > 0:
        reasons.append("current promotion signal")
    if bool(row.get("recommend_coupon", False)) or row.get("coupon_score", 0) > 0:
        reasons.append("coupon candidate")
    if row.get("cooccurrence_score", 0) > 0:
        reasons.append("co-occurs with past basket items")

    if not reasons:
        return "high model score based on household purchase history"

    return "; ".join(reasons)


def attach_reasons(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Attach a `reason` column to a recommendation DataFrame."""
    if recommendations.empty:
        result = recommendations.copy()
        result["reason"] = []
        return result

    result = recommendations.copy()
    result["reason"] = result.apply(build_recommendation_reason, axis=1)
    return result

