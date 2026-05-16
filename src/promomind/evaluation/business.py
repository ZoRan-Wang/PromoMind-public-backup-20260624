"""Business-facing recommendation metrics."""

from __future__ import annotations

import pandas as pd

from promomind.models.candidates import ITEM_COL, USER_COL
from promomind.evaluation.ranking import _top_k


def business_utility_at_k(
    recommendations: pd.DataFrame,
    item_values: pd.DataFrame | None = None,
    ground_truth: pd.DataFrame | None = None,
    k: int = 10,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
    revenue_col: str = "expected_revenue",
    discount_col: str = "discount_cost",
) -> float:
    """Average top-k business utility using revenue minus discount cost.

    ``item_values`` can provide item-level revenue/cost columns. If the columns
    already exist on ``recommendations``, pass ``item_values=None``.

    When ``ground_truth`` is provided, utility is counted only for recommended
    items that were actually purchased by that user in the evaluation window.
    Without ground truth, the function returns the average utility of the top-k
    recommendation list itself, which is useful for demo scenarios.
    """

    recs = _top_k(recommendations.rename(columns={user_col: USER_COL, item_col: ITEM_COL}), k)
    if item_values is not None:
        recs = recs.merge(
            item_values[[item_col, revenue_col, discount_col]].rename(columns={item_col: ITEM_COL}),
            on=ITEM_COL,
            how="left",
            suffixes=("", "_item"),
        )

    for col in [revenue_col, discount_col]:
        if col not in recs.columns:
            recs[col] = 0.0
        recs[col] = pd.to_numeric(recs[col], errors="coerce").fillna(0.0)

    if ground_truth is not None:
        truth = ground_truth[[user_col, item_col]].rename(
            columns={user_col: USER_COL, item_col: ITEM_COL}
        )
        truth = truth.drop_duplicates()
        recs = recs.merge(truth.assign(_hit=True), on=[USER_COL, ITEM_COL], how="left")
        recs = recs[recs["_hit"].fillna(False)]

    if recs.empty:
        return 0.0
    per_row = recs[revenue_col] - recs[discount_col]
    per_user = per_row.groupby(recs[USER_COL]).sum()
    return float(per_user.mean()) if not per_user.empty else 0.0
