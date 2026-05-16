"""Promotion-aware reranking for candidate recommendation lists."""

from __future__ import annotations

import pandas as pd

from promomind.models.candidates import ITEM_COL, SCORE_COL, USER_COL, validate_candidates


def promotion_aware_rerank(
    candidates: pd.DataFrame,
    item_features: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    k: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.2,
    lambda_discount: float = 0.1,
    rho: float = 0.05,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
    score_col: str = SCORE_COL,
    category_col: str = "category",
    promotion_col: str = "promotion_score",
    coupon_col: str = "coupon_score",
    discount_cost_col: str = "discount_cost",
) -> pd.DataFrame:
    """Rerank candidates with promotion, coupon, cost, and diversity signals.

    Formula:
        ``final_score = alpha * score + beta * promotion_score
        + gamma * coupon_score - lambda_discount * discount_cost
        + rho * diversity_bonus``

    ``diversity_bonus`` is assigned greedily within each user list: an item gets
    1.0 when its category has not appeared earlier in that user's base-score
    ordering, else 0.0. Missing feature columns default to zero.
    """

    if k is not None and k <= 0:
        raise ValueError("k must be positive when provided")

    schema_candidates = candidates.rename(
        columns={user_col: USER_COL, item_col: ITEM_COL, score_col: SCORE_COL}
    )
    reranked = validate_candidates(schema_candidates)

    if item_features is not None:
        feature_cols = [item_col] + [
            col
            for col in [category_col, promotion_col, coupon_col, discount_cost_col]
            if col in item_features.columns
        ]
        reranked = reranked.merge(
            item_features[feature_cols].drop_duplicates(subset=[item_col]).rename(
                columns={item_col: ITEM_COL}
            ),
            on=ITEM_COL,
            how="left",
        )

    if user_features is not None:
        feature_cols = [user_col] + [
            col
            for col in [promotion_col, coupon_col, discount_cost_col]
            if col in user_features.columns and col not in reranked.columns
        ]
        if len(feature_cols) > 1:
            reranked = reranked.merge(
                user_features[feature_cols].drop_duplicates(subset=[user_col]).rename(
                    columns={user_col: USER_COL}
                ),
                on=USER_COL,
                how="left",
            )

    for col in [promotion_col, coupon_col, discount_cost_col]:
        if col not in reranked.columns:
            reranked[col] = 0.0
        reranked[col] = pd.to_numeric(reranked[col], errors="coerce").fillna(0.0)

    if category_col not in reranked.columns:
        reranked[category_col] = reranked[ITEM_COL]

    base_ordered = reranked.sort_values([USER_COL, SCORE_COL, ITEM_COL], ascending=[True, False, True])
    seen_categories: dict[object, set] = {}
    diversity_bonus = []
    for row in base_ordered[[USER_COL, category_col]].itertuples(index=False):
        seen = seen_categories.setdefault(row.user_id, set())
        bonus = 0.0 if row[1] in seen else 1.0
        seen.add(row[1])
        diversity_bonus.append(bonus)
    base_ordered["diversity_bonus"] = diversity_bonus

    reranked = base_ordered
    reranked["final_score"] = (
        alpha * reranked[SCORE_COL]
        + beta * reranked[promotion_col]
        + gamma * reranked[coupon_col]
        - lambda_discount * reranked[discount_cost_col]
        + rho * reranked["diversity_bonus"]
    )
    reranked = reranked.sort_values(
        [USER_COL, "final_score", SCORE_COL, ITEM_COL],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    reranked["rank"] = reranked.groupby(USER_COL).cumcount() + 1
    if k is not None:
        reranked = reranked[reranked["rank"] <= k]

    return reranked.reset_index(drop=True)

