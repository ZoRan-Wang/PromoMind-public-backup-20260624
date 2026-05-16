"""Simple popularity baselines for PromoMind.

These implementations are intentionally transparent and DataFrame-oriented so
they can serve as assignment baselines before heavier recommenders are wired in.
"""

from __future__ import annotations

import pandas as pd

from .candidates import ITEM_COL, SCORE_COL, USER_COL, ensure_columns, sort_candidates


def _item_popularity(
    interactions: pd.DataFrame,
    item_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    if weight_col:
        ensure_columns(interactions, [item_col, weight_col], "interactions")
        popularity = interactions.groupby(item_col, as_index=False)[weight_col].sum()
        popularity = popularity.rename(columns={weight_col: SCORE_COL})
    else:
        ensure_columns(interactions, [item_col], "interactions")
        popularity = interactions.groupby(item_col).size().reset_index(name=SCORE_COL)

    return popularity.rename(columns={item_col: ITEM_COL})


def popularity_recommendations(
    interactions: pd.DataFrame,
    users: pd.Series | list | None = None,
    k: int = 10,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
    weight_col: str | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Recommend globally popular items to each user.

    Args:
        interactions: Historical rows containing at least user and item columns.
        users: Optional user IDs to score. Defaults to users seen in interactions.
        k: Number of recommendations per user.
        user_col: Name of the user column in ``interactions``.
        item_col: Name of the item column in ``interactions``.
        weight_col: Optional column to sum instead of using row counts.
        exclude_seen: Remove items already observed for each user.

    Returns:
        DataFrame with ``user_id``, ``item_id``, ``score``, and ``rank``.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    ensure_columns(interactions, [user_col, item_col], "interactions")

    user_values = pd.Series(users if users is not None else interactions[user_col].dropna().unique())
    popularity = _item_popularity(interactions, item_col=item_col, weight_col=weight_col)

    candidates = user_values.rename(USER_COL).to_frame().merge(popularity, how="cross")

    if exclude_seen:
        seen = interactions[[user_col, item_col]].drop_duplicates().rename(
            columns={user_col: USER_COL, item_col: ITEM_COL}
        )
        candidates = candidates.merge(seen.assign(_seen=True), on=[USER_COL, ITEM_COL], how="left")
        candidates = candidates[candidates["_seen"].isna()].drop(columns="_seen")

    return sort_candidates(candidates, k=k)


def category_popularity_recommendations(
    interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    users: pd.Series | list | None = None,
    k: int = 10,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
    category_col: str = "category",
    weight_col: str | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Recommend popular items weighted by each user's historical categories.

    The score is ``item_popularity * user_category_affinity``. Users without
    category history fall back to global category counts.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    ensure_columns(interactions, [user_col, item_col], "interactions")
    ensure_columns(item_features, [item_col, category_col], "item_features")

    features = item_features[[item_col, category_col]].drop_duplicates().rename(
        columns={item_col: ITEM_COL}
    )
    item_pop = _item_popularity(interactions, item_col=item_col, weight_col=weight_col)
    item_pool = features.merge(item_pop, on=ITEM_COL, how="left")
    item_pool[SCORE_COL] = item_pool[SCORE_COL].fillna(0.0)

    enriched = interactions[[user_col, item_col]].merge(
        item_features[[item_col, category_col]], on=item_col, how="left"
    )
    category_counts = (
        enriched.dropna(subset=[category_col])
        .groupby([user_col, category_col])
        .size()
        .reset_index(name="category_affinity")
        .rename(columns={user_col: USER_COL})
    )
    global_category = (
        enriched.dropna(subset=[category_col])
        .groupby(category_col)
        .size()
        .reset_index(name="category_affinity")
    )

    user_values = pd.Series(users if users is not None else interactions[user_col].dropna().unique())
    all_users = user_values.rename(USER_COL).to_frame()

    candidates = all_users.merge(category_counts, on=USER_COL, how="left")
    cold_users = candidates[candidates["category_affinity"].isna()][[USER_COL]].drop_duplicates()
    warm_candidates = candidates.dropna(subset=["category_affinity"])
    cold_candidates = cold_users.merge(global_category, how="cross")
    category_candidates = pd.concat([warm_candidates, cold_candidates], ignore_index=True)

    candidates = category_candidates.merge(item_pool, on=category_col, how="inner")
    candidates[SCORE_COL] = candidates[SCORE_COL] * candidates["category_affinity"]
    candidates = candidates[[USER_COL, ITEM_COL, SCORE_COL, category_col]]

    if exclude_seen:
        seen = interactions[[user_col, item_col]].drop_duplicates().rename(
            columns={user_col: USER_COL, item_col: ITEM_COL}
        )
        candidates = candidates.merge(seen.assign(_seen=True), on=[USER_COL, ITEM_COL], how="left")
        candidates = candidates[candidates["_seen"].isna()].drop(columns="_seen")

    return sort_candidates(candidates, k=k)

