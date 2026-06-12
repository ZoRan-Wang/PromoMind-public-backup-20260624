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
    popularity = popularity.sort_values([SCORE_COL, ITEM_COL], ascending=[False, True])
    popular_rows = list(popularity.itertuples(index=False, name=None))

    if exclude_seen:
        seen_by_user = (
            interactions[[user_col, item_col]]
            .drop_duplicates()
            .groupby(user_col)[item_col]
            .apply(set)
            .to_dict()
        )
    else:
        seen_by_user = {}

    records: list[dict[str, object]] = []
    for user in user_values:
        seen = seen_by_user.get(user, set())
        added = 0
        for item, score in popular_rows:
            if item in seen:
                continue
            records.append({USER_COL: user, ITEM_COL: item, SCORE_COL: score})
            added += 1
            if added >= k:
                break

    return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)


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

    # Grocery category metadata can be sparse or too granular. If a household's
    # preferred categories do not leave enough unseen products, fill from global
    # popularity so the baseline remains evaluable for every warm user.
    fallback = popularity_recommendations(
        interactions,
        users=user_values,
        k=k,
        user_col=user_col,
        item_col=item_col,
        weight_col=weight_col,
        exclude_seen=exclude_seen,
    ).drop(columns=["rank"], errors="ignore")
    fallback[category_col] = "global_fallback"
    if not fallback.empty:
        positive_primary = candidates[SCORE_COL][candidates[SCORE_COL] > 0]
        if not positive_primary.empty:
            fallback[SCORE_COL] = fallback[SCORE_COL] * positive_primary.min() * 1e-6
        else:
            fallback[SCORE_COL] = fallback[SCORE_COL] * 1e-6

    combined = pd.concat([candidates, fallback], ignore_index=True)
    combined = combined.drop_duplicates([USER_COL, ITEM_COL], keep="first")
    return sort_candidates(combined, k=k)
