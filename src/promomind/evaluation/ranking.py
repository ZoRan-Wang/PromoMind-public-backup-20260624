"""Basic ranking metrics over pandas DataFrames."""

from __future__ import annotations

import math

import pandas as pd

from promomind.models.candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates


def _top_k(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be positive")
    if "rank" in recommendations.columns:
        return recommendations[recommendations["rank"] <= k].copy()
    return sort_candidates(recommendations, k=k)


def _truth_sets(ground_truth: pd.DataFrame, user_col: str, item_col: str) -> dict[object, set]:
    return ground_truth.groupby(user_col)[item_col].apply(set).to_dict()


def recall_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
) -> float:
    """Mean per-user Recall@K."""

    recs = _top_k(recommendations.rename(columns={user_col: USER_COL, item_col: ITEM_COL}), k)
    truth = _truth_sets(ground_truth, user_col, item_col)
    recalls = []
    for user, relevant in truth.items():
        if not relevant:
            continue
        recommended = set(recs.loc[recs[USER_COL] == user, ITEM_COL])
        recalls.append(len(recommended & relevant) / len(relevant))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def ndcg_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
) -> float:
    """Mean binary NDCG@K."""

    recs = _top_k(recommendations.rename(columns={user_col: USER_COL, item_col: ITEM_COL}), k)
    truth = _truth_sets(ground_truth, user_col, item_col)
    scores = []
    for user, relevant in truth.items():
        if not relevant:
            continue
        user_recs = recs.loc[recs[USER_COL] == user].sort_values("rank")
        dcg = 0.0
        for rank, item in enumerate(user_recs[ITEM_COL].head(k), start=1):
            if item in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def coverage(
    recommendations: pd.DataFrame,
    catalog_items: pd.Series | list | set,
    k: int | None = None,
    item_col: str = ITEM_COL,
) -> float:
    """Share of catalog items appearing in recommendations."""

    catalog = set(catalog_items)
    if not catalog:
        return 0.0
    recs = _top_k(recommendations, k) if k is not None else recommendations
    return len(set(recs[item_col]) & catalog) / len(catalog)


def diversity(
    recommendations: pd.DataFrame,
    item_features: pd.DataFrame,
    k: int | None = None,
    item_col: str = ITEM_COL,
    category_col: str = "category",
) -> float:
    """Average per-user category diversity as unique categories divided by list length."""

    recs = _top_k(recommendations, k) if k is not None else recommendations.copy()
    merged = recs.merge(item_features[[item_col, category_col]], on=item_col, how="left")
    values = []
    for _, group in merged.groupby(USER_COL):
        if len(group) == 0:
            continue
        values.append(group[category_col].nunique(dropna=True) / len(group))
    return float(sum(values) / len(values)) if values else 0.0


def novelty(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int | None = None,
    item_col: str = ITEM_COL,
) -> float:
    """Average self-information of recommended items based on historical popularity."""

    recs = _top_k(recommendations, k) if k is not None else recommendations.copy()
    if interactions.empty or recs.empty:
        return 0.0
    popularity = interactions[item_col].value_counts(normalize=True).to_dict()
    values = [-math.log2(popularity.get(item, 1 / len(popularity))) for item in recs[item_col]]
    return float(sum(values) / len(values)) if values else 0.0

