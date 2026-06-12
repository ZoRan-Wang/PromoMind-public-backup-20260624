"""Next-basket recommenders focused on repeat grocery behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from promomind.data import schema

from .candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates


def _top_k_from_scores(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return []
    if finite.size > k:
        selected = finite[np.argpartition(-scores[finite], k - 1)[:k]]
    else:
        selected = finite
    order = selected[np.lexsort((selected, -scores[selected]))]
    return [(int(item_idx), float(scores[item_idx])) for item_idx in order[:k]]


def _ordered_unique(values: pd.Series) -> list[Hashable]:
    return sorted(values.dropna().unique().tolist(), key=lambda value: str(value))


def _global_scores(
    interactions: pd.DataFrame,
    item_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    if weight_col and weight_col in interactions.columns:
        global_items = interactions.groupby(item_col, as_index=False)[weight_col].sum()
        return global_items.rename(columns={weight_col: SCORE_COL})
    return interactions.groupby(item_col).size().reset_index(name=SCORE_COL)


@dataclass
class PersonalTopFrequencyRecommender:
    """Recommend a household's most frequently purchased products.

    This is a strong next-basket baseline for grocery data because repeat
    consumption is a central part of the task, not leakage.
    """

    recency_weight: float = 0.01

    def __post_init__(self) -> None:
        self.user_scores_: dict[Hashable, list[tuple[Hashable, float]]] = {}
        self.global_scores_: list[tuple[Hashable, float]] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
        time_col: str | None = schema.WEEK,
    ) -> "PersonalTopFrequencyRecommender":
        """Fit per-household frequency tables with a small recency tie-breaker."""

        required = [user_col, item_col]
        missing = [col for col in required if col not in interactions.columns]
        if missing:
            raise ValueError(f"interactions is missing required columns: {missing}")

        work = interactions.dropna(subset=[user_col, item_col]).copy()
        if weight_col and weight_col in work.columns:
            work["_weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
        else:
            work["_weight"] = 1.0

        aggregations: dict[str, tuple[str, str]] = {"frequency_score": ("_weight", "sum")}
        if time_col and time_col in work.columns:
            work["_time"] = pd.to_numeric(work[time_col], errors="coerce").fillna(0.0)
            aggregations["last_time"] = ("_time", "max")
        else:
            work["_time"] = 0.0
            aggregations["last_time"] = ("_time", "max")

        scores = work.groupby([user_col, item_col], as_index=False).agg(**aggregations)
        scores[SCORE_COL] = scores["frequency_score"] + self.recency_weight * scores["last_time"]
        scores = scores.sort_values([user_col, SCORE_COL, item_col], ascending=[True, False, True])

        self.user_scores_ = {
            user: list(zip(group[item_col], group[SCORE_COL], strict=False))
            for user, group in scores.groupby(user_col, sort=False)
        }

        global_scores = _global_scores(work, item_col=item_col, weight_col="_weight")
        global_scores = global_scores.sort_values([SCORE_COL, item_col], ascending=[False, True])
        self.global_scores_ = list(zip(global_scores[item_col], global_scores[SCORE_COL], strict=False))
        return self

    def recommend(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        exclude_seen: bool = False,
    ) -> pd.DataFrame:
        """Return top-k personal-frequency candidates."""

        if k <= 0:
            raise ValueError("k must be positive")

        records: list[dict[str, object]] = []
        for user in users:
            added_items: set[Hashable] = set()
            user_items = self.user_scores_.get(user, [])
            seen_items = {item for item, _ in user_items}
            for item, score in user_items:
                if exclude_seen:
                    continue
                records.append({USER_COL: user, ITEM_COL: item, SCORE_COL: float(score)})
                added_items.add(item)
                if len(added_items) >= k:
                    break

            if len(added_items) < k:
                for item, score in self.global_scores_:
                    if item in added_items or (exclude_seen and item in seen_items):
                        continue
                    records.append({USER_COL: user, ITEM_COL: item, SCORE_COL: float(score) * 1e-6})
                    added_items.add(item)
                    if len(added_items) >= k:
                        break

        return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)


@dataclass
class TIFUKNNRecommender:
    """TIFU-KNN style next-basket model.

    The model builds a time-decayed personalized item-frequency matrix, finds
    nearest households by cosine similarity, and combines personal frequency
    with neighbor frequency. It follows the practical TIFU-KNN idea while using
    the repository's household-week-basket schema.
    """

    n_neighbors: int = 100
    alpha: float = 0.7
    basket_decay: float = 0.95

    def __post_init__(self) -> None:
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < self.basket_decay <= 1:
            raise ValueError("basket_decay must be in (0, 1]")

        self.users_: list[Hashable] = []
        self.items_: list[Hashable] = []
        self.user_index_: dict[Hashable, int] = {}
        self.item_index_: dict[Hashable, int] = {}
        self.score_matrix_: sparse.csr_matrix | None = None
        self.global_scores_: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
        basket_col: str | None = schema.BASKET_ID,
        time_col: str | None = schema.WEEK,
    ) -> "TIFUKNNRecommender":
        """Fit the time-decayed personal plus neighbor item-frequency scores."""

        required = [user_col, item_col]
        missing = [col for col in required if col not in interactions.columns]
        if missing:
            raise ValueError(f"interactions is missing required columns: {missing}")

        work = interactions.dropna(subset=[user_col, item_col]).copy()
        if work.empty:
            self.score_matrix_ = sparse.csr_matrix((0, 0), dtype=np.float32)
            self.global_scores_ = np.array([], dtype=np.float32)
            return self

        if weight_col and weight_col in work.columns:
            work["_weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
        else:
            work["_weight"] = 1.0

        self.users_ = _ordered_unique(work[user_col])
        self.items_ = _ordered_unique(work[item_col])
        self.user_index_ = {user: idx for idx, user in enumerate(self.users_)}
        self.item_index_ = {item: idx for idx, item in enumerate(self.items_)}

        pif = self._build_time_decayed_matrix(
            work,
            user_col=user_col,
            item_col=item_col,
            basket_col=basket_col,
            time_col=time_col,
        )
        neighbor_scores = self._build_neighbor_scores(pif)
        self.score_matrix_ = (pif.multiply(self.alpha) + neighbor_scores.multiply(1 - self.alpha)).tocsr()
        self.global_scores_ = np.asarray(pif.sum(axis=0)).ravel().astype(np.float32)
        return self

    def _build_time_decayed_matrix(
        self,
        work: pd.DataFrame,
        user_col: str,
        item_col: str,
        basket_col: str | None,
        time_col: str | None,
    ) -> sparse.csr_matrix:
        order_cols = [user_col]
        if time_col and time_col in work.columns:
            work["_time"] = pd.to_numeric(work[time_col], errors="coerce").fillna(0.0)
            order_cols.append("_time")
        else:
            work["_time"] = 0.0
            order_cols.append("_time")

        if basket_col and basket_col in work.columns:
            order_cols.append(basket_col)
            basket_keys = [user_col, basket_col, "_time"]
        else:
            work["_basket_proxy"] = np.arange(len(work))
            order_cols.append("_basket_proxy")
            basket_keys = [user_col, "_basket_proxy", "_time"]

        baskets = work[basket_keys].drop_duplicates().sort_values(order_cols)
        baskets["_basket_pos"] = baskets.groupby(user_col).cumcount()
        baskets["_n_baskets"] = baskets.groupby(user_col)["_basket_pos"].transform("max") + 1
        work = work.merge(baskets, on=basket_keys, how="left")
        work["_baskets_ago"] = work["_n_baskets"] - 1 - work["_basket_pos"]
        work["_decayed_weight"] = work["_weight"] * np.power(
            self.basket_decay,
            work["_baskets_ago"].to_numpy(dtype=np.float32),
        )

        aggregated = (
            work.groupby([user_col, item_col], observed=True, as_index=False)["_decayed_weight"]
            .sum()
            .reset_index(drop=True)
        )
        rows = aggregated[user_col].map(self.user_index_).to_numpy(dtype=np.int64)
        cols = aggregated[item_col].map(self.item_index_).to_numpy(dtype=np.int64)
        data = aggregated["_decayed_weight"].to_numpy(dtype=np.float32)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.users_), len(self.items_)))
        matrix.sum_duplicates()
        return matrix

    def _build_neighbor_scores(self, pif: sparse.csr_matrix) -> sparse.csr_matrix:
        if pif.shape[0] == 0:
            return pif.copy()

        norms = np.sqrt(pif.multiply(pif).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        normalized = pif.multiply(1.0 / norms[:, None]).tocsr()
        similarity = (normalized @ normalized.T).toarray().astype(np.float32)
        np.fill_diagonal(similarity, 0.0)

        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for user_idx in range(similarity.shape[0]):
            row = similarity[user_idx]
            if self.n_neighbors < len(row):
                selected = np.argpartition(-row, self.n_neighbors)[: self.n_neighbors]
            else:
                selected = np.arange(len(row))
            selected = selected[row[selected] > 0]
            selected = selected[np.argsort(-row[selected], kind="mergesort")]
            data.extend(row[selected].tolist())
            indices.extend(selected.tolist())
            indptr.append(len(data))

        neighbors = sparse.csr_matrix((data, indices, indptr), shape=similarity.shape)
        neighbor_scores = neighbors @ pif
        denom = np.asarray(neighbors.sum(axis=1)).ravel()
        denom[denom == 0] = 1.0
        return neighbor_scores.multiply(1.0 / denom[:, None]).tocsr()

    def recommend(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        exclude_seen: bool = False,
    ) -> pd.DataFrame:
        """Return top-k TIFU-KNN candidates."""

        if k <= 0:
            raise ValueError("k must be positive")
        if self.score_matrix_ is None or self.global_scores_ is None:
            raise RuntimeError("Call fit() before recommend().")

        records: list[dict[str, object]] = []
        for user in users:
            if user not in self.user_index_:
                scores = self.global_scores_.copy() * 1e-6
                source_row = None
            else:
                user_idx = self.user_index_[user]
                scores = self.score_matrix_.getrow(user_idx).toarray().ravel().astype(np.float32)
                source_row = self.score_matrix_.getrow(user_idx)
            if exclude_seen and source_row is not None:
                scores[source_row.indices] = -np.inf

            for item_idx, score in _top_k_from_scores(scores, k):
                if score <= 0:
                    continue
                records.append(
                    {
                        USER_COL: user,
                        ITEM_COL: self.items_[item_idx],
                        SCORE_COL: score,
                    }
                )

        return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)
