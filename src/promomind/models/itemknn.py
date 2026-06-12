"""Item-based k-nearest-neighbor recommender for implicit grocery baskets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates
from .sparse import InteractionMatrix, build_interaction_matrix


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return matrix.multiply(1.0 / norms[:, None]).tocsr()


def _keep_top_n_by_row(matrix: sparse.csr_matrix, n: int) -> sparse.csr_matrix:
    if n <= 0:
        raise ValueError("n must be positive")

    matrix = matrix.tocsr()
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]

    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        if len(row_data) > n:
            keep = np.argpartition(-row_data, n - 1)[:n]
            order = keep[np.argsort(-row_data[keep], kind="mergesort")]
            row_data = row_data[order]
            row_indices = row_indices[order]
        else:
            order = np.argsort(-row_data, kind="mergesort")
            row_data = row_data[order]
            row_indices = row_indices[order]

        data.extend(row_data.tolist())
        indices.extend(row_indices.tolist())
        indptr.append(len(data))

    return sparse.csr_matrix((data, indices, indptr), shape=matrix.shape)


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


@dataclass
class ItemKNNRecommender:
    """Recommend items similar to a household's previously purchased items."""

    max_similar_items: int = 100
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.matrix_: InteractionMatrix | None = None
        self.similarity_: sparse.csr_matrix | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
    ) -> "ItemKNNRecommender":
        """Fit item-item cosine similarities from an implicit user-item graph."""

        self.matrix_ = build_interaction_matrix(
            interactions,
            user_col=user_col,
            item_col=item_col,
            weight_col=weight_col,
            alpha=self.alpha,
        )
        if self.matrix_.n_items == 0:
            self.similarity_ = sparse.csr_matrix((0, 0), dtype=np.float64)
            return self

        item_vectors = _row_normalize(self.matrix_.item_users)
        similarity = (item_vectors @ item_vectors.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity_ = _keep_top_n_by_row(similarity, self.max_similar_items)
        return self

    def recommend(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return ``user_id,item_id,score,rank`` recommendations."""

        if k <= 0:
            raise ValueError("k must be positive")
        if self.matrix_ is None or self.similarity_ is None:
            raise RuntimeError("Call fit() before recommend().")

        records: list[dict[str, object]] = []
        for user in users:
            if user not in self.matrix_.user_index:
                continue
            user_idx = self.matrix_.user_index[user]
            raw_scores = self.matrix_.user_items[user_idx] @ self.similarity_
            scores = np.asarray(raw_scores.toarray()).ravel()
            if exclude_seen:
                seen = self.matrix_.user_items[user_idx].indices
                scores[seen] = -np.inf

            for item_idx, score in _top_k_from_scores(scores, k):
                records.append(
                    {
                        USER_COL: user,
                        ITEM_COL: self.matrix_.index_item[item_idx],
                        SCORE_COL: score,
                    }
                )

        return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)
