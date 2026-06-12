"""Implicit-feedback ALS recommender for PromoMind candidate generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd

from .candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates
from .sparse import InteractionMatrix, build_interaction_matrix


def _optional_implicit_als():
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        return None
    return AlternatingLeastSquares


def _blas_thread_context():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(1, "blas")


def _solve_regularized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


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
class ImplicitALSRecommender:
    """Small ALS adapter with an ``implicit`` backend and scipy fallback.

    The native fallback implements the Hu-Koren-Volinsky implicit ALS update
    directly. It is slower than the optional ``implicit`` package, but it keeps
    the project runnable in a clean course environment.
    """

    factors: int = 64
    regularization: float = 0.05
    iterations: int = 20
    alpha: float = 20.0
    random_state: int = 42
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.factors <= 0:
            raise ValueError("factors must be positive")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.backend not in {"auto", "implicit", "native"}:
            raise ValueError("backend must be one of: auto, implicit, native")

        self.matrix_: InteractionMatrix | None = None
        self.model = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.backend_: str | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
    ) -> "ImplicitALSRecommender":
        """Fit ALS on implicit user-item feedback."""

        self.matrix_ = build_interaction_matrix(
            interactions,
            user_col=user_col,
            item_col=item_col,
            weight_col=weight_col,
            alpha=self.alpha,
        )
        if self.matrix_.n_users == 0 or self.matrix_.n_items == 0:
            self.backend_ = "native"
            self.user_factors_ = np.empty((self.matrix_.n_users, self.factors))
            self.item_factors_ = np.empty((self.matrix_.n_items, self.factors))
            return self

        AlternatingLeastSquares = _optional_implicit_als()
        if self.backend in {"auto", "implicit"} and AlternatingLeastSquares is not None:
            self.backend_ = "implicit"
            with _blas_thread_context():
                self.model = AlternatingLeastSquares(
                    factors=self.factors,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    random_state=self.random_state,
                )
                self.model.fit(self.matrix_.user_items, show_progress=False)
            return self

        if self.backend == "implicit":
            raise ImportError(
                "backend='implicit' requires the optional implicit package. "
                "Install with `python -m pip install implicit`, or use backend='native'."
            )

        self.backend_ = "native"
        self._fit_native()
        return self

    def _fit_native(self) -> None:
        if self.matrix_ is None:
            raise RuntimeError("No interaction matrix is available.")

        rng = np.random.default_rng(self.random_state)
        user_items = self.matrix_.user_items.tocsr().astype(np.float64)
        item_users = self.matrix_.item_users.tocsr().astype(np.float64)
        n_users, n_items = user_items.shape

        user_factors = 0.01 * rng.standard_normal((n_users, self.factors))
        item_factors = 0.01 * rng.standard_normal((n_items, self.factors))
        reg_eye = self.regularization * np.eye(self.factors)

        for _ in range(self.iterations):
            yty = item_factors.T @ item_factors
            for user_idx in range(n_users):
                start, end = user_items.indptr[user_idx], user_items.indptr[user_idx + 1]
                item_idx = user_items.indices[start:end]
                confidence_minus_one = user_items.data[start:end]
                if item_idx.size == 0:
                    continue
                factors_i = item_factors[item_idx]
                a = yty + (factors_i.T * confidence_minus_one) @ factors_i + reg_eye
                b = factors_i.T @ (1.0 + confidence_minus_one)
                user_factors[user_idx] = _solve_regularized(a, b)

            xtx = user_factors.T @ user_factors
            for item_idx in range(n_items):
                start, end = item_users.indptr[item_idx], item_users.indptr[item_idx + 1]
                user_idx = item_users.indices[start:end]
                confidence_minus_one = item_users.data[start:end]
                if user_idx.size == 0:
                    continue
                factors_u = user_factors[user_idx]
                a = xtx + (factors_u.T * confidence_minus_one) @ factors_u + reg_eye
                b = factors_u.T @ (1.0 + confidence_minus_one)
                item_factors[item_idx] = _solve_regularized(a, b)

        self.user_factors_ = user_factors
        self.item_factors_ = item_factors

    def recommend(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-k ALS candidates as ``user_id,item_id,score,rank``."""

        if k <= 0:
            raise ValueError("k must be positive")
        if self.matrix_ is None or self.backend_ is None:
            raise RuntimeError("Call fit() before recommend().")

        records: list[dict[str, object]] = []
        for user in users:
            if user not in self.matrix_.user_index:
                continue
            user_idx = self.matrix_.user_index[user]
            if self.backend_ == "implicit":
                item_ids, scores = self.model.recommend(
                    user_idx,
                    self.matrix_.user_items[user_idx],
                    N=k,
                    filter_already_liked_items=exclude_seen,
                )
                for item_idx, score in zip(item_ids, scores, strict=False):
                    records.append(
                        {
                            USER_COL: user,
                            ITEM_COL: self.matrix_.index_item[int(item_idx)],
                            SCORE_COL: float(score),
                        }
                    )
                continue

            if self.user_factors_ is None or self.item_factors_ is None:
                raise RuntimeError("Native ALS factors are not available.")
            scores = self.user_factors_[user_idx] @ self.item_factors_.T
            scores = scores.astype(np.float64, copy=True)
            if exclude_seen:
                scores[self.matrix_.user_items[user_idx].indices] = -np.inf
            for item_idx, score in _top_k_from_scores(scores, k):
                records.append(
                    {
                        USER_COL: user,
                        ITEM_COL: self.matrix_.index_item[item_idx],
                        SCORE_COL: score,
                    }
                )

        return sort_candidates(pd.DataFrame(records, columns=[USER_COL, ITEM_COL, SCORE_COL]), k=k)
