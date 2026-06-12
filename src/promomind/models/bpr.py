"""Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd

from .candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates
from .sparse import InteractionMatrix, build_interaction_matrix


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
class BPRRecommender:
    """Simple BPR-SGD recommender for implicit feedback.

    This implementation is intentionally compact and dependency-light. It is
    suitable for baselines and course experiments; teams can replace it with a
    faster library backend later without changing the candidate output schema.
    """

    factors: int = 64
    learning_rate: float = 0.03
    regularization: float = 0.01
    epochs: int = 5
    samples_per_epoch: int | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.factors <= 0:
            raise ValueError("factors must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        self.matrix_: InteractionMatrix | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
    ) -> "BPRRecommender":
        """Fit pairwise matrix factorization on observed user-item positives."""

        self.matrix_ = build_interaction_matrix(
            interactions,
            user_col=user_col,
            item_col=item_col,
            weight_col=weight_col,
            alpha=1.0,
        )
        n_users, n_items = self.matrix_.user_items.shape
        rng = np.random.default_rng(self.random_state)
        self.user_factors_ = 0.01 * rng.standard_normal((n_users, self.factors))
        self.item_factors_ = 0.01 * rng.standard_normal((n_items, self.factors))

        if n_users == 0 or n_items <= 1:
            return self

        positives = [
            set(self.matrix_.user_items[user_idx].indices.tolist()) for user_idx in range(n_users)
        ]
        trainable_users = [idx for idx, items in enumerate(positives) if 0 < len(items) < n_items]
        if not trainable_users:
            return self

        samples = self.samples_per_epoch or int(self.matrix_.user_items.nnz)
        samples = max(samples, len(trainable_users))

        for _ in range(self.epochs):
            for _ in range(samples):
                user_idx = int(rng.choice(trainable_users))
                pos_items = tuple(positives[user_idx])
                pos_idx = int(rng.choice(pos_items))
                neg_idx = int(rng.integers(0, n_items))
                attempts = 0
                while neg_idx in positives[user_idx] and attempts < 100:
                    neg_idx = int(rng.integers(0, n_items))
                    attempts += 1
                if neg_idx in positives[user_idx]:
                    continue

                self._update_triplet(user_idx, pos_idx, neg_idx)

        return self

    def _update_triplet(self, user_idx: int, pos_idx: int, neg_idx: int) -> None:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("BPR factors are not initialized.")

        user_vec = self.user_factors_[user_idx].copy()
        pos_vec = self.item_factors_[pos_idx].copy()
        neg_vec = self.item_factors_[neg_idx].copy()

        score_diff = float(user_vec @ (pos_vec - neg_vec))
        gradient = 1.0 / (1.0 + np.exp(score_diff))

        self.user_factors_[user_idx] += self.learning_rate * (
            gradient * (pos_vec - neg_vec) - self.regularization * user_vec
        )
        self.item_factors_[pos_idx] += self.learning_rate * (
            gradient * user_vec - self.regularization * pos_vec
        )
        self.item_factors_[neg_idx] += self.learning_rate * (
            -gradient * user_vec - self.regularization * neg_vec
        )

    def recommend(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-k BPR candidates as ``user_id,item_id,score,rank``."""

        if k <= 0:
            raise ValueError("k must be positive")
        if self.matrix_ is None or self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("Call fit() before recommend().")

        records: list[dict[str, object]] = []
        for user in users:
            if user not in self.matrix_.user_index:
                continue
            user_idx = self.matrix_.user_index[user]
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
