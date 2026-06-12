"""Sparse interaction matrix utilities for candidate-generation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
from scipy import sparse

from .candidates import ITEM_COL, USER_COL, ensure_columns


@dataclass(frozen=True)
class InteractionMatrix:
    """Sparse user-item matrix plus reversible ID mappings."""

    user_items: sparse.csr_matrix
    item_users: sparse.csr_matrix
    user_index: dict[Hashable, int]
    item_index: dict[Hashable, int]
    index_user: dict[int, Hashable]
    index_item: dict[int, Hashable]

    @property
    def n_users(self) -> int:
        return self.user_items.shape[0]

    @property
    def n_items(self) -> int:
        return self.user_items.shape[1]


def _ordered_unique(values: pd.Series) -> list[Hashable]:
    """Return deterministic unique IDs without forcing IDs to be numeric."""

    return sorted(values.dropna().unique().tolist(), key=lambda value: str(value))


def build_interaction_matrix(
    interactions: pd.DataFrame,
    user_col: str = USER_COL,
    item_col: str = ITEM_COL,
    weight_col: str | None = None,
    alpha: float = 1.0,
) -> InteractionMatrix:
    """Build a CSR user-item matrix from implicit feedback rows.

    Duplicate user-item rows are summed. When ``weight_col`` is omitted, each
    observed interaction contributes one unit of implicit feedback.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    required = [user_col, item_col] + ([weight_col] if weight_col else [])
    ensure_columns(interactions, required, "interactions")

    work = interactions[required].dropna(subset=[user_col, item_col]).copy()
    if work.empty:
        empty = sparse.csr_matrix((0, 0), dtype=np.float64)
        return InteractionMatrix(empty, empty.T.tocsr(), {}, {}, {}, {})

    if weight_col:
        work["_weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    else:
        work["_weight"] = 1.0
    work = work[work["_weight"] > 0]
    if work.empty:
        empty = sparse.csr_matrix((0, 0), dtype=np.float64)
        return InteractionMatrix(empty, empty.T.tocsr(), {}, {}, {}, {})

    aggregated = (
        work.groupby([user_col, item_col], observed=True, as_index=False)["_weight"]
        .sum()
        .reset_index(drop=True)
    )

    users = _ordered_unique(aggregated[user_col])
    items = _ordered_unique(aggregated[item_col])
    user_index = {user: idx for idx, user in enumerate(users)}
    item_index = {item: idx for idx, item in enumerate(items)}
    index_user = {idx: user for user, idx in user_index.items()}
    index_item = {idx: item for item, idx in item_index.items()}

    row = aggregated[user_col].map(user_index).to_numpy(dtype=np.int64)
    col = aggregated[item_col].map(item_index).to_numpy(dtype=np.int64)
    data = (aggregated["_weight"].to_numpy(dtype=np.float64) * alpha).astype(np.float64)

    user_items = sparse.csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    user_items.sum_duplicates()
    return InteractionMatrix(
        user_items=user_items,
        item_users=user_items.T.tocsr(),
        user_index=user_index,
        item_index=item_index,
        index_user=index_user,
        index_item=index_item,
    )
