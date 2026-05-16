"""Shared candidate recommendation schema helpers.

The canonical candidate output is a pandas DataFrame with at least:

- ``user_id``: household/customer identifier
- ``item_id``: product identifier
- ``score``: model score where larger means more relevant

Most functions in this package preserve additional columns such as ``rank``,
``category``, ``promotion_score``, ``coupon_score``, or business metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


USER_COL = "user_id"
ITEM_COL = "item_id"
SCORE_COL = "score"
RANK_COL = "rank"
REQUIRED_CANDIDATE_COLUMNS = (USER_COL, ITEM_COL, SCORE_COL)


@dataclass(frozen=True)
class CandidateSchema:
    """Names used by PromoMind candidate DataFrames."""

    user_col: str = USER_COL
    item_col: str = ITEM_COL
    score_col: str = SCORE_COL
    rank_col: str = RANK_COL

    @property
    def required_columns(self) -> tuple[str, str, str]:
        return (self.user_col, self.item_col, self.score_col)


def validate_candidates(
    candidates: pd.DataFrame,
    schema: CandidateSchema = CandidateSchema(),
    allow_null_scores: bool = False,
) -> pd.DataFrame:
    """Validate and return a shallow copy of a candidate DataFrame.

    Raises:
        ValueError: if required columns are missing or scores are invalid.
    """

    missing = [col for col in schema.required_columns if col not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate DataFrame is missing required columns: {missing}")

    validated = candidates.copy()
    validated[schema.score_col] = pd.to_numeric(validated[schema.score_col], errors="coerce")

    if not allow_null_scores and validated[schema.score_col].isna().any():
        raise ValueError(f"Candidate column '{schema.score_col}' contains null/non-numeric scores")

    return validated


def sort_candidates(
    candidates: pd.DataFrame,
    k: int | None = None,
    schema: CandidateSchema = CandidateSchema(),
) -> pd.DataFrame:
    """Sort candidates by user and descending score, then assign 1-based ranks."""

    if k is not None and k <= 0:
        raise ValueError("k must be positive when provided")

    sorted_df = validate_candidates(candidates, schema=schema).sort_values(
        [schema.user_col, schema.score_col, schema.item_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    sorted_df[schema.rank_col] = sorted_df.groupby(schema.user_col).cumcount() + 1

    if k is not None:
        sorted_df = sorted_df[sorted_df[schema.rank_col] <= k]

    return sorted_df.reset_index(drop=True)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    """Raise a readable error when a DataFrame is missing expected columns."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")

