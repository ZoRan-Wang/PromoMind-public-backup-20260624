"""Wrapper stub for an implicit-feedback ALS recommender.

Expected recommendation output:

```
user_id | item_id | score | rank
```

where ``score`` is the ALS model's predicted preference and larger values rank
higher. This wrapper keeps the project interface stable even when the optional
``implicit`` dependency has not been installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .candidates import ITEM_COL, SCORE_COL, USER_COL, sort_candidates


def _require_implicit():
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise ImportError(
            "ImplicitALSRecommender requires the optional 'implicit' package. "
            "Install it with `pip install implicit scipy` before fitting ALS, "
            "or use the popularity baselines for a dependency-light run."
        ) from exc
    return AlternatingLeastSquares


@dataclass
class ImplicitALSRecommender:
    """Small adapter around ``implicit.als.AlternatingLeastSquares``.

    The current implementation documents the intended interface and performs
    dependency checks. Matrix-building details can be extended once the final
    cleaned interaction schema is fixed by the data workstream.
    """

    factors: int = 64
    regularization: float = 0.01
    iterations: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = None
        self.user_index_: dict[object, int] = {}
        self.item_index_: dict[object, int] = {}
        self.index_user_: dict[int, object] = {}
        self.index_item_: dict[int, object] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        weight_col: str | None = None,
    ) -> "ImplicitALSRecommender":
        """Fit ALS on implicit user-item interactions.

        Raises a helpful ImportError when ``implicit`` or ``scipy`` is missing.
        A complete implementation should convert interactions into a scipy CSR
        item-user matrix, call ``model.fit(...)``, and populate ID mappings.
        """

        AlternatingLeastSquares = _require_implicit()
        try:
            import scipy.sparse as sparse  # noqa: F401
        except ImportError as exc:
            raise ImportError("ImplicitALSRecommender also requires scipy sparse matrices.") from exc

        required = [user_col, item_col] + ([weight_col] if weight_col else [])
        missing = [col for col in required if col not in interactions.columns]
        if missing:
            raise ValueError(f"interactions is missing required columns: {missing}")

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        raise NotImplementedError(
            "ALS matrix construction is intentionally left as a project stub. "
            "Expected output from recommend() is a DataFrame with user_id, item_id, score, rank."
        )

    def recommend(self, users: list, k: int = 10) -> pd.DataFrame:
        """Return top-k ALS candidates as ``user_id,item_id,score,rank``.

        This method is a placeholder until ``fit`` builds model factors and ID
        mappings. It raises a clear error instead of silently returning junk.
        """

        if self.model is None:
            raise RuntimeError("Call fit() before recommend(), or use a baseline recommender.")
        if k <= 0:
            raise ValueError("k must be positive")

        empty = pd.DataFrame(columns=[USER_COL, ITEM_COL, SCORE_COL])
        return sort_candidates(empty, k=k)

