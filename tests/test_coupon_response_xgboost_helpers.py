import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_coupon_response_xgboost_ranker as xgbr  # noqa: E402


def test_normalize_scores_by_event_is_group_local():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b"],
            "product_id": [1, 2, 1, 2],
        }
    )

    normalized = xgbr._normalize_scores_by_event(frame, [10.0, 20.0, 100.0, 200.0])

    assert np.allclose(normalized, [0.0, 1.0, 0.0, 1.0])


def test_blend_weights_grid_includes_pure_heuristic_and_pure_xgb():
    weights = xgbr._candidate_blend_weights(enabled=True, step=0.25)

    assert weights == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_blend_scores_respects_selected_xgb_weight():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a"],
            "product_id": [1, 2],
        }
    )

    blended = xgbr._blend_scores(
        frame,
        xgb_scores=[0.0, 10.0],
        heuristic_scores=[10.0, 0.0],
        xgb_blend_weight=0.25,
    )

    assert math.isclose(float(blended[0]), 0.75)
    assert math.isclose(float(blended[1]), 0.25)
