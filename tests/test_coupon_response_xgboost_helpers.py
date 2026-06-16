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


def test_coupon_family_features_use_prior_same_coupon_upc_history():
    features = pd.DataFrame(
        {
            "campaign_id": [1, 1],
            "coupon_start_date": ["2026-01-10", "2026-01-10"],
            "household_id": [5, 5],
            "product_id": [100, 101],
            "user_product_count": [0, 1],
        }
    )
    sources = {
        "coupons": pd.DataFrame(
            {
                "campaign_id": [1, 1],
                "coupon_upc": [9000, 9000],
                "product_id": [100, 101],
            }
        ),
        "transactions": pd.DataFrame(
            {
                "household_id": [5, 5],
                "product_id": [101, 100],
                "transaction_timestamp": pd.to_datetime(["2026-01-01", "2026-01-11"]),
            }
        ),
    }

    out = xgbr.add_coupon_family_features(features, sources)

    assert out.loc[0, "coupon_family_match"] == 1.0
    assert math.isclose(out.loc[0, "coupon_family_count_log"], math.log1p(1.0))
    assert math.isclose(out.loc[0, "coupon_family_substitute_signal"], math.log1p(1.0))
    assert out.loc[1, "coupon_family_substitute_signal"] == 0.0


def test_redemption_features_ignore_future_redemptions(tmp_path):
    features = pd.DataFrame(
        {
            "campaign_id": [2],
            "coupon_start_date": ["2026-01-10"],
            "household_id": [7],
            "product_id": [200],
            "product_category": ["SOUP"],
        }
    )
    raw_dir = tmp_path
    pd.DataFrame(
        {
            "household_id": [7, 7],
            "coupon_upc": [3000, 3000],
            "campaign_id": [1, 2],
            "redemption_date": ["2026-01-01", "2026-01-12"],
        }
    ).to_csv(raw_dir / "coupon_redemptions.csv", index=False)
    sources = {
        "coupons": pd.DataFrame(
            {
                "campaign_id": [1, 2],
                "coupon_upc": [3000, 3000],
                "product_id": [200, 200],
            }
        ),
        "products": pd.DataFrame(
            {
                "product_id": [200],
                "product_category": ["SOUP"],
            }
        ),
    }

    out = xgbr.add_redemption_features(features, sources, raw_dir)

    assert math.isclose(out.loc[0, "household_redemption_count_log"], math.log1p(1.0))
    assert math.isclose(out.loc[0, "household_coupon_upc_redemption_log"], math.log1p(1.0))
    assert math.isclose(out.loc[0, "household_product_redemption_log"], math.log1p(1.0))
