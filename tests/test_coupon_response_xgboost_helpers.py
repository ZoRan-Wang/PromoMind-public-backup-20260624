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


def test_rank_fusion_configs_are_disabled_by_default():
    configs = xgbr._rank_fusion_configs(enabled=False)

    assert configs == [xgbr.NO_RANK_FUSION]


def test_rank_fusion_scores_can_combine_xgb_and_heuristic_ranks():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a"],
            "product_id": [1, 2],
            "base_signal": [0.0, 0.0],
            "global_signal": [0.0, 0.0],
        }
    )
    config = {
        "rank_fusion_method": "rrf",
        "rank_fusion_c": 10.0,
        "rank_fusion_xgb_weight": 0.5,
        "rank_fusion_heuristic_weight": 0.5,
        "rank_fusion_base_weight": 0.0,
        "rank_fusion_global_weight": 0.0,
    }

    scores = xgbr._rank_fusion_scores(
        frame,
        xgb_scores=[1.0, 0.0],
        heuristic_scores=[0.0, 1.0],
        config=config,
    )

    assert np.allclose(scores, [0.5 / 11.0 + 0.5 / 12.0, 0.5 / 12.0 + 0.5 / 11.0])


def test_select_ensemble_configs_uses_top_validation_rows():
    search = pd.DataFrame(
        {
            "n_estimators": [120, 180, 250],
            "learning_rate": [0.03, 0.03, 0.03],
            "max_depth": [2, 2, 2],
            "objective": ["rank:ndcg", "rank:ndcg", "rank:ndcg"],
            "positive_train_events_only": [False, False, False],
            "subsample": [0.9, 0.9, 0.9],
            "colsample_bytree": [0.9, 0.9, 0.9],
            "recall_at_20": [0.30, 0.32, 0.31],
            "ndcg_at_10": [0.20, 0.19, 0.25],
            "recall_at_10": [0.10, 0.20, 0.15],
        }
    )

    configs = xgbr._select_ensemble_configs(search, "recall_at_20", 2)

    assert [config["n_estimators"] for config in configs] == [180, 250]


def test_pull_forward_timing_labels_grade_middle_highest():
    features = pd.DataFrame(
        {
            "event_id": ["a", "a", "a", "a"],
            "product_id": [1, 2, 3, 4],
            "coupon_start_date": ["2026-01-10"] * 4,
            "label": [1.0, 1.0, 1.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "event_id": ["a", "a", "a"],
            "product_id": [1, 2, 3],
            "observed_purchase_time": ["2026-01-10 12:00:00", "2026-01-12", "2026-01-14"],
        }
    )

    out = xgbr.apply_label_scheme(features, truth, "pull_forward_timing")

    assert out["label"].tolist() == [2.0, 3.0, 2.0, 0.0]


def test_pull_forward_interval_labels_use_repurchase_cadence_window():
    features = pd.DataFrame(
        {
            "event_id": ["a", "a", "a", "a"],
            "product_id": [1, 2, 3, 4],
            "coupon_start_date": ["2026-01-10"] * 4,
            "days_since_last": [8.0, 7.0, 9.0, 8.0],
            "median_interval_days": [10.0, 10.0, 10.0, 10.0],
            "label": [1.0, 1.0, 1.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "event_id": ["a", "a", "a"],
            "product_id": [1, 2, 3],
            "observed_purchase_time": ["2026-01-11", "2026-01-10 12:00:00", "2026-01-13"],
        }
    )

    out = xgbr.apply_label_scheme(
        features,
        truth,
        "pull_forward_interval",
        pull_forward_min_days=-1.0,
        pull_forward_max_days=2.0,
    )

    assert out["label"].tolist() == [3.0, 2.0, 2.0, 0.0]


def test_expected_lead_timing_labels_prefer_one_to_two_day_coupon_lead():
    features = pd.DataFrame(
        {
            "event_id": ["a", "a", "a", "a"],
            "product_id": [1, 2, 3, 4],
            "coupon_start_date": ["2026-01-10"] * 4,
            "days_since_last": [8.5, 9.5, 6.0, 8.0],
            "median_interval_days": [10.0, 10.0, 10.0, 10.0],
            "label": [1.0, 1.0, 1.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "event_id": ["a", "a", "a"],
            "product_id": [1, 2, 3],
            "observed_purchase_time": ["2026-01-11", "2026-01-11", "2026-01-11"],
        }
    )

    out = xgbr.apply_label_scheme(
        features,
        truth,
        "expected_lead_timing",
        expected_lead_min_days=1.0,
        expected_lead_max_days=2.0,
    )

    assert out["label"].tolist() == [3.0, 2.0, 2.0, 0.0]


def test_text_embedding_features_use_prior_product_text_history():
    features = pd.DataFrame(
        {
            "event_id": ["a", "a"],
            "campaign_id": [1, 1],
            "coupon_start_date": ["2026-01-10", "2026-01-10"],
            "household_id": [5, 5],
            "product_id": [101, 201],
            "label": [0.0, 0.0],
        }
    )
    sources = {
        "products": pd.DataFrame(
            {
                "product_id": [100, 101, 200, 201],
                "department": ["GROCERY", "GROCERY", "PET", "PET"],
                "brand": ["Private", "Private", "National", "National"],
                "product_category": ["APPLE SAUCE", "APPLE SAUCE", "DOG FOOD", "DOG FOOD"],
                "product_type": ["APPLE CINNAMON CUP", "APPLE CINNAMON JAR", "DRY DOG CHICKEN", "DRY DOG BEEF"],
                "package_size": ["4 OZ", "4 OZ", "5 LB", "5 LB"],
            }
        ),
        "transactions": pd.DataFrame(
            {
                "household_id": [5, 5],
                "product_id": [100, 200],
                "transaction_timestamp": pd.to_datetime(["2026-01-01", "2026-01-11"]),
            }
        ),
    }

    out = xgbr.add_text_embedding_features(features, sources, components=2, max_features=64)

    apple_similarity = out.loc[out["product_id"].eq(101), "text_embedding_similarity"].iloc[0]
    dog_similarity = out.loc[out["product_id"].eq(201), "text_embedding_similarity"].iloc[0]
    assert apple_similarity > dog_similarity
    assert out["text_embedding_has_profile"].tolist() == [1.0, 1.0]
    assert out["text_embedding_history_count_log"].nunique() == 1


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
