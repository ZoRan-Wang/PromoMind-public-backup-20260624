import math

import pandas as pd

from promomind.evaluation.business import business_utility_at_k
from promomind.evaluation.ranking import ndcg_at_k, recall_at_k
from promomind.rerank.promotion import promotion_aware_rerank


def test_recall_at_k_uses_per_user_truth_sets():
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "b", "c"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    truth = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "item_id": ["a", "x", "z"]})

    assert recall_at_k(recommendations, truth, k=1) == 0.25


def test_ndcg_at_k_rewards_higher_rank_hits():
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["miss", "hit", "other"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    truth = pd.DataFrame({"user_id": ["u1"], "item_id": ["hit"]})

    assert math.isclose(ndcg_at_k(recommendations, truth, k=3), 1 / math.log2(3))


def test_business_utility_at_k_is_revenue_minus_discount_per_user_average():
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "b", "c"],
            "score": [3, 2, 1],
            "expected_revenue": [5.0, 4.0, 10.0],
            "discount_cost": [1.0, 1.5, 3.0],
        }
    )

    assert business_utility_at_k(recommendations, k=2) == 6.75


def test_business_utility_at_k_can_count_hits_only():
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "b", "c"],
            "score": [3, 2, 1],
            "expected_revenue": [5.0, 4.0, 10.0],
            "discount_cost": [1.0, 1.5, 3.0],
        }
    )
    truth = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["b", "missing"]})

    assert business_utility_at_k(recommendations, ground_truth=truth, k=2) == 2.5


def test_promotion_reranker_can_lift_promoted_candidate():
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["regular", "promo"],
            "score": [1.0, 0.8],
        }
    )
    features = pd.DataFrame(
        {
            "item_id": ["regular", "promo"],
            "category": ["pantry", "snacks"],
            "promotion_score": [0.0, 2.0],
            "coupon_score": [0.0, 0.0],
            "discount_cost": [0.0, 0.0],
        }
    )

    reranked = promotion_aware_rerank(candidates, features, beta=0.3, rho=0.0, k=1)

    assert reranked.loc[0, "item_id"] == "promo"
    assert reranked.loc[0, "rank"] == 1
