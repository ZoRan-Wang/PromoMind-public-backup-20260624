import pandas as pd
import pytest

from promomind.models.als import ImplicitALSRecommender
from promomind.models.baselines import category_popularity_recommendations
from promomind.models.bpr import BPRRecommender
from promomind.models.candidates import ITEM_COL, RANK_COL, SCORE_COL, USER_COL
from promomind.models.itemknn import ItemKNNRecommender
from promomind.models.next_basket import PersonalTopFrequencyRecommender, TIFUKNNRecommender
from promomind.models.sparse import build_interaction_matrix


def _training_interactions():
    return pd.DataFrame(
        {
            "household_id": ["H1", "H1", "H2", "H2", "H3", "H3", "H4"],
            "product_id": ["P1", "P2", "P1", "P3", "P2", "P3", "P4"],
            "quantity": [1, 2, 1, 1, 1, 3, 1],
        }
    )


def _assert_candidate_schema(candidates):
    for column in [USER_COL, ITEM_COL, SCORE_COL, RANK_COL]:
        assert column in candidates.columns
    assert candidates[SCORE_COL].notna().all()
    assert candidates[RANK_COL].min() == 1


def test_sparse_loader_builds_stable_mappings_and_sums_duplicates():
    interactions = pd.DataFrame(
        {
            "household_id": ["H2", "H1", "H1"],
            "product_id": ["P2", "P1", "P1"],
            "quantity": [1, 2, 3],
        }
    )

    matrix = build_interaction_matrix(
        interactions,
        user_col="household_id",
        item_col="product_id",
        weight_col="quantity",
    )

    assert matrix.user_items.shape == (2, 2)
    assert matrix.user_index == {"H1": 0, "H2": 1}
    assert matrix.item_index == {"P1": 0, "P2": 1}
    assert matrix.user_items[matrix.user_index["H1"], matrix.item_index["P1"]] == 5


def test_itemknn_recommends_unseen_items_with_candidate_schema():
    train = _training_interactions()

    model = ItemKNNRecommender(max_similar_items=3).fit(
        train,
        user_col="household_id",
        item_col="product_id",
        weight_col="quantity",
    )
    candidates = model.recommend(["H1"], k=2)

    _assert_candidate_schema(candidates)
    assert set(candidates[ITEM_COL]).isdisjoint({"P1", "P2"})


def test_category_popularity_falls_back_when_categories_have_no_unseen_items():
    train = pd.DataFrame(
        {
            "household_id": ["H1", "H2", "H2"],
            "product_id": ["P1", "P2", "P3"],
            "quantity": [1, 1, 1],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "category": ["only_h1", "other", "another"],
        }
    )

    candidates = category_popularity_recommendations(
        train,
        products,
        users=["H1"],
        k=2,
        user_col="household_id",
        item_col="product_id",
        category_col="category",
        weight_col="quantity",
    )

    _assert_candidate_schema(candidates)
    assert len(candidates) == 2
    assert set(candidates[ITEM_COL]).isdisjoint({"P1"})


def test_native_als_recommends_unseen_items_with_candidate_schema():
    train = _training_interactions()

    model = ImplicitALSRecommender(
        factors=3,
        regularization=0.05,
        iterations=2,
        alpha=5,
        backend="native",
        random_state=7,
    ).fit(train, user_col="household_id", item_col="product_id", weight_col="quantity")
    candidates = model.recommend(["H1"], k=2)

    _assert_candidate_schema(candidates)
    assert set(candidates[ITEM_COL]).isdisjoint({"P1", "P2"})


def test_implicit_als_backend_recommends_when_dependency_is_installed():
    pytest.importorskip("implicit")
    train = _training_interactions()

    model = ImplicitALSRecommender(
        factors=3,
        regularization=0.05,
        iterations=2,
        alpha=5,
        backend="implicit",
        random_state=7,
    ).fit(train, user_col="household_id", item_col="product_id", weight_col="quantity")
    candidates = model.recommend(["H1"], k=2)

    _assert_candidate_schema(candidates)
    assert set(candidates[ITEM_COL]).isdisjoint({"P1", "P2"})


def test_bpr_recommends_unseen_items_with_candidate_schema():
    train = _training_interactions()

    model = BPRRecommender(
        factors=4,
        learning_rate=0.05,
        regularization=0.01,
        epochs=2,
        samples_per_epoch=20,
        random_state=11,
    ).fit(train, user_col="household_id", item_col="product_id", weight_col="quantity")
    candidates = model.recommend(["H1"], k=2)

    _assert_candidate_schema(candidates)
    assert set(candidates[ITEM_COL]).isdisjoint({"P1", "P2"})


def test_personal_top_frequency_prioritizes_repeat_purchases():
    train = _training_interactions()

    model = PersonalTopFrequencyRecommender().fit(
        train,
        user_col="household_id",
        item_col="product_id",
        weight_col="quantity",
    )
    candidates = model.recommend(["H1"], k=2)

    _assert_candidate_schema(candidates)
    assert list(candidates[ITEM_COL].head(2)) == ["P2", "P1"]


def test_personal_top_frequency_can_exclude_seen_items():
    train = _training_interactions()

    model = PersonalTopFrequencyRecommender().fit(
        train,
        user_col="household_id",
        item_col="product_id",
        weight_col="quantity",
    )
    candidates = model.recommend(["H1"], k=2, exclude_seen=True)

    _assert_candidate_schema(candidates)
    assert set(candidates[ITEM_COL]).isdisjoint({"P1", "P2"})


def test_tifu_knn_recommends_with_candidate_schema():
    train = _training_interactions().assign(
        basket_id=["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        week=[1, 2, 1, 2, 1, 2, 3],
    )

    model = TIFUKNNRecommender(n_neighbors=2, alpha=0.7, basket_decay=0.95).fit(
        train,
        user_col="household_id",
        item_col="product_id",
        weight_col="quantity",
        basket_col="basket_id",
        time_col="week",
    )
    candidates = model.recommend(["H1"], k=3)

    _assert_candidate_schema(candidates)
    assert len(candidates) > 0
