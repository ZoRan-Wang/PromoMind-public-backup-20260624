"""Run a coupon-response reranker on campaign-aware coupon candidates.

This runner is intentionally different from the ordinary next-basket benchmark:
it evaluates whether exposed households buy campaign coupon products within
five days after the campaign start date.

The ranker expands the SOTA next-basket candidate list with active campaign
coupon products, then scores candidates with time-aware repeat and category
features. We tune weights on validation campaigns and report a held-out test
score using the fixed best validation configuration.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from promomind.data import schema  # noqa: E402

EVENT_COL = "event_id"
MODEL_NAME = "coupon_response_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coupon-response ranking experiment.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--base-candidates", type=Path, default=REPO_ROOT / "outputs" / "candidates_sota_ensemble.csv")
    parser.add_argument("--max-global-products", type=int, default=1000)
    parser.add_argument("--max-candidates-per-event", type=int, default=600)
    parser.add_argument("--base-k", type=int, default=50)
    parser.add_argument("--top-categories", type=int, default=3)
    parser.add_argument("--category-products", type=int, default=250)
    parser.add_argument("--eval-k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--primary-metric", default="positive_event_hit_rate_at_10")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def _normalize_ids(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        out = out.dropna(subset=[column])
        out[column] = out[column].astype(int)
    return out


def load_sources(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    transactions = _read_csv(args.raw_dir / "transactions.csv", parse_dates=["transaction_timestamp"])
    transactions = _normalize_ids(transactions, [schema.HOUSEHOLD_ID, schema.PRODUCT_ID])

    campaigns = _read_csv(args.raw_dir / "campaigns.csv")
    campaigns = _normalize_ids(campaigns, ["campaign_id", schema.HOUSEHOLD_ID])
    campaign_desc = _read_csv(
        args.raw_dir / "campaign_descriptions.csv",
        parse_dates=["start_date", "end_date"],
    )
    campaign_desc = _normalize_ids(campaign_desc, ["campaign_id"])
    campaigns = campaigns.merge(campaign_desc, on="campaign_id", how="left")

    coupons = _read_csv(args.raw_dir / "coupons.csv")
    coupons = _normalize_ids(coupons, ["campaign_id", "coupon_upc", schema.PRODUCT_ID])
    coupons = coupons.drop_duplicates(["campaign_id", "coupon_upc", schema.PRODUCT_ID])

    products = _read_csv(args.raw_dir / "products.csv")
    products = _normalize_ids(products, [schema.PRODUCT_ID])
    products["product_category"] = products["product_category"].fillna("UNKNOWN").astype(str)
    products["brand"] = products["brand"].fillna("UNKNOWN").astype(str)

    candidates = _read_csv(args.base_candidates)
    candidates = candidates.rename(columns={"base_rank": "source_rank", "base_score": "source_score"})
    candidates = _normalize_ids(candidates, [schema.HOUSEHOLD_ID, schema.PRODUCT_ID])
    if "source_rank" not in candidates.columns:
        raise ValueError(f"{args.base_candidates} must contain base_rank")
    candidates = candidates[pd.to_numeric(candidates["source_rank"], errors="coerce") <= args.base_k].copy()
    candidates["source_rank"] = pd.to_numeric(candidates["source_rank"], errors="coerce")
    candidates["source_score"] = pd.to_numeric(candidates.get("source_score", 0.0), errors="coerce").fillna(0.0)

    val = _read_csv(args.processed_dir / "transactions_val.csv", parse_dates=["transaction_timestamp"])
    test = _read_csv(args.processed_dir / "transactions_test.csv", parse_dates=["transaction_timestamp"])

    return {
        "transactions": transactions,
        "campaigns": campaigns,
        "coupons": coupons,
        "products": products,
        "candidates": candidates,
        "val": val,
        "test": test,
    }


def make_eval_events(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    campaigns = sources["campaigns"].copy()
    val_start = pd.Timestamp(sources["val"]["transaction_timestamp"].min())
    test_start = pd.Timestamp(sources["test"]["transaction_timestamp"].min())
    max_observed = pd.Timestamp(sources["transactions"]["transaction_timestamp"].max())
    campaigns["success_window_end"] = campaigns["start_date"] + pd.Timedelta(days=5)

    val_mask = (campaigns["start_date"] >= val_start) & (campaigns["start_date"] < test_start)
    test_mask = (campaigns["start_date"] >= test_start) & (campaigns["success_window_end"] <= max_observed)
    events = pd.concat(
        [
            campaigns.loc[val_mask].assign(split="validation"),
            campaigns.loc[test_mask].assign(split="test"),
        ],
        ignore_index=True,
    )
    events[EVENT_COL] = (
        events[schema.HOUSEHOLD_ID].astype(str)
        + "_"
        + events["campaign_id"].astype(str)
        + "_"
        + events["start_date"].dt.strftime("%Y%m%d")
    )
    return events[
        [
            EVENT_COL,
            "split",
            schema.HOUSEHOLD_ID,
            "campaign_id",
            "campaign_type",
            "start_date",
            "success_window_end",
        ]
    ].drop_duplicates(EVENT_COL)


def build_truth(events: pd.DataFrame, transactions: pd.DataFrame, coupons: pd.DataFrame) -> pd.DataFrame:
    truth_rows: list[dict[str, Any]] = []
    for campaign_id, campaign_events in events.groupby("campaign_id", sort=True):
        active_products = set(coupons.loc[coupons["campaign_id"] == campaign_id, schema.PRODUCT_ID])
        if not active_products:
            continue
        start = pd.Timestamp(campaign_events["start_date"].iloc[0])
        end = pd.Timestamp(campaign_events["success_window_end"].iloc[0])
        households = set(campaign_events[schema.HOUSEHOLD_ID])
        window = transactions[
            transactions[schema.HOUSEHOLD_ID].isin(households)
            & transactions[schema.PRODUCT_ID].isin(active_products)
            & (transactions["transaction_timestamp"] >= start)
            & (transactions["transaction_timestamp"] <= end)
        ].copy()
        if window.empty:
            continue
        event_lookup = campaign_events.set_index(schema.HOUSEHOLD_ID)[EVENT_COL].to_dict()
        window[EVENT_COL] = window[schema.HOUSEHOLD_ID].map(event_lookup)
        positive = window.sort_values("transaction_timestamp").drop_duplicates([EVENT_COL, schema.PRODUCT_ID])
        for row in positive[[EVENT_COL, schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]].itertuples(index=False):
            truth_rows.append(
                {
                    EVENT_COL: row.event_id,
                    schema.HOUSEHOLD_ID: int(row.household_id),
                    "campaign_id": int(campaign_id),
                    schema.PRODUCT_ID: int(row.product_id),
                    "observed_purchase_time": pd.Timestamp(row.transaction_timestamp).isoformat(),
                }
            )
    return pd.DataFrame(truth_rows)


def _median_interval_days(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    work = frame.sort_values([schema.HOUSEHOLD_ID, schema.PRODUCT_ID, "transaction_timestamp"]).copy()
    work["gap_days"] = (
        work.groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID])["transaction_timestamp"].diff().dt.total_seconds()
        / 86400.0
    )
    return work.dropna(subset=["gap_days"]).groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID])["gap_days"].median()


def build_feature_table(
    sources: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    max_global_products: int,
    max_candidates_per_event: int,
    top_categories: int,
    category_products: int,
) -> pd.DataFrame:
    transactions = sources["transactions"]
    coupons = sources["coupons"]
    products = sources["products"]
    candidates = sources["candidates"]
    product_category = products.set_index(schema.PRODUCT_ID)["product_category"].to_dict()

    feature_rows: list[dict[str, Any]] = []
    for campaign_id, campaign_events in events.groupby("campaign_id", sort=True):
        start = pd.Timestamp(campaign_events["start_date"].iloc[0])
        active_products = sorted(set(coupons.loc[coupons["campaign_id"] == campaign_id, schema.PRODUCT_ID]))
        if not active_products:
            continue
        active_set = set(active_products)
        active_categories = {product: product_category.get(product, "UNKNOWN") for product in active_products}
        households = set(campaign_events[schema.HOUSEHOLD_ID])

        history = transactions[transactions["transaction_timestamp"] < start].copy()
        global_counts = history.groupby(schema.PRODUCT_ID).size()
        active_global_counts = global_counts.reindex(active_products, fill_value=0).astype(float)
        max_global = float(np.log1p(active_global_counts.max())) if active_global_counts.max() > 0 else 1.0
        global_signal = (np.log1p(active_global_counts) / max_global).to_dict()
        top_global = set(active_global_counts.sort_values(ascending=False).head(max_global_products).index.astype(int))

        product_discount = (
            history.assign(
                discount_value=history[["retail_disc", "coupon_disc", "coupon_match_disc"]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .sum(axis=1)
            )
            .groupby(schema.PRODUCT_ID)["discount_value"]
            .mean()
            .reindex(active_products, fill_value=0.0)
        )
        max_discount = float(product_discount.max()) if product_discount.max() > 0 else 1.0
        discount_signal = (product_discount / max_discount).to_dict()

        category_to_products: dict[str, list[int]] = {}
        active_by_global = active_global_counts.sort_values(ascending=False).index.astype(int).tolist()
        for product_id in active_by_global:
            category = active_categories.get(product_id, "UNKNOWN")
            bucket = category_to_products.setdefault(category, [])
            if len(bucket) < category_products:
                bucket.append(product_id)

        household_history = history[history[schema.HOUSEHOLD_ID].isin(households)].copy()
        household_active_history = household_history[household_history[schema.PRODUCT_ID].isin(active_set)].copy()

        hp_counts = household_active_history.groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID]).size()
        hp_last = household_active_history.groupby([schema.HOUSEHOLD_ID, schema.PRODUCT_ID])["transaction_timestamp"].max()
        hp_interval = _median_interval_days(household_active_history)

        household_history["product_category"] = household_history[schema.PRODUCT_ID].map(product_category).fillna("UNKNOWN")
        cat_counts = household_history.groupby([schema.HOUSEHOLD_ID, "product_category"]).size()
        total_counts = household_history.groupby(schema.HOUSEHOLD_ID).size()

        base_campaign = candidates[
            candidates[schema.HOUSEHOLD_ID].isin(households)
            & candidates[schema.PRODUCT_ID].isin(active_set)
        ].copy()
        base_campaign["base_signal"] = 1.0 / base_campaign["source_rank"].astype(float)
        base_signal = base_campaign.set_index([schema.HOUSEHOLD_ID, schema.PRODUCT_ID])["base_signal"].to_dict()
        base_products = base_campaign.groupby(schema.HOUSEHOLD_ID)[schema.PRODUCT_ID].apply(set).to_dict()

        for event in campaign_events.itertuples(index=False):
            household_id = int(getattr(event, schema.HOUSEHOLD_ID))
            user_products = {
                int(product)
                for hh, product in hp_counts.index
                if int(hh) == household_id
            }
            user_category_scores = {}
            if household_id in total_counts.index and int(total_counts.loc[household_id]) > 0:
                user_cat_slice = cat_counts.loc[household_id] if household_id in cat_counts.index.get_level_values(0) else {}
                if hasattr(user_cat_slice, "sort_values"):
                    for category, value in user_cat_slice.sort_values(ascending=False).head(top_categories).items():
                        user_category_scores[str(category)] = float(value) / float(total_counts.loc[household_id])

            category_candidates: set[int] = set()
            for category in user_category_scores:
                category_candidates.update(category_to_products.get(category, [])[:category_products])

            candidate_products = set()
            candidate_products.update(top_global)
            candidate_products.update(base_products.get(household_id, set()))
            candidate_products.update(user_products)
            candidate_products.update(category_candidates)
            candidate_products &= active_set
            if not candidate_products:
                candidate_products = set(list(active_global_counts.sort_values(ascending=False).index.astype(int))[:max_candidates_per_event])

            ranked_for_cap = []
            for product_id in candidate_products:
                category = active_categories.get(product_id, "UNKNOWN")
                count = int(hp_counts.get((household_id, product_id), 0))
                category_affinity = user_category_scores.get(category, 0.0)
                priority = (
                    3.0 * float((household_id, product_id) in base_signal)
                    + 2.0 * float(count > 0)
                    + category_affinity
                    + float(global_signal.get(product_id, 0.0))
                )
                ranked_for_cap.append((priority, product_id))
            ranked_for_cap.sort(reverse=True)
            capped_products = [product_id for _, product_id in ranked_for_cap[:max_candidates_per_event]]

            max_user_count = max([int(hp_counts.get((household_id, pid), 0)) for pid in capped_products] + [1])
            max_user_log = math.log1p(max_user_count)
            for product_id in capped_products:
                count = int(hp_counts.get((household_id, product_id), 0))
                repeat_signal = math.log1p(count) / max_user_log if max_user_log else 0.0
                last_time = hp_last.get((household_id, product_id))
                median_interval = hp_interval.get((household_id, product_id), np.nan)
                if pd.notna(last_time):
                    days_since_last = max(0.0, (start - pd.Timestamp(last_time)).total_seconds() / 86400.0)
                else:
                    days_since_last = np.nan
                if count >= 2 and pd.notna(median_interval) and median_interval > 0 and pd.notna(days_since_last):
                    scale = max(float(median_interval), 7.0)
                    cadence_signal = math.exp(-abs(days_since_last - float(median_interval)) / scale)
                elif count == 1 and pd.notna(days_since_last):
                    cadence_signal = 0.5 * math.exp(-days_since_last / 45.0)
                else:
                    cadence_signal = 0.0

                category = active_categories.get(product_id, "UNKNOWN")
                feature_rows.append(
                    {
                        EVENT_COL: getattr(event, EVENT_COL),
                        "split": getattr(event, "split"),
                        schema.HOUSEHOLD_ID: household_id,
                        "campaign_id": int(campaign_id),
                        "campaign_type": getattr(event, "campaign_type"),
                        "coupon_start_date": pd.Timestamp(start).date().isoformat(),
                        "predicted_purchase_time": pd.Timestamp(getattr(event, "success_window_end")).isoformat(),
                        schema.PRODUCT_ID: int(product_id),
                        "base_signal": float(base_signal.get((household_id, product_id), 0.0)),
                        "repeat_signal": float(repeat_signal),
                        "cadence_signal": float(cadence_signal),
                        "category_signal": float(user_category_scores.get(category, 0.0)),
                        "global_signal": float(global_signal.get(product_id, 0.0)),
                        "discount_signal": float(discount_signal.get(product_id, 0.0)),
                        "product_category": category,
                        "user_product_count": count,
                        "days_since_last": days_since_last,
                        "median_interval_days": median_interval,
                    }
                )
    return pd.DataFrame(feature_rows)


def weight_grid(seed: int) -> list[dict[str, float]]:
    feature_names = ["base_signal", "repeat_signal", "cadence_signal", "category_signal", "global_signal", "discount_signal"]
    configs: list[dict[str, float]] = [
        {"base_signal": 1.0},
        {"global_signal": 1.0},
        {"repeat_signal": 0.45, "cadence_signal": 0.25, "category_signal": 0.2, "global_signal": 0.1},
        {"base_signal": 0.25, "repeat_signal": 0.25, "cadence_signal": 0.2, "category_signal": 0.15, "global_signal": 0.15},
        {"base_signal": 0.1, "repeat_signal": 0.35, "cadence_signal": 0.35, "category_signal": 0.1, "global_signal": 0.1},
        {"repeat_signal": 0.3, "cadence_signal": 0.25, "category_signal": 0.15, "global_signal": 0.2, "discount_signal": 0.1},
    ]
    rng = np.random.default_rng(seed)
    for alpha in [
        np.array([1.5, 2.0, 2.0, 1.5, 1.0, 0.5]),
        np.array([0.8, 2.5, 2.5, 1.2, 1.0, 0.4]),
        np.array([2.0, 1.0, 1.0, 1.0, 2.0, 0.2]),
    ]:
        for weights in rng.dirichlet(alpha, size=20):
            configs.append(dict(zip(feature_names, weights.astype(float), strict=False)))

    normalized = []
    seen = set()
    for config in configs:
        dense = {name: float(config.get(name, 0.0)) for name in feature_names}
        total = sum(dense.values())
        if total <= 0:
            continue
        dense = {name: value / total for name, value in dense.items()}
        key = tuple(round(dense[name], 4) for name in feature_names)
        if key not in seen:
            seen.add(key)
            normalized.append(dense)
    return normalized


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but PyTorch CUDA is not available.")
    return "cpu"


def compute_weighted_score(features: pd.DataFrame, weights: dict[str, float], device: str) -> np.ndarray:
    feature_names = ["base_signal", "repeat_signal", "cadence_signal", "category_signal", "global_signal", "discount_signal"]
    matrix = features[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    weight_vector = np.array([float(weights.get(name, 0.0)) for name in feature_names], dtype=np.float32)
    if device == "cuda":
        import torch

        with torch.inference_mode():
            scores = torch.as_tensor(matrix, device="cuda").matmul(torch.as_tensor(weight_vector, device="cuda"))
            return scores.detach().cpu().numpy()
    return matrix @ weight_vector


def score_and_rank(
    features: pd.DataFrame,
    weights: dict[str, float],
    k: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    out = features.copy()
    out["final_score"] = compute_weighted_score(out, weights, device=device)
    out = out.sort_values([EVENT_COL, "final_score", schema.PRODUCT_ID], ascending=[True, False, True])
    out["rank"] = out.groupby(EVENT_COL).cumcount() + 1
    if k is not None:
        out = out[out["rank"] <= k].copy()
    return out


def evaluate_ranked(
    ranked: pd.DataFrame,
    truth: pd.DataFrame,
    events: pd.DataFrame,
    eval_ks: list[int],
) -> dict[str, float]:
    truth_sets = truth.groupby(EVENT_COL)[schema.PRODUCT_ID].apply(lambda items: set(map(int, items))).to_dict()
    all_event_ids = events[EVENT_COL].tolist()
    positive_event_ids = [event_id for event_id in all_event_ids if truth_sets.get(event_id)]
    by_event = {event_id: group for event_id, group in ranked.groupby(EVENT_COL, sort=False)}

    row: dict[str, float] = {
        "n_events": float(len(all_event_ids)),
        "positive_events": float(len(positive_event_ids)),
        "n_candidates": float(len(ranked)),
    }
    for k in eval_ks:
        recalls = []
        ndcgs = []
        all_hits = 0
        positive_hits = 0
        precision_denominator = max(1, len(all_event_ids) * k)
        hit_items = 0
        for event_id in all_event_ids:
            relevant = truth_sets.get(event_id, set())
            group = by_event.get(event_id)
            recommended = []
            if group is not None:
                recommended = group.sort_values("rank")[schema.PRODUCT_ID].head(k).astype(int).tolist()
            hits = [item for item in recommended if item in relevant]
            hit_items += len(set(hits))
            if hits:
                all_hits += 1
                if relevant:
                    positive_hits += 1
            if relevant:
                recalls.append(len(set(hits)) / len(relevant))
                dcg = 0.0
                for rank, item in enumerate(recommended, start=1):
                    if item in relevant:
                        dcg += 1.0 / math.log2(rank + 1)
                ideal_hits = min(len(relevant), k)
                idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
                ndcgs.append(dcg / idcg if idcg else 0.0)
        row[f"recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
        row[f"ndcg_at_{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        row[f"event_hit_rate_at_{k}"] = all_hits / len(all_event_ids) if all_event_ids else 0.0
        row[f"positive_event_hit_rate_at_{k}"] = positive_hits / len(positive_event_ids) if positive_event_ids else 0.0
        row[f"precision_at_{k}"] = hit_items / precision_denominator
    return row


def attach_product_metadata(ranked: pd.DataFrame, products: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    cols = [schema.PRODUCT_ID, "department", "brand", "product_category", "product_type", "package_size"]
    available = [column for column in cols if column in products.columns]
    out = ranked.rename(columns={"product_category": "scoring_product_category"}).merge(
        products[available].drop_duplicates(schema.PRODUCT_ID),
        on=schema.PRODUCT_ID,
        how="left",
    )
    name_parts = [
        out.get("brand", pd.Series("", index=out.index)).fillna("").astype(str),
        out.get("product_category", pd.Series("", index=out.index)).fillna("").astype(str),
        out.get("product_type", pd.Series("", index=out.index)).fillna("").astype(str),
    ]
    out["product_name"] = (
        name_parts[0].str.strip()
        + " / "
        + name_parts[1].str.strip()
        + " / "
        + name_parts[2].str.strip()
    ).str.replace(r"( / )+$", "", regex=True)
    if not truth.empty:
        truth_flags = truth[[EVENT_COL, schema.PRODUCT_ID, "observed_purchase_time"]].drop_duplicates(
            [EVENT_COL, schema.PRODUCT_ID]
        )
        out = out.merge(truth_flags, on=[EVENT_COL, schema.PRODUCT_ID], how="left")
    else:
        out["observed_purchase_time"] = ""
    out["success_within_5d_observed"] = out["observed_purchase_time"].notna() & (out["observed_purchase_time"] != "")
    out["coupon_eligible"] = True
    out["model_name"] = MODEL_NAME
    out["recommend_coupon"] = True
    out = out.rename(columns={"rank": "final_rank"})
    return out


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    eval_ks = sorted(set(args.eval_k))
    max_k = max(eval_ks)
    device = resolve_device(args.device)
    print(f"Scoring device: {device}")

    sources = load_sources(args)
    events = make_eval_events(sources)
    feature_path = args.outputs_dir / "coupon_response_features.csv"
    truth_path = args.outputs_dir / "coupon_response_truth.csv"
    if args.reuse_features and feature_path.exists() and truth_path.exists():
        features = _read_csv(feature_path)
        truth = _read_csv(truth_path)
    else:
        truth = build_truth(events, sources["transactions"], sources["coupons"])
        features = build_feature_table(
            sources,
            events,
            max_global_products=args.max_global_products,
            max_candidates_per_event=args.max_candidates_per_event,
            top_categories=args.top_categories,
            category_products=args.category_products,
        )
    if features.empty:
        raise RuntimeError("No coupon response candidates were generated.")

    features.to_csv(feature_path, index=False)
    truth.to_csv(truth_path, index=False)

    search_rows = []
    configs = weight_grid(args.seed)
    val_events = events[events["split"] == "validation"].copy()
    val_truth = truth[truth[EVENT_COL].isin(set(val_events[EVENT_COL]))].copy()
    val_features = features[features["split"] == "validation"].copy()
    for idx, weights in enumerate(configs):
        ranked = score_and_rank(val_features, weights, k=max_k, device=device)
        row = {"model_name": MODEL_NAME, "config_id": idx}
        row.update({f"w_{key}": value for key, value in weights.items()})
        row.update(evaluate_ranked(ranked, val_truth, val_events, eval_ks))
        search_rows.append(row)

    if not search_rows:
        raise RuntimeError("No validation ranking rows were generated.")
    if args.primary_metric not in search_rows[0]:
        raise ValueError(f"--primary-metric must be one of: {sorted(search_rows[0])}")

    best_idx = max(
        range(len(search_rows)),
        key=lambda i: (
            float(search_rows[i].get(args.primary_metric, 0.0)),
            float(search_rows[i].get("ndcg_at_10", 0.0)),
            float(search_rows[i].get("recall_at_10", 0.0)),
        ),
    )
    best_weights = {
        key.removeprefix("w_"): float(value)
        for key, value in search_rows[best_idx].items()
        if key.startswith("w_")
    }

    comparison_configs = [
        ("coupon_base_intersection", {"base_signal": 1.0}),
        ("coupon_global_popularity", {"global_signal": 1.0}),
        (
            "coupon_repeat_cadence",
            {"repeat_signal": 0.45, "cadence_signal": 0.25, "category_signal": 0.2, "global_signal": 0.1},
        ),
        (MODEL_NAME, best_weights),
    ]

    ranked_all = score_and_rank(features, best_weights, k=max_k, device=device)
    comparison_rows = []
    for model_name, weights in comparison_configs:
        ranked_for_model = ranked_all if model_name == MODEL_NAME else score_and_rank(features, weights, k=max_k, device=device)
        for split in ["validation", "test"]:
            split_events = events[events["split"] == split].copy()
            split_truth = truth[truth[EVENT_COL].isin(set(split_events[EVENT_COL]))].copy()
            split_ranked = ranked_for_model[ranked_for_model[EVENT_COL].isin(set(split_events[EVENT_COL]))].copy()
            row = {"model_name": model_name, "split": split}
            row.update({f"w_{key}": value for key, value in weights.items()})
            row.update(evaluate_ranked(split_ranked, split_truth, split_events, eval_ks))
            comparison_rows.append(row)

    ranked_out = attach_product_metadata(ranked_all, sources["products"], truth)
    ranked_out.to_csv(args.outputs_dir / "candidates_coupon_response_ranker.csv", index=False)
    ranked_out.to_csv(args.outputs_dir / "reranked_recommendations.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.outputs_dir / "coupon_response_weight_search.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.outputs_dir / "coupon_response_model_comparison.csv", index=False)

    print(f"Wrote {args.outputs_dir / 'coupon_response_features.csv'} ({len(features)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_truth.csv'} ({len(truth)} rows)")
    print(f"Wrote {args.outputs_dir / 'candidates_coupon_response_ranker.csv'} ({len(ranked_out)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_weight_search.csv'} ({len(search_rows)} rows)")
    print(f"Wrote {args.outputs_dir / 'coupon_response_model_comparison.csv'} ({len(comparison_rows)} rows)")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
