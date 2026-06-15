"""Build the coupon-timing demo artifacts.

This script intentionally uses external recommender code for the model step:

* TBP/TARS external repo:
  https://github.com/GiulioRossetti/tbp-next-basket

The repository is Python 2 code, so the script converts the required files to a
local ignored directory at runtime and applies two compatibility guards for
modern Python/Numpy execution. The recommender algorithm, model API, and train
partition defaults remain TBP's defaults from the external project.

Coupon-label definition used here:

    success = the household bought the coupon product within 5 days after the
              campaign start date.

Generated files:

* outputs/coupon_timing_training_labels.csv
* outputs/demo_time_name_recommendations.csv
* outputs/demo_history_input.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
OUTPUTS = REPO_ROOT / "outputs"
EXTERNAL = REPO_ROOT / "external"
TBP_REPO = EXTERNAL / "tbp-next-basket"
TBP_CONVERTED = EXTERNAL / ".converted_tbp"
TBP_URL = "https://github.com/GiulioRossetti/tbp-next-basket.git"
TBP_SOURCE = "GiulioRossetti/tbp-next-basket"


@dataclass(frozen=True)
class CouponEvent:
    household_id: int
    campaign_id: int
    coupon_start: pd.Timestamp
    coupon_end: pd.Timestamp
    baskets_before: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coupon timing demo outputs.")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL)
    parser.add_argument("--demo-events", type=int, default=3)
    parser.add_argument("--prediction-length", type=int, default=5)
    parser.add_argument("--label-negative-ratio", type=int, default=4)
    parser.add_argument("--max-label-rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bootstrap", action="store_true", help="Do not clone external TBP repo if missing.")
    return parser.parse_args()


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def ensure_tbp_repo(external_dir: Path, no_bootstrap: bool) -> Path:
    repo = external_dir / "tbp-next-basket"
    if repo.exists():
        return repo
    if no_bootstrap:
        raise FileNotFoundError(
            f"External TBP repo missing at {repo}. Clone {TBP_URL} or rerun without --no-bootstrap."
        )
    external_dir.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", TBP_URL, str(repo)], cwd=REPO_ROOT)
    return repo


def convert_tbp_runtime(repo: Path, converted_dir: Path) -> Path:
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package

    if converted_dir.exists():
        shutil.rmtree(converted_dir)
    files = [
        "tbp/tbp.py",
        "tbp/tars.py",
        "tbp/__init__.py",
        "utils/data_management.py",
        "utils/__init__.py",
        "evaluation/evaluation_measures.py",
        "evaluation/calculate_aggregate_statistics.py",
        "evaluation/__init__.py",
    ]
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    for relative in files:
        source = repo / relative
        target = converted_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        if relative == "tbp/tars.py":
            text = text.replace(
                "    h = 2.0 * iqr / n**(1.0/3.0)\n"
                "    k = math.ceil((np.max(x) - np.min(x))/h)\n",
                "    h = 2.0 * iqr / n**(1.0/3.0)\n"
                "    if h == 0 or np.isnan(h):\n"
                "        return float('inf')\n"
                "    k = math.ceil((np.max(x) - np.min(x))/h)\n",
            )
            text = text.replace(
                "nbr_bins = np.round(estimate_nbr_bins(self.item_estimated_nbr_periods.values()))",
                "nbr_bins = int(np.round(estimate_nbr_bins(self.item_estimated_nbr_periods.values())))",
            )
            text = text.replace(
                "nbr_bins = np.round(estimate_nbr_bins(self.item_nbr_periods.values()))",
                "nbr_bins = int(np.round(estimate_nbr_bins(self.item_nbr_periods.values())))",
            )
        converted = str(tool.refactor_string(text, str(source)))
        target.write_text(converted, encoding="utf-8")
    return converted_dir


def import_tbp(converted_dir: Path):
    path = str(converted_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from tbp.tbp import TBP
    from utils.data_management import remap_items_with_data, split_train_test

    return TBP, split_train_test, remap_items_with_data


def product_name(row: pd.Series) -> str:
    pieces = [
        str(row.get("brand", "") or "").strip(),
        str(row.get("product_category", "") or "").strip(),
        str(row.get("product_type", "") or "").strip(),
    ]
    pieces = [piece for piece in pieces if piece and piece.lower() != "nan"]
    return " / ".join(pieces) if pieces else str(int(row["product_id"]))


def load_sources(raw_dir: Path) -> dict[str, pd.DataFrame]:
    transactions = _read_csv(raw_dir / "transactions.csv", parse_dates=["transaction_timestamp"])
    campaigns = _read_csv(raw_dir / "campaigns.csv")
    campaign_desc = _read_csv(raw_dir / "campaign_descriptions.csv", parse_dates=["start_date", "end_date"])
    coupons = _read_csv(raw_dir / "coupons.csv")
    products = _read_csv(raw_dir / "products.csv")
    products["product_id_str"] = products["product_id"].astype(int).astype(str)
    products["product_name"] = products.apply(product_name, axis=1)
    return {
        "transactions": transactions,
        "campaigns": campaigns,
        "campaign_desc": campaign_desc,
        "coupons": coupons,
        "products": products,
    }


def build_coupon_labels(
    sources: dict[str, pd.DataFrame],
    negative_ratio: int,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = pd.Series(range(10_000)).sample(frac=1.0, random_state=seed).to_list()
    rng_idx = 0
    raw_cap = max_rows * max(2, negative_ratio + 2) if max_rows > 0 else None
    transactions = sources["transactions"].copy()
    transactions["product_id"] = transactions["product_id"].astype(int)
    campaigns = sources["campaigns"].merge(sources["campaign_desc"], on="campaign_id", how="left")
    coupons = sources["coupons"].copy()
    coupons["product_id"] = coupons["product_id"].astype(int)
    product_lookup = sources["products"][["product_id", "product_name", "department", "product_category"]]

    rows: list[dict[str, Any]] = []
    positives_seen: set[tuple[int, int, int]] = set()

    for campaign_id, campaign_group in campaigns.groupby("campaign_id", sort=True):
        coupon_group = coupons[coupons["campaign_id"] == campaign_id]
        if coupon_group.empty:
            continue
        start = pd.Timestamp(campaign_group["start_date"].iloc[0])
        end = start + pd.Timedelta(days=5)
        households = set(campaign_group["household_id"].dropna().astype(int).tolist())
        coupon_products = coupon_group[["coupon_upc", "product_id"]].drop_duplicates()
        product_set = set(coupon_products["product_id"].tolist())

        window = transactions[
            transactions["household_id"].astype(int).isin(households)
            & transactions["product_id"].isin(product_set)
            & (transactions["transaction_timestamp"] >= start)
            & (transactions["transaction_timestamp"] <= end)
        ]
        positive = (
            window.sort_values("transaction_timestamp")
            .drop_duplicates(["household_id", "product_id"])
            [["household_id", "product_id", "transaction_timestamp"]]
        )
        for hit in positive.itertuples(index=False):
            coupon_upc = coupon_products.loc[coupon_products["product_id"] == int(hit.product_id), "coupon_upc"].iloc[0]
            key = (int(hit.household_id), int(campaign_id), int(hit.product_id))
            positives_seen.add(key)
            rows.append(
                {
                    "household_id": int(hit.household_id),
                    "campaign_id": int(campaign_id),
                    "coupon_upc": int(coupon_upc),
                    "product_id": int(hit.product_id),
                    "coupon_start_date": start.date().isoformat(),
                    "success_window_end": end.date().isoformat(),
                    "success_within_5d": True,
                    "observed_purchase_time": pd.Timestamp(hit.transaction_timestamp).isoformat(),
                    "label_source": "campaign_start_to_purchase_within_5_days",
                }
            )

        target_negatives = min(len(positive) * max(1, negative_ratio) + 20, 500)
        if target_negatives <= 0:
            continue
        household_list = sorted(households)
        product_rows = coupon_products.drop_duplicates("product_id")
        negatives_added = 0
        for _ in range(target_negatives * 3):
            if raw_cap is not None and len(rows) >= raw_cap:
                break
            if not household_list or product_rows.empty:
                break
            household = household_list[rng[rng_idx % len(rng)] % len(household_list)]
            rng_idx += 1
            product_row = product_rows.iloc[rng[rng_idx % len(rng)] % len(product_rows)]
            rng_idx += 1
            product_id = int(product_row["product_id"])
            key = (household, int(campaign_id), product_id)
            if key in positives_seen:
                continue
            rows.append(
                {
                    "household_id": household,
                    "campaign_id": int(campaign_id),
                    "coupon_upc": int(product_row["coupon_upc"]),
                    "product_id": product_id,
                    "coupon_start_date": start.date().isoformat(),
                    "success_window_end": end.date().isoformat(),
                    "success_within_5d": False,
                    "observed_purchase_time": "",
                    "label_source": "campaign_start_to_purchase_within_5_days",
                }
            )
            negatives_added += 1
            if negatives_added >= target_negatives:
                break
        if raw_cap is not None and len(rows) >= raw_cap:
            break

    labels = pd.DataFrame(rows).drop_duplicates(["household_id", "campaign_id", "product_id"])
    if labels.empty:
        return labels

    if max_rows > 0 and len(labels) > max_rows:
        positive = labels[labels["success_within_5d"]]
        negative = labels[~labels["success_within_5d"]]
        desired_positive = min(
            len(positive),
            max(1, max_rows // (max(1, negative_ratio) + 1)),
        )
        desired_negative = min(len(negative), max_rows - desired_positive)
        sampled_parts = []
        if desired_positive:
            sampled_parts.append(positive.sample(n=desired_positive, random_state=seed))
        if desired_negative:
            sampled_parts.append(negative.sample(n=desired_negative, random_state=seed + 1))
        labels = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else labels.head(0)
        remaining_slots = max_rows - len(labels)
        if remaining_slots > 0:
            remaining = labels.iloc[0:0]
            chosen_index = labels.index
            all_remaining = pd.DataFrame(rows).drop_duplicates(
                ["household_id", "campaign_id", "product_id"]
            ).drop(index=chosen_index, errors="ignore")
            if not all_remaining.empty:
                remaining = all_remaining.sample(
                    n=min(remaining_slots, len(all_remaining)),
                    random_state=seed + 2,
                )
            labels = pd.concat([labels, remaining], ignore_index=False)
        labels = labels.sort_values(["campaign_id", "household_id", "product_id"]).reset_index(drop=True)

    labels = labels.merge(product_lookup, on="product_id", how="left")
    return labels.head(max_rows) if max_rows > 0 else labels


def choose_demo_events(sources: dict[str, pd.DataFrame], n_events: int) -> list[CouponEvent]:
    transactions = sources["transactions"][["household_id", "basket_id", "transaction_timestamp"]].drop_duplicates()
    campaigns = sources["campaigns"].merge(sources["campaign_desc"], on="campaign_id", how="left")
    candidates: list[CouponEvent] = []
    for event in campaigns.itertuples(index=False):
        start = pd.Timestamp(event.start_date)
        if start < pd.Timestamp("2017-03-01"):
            continue
        baskets_before = int(
            (
                (transactions["household_id"].astype(int) == int(event.household_id))
                & (transactions["transaction_timestamp"] < start)
            ).sum()
        )
        if 10 <= baskets_before <= 35:
            candidates.append(
                CouponEvent(
                    household_id=int(event.household_id),
                    campaign_id=int(event.campaign_id),
                    coupon_start=start,
                    coupon_end=pd.Timestamp(event.end_date),
                    baskets_before=baskets_before,
                )
            )
    candidates = sorted(candidates, key=lambda item: (item.baskets_before, item.coupon_start, item.household_id))
    if not candidates:
        return []
    step = max(1, len(candidates) // max(1, n_events))
    return candidates[::step][:n_events]


def tbp_customer_from_history(transactions: pd.DataFrame, household_id: int, cutoff: pd.Timestamp) -> dict[str, Any]:
    history = transactions[
        (transactions["household_id"].astype(int) == household_id)
        & (transactions["transaction_timestamp"] < cutoff)
    ].sort_values("transaction_timestamp")
    customer = {"customer_id": household_id, "data": {}}
    for basket_id, group in history.groupby("basket_id", sort=False):
        stamp = pd.Timestamp(group["transaction_timestamp"].iloc[0])
        basket_key = stamp.strftime("%Y_%m_%d_%H_%M") + f"_{int(basket_id)}"
        customer["data"][basket_key] = {
            "anno": int(stamp.year),
            "mese_n": int(stamp.month),
            "giorno_n": int(stamp.day),
            "ora": int(stamp.hour),
            "minuto": int(stamp.minute),
            "basket": {
                str(int(row.product_id)): [float(row.quantity)]
                for row in group[["product_id", "quantity"]].itertuples(index=False)
            },
        }
    return customer


def predict_event(
    event: CouponEvent,
    sources: dict[str, pd.DataFrame],
    tbp_api: tuple[Any, Any, Any],
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    TBP, split_train_test, remap_items_with_data = tbp_api
    transactions = sources["transactions"]
    customer = tbp_customer_from_history(transactions, event.household_id, event.coupon_start)
    train, _ = split_train_test(
        {event.household_id: customer},
        split_mode="loo",
        min_number_of_basket=10,
        min_basket_size=1,
        max_basket_size=float("inf"),
        min_item_occurrences=2,
        item2category=None,
    )
    if event.household_id not in train:
        return pd.DataFrame(), pd.DataFrame()

    train, new2old, _ = remap_items_with_data(train)
    model = TBP().build_model(train[event.household_id])
    prediction_time = event.coupon_start + pd.Timedelta(days=5)
    predicted = model.predict(
        train[event.household_id]["data"],
        prediction_time.to_pydatetime(),
        nbr_patterns=None,
        pred_length=prediction_length,
    )
    predicted_products = [int(new2old[item]) for item in predicted]

    products = sources["products"][
        ["product_id", "product_name", "department", "product_category", "brand"]
    ].copy()
    coupons = sources["coupons"]
    active_products = set(coupons.loc[coupons["campaign_id"] == event.campaign_id, "product_id"].astype(int))
    purchase_window = transactions[
        (transactions["household_id"].astype(int) == event.household_id)
        & (transactions["product_id"].astype(int).isin(predicted_products))
        & (transactions["transaction_timestamp"] >= event.coupon_start)
        & (transactions["transaction_timestamp"] <= prediction_time)
    ].sort_values("transaction_timestamp")
    observed = (
        purchase_window.drop_duplicates("product_id")
        .set_index(purchase_window["product_id"].astype(int))["transaction_timestamp"]
        .to_dict()
    )

    rows = []
    for rank, product_id in enumerate(predicted_products, start=1):
        rows.append(
            {
                "household_id": event.household_id,
                "campaign_id": event.campaign_id,
                "coupon_start_date": event.coupon_start.date().isoformat(),
                "predicted_purchase_time": prediction_time.isoformat(),
                "rank": rank,
                "product_id": product_id,
                "source_model": "TBP/TARS external",
                "source_repo": TBP_SOURCE,
                "coupon_eligible": product_id in active_products,
                "success_within_5d_observed": product_id in observed,
                "observed_purchase_time": pd.Timestamp(observed[product_id]).isoformat()
                if product_id in observed
                else "",
                "baskets_before_coupon": event.baskets_before,
            }
        )
    recs = pd.DataFrame(rows).merge(products, on="product_id", how="left")

    history_rows = []
    history = transactions[
        (transactions["household_id"].astype(int) == event.household_id)
        & (transactions["transaction_timestamp"] < event.coupon_start)
    ].sort_values("transaction_timestamp")
    for row in history.tail(40).itertuples(index=False):
        history_rows.append(
            {
                "household_id": int(row.household_id),
                "campaign_id": event.campaign_id,
                "basket_id": int(row.basket_id),
                "purchase_time": pd.Timestamp(row.transaction_timestamp).isoformat(),
                "product_id": int(row.product_id),
            }
        )
    history_frame = pd.DataFrame(history_rows).merge(products, on="product_id", how="left")
    return recs, history_frame


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    tbp_repo = ensure_tbp_repo(args.external_dir, args.no_bootstrap)
    converted = convert_tbp_runtime(tbp_repo, args.external_dir / ".converted_tbp")
    tbp_api = import_tbp(converted)
    sources = load_sources(args.raw_dir)

    labels = build_coupon_labels(
        sources,
        negative_ratio=args.label_negative_ratio,
        max_rows=args.max_label_rows,
        seed=args.seed,
    )
    labels.to_csv(args.outputs_dir / "coupon_timing_training_labels.csv", index=False)
    print(f"Wrote {args.outputs_dir / 'coupon_timing_training_labels.csv'} ({len(labels)} rows)")
    print(labels["success_within_5d"].value_counts(dropna=False).to_string())

    events = choose_demo_events(sources, args.demo_events)
    all_recs = []
    all_history = []
    for event in events:
        recs, history = predict_event(event, sources, tbp_api, args.prediction_length)
        if not recs.empty:
            all_recs.append(recs)
        if not history.empty:
            all_history.append(history)

    recommendations = pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()
    history_input = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    recommendations.to_csv(args.outputs_dir / "demo_time_name_recommendations.csv", index=False)
    history_input.to_csv(args.outputs_dir / "demo_history_input.csv", index=False)
    print(f"Wrote {args.outputs_dir / 'demo_time_name_recommendations.csv'} ({len(recommendations)} rows)")
    print(f"Wrote {args.outputs_dir / 'demo_history_input.csv'} ({len(history_input)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
