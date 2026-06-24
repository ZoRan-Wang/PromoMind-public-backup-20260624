"""Local browser demo backend for PromoMind.

Run from the repository root:

    python app/web_demo/server.py --port 8765
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
OUTPUTS = ROOT / "outputs"
DATA_PROCESSED = ROOT / "data" / "processed"

RECOMMENDATIONS_PATH = OUTPUTS / "reranked_recommendations.csv"
DEMO_TIMING_PATH = OUTPUTS / "demo_time_name_recommendations.csv"
TAIL_FUSION_METRICS_PATH = OUTPUTS / "coupon_response_tail_fusion_model_comparison.csv"
FINAL_MODEL_METRICS_PATH = OUTPUTS / "coupon_response_final_model_comparison.csv"
TRANSACTIONS_PATH = DATA_PROCESSED / "transactions_clean.csv"
PRODUCT_FEATURES_PATH = DATA_PROCESSED / "product_features.csv"
HISTORY_LIMIT = 30

PRESENTATION_PRESETS = [
    {
        "event_id": "955_19_20171115",
        "label": "Max hit baby-food run",
        "tag": "HIT 4",
        "story": "Four Top-10 products were observed in the held-out response window.",
        "tone": "hit",
        "coupon_slots": 10,
    },
    {
        "event_id": "1074_22_20171206",
        "label": "Cheese repeat basket",
        "tag": "HIT 3",
        "story": "Repeat cheese and refrigerated dough items convert within the five-day window.",
        "tone": "hit",
        "coupon_slots": 10,
    },
    {
        "event_id": "1418_19_20171115",
        "label": "Baby-food repeat",
        "tag": "HIT 3",
        "story": "Prior baby-food behavior drives multiple observed Top-10 purchases.",
        "tone": "hit",
        "coupon_slots": 10,
    },
    {
        "event_id": "1804_22_20171206",
        "label": "Seasonal card hit",
        "tag": "HIT 3",
        "story": "The campaign captures greeting-card demand and refrigerated juice purchases.",
        "tone": "hit",
        "coupon_slots": 10,
    },
    {
        "event_id": "1216_20_20171127",
        "label": "Reasonable miss laundry",
        "tag": "NO HIT",
        "story": "Laundry detergent is a planned household-care category, so a five-day window can miss intent.",
        "tone": "miss",
        "coupon_slots": 10,
    },
    {
        "event_id": "1142_22_20171206",
        "label": "Reasonable miss cards",
        "tag": "NO HIT",
        "story": "Greeting cards are seasonal and occasion-driven, so the recommendation can be sensible without immediate purchase.",
        "tone": "miss",
        "coupon_slots": 10,
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def as_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return 0.0
    return float(value)


def as_int(value: str) -> int:
    if value == "" or value.lower() == "nan":
        return 0
    return int(float(value))


def as_bool(value: str) -> bool:
    return value.lower() == "true" or value == "1" or value == "1.0"


def product_key(value: str) -> str:
    return str(as_int(value))


def metric_row(rows: list[dict[str, str]], model_name: str, split: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("model_name") == model_name and row.get("split") == split]
    return matches[0]


class DemoData:
    def __init__(self) -> None:
        self.recommendations = read_csv_rows(RECOMMENDATIONS_PATH)
        self.demo_timing = read_csv_rows(DEMO_TIMING_PATH)
        self.tail_metrics = read_csv_rows(TAIL_FUSION_METRICS_PATH)
        self.final_metrics = read_csv_rows(FINAL_MODEL_METRICS_PATH)
        self.product_lookup = self._build_product_lookup()
        self.events = self._build_events()
        self.demo_events = self._build_demo_events()
        self.household_history = self._build_household_history()
        self.metric_summary = self._build_metric_summary()

    def _build_product_lookup(self) -> dict[str, dict[str, str]]:
        products = {}
        for row in iter_csv_rows(PRODUCT_FEATURES_PATH):
            product_name = f"{row['brand']} / {row['product_category']} / {row['product_type']}"
            products[product_key(row["product_id"])] = {
                "product_name": product_name,
                "department": row["department"],
                "product_category": row["product_category"],
                "brand": row["brand"],
            }
        return products

    def _build_events(self) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        test_rows = [row for row in self.recommendations if row.get("split") == "test"]
        for row in test_rows:
            grouped[row["event_id"]].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda item: as_int(item["rank"]))
        return dict(sorted(grouped.items()))

    def _build_demo_events(self) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.demo_timing:
            event_id = f"demo_{row['household_id']}_{row['campaign_id']}"
            grouped[event_id].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda item: as_int(item["rank"]))
        return dict(sorted(grouped.items()))

    def _build_household_history(self) -> dict[str, list[dict[str, object]]]:
        households = self._relevant_households()
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in iter_csv_rows(TRANSACTIONS_PATH):
            household_id = row["household_id"]
            if household_id in households:
                product = self.product_lookup[product_key(row["product_id"])]
                grouped[household_id].append(
                    {
                        "basket_id": row["basket_id"],
                        "purchase_time": row["transaction_timestamp"],
                        "product_id": as_int(row["product_id"]),
                        "product_name": product["product_name"],
                        "department": product["department"],
                        "product_category": product["product_category"],
                        "brand": product["brand"],
                        "quantity": as_float(row["quantity"]),
                        "sales_value": as_float(row["sales_value"]),
                    }
                )
        for rows in grouped.values():
            rows.sort(key=lambda item: str(item["purchase_time"]))
        return grouped

    def _relevant_households(self) -> set[str]:
        households = set()
        for rows in self.events.values():
            households.add(rows[0]["household_id"])
        for rows in self.demo_events.values():
            households.add(rows[0]["household_id"])
        return households

    def _build_metric_summary(self) -> dict[str, object]:
        baseline = metric_row(self.final_metrics, "coupon_base_intersection", "test")
        xgboost = metric_row(self.final_metrics, "coupon_response_xgboost_ranker", "test")
        tail_fusion = metric_row(self.tail_metrics, "coupon_response_tail_fusion", "test")
        rows = [
            self.metric_payload("Candidate-only", baseline),
            self.metric_payload("XGBoost LTR", xgboost),
            self.metric_payload("Final tail fusion", tail_fusion),
        ]
        return {
            "rows": rows,
            "headline": {
                "recall_at_10": rows[-1]["recall_at_10"],
                "ndcg_at_10": rows[-1]["ndcg_at_10"],
                "positive_hit_at_10": rows[-1]["positive_event_hit_rate_at_10"],
                "recall_lift_vs_baseline": rows[-1]["recall_at_10"] - rows[0]["recall_at_10"],
                "hit_lift_vs_baseline": rows[-1]["positive_event_hit_rate_at_10"]
                - rows[0]["positive_event_hit_rate_at_10"],
            },
        }

    @staticmethod
    def metric_payload(label: str, row: dict[str, str]) -> dict[str, object]:
        return {
            "label": label,
            "recall_at_10": as_float(row["recall_at_10"]),
            "ndcg_at_10": as_float(row["ndcg_at_10"]),
            "positive_event_hit_rate_at_10": as_float(row["positive_event_hit_rate_at_10"]),
            "recall_at_20": as_float(row["recall_at_20"]),
            "ndcg_at_20": as_float(row["ndcg_at_20"]),
            "positive_event_hit_rate_at_20": as_float(row["positive_event_hit_rate_at_20"]),
        }

    def event_options(self) -> list[dict[str, object]]:
        demo_options = []
        for event_id, rows in self.demo_events.items():
            first = rows[0]
            demo_options.append(
                {
                    "event_id": event_id,
                    "household_id": as_int(first["household_id"]),
                    "campaign_id": as_int(first["campaign_id"]),
                    "campaign_type": "Demo",
                    "coupon_start_date": first["coupon_start_date"],
                    "predicted_purchase_time": first["predicted_purchase_time"],
                    "top_product_name": first["product_name"],
                    "top_product_category": first["product_category"],
                    "positive_in_top10": sum(
                        1 for row in rows[:10] if as_bool(row["success_within_5d_observed"])
                    ),
                    "coupon_eligible_top10": sum(1 for row in rows[:10] if as_bool(row["coupon_eligible"])),
                    "event_group": "history_showcase",
                }
            )
        final_options = []
        for event_id, rows in self.events.items():
            first = rows[0]
            top10 = rows[:10]
            final_options.append(
                {
                    "event_id": event_id,
                    "household_id": as_int(first["household_id"]),
                    "campaign_id": as_int(first["campaign_id"]),
                    "campaign_type": first["campaign_type"],
                    "coupon_start_date": first["coupon_start_date"],
                    "predicted_purchase_time": first["predicted_purchase_time"],
                    "top_product_name": first["product_name"],
                    "top_product_category": first["product_category"],
                    "positive_in_top10": sum(
                        1 for row in top10 if as_bool(row["success_within_5d_observed"])
                    ),
                    "coupon_eligible_top10": sum(1 for row in top10 if as_bool(row["coupon_eligible"])),
                    "event_group": "heldout_test",
                }
            )
        final_options.sort(
            key=lambda row: (
                -as_int(str(row["positive_in_top10"])),
                as_int(str(row["household_id"])),
                as_int(str(row["campaign_id"])),
            )
        )
        demo_options.sort(key=lambda row: as_int(str(row["household_id"])))
        return final_options + demo_options

    def recommendation_payload(self, event_id: str, coupon_slots: int) -> dict[str, object]:
        if event_id in self.demo_events:
            return self.demo_payload(event_id, coupon_slots)
        rows = self.events[event_id]
        first = rows[0]
        history, history_summary = self.recent_history(first["household_id"], first["coupon_start_date"])
        top20 = rows[:20]
        top10 = rows[:10]
        recommended = [self.row_payload(row, coupon_slots) for row in top20]
        return {
            "event": {
                "event_id": event_id,
                "household_id": as_int(first["household_id"]),
                "campaign_id": as_int(first["campaign_id"]),
                "campaign_type": first["campaign_type"],
                "coupon_start_date": first["coupon_start_date"],
                "predicted_purchase_time": first["predicted_purchase_time"],
                "model_name": first["model_name"],
            },
            "kpis": {
                "top10_positive_hits": sum(1 for row in top10 if as_float(row["label"]) > 0),
                "top10_observed_success": sum(
                    1 for row in top10 if as_bool(row["success_within_5d_observed"])
                ),
                "coupon_slots": coupon_slots,
                "coupon_eligible_in_slots": sum(
                    1
                    for row in rows[:coupon_slots]
                    if as_bool(row["coupon_eligible"])
                ),
                "avg_top10_score": sum(as_float(row["final_score"]) for row in top10) / len(top10),
            },
            "history_summary": history_summary,
            "history": [self.history_payload(row) for row in history],
            "recommendations": recommended,
        }

    def demo_payload(self, event_id: str, coupon_slots: int) -> dict[str, object]:
        rows = self.demo_events[event_id]
        first = rows[0]
        history, history_summary = self.recent_history(first["household_id"], first["coupon_start_date"])
        top10 = rows[:10]
        return {
            "event": {
                "event_id": event_id,
                "household_id": as_int(first["household_id"]),
                "campaign_id": as_int(first["campaign_id"]),
                "campaign_type": "Showcase event",
                "coupon_start_date": first["coupon_start_date"],
                "predicted_purchase_time": first["predicted_purchase_time"],
                "model_name": first["source_model"],
            },
            "kpis": {
                "top10_positive_hits": sum(
                    1 for row in top10 if as_bool(row["success_within_5d_observed"])
                ),
                "top10_observed_success": sum(
                    1 for row in top10 if as_bool(row["success_within_5d_observed"])
                ),
                "coupon_slots": coupon_slots,
                "coupon_eligible_in_slots": sum(
                    1
                    for row in rows[:coupon_slots]
                    if as_bool(row["coupon_eligible"])
                ),
                "avg_top10_score": sum(1 / as_int(row["rank"]) for row in top10) / len(top10),
            },
            "history_summary": history_summary,
            "history": [self.history_payload(row) for row in history],
            "recommendations": [self.demo_row_payload(row, coupon_slots) for row in rows[:20]],
        }

    def recent_history(self, household_id: str, coupon_start_date: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        all_rows = self.household_history.get(household_id, [])
        prior_rows = [row for row in all_rows if str(row["purchase_time"]) < coupon_start_date]
        return prior_rows[-HISTORY_LIMIT:], {
            "shown": min(len(prior_rows), HISTORY_LIMIT),
            "total_before_coupon": len(prior_rows),
            "cutoff": coupon_start_date,
        }

    @staticmethod
    def history_payload(row: dict[str, object]) -> dict[str, object]:
        return {
            "basket_id": row["basket_id"],
            "purchase_time": row["purchase_time"],
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "department": row["department"],
            "product_category": row["product_category"],
            "brand": row["brand"],
            "quantity": row["quantity"],
            "sales_value": row["sales_value"],
        }

    @staticmethod
    def row_payload(row: dict[str, str], coupon_slots: int) -> dict[str, object]:
        rank = as_int(row["rank"])
        return {
            "rank": rank,
            "product_id": as_int(row["product_id"]),
            "product_name": row["product_name"],
            "department": row["department"],
            "product_category": row["product_category"],
            "brand": row["brand"],
            "final_score": as_float(row["final_score"]),
            "coupon_eligible": as_bool(row["coupon_eligible"]),
            "recommend_coupon": rank <= coupon_slots,
            "observed_success": as_bool(row["success_within_5d_observed"]),
            "observed_purchase_time": row["observed_purchase_time"],
            "fusion_source": row["fusion_source"],
            "repeat_signal": as_float(row["repeat_signal"]),
            "cadence_signal": as_float(row["cadence_signal"]),
            "global_signal": as_float(row["global_signal"]),
            "label": as_float(row["label"]),
            "reason": reason_for(row),
        }

    @staticmethod
    def demo_row_payload(row: dict[str, str], coupon_slots: int) -> dict[str, object]:
        rank = as_int(row["rank"])
        return {
            "rank": rank,
            "product_id": as_int(row["product_id"]),
            "product_name": row["product_name"],
            "department": row["department"],
            "product_category": row["product_category"],
            "brand": row["brand"],
            "final_score": 1 / rank,
            "coupon_eligible": as_bool(row["coupon_eligible"]),
            "recommend_coupon": rank <= coupon_slots,
            "observed_success": as_bool(row["success_within_5d_observed"]),
            "observed_purchase_time": row["observed_purchase_time"],
            "fusion_source": row["source_model"],
            "repeat_signal": 0.0,
            "cadence_signal": 0.0,
            "global_signal": 0.0,
            "label": 0.0,
            "reason": demo_reason_for(row),
        }


def reason_for(row: dict[str, str]) -> str:
    reasons = []
    if as_bool(row["has_prior_product"]):
        reasons.append("prior household purchase")
    if as_float(row["cadence_signal"]) >= 0.5:
        reasons.append("repurchase timing match")
    if as_float(row["global_signal"]) >= 0.7:
        reasons.append("strong campaign response signal")
    if as_bool(row["coupon_eligible"]):
        reasons.append("coupon eligible")
    if row["fusion_source"] == "category_tail":
        reasons.append("category tail coverage")
    if not reasons:
        reasons.append("ranked by response score")
    return ", ".join(reasons[:3])


def demo_reason_for(row: dict[str, str]) -> str:
    reasons = ["time-product pair"]
    if as_bool(row["coupon_eligible"]):
        reasons.append("coupon eligible")
    if as_bool(row["success_within_5d_observed"]):
        reasons.append("observed within 5 days")
    reasons.append(row["source_model"])
    return ", ".join(reasons[:3])


DATA = DemoData()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/static/style.css":
            self.send_static("style.css", "text/css; charset=utf-8")
        elif parsed.path == "/static/app.js":
            self.send_static("app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/api/bootstrap":
            self.send_json(
                {
                    "metrics": DATA.metric_summary,
                    "events": DATA.event_options(),
                    "presets": PRESENTATION_PRESETS,
                }
            )
        elif parsed.path == "/api/recommendations":
            query = parse_qs(parsed.query)
            event_id = query["event_id"][0]
            coupon_slots = as_int(query.get("coupon_slots", ["3"])[0])
            self.send_json(DATA.recommendation_payload(event_id, coupon_slots))
        else:
            self.send_response(404)
            self.end_headers()

    def send_static(self, filename: str, content_type: str) -> None:
        body = (STATIC_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PromoMind local web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PromoMind demo running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
