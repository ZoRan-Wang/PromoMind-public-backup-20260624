"""Build all local artifacts required by the PromoMind web demo.

This script starts from the committed Complete Journey RDS/RDA files under
data/raw/completejourney and regenerates the ignored local CSV artifacts used by
the browser demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyreadr

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ORIGINAL = REPO_ROOT / "data" / "raw" / "completejourney"
RAW_CSV = REPO_ROOT / "data" / "raw"
OUTPUTS = REPO_ROOT / "outputs"

RAW_EXPORTS = [
    ("transactions.rds", "transactions.csv"),
    ("promotions.rds", "promotions.csv"),
    ("products.rda", "products.csv"),
    ("demographics.rda", "demographics.csv"),
    ("coupons.rda", "coupons.csv"),
    ("coupon_redemptions.rda", "coupon_redemptions.csv"),
    ("campaigns.rda", "campaigns.csv"),
    ("campaign_descriptions.rda", "campaign_descriptions.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-raw-only",
        action="store_true",
        help="Only export data/raw/*.csv from committed RDS/RDA source files.",
    )
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run pytest, compileall, and frontend syntax checks after artifact generation.",
    )
    return parser.parse_args()


def read_r_table(path: Path):
    objects = pyreadr.read_r(str(path))
    if len(objects) != 1:
        raise ValueError(f"Expected one object in {path}, found {list(objects)}")
    return next(iter(objects.values()))


def export_raw_csvs() -> None:
    RAW_CSV.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in RAW_EXPORTS:
        source = RAW_ORIGINAL / source_name
        target = RAW_CSV / target_name
        if not source.exists():
            raise FileNotFoundError(f"Missing committed source file: {source}")
        frame = read_r_table(source)
        frame.to_csv(target, index=False)
        print(f"Wrote {target} ({len(frame)} rows)")


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def copy_output(source_name: str, target_name: str) -> None:
    source = OUTPUTS / source_name
    target = OUTPUTS / target_name
    if not source.exists():
        raise FileNotFoundError(f"Missing generated output: {source}")
    shutil.copyfile(source, target)
    print(f"Copied {source} -> {target}")


def write_manifest() -> None:
    required = [
        "reranked_recommendations.csv",
        "demo_time_name_recommendations.csv",
        "coupon_response_tail_fusion_model_comparison.csv",
        "coupon_response_final_model_comparison.csv",
        "coupon_response_heuristic_model_comparison_zixun.csv",
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_outputs": required,
        "web_demo_command": "python app/web_demo/server.py --port 8766",
        "web_demo_url": "http://127.0.0.1:8766/",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "web_demo_artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def build_artifacts(run_checks: bool) -> None:
    py = sys.executable
    run([py, "scripts/clean_completejourney.py", "--top-products", "10000"])
    run([py, "scripts/run_candidate_models.py", "--models", "all", "--als-backend", "auto"])
    run([py, "scripts/run_cornac_nbr_models.py"])
    run([py, "scripts/run_sota_ensemble.py"])
    run(
        [
            py,
            "scripts/run_coupon_response_xgboost_ranker.py",
            "--device",
            "auto",
            "--search",
            "--label-scheme",
            "pull_forward_interval",
            "--pull-forward-min-days",
            "-1",
            "--pull-forward-max-days",
            "2",
            "--primary-metric",
            "recall_at_20",
        ]
    )
    copy_output(
        "candidates_coupon_response_xgboost_ranker.csv",
        "candidates_coupon_response_xgboost_ranker_pf_interval_best.csv",
    )
    copy_output(
        "coupon_response_xgboost_model_comparison.csv",
        "coupon_response_xgboost_model_comparison_pf_interval_best.csv",
    )
    run(
        [
            py,
            "scripts/run_coupon_response_xgboost_ranker.py",
            "--reuse-features",
            "--device",
            "auto",
            "--search",
            "--label-scheme",
            "pull_forward_interval",
            "--pull-forward-min-days",
            "-1",
            "--pull-forward-max-days",
            "2",
            "--primary-metric",
            "recall_at_20",
            "--use-category-embedding-features",
        ]
    )
    copy_output(
        "candidates_coupon_response_xgboost_ranker.csv",
        "candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv",
    )
    copy_output(
        "coupon_response_xgboost_model_comparison.csv",
        "coupon_response_xgboost_model_comparison_pf_interval_category_embedding.csv",
    )
    run(
        [
            py,
            "scripts/run_coupon_response_tail_fusion.py",
            "--primary-candidates",
            "outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv",
            "--secondary-candidates",
            "outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv",
            "--primary-metric",
            "recall_at_20",
            "--selection-profile",
            "tail_recall",
            "--preserve-min-rank",
            "7",
            "--preserve-max-rank",
            "12",
        ]
    )
    run(
        [
            py,
            "scripts/run_coupon_response_reranking.py",
            "--candidates",
            "outputs/candidates_coupon_response_tail_fusion.csv",
            "--truth",
            "outputs/coupon_response_all_truth.csv",
            "--output-name",
            "reranked_C.csv",
            "--eval-split",
            "both",
        ]
    )
    run([py, "scripts/run_coupon_response_ranker.py", "--reuse-features", "--device", "auto", "--primary-metric", "ndcg_at_10"])
    copy_output("coupon_response_model_comparison.csv", "coupon_response_heuristic_model_comparison_zixun.csv")
    run(
        [
            py,
            "scripts/run_coupon_response_tail_fusion.py",
            "--primary-candidates",
            "outputs/candidates_coupon_response_xgboost_ranker_pf_interval_category_embedding.csv",
            "--secondary-candidates",
            "outputs/candidates_coupon_response_xgboost_ranker_pf_interval_best.csv",
            "--primary-metric",
            "recall_at_20",
            "--selection-profile",
            "tail_recall",
            "--preserve-min-rank",
            "7",
            "--preserve-max-rank",
            "12",
        ]
    )
    run([py, "scripts/build_coupon_timing_demo.py", "--demo-events", "3", "--prediction-length", "5", "--max-label-rows", "5000"])
    write_manifest()
    if run_checks:
        run([py, "-m", "pytest", "-q"])
        run([py, "-m", "compileall", "scripts", "src", "tests", "app"])
        run(["node", "--check", "app/web_demo/static/app.js"])


def main() -> int:
    args = parse_args()
    export_raw_csvs()
    if not args.export_raw_only:
        build_artifacts(run_checks=args.run_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
