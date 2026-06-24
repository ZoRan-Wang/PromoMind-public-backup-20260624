"""Restore ignored local data/model artifacts from committed cache packages."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "artifacts" / "local_cache"
PACKAGES = [
    "PromoMind_raw_csv_cache_20260624_zixun.zip",
    "PromoMind_processed_cache_20260624_zixun.zip",
    "PromoMind_outputs_core_20260624_zixun.zip",
    "PromoMind_outputs_coupon_response_all_features_20260624_zixun.zip",
    "PromoMind_outputs_coupon_response_features_20260624_zixun.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove existing ignored artifact files before extraction.")
    parser.add_argument("--list", action="store_true", help="List package names and contents without extracting.")
    return parser.parse_args()


def clean_outputs() -> None:
    for path in [REPO_ROOT / "outputs", REPO_ROOT / "data" / "processed"]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (REPO_ROOT / "data" / "raw").glob("*.csv"):
        path.unlink()


def list_packages() -> None:
    for package in PACKAGES:
        path = CACHE_DIR / package
        print(f"\n{package}")
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                print(f"  {name}")


def extract_packages() -> None:
    for package in PACKAGES:
        path = CACHE_DIR / package
        print(f"Extracting {path}")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(REPO_ROOT)


def main() -> int:
    args = parse_args()
    if args.list:
        list_packages()
        return 0
    if args.clean:
        clean_outputs()
    extract_packages()
    print("\nRestored local artifacts. Run: python app/web_demo/server.py --port 8766")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
