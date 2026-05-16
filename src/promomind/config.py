"""Shared project paths and default constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_TRAIN_END_WEEK = 40
DEFAULT_VALID_END_WEEK = 46
DEFAULT_TEST_END_WEEK = 53

