"""Centralized path constants.

Keeping these in one place makes the raw/interim/processed/metadata
separation enforceable in code review: nothing outside this module should
need to know the literal directory layout, and in particular nothing should
ever construct a write path under RAW_ROOT.
"""
from __future__ import annotations

from pathlib import Path

# backend/src/ingest/paths.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = BACKEND_ROOT / "data"

RAW_ROOT = DATA_ROOT / "raw" / "ipinyou.contest.dataset"  # READ-ONLY, never write here
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
METADATA_ROOT = DATA_ROOT / "metadata"

SEASON_DIRS = {2: RAW_ROOT / "training2nd", 3: RAW_ROOT / "training3rd"}

FILE_TYPES = ("bid", "imp", "clk", "conv")
FILE_TYPE_NAMES = {"bid": "bids", "imp": "impressions", "clk": "clicks", "conv": "conversions"}
