"""Centralized path constants.

Keeping these in one place makes the raw/interim/processed/metadata
separation enforceable in code review: nothing outside this module should
need to know the literal directory layout, and in particular nothing should
ever construct a write path under RAW_ROOT.

SAMPLE_ROOT / output_root(): a `--sample` dev run must never write into, or
influence the "skip if output exists" logic of, the real `PROCESSED_ROOT`
partitions -- a 20k-row sample file living at the exact same path as a real
partition file is indistinguishable from a truncated real run, and a later
full run would silently skip re-generating it. So sample output gets its own
root entirely, under `INTERIM_ROOT` (it is exactly the kind of disposable,
regeneratable-on-demand intermediate that directory is for), mirroring the
same `<file_type_dir>/season=.../date=.../part-0.parquet` layout used under
`PROCESSED_ROOT`. Callers (ingest.py, validate.py, dataset.py) ask this
module for the right root via `output_root(sample)` rather than branching on
`--sample` themselves and reconstructing the literal path -- same
"nothing outside this module knows the layout" property extended to cover
the sample/production choice, not just raw/interim/processed/metadata.
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

# Dedicated root for `--sample` dev runs -- see module docstring. Lives under
# INTERIM_ROOT (disposable/regeneratable), never under PROCESSED_ROOT.
SAMPLE_ROOT = INTERIM_ROOT / "sample"

SEASON_DIRS = {2: RAW_ROOT / "training2nd", 3: RAW_ROOT / "training3rd"}

FILE_TYPES = ("bid", "imp", "clk", "conv")
FILE_TYPE_NAMES = {"bid": "bids", "imp": "impressions", "clk": "clicks", "conv": "conversions"}


def output_root(sample: bool) -> Path:
    """The Parquet output root to write/read for a given `--sample` mode.

    The single place that decides "sample run -> SAMPLE_ROOT, real run ->
    PROCESSED_ROOT" -- everything downstream of ingestion (validate.py,
    dataset.py) should call this instead of re-deriving the choice, so the
    two roots can never drift apart or get inverted in one caller but not
    another.
    """
    return SAMPLE_ROOT if sample else PROCESSED_ROOT
