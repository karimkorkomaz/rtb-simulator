"""Temporal train/val/test split boundaries for CTR modelling.

**Split temporally, never randomly.** Row order in the ingested Parquet
carries no information about "held-out"-ness, so a random split would let
rows from the same day (same campaigns, same traffic conditions) appear on
both sides of the split -- an optimistic, non-representative estimate of
how the model will perform on genuinely future traffic. Splitting on
`date` avoids that by construction: the model never sees a val/test-day
row until evaluation time.

Boundaries are explicit, named constants here -- not literals scattered
through training/evaluation scripts, and not inferred from row counts --
per `backend/src/ingest/README.md`, "Where train/val/test splitting and
downsampling would slot in":

    Season 2 (2013-06-06 .. 2013-06-12, 7 ingested dates): train on days
    1-5 (06-06..06-10), validate on day 6 (06-11), test on day 7 (06-12).

Season 3 (2013-10-19 .. 2013-10-27) is an **out-of-time** evaluation set,
run four months later against different campaigns (see
`docs/analysis/eda-findings.md`). It is deliberately **not** pooled with
season 2 -- pooling would hide exactly the distribution shift that is
itself worth reporting. This module can express a split for any season
(see `SEASON_SPLITS` / `get_split_boundaries()`), but season 3's boundaries
are not yet decided here: `get_split_boundaries(3)` raises `NotImplementedError`
until a later phase deliberately defines how season 3 is used (e.g. as a
single held-out OOT evaluation block, not itself train/val/test split).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import duckdb
import pandas as pd

from ..features import dataset

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitBoundaries:
    """Inclusive `(start, end)` `YYYY-MM-DD` date ranges for each split."""

    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]

    def range_for(self, split: SplitName) -> tuple[str, str]:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        raise ValueError(f"unknown split name {split!r}; expected 'train', 'val', or 'test'")


# ---------------------------------------------------------------------------
# Season 2: train days 1-5, validate day 6, test day 7 (README-specified
# shape). All seven ingested season-2 dates are accounted for exactly once
# -- no gap, no overlap.
# ---------------------------------------------------------------------------
SEASON_2_TRAIN_START = "2013-06-06"
SEASON_2_TRAIN_END = "2013-06-10"
SEASON_2_VAL_DATE = "2013-06-11"
SEASON_2_TEST_DATE = "2013-06-12"

SEASON_2_SPLIT = SplitBoundaries(
    train=(SEASON_2_TRAIN_START, SEASON_2_TRAIN_END),
    val=(SEASON_2_VAL_DATE, SEASON_2_VAL_DATE),
    test=(SEASON_2_TEST_DATE, SEASON_2_TEST_DATE),
)

# Season 3 spans 2013-10-19..2013-10-27 (9 ingested dates) and is reserved
# as an out-of-time evaluation block, run once the season-2-trained model
# is finalized -- deliberately NOT given train/val/test boundaries here.
# `get_split_boundaries(3)` raises rather than guessing a shape for it.
SEASON_SPLITS: dict[int, SplitBoundaries] = {
    2: SEASON_2_SPLIT,
}


def get_split_boundaries(season: int) -> SplitBoundaries:
    """Look up the named split boundaries for `season`.

    Raises `NotImplementedError` for season 3 (and any season with no
    defined split) rather than silently inventing a shape -- season 3 is
    out-of-time evaluation, to be wired up as its own deliberate decision,
    not pooled with or split like season 2.
    """
    if season not in SEASON_SPLITS:
        if season == 3:
            raise NotImplementedError(
                "get_split_boundaries(3): season 3 is reserved as an "
                "out-of-time evaluation set (2013-10-19..2013-10-27, four "
                "months after season 2, different campaigns -- see "
                "docs/analysis/eda-findings.md) and is deliberately NOT "
                "pooled with season 2 or given train/val/test boundaries "
                "here. Evaluate a season-2-trained model against season 3 "
                "as a single out-of-time block via "
                "`features.dataset.load_impression_features(seasons=(3,), "
                "...)` directly, once that stage is built."
            )
        raise NotImplementedError(
            f"get_split_boundaries({season!r}): no split boundaries are "
            f"defined for season {season!r}. Known seasons: "
            f"{sorted(SEASON_SPLITS)!r}."
        )
    return SEASON_SPLITS[season]


def load_split(
    season: int,
    split: SplitName,
    *,
    columns: Optional[Iterable[str]] = None,
    row_limit: Optional[int] = None,
    processed_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Materialize `(X, y)` for one split of one season.

    Goes through `features.dataset.load_impression_features()` (the single
    sanctioned feature loader -- see that module's docstring) with the
    split's date range pushed down via `date_range=`, so the allowlist
    (`schema.feature_columns()`) and the leak-safe click-label join are
    applied exactly as they are everywhere else in this codebase. This
    function only decides *which date range* maps to *which split name*;
    it does not downsample (see `modeling.downsample`) and does not itself
    decide the boundary values (see `SEASON_SPLITS` above).
    """
    boundaries = get_split_boundaries(season)
    date_range = boundaries.range_for(split)
    return dataset.load_impression_features(
        seasons=(season,),
        columns=columns,
        row_limit=row_limit,
        processed_root=processed_root,
        date_range=date_range,
    )


def verify_split_counts(
    season: int,
    *,
    processed_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Cross-check per-split impression/click counts and positive rates
    directly against the Parquet output via DuckDB, independent of
    `load_split()` above (same underlying join logic, but expressed as its
    own SQL query so a bug in one is unlikely to be masked by the other).
    Returns one row per split with `n_impressions`, `n_clicks`,
    `positive_rate`.
    """
    boundaries = get_split_boundaries(season)
    from ..ingest import paths as ingest_paths

    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_glob = str(root / "impressions" / "**" / "*.parquet")
    clk_glob = str(root / "clicks" / "**" / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = []
    for split_name in ("train", "val", "test"):
        start, end = boundaries.range_for(split_name)
        q = f"""
            SELECT
                COUNT(*) AS n_impressions,
                SUM(CASE WHEN c.bidid IS NULL THEN 0 ELSE 1 END) AS n_clicks
            FROM read_parquet('{imp_glob}', hive_partitioning=1) i
            LEFT JOIN (
                SELECT DISTINCT bidid FROM read_parquet('{clk_glob}', hive_partitioning=1)
            ) c USING (bidid)
            WHERE i.season = {int(season)}
              AND i.date BETWEEN DATE '{start}' AND DATE '{end}'
        """
        result = con.execute(q).df().iloc[0]
        n_imp = int(result["n_impressions"])
        n_clk = int(result["n_clicks"])
        rows.append(
            {
                "season": season,
                "split": split_name,
                "date_start": start,
                "date_end": end,
                "n_impressions": n_imp,
                "n_clicks": n_clk,
                "positive_rate": n_clk / n_imp if n_imp else float("nan"),
            }
        )
    return pd.DataFrame(rows)
