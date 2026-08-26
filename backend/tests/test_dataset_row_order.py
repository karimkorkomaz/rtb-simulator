"""Regression test for a real reproducibility bug found while building the
CTR modelling stage: `features.dataset.load_impression_features()` had no
row-order guarantee, so DuckDB's parallel Parquet scan could (and did, in
practice -- caught via a logistic-regression hyperparameter selection that
picked a different `C` across two otherwise-identical `--full` runs)
materialize the same set of rows in a DIFFERENT order across separate
calls. Any seeded, positionally-indexed downstream sampling (e.g.
`modeling.downsample.downsample_negatives()`) then silently samples
different actual rows despite an identical seed -- exactly the kind of
thing "fixed random seed everywhere, every number must regenerate" is
supposed to rule out.

Fixed by carrying `bidid`/`timestamp` through as hidden columns and doing
a single in-memory pandas sort on them after materializing (see
`features/dataset.py`'s docstrings for why this is done in pandas rather
than as a SQL `ORDER BY`: the SQL-level sort was measured to be
prohibitively slow -- 20+ minutes and still running, vs. ~11-30s for the
pandas approach used here -- for a wide, ~8.8M-row table).

**This guarantee only covers `row_limit=None` (the full row set).** With
`row_limit` set, DuckDB's `LIMIT` (with no SQL-level `ORDER BY`) can select
a genuinely different SUBSET of rows across separate calls -- the pandas
sort only fixes the ORDER of whatever subset came back, not which rows
those are. That's an accepted, documented limitation of `row_limit`
(dev-mode fast iteration only -- see `modeling.train_lr`'s `--dev`
docstring note); it does not affect `--full` runs, which never pass
`row_limit`, and which is what gets reported.

Deliberately stdlib `unittest`, same style as the rest of `backend/tests/`.
Gated with `skipUnless` (same pattern as
`test_feature_allowlist.TestFeatureColumnsRealParquetSchema`) since it
needs the real ingested Parquet data, which is git-ignored and not present
on a clean clone.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.src.ingest import paths  # noqa: E402
from backend.src.modeling import split  # noqa: E402


@unittest.skipUnless(
    (paths.PROCESSED_ROOT / "impressions").exists(),
    "backend/data/processed/impressions is git-ignored and not present on "
    "a clean clone -- run `python -m src.ingest.ingest` from backend/ to "
    "generate it and enable this test.",
)
class TestRowOrderIsDeterministicAcrossCalls(unittest.TestCase):
    """Uses the season-2 `test` split (a single ~1.66M-row date partition,
    the smallest full -- i.e. `row_limit=None` -- split available) rather
    than a `row_limit`-truncated load, precisely because `row_limit`
    itself is NOT covered by this guarantee (see module docstring).
    """

    @classmethod
    def setUpClass(cls):
        cls.X1, cls.y1 = split.load_split(2, "test")
        cls.X2, cls.y2 = split.load_split(2, "test")

    def test_two_separate_full_loads_return_identical_row_order(self):
        self.assertTrue(
            self.X1["timestamp"].equals(self.X2["timestamp"]),
            "two back-to-back full (row_limit=None) loads of the same "
            "split returned rows in a different order -- the "
            "deterministic-ordering guarantee in "
            "features.dataset.load_impression_features() regressed.",
        )

    def test_labels_align_with_the_same_deterministic_order(self):
        self.assertTrue((self.y1.to_numpy() == self.y2.to_numpy()).all())

    def test_row_count_matches_the_independently_verified_figure(self):
        # Cross-check against modeling.split.verify_split_counts()'s
        # independent SQL query (see test_split_boundaries.py / the task's
        # own verified numbers) rather than trusting this loader alone.
        self.assertEqual(len(self.X1), 1_657_338)
        self.assertEqual(int(self.y1.sum()), 1369)


if __name__ == "__main__":
    unittest.main()
