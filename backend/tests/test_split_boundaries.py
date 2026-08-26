"""Tests for `backend/src/modeling/split.py` -- the temporal train/val/test
split boundaries.

Deliberately stdlib `unittest` (no pytest dependency), same style as
`test_feature_allowlist.py`; see that file's module docstring for why.
These tests do NOT touch the ingested Parquet data (no `skipUnless` gate
needed) -- they pin the boundary VALUES and the split-name contract, which
must hold regardless of whether `data/processed/` exists locally.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.src.modeling import split  # noqa: E402


class TestSeason2SplitBoundaries(unittest.TestCase):
    """Pins the exact dates: train days 1-5, validate day 6, test day 7,
    as specified by `backend/src/ingest/README.md`.
    """

    def setUp(self):
        self.boundaries = split.get_split_boundaries(2)

    def test_train_is_first_five_days(self):
        self.assertEqual(self.boundaries.train, ("2013-06-06", "2013-06-10"))

    def test_val_is_day_six(self):
        self.assertEqual(self.boundaries.val, ("2013-06-11", "2013-06-11"))

    def test_test_is_day_seven(self):
        self.assertEqual(self.boundaries.test, ("2013-06-12", "2013-06-12"))

    def test_range_for_dispatches_correctly(self):
        self.assertEqual(self.boundaries.range_for("train"), self.boundaries.train)
        self.assertEqual(self.boundaries.range_for("val"), self.boundaries.val)
        self.assertEqual(self.boundaries.range_for("test"), self.boundaries.test)

    def test_range_for_rejects_unknown_split_name(self):
        with self.assertRaises(ValueError):
            self.boundaries.range_for("holdout")  # type: ignore[arg-type]

    def test_all_seven_ingested_dates_covered_exactly_once_no_gap_no_overlap(self):
        # The seven ingested season-2 dates, per the task brief and
        # `backend/data/metadata/ingestion_row_counts.json`.
        all_dates = [date(2013, 6, 6) + timedelta(days=i) for i in range(7)]
        covered: set[date] = set()
        for split_name in ("train", "val", "test"):
            start_s, end_s = self.boundaries.range_for(split_name)
            start = date.fromisoformat(start_s)
            end = date.fromisoformat(end_s)
            self.assertLessEqual(start, end)
            d = start
            while d <= end:
                self.assertNotIn(
                    d, covered,
                    f"{d} is covered by more than one split -- train/val/test overlap",
                )
                covered.add(d)
                d += timedelta(days=1)
        self.assertEqual(
            covered, set(all_dates),
            "train+val+test must cover exactly the seven ingested season-2 "
            "dates, no gap and no overlap",
        )

    def test_val_and_test_are_strictly_after_train(self):
        # The core "temporal, never random" guarantee: nothing in val/test
        # precedes anything in train.
        train_end = date.fromisoformat(self.boundaries.train[1])
        val_start = date.fromisoformat(self.boundaries.val[0])
        test_start = date.fromisoformat(self.boundaries.test[0])
        self.assertLess(train_end, val_start)
        self.assertLess(val_start, test_start)


class TestSeason3IsNotYetSplit(unittest.TestCase):
    """Season 3 is out-of-time evaluation and must NOT be silently pooled
    with, or split like, season 2 -- calling `get_split_boundaries(3)`
    must raise, not guess a shape.
    """

    def test_season_3_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            split.get_split_boundaries(3)

    def test_season_3_error_message_explains_out_of_time_status(self):
        with self.assertRaises(NotImplementedError) as ctx:
            split.get_split_boundaries(3)
        message = str(ctx.exception)
        self.assertIn("out-of-time", message)
        self.assertIn("NOT pooled", message)

    def test_unknown_season_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            split.get_split_boundaries(99)


class TestSplitBoundariesDataclass(unittest.TestCase):
    def test_is_frozen(self):
        boundaries = split.get_split_boundaries(2)
        with self.assertRaises(Exception):
            boundaries.train = ("2000-01-01", "2000-01-01")  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
