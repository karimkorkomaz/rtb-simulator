"""Tests for `backend/src/modeling/downsample.py` -- negative downsampling
and its recalibration correction. These are the two things that silently
invalidate the thesis if wrong (an unrecorded/incorrect sampling rate
means every downstream probability is mis-scaled with no way to detect
it), so they get direct, deterministic unit coverage independent of the
real dataset.

Deliberately stdlib `unittest`, same style as `test_feature_allowlist.py`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.src.modeling.downsample import (  # noqa: E402
    downsample_negatives,
    recalibrate_probability,
)


def _toy_frame(n_positive: int, n_negative: int) -> tuple[pd.DataFrame, pd.Series]:
    n = n_positive + n_negative
    X = pd.DataFrame({"feature": np.arange(n)})
    y = pd.Series([1] * n_positive + [0] * n_negative, name="click")
    return X, y


class TestDownsampleNegatives(unittest.TestCase):
    def test_all_positives_kept(self):
        X, y = _toy_frame(n_positive=50, n_negative=10_000)
        X_ds, y_ds, info = downsample_negatives(X, y, rate=0.1, seed=42)
        self.assertEqual(int(y_ds.sum()), 50)
        self.assertEqual(info.n_positive, 50)

    def test_negative_count_matches_requested_rate(self):
        X, y = _toy_frame(n_positive=50, n_negative=10_000)
        X_ds, y_ds, info = downsample_negatives(X, y, rate=0.1, seed=42)
        n_negative_kept = int((y_ds == 0).sum())
        self.assertEqual(n_negative_kept, 1000)  # round(0.1 * 10_000)
        self.assertEqual(info.n_negative_sampled, 1000)
        self.assertEqual(info.n_negative_true, 10_000)

    def test_rate_one_keeps_everything(self):
        X, y = _toy_frame(n_positive=20, n_negative=500)
        X_ds, y_ds, info = downsample_negatives(X, y, rate=1.0, seed=1)
        self.assertEqual(len(y_ds), len(y))
        self.assertEqual(info.n_negative_sampled, info.n_negative_true)

    def test_no_negatives_sampled_more_than_once(self):
        X, y = _toy_frame(n_positive=10, n_negative=1000)
        X_ds, y_ds, info = downsample_negatives(X, y, rate=0.3, seed=7)
        self.assertEqual(len(X_ds.index), len(set(X_ds.index)))

    def test_reproducible_given_same_seed(self):
        X, y = _toy_frame(n_positive=30, n_negative=5000)
        X_ds_a, y_ds_a, _ = downsample_negatives(X, y, rate=0.2, seed=99)
        X_ds_b, y_ds_b, _ = downsample_negatives(X, y, rate=0.2, seed=99)
        self.assertTrue(X_ds_a.index.equals(X_ds_b.index))

    def test_different_seeds_give_different_samples(self):
        X, y = _toy_frame(n_positive=30, n_negative=5000)
        X_ds_a, _, _ = downsample_negatives(X, y, rate=0.2, seed=1)
        X_ds_b, _, _ = downsample_negatives(X, y, rate=0.2, seed=2)
        self.assertFalse(X_ds_a.index.equals(X_ds_b.index))

    def test_info_records_positive_rate_before_and_after(self):
        X, y = _toy_frame(n_positive=100, n_negative=9900)  # true rate 1%
        X_ds, y_ds, info = downsample_negatives(X, y, rate=0.1, seed=3)
        self.assertAlmostEqual(info.positive_rate_before, 0.01)
        # after: 100 positive / (100 + round(0.1*9900)=990) = 100/1090
        self.assertAlmostEqual(info.positive_rate_after, 100 / 1090)
        self.assertGreater(info.positive_rate_after, info.positive_rate_before)

    def test_rejects_out_of_range_rate(self):
        X, y = _toy_frame(n_positive=5, n_negative=5)
        with self.assertRaises(ValueError):
            downsample_negatives(X, y, rate=0.0, seed=1)
        with self.assertRaises(ValueError):
            downsample_negatives(X, y, rate=1.5, seed=1)

    def test_mismatched_lengths_raise(self):
        X, y = _toy_frame(n_positive=5, n_negative=5)
        with self.assertRaises(ValueError):
            downsample_negatives(X, y.iloc[:-1], rate=0.5, seed=1)

    def test_never_applied_implicitly_to_full_rate_data_is_a_noop_on_positives(self):
        # Sanity: at rate=1.0, downsampling is a pure identity for a
        # val/test-shaped call (documents the "never call this on val/test"
        # rule structurally, even though nothing here prevents misuse --
        # the module docstring is the actual enforcement).
        X, y = _toy_frame(n_positive=8, n_negative=800)
        X_ds, y_ds, info = downsample_negatives(X, y, rate=1.0, seed=0)
        pd.testing.assert_frame_equal(X_ds.sort_index(), X.sort_index())


class TestRecalibrateProbability(unittest.TestCase):
    def test_rate_one_is_identity(self):
        p = np.array([0.01, 0.5, 0.99, 0.0001])
        q = recalibrate_probability(p, sampling_rate=1.0)
        np.testing.assert_array_equal(q, p)

    def test_scalar_input_returns_scalar(self):
        q = recalibrate_probability(0.5, sampling_rate=0.1)
        self.assertIsInstance(q, float)

    def test_downsampling_inflates_so_correction_reduces_probability(self):
        # A downsampled-train model's raw output should always be corrected
        # DOWNWARD relative to the true rate (since downsampling negatives
        # inflates the apparent positive rate) -- except at the boundaries
        # p=0/p=1, where it is a no-op.
        p = 0.5
        q = recalibrate_probability(p, sampling_rate=0.1)
        self.assertLess(q, p)

    def test_matches_hand_derived_example(self):
        # True population: 1 positive per 100 negatives among rows with
        # this feature value (p_true = 1/101). Downsample negatives to
        # keep 10% (w=0.1): downsampled counts become 1 positive, 10
        # negatives, so a model fit on the downsampled data would predict
        # p_ds = 1/11. Recalibrating p_ds with w=0.1 should recover
        # (approximately) p_true = 1/101.
        p_true = 1 / 101
        w = 0.1
        p_ds = 1 / 11
        q = recalibrate_probability(p_ds, sampling_rate=w)
        self.assertAlmostEqual(q, p_true, places=9)

    def test_rejects_out_of_range_rate(self):
        with self.assertRaises(ValueError):
            recalibrate_probability(0.5, sampling_rate=0.0)
        with self.assertRaises(ValueError):
            recalibrate_probability(0.5, sampling_rate=1.2)

    def test_series_input_preserves_index(self):
        p = pd.Series([0.1, 0.2, 0.3], index=[10, 20, 30], name="p")
        q = recalibrate_probability(p, sampling_rate=0.5)
        self.assertIsInstance(q, pd.Series)
        self.assertListEqual(list(q.index), [10, 20, 30])

    def test_monotonic_in_p(self):
        # Recalibration must preserve rank order -- it is what makes AUC
        # invariant to whether it's applied before scoring.
        p = np.linspace(0.001, 0.999, 50)
        q = recalibrate_probability(p, sampling_rate=0.05)
        self.assertTrue(np.all(np.diff(q) > 0))


if __name__ == "__main__":
    unittest.main()
