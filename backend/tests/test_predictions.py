"""Tests for `backend/src/simulation/predictions.py` -- the correction-
order guard (downsampling recalibration THEN isotonic, asserted every
run) and the bidid-less-predictions-to-pool alignment guard.

Deliberately stdlib `unittest`, same style as `test_replay.py` /
`test_downsample.py`. All tests here use small, hand-built fixtures
(a fake calibrator with a `.predict()` method, tiny frames) so the suite
runs in milliseconds without the real ingested Parquet data or model
artifacts.
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

from backend.src.simulation import predictions as preds_mod  # noqa: E402
from backend.src.simulation import replay  # noqa: E402
from backend.src.modeling.downsample import recalibrate_probability  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


class _SquareRootCalibrator:
    """A fake, deterministic, monotone "calibrator" (`p -> sqrt(p)`) --
    stands in for a fitted `sklearn.isotonic.IsotonicRegression` in tests
    that only need `.predict()`, without fitting a real one. Monotone
    (increasing) so it's a plausible stand-in for isotonic regression's
    shape constraint, and NOT the identity function, so it actually
    changes the input (needed for the "reverse order differs" check to
    be a meaningful test at all).
    """

    def predict(self, p):
        return np.sqrt(np.asarray(p, dtype=np.float64))


class _IdentityCalibrator:
    """A degenerate "calibrator" that changes nothing -- used to
    construct the deliberately-vacuous case that
    `assert_correction_order()`'s check 3 must catch.
    """

    def predict(self, p):
        return np.asarray(p, dtype=np.float64)


def _artifact(calibrator, *, fit_split: str = "season2_val", sampling_rate: float = 0.02) -> dict:
    return {
        "calibrator": calibrator,
        "fit_on": {"split": fit_split},
        "sampling_rate": sampling_rate,
    }


def _correctly_ordered_predictions(
    *, sampling_rate: float = 0.02, calibrator=None, n: int = 50, seed: int = 0
) -> pd.DataFrame:
    """Build a predictions frame whose p_calibrated/p_isotonic were
    genuinely produced by the correct-order chain, so
    `assert_correction_order()` should PASS against it.
    """
    calibrator = calibrator or _SquareRootCalibrator()
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.001, 0.5, size=n)
    p_calibrated = recalibrate_probability(p_raw, sampling_rate)
    p_isotonic = calibrator.predict(p_calibrated)
    return pd.DataFrame(
        {
            "y_true": np.zeros(n, dtype=np.int64),
            "p_raw": p_raw,
            "p_calibrated": p_calibrated,
            "advertiser": ["1458"] * n,
            "timestamp": pd.to_datetime(["2013-06-12"] * n),
            "p_isotonic": p_isotonic,
        }
    )


# ---------------------------------------------------------------------------
# assert_correction_order(): the correct-order chain passes
# ---------------------------------------------------------------------------


class TestAssertCorrectionOrderHappyPath(unittest.TestCase):
    def test_correct_order_passes(self):
        predictions = _correctly_ordered_predictions()
        artifact = _artifact(_SquareRootCalibrator())
        # Must not raise.
        preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)


# ---------------------------------------------------------------------------
# Check 1: recalibrate_probability(p_raw) must reproduce p_calibrated
# ---------------------------------------------------------------------------


class TestCheckOneRecalibrationMismatch(unittest.TestCase):
    def test_wrong_sampling_rate_raises(self):
        predictions = _correctly_ordered_predictions(sampling_rate=0.02)
        artifact = _artifact(_SquareRootCalibrator())
        with self.assertRaises(AssertionError) as ctx:
            preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.5)
        self.assertIn("p_calibrated", str(ctx.exception))

    def test_corrupted_p_calibrated_column_raises(self):
        predictions = _correctly_ordered_predictions(sampling_rate=0.02)
        predictions = predictions.copy()
        predictions["p_calibrated"] = predictions["p_calibrated"] * 3.0  # simulate corruption
        artifact = _artifact(_SquareRootCalibrator())
        with self.assertRaises(AssertionError):
            preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)


# ---------------------------------------------------------------------------
# Check 2: isotonic(recomputed p_calibrated) must reproduce p_isotonic
# ---------------------------------------------------------------------------


class TestCheckTwoIsotonicMismatch(unittest.TestCase):
    def test_isotonic_applied_to_wrong_input_raises(self):
        # p_isotonic on disk was produced by a DIFFERENT calibrator
        # (identity) than the one we're about to check against
        # (sqrt) -- must be caught.
        predictions = _correctly_ordered_predictions(calibrator=_IdentityCalibrator())
        artifact = _artifact(_SquareRootCalibrator())  # mismatched calibrator
        with self.assertRaises(AssertionError) as ctx:
            preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)
        self.assertIn("p_isotonic", str(ctx.exception))


# ---------------------------------------------------------------------------
# Check 3: the reverse-order composition must be MATERIALLY different --
# this is the "not vacuous" guard itself.
# ---------------------------------------------------------------------------


class TestCheckThreeReverseOrderNotVacuous(unittest.TestCase):
    def test_identity_calibrator_makes_the_check_vacuous_and_is_caught(self):
        # With an IDENTITY calibrator, isotonic(p_calibrated) ==
        # p_calibrated == isotonic(p_raw)-then-recalibrated is NOT
        # necessarily identical -- recalibrate_probability is nonlinear,
        # so composing it after an identity "isotonic" in the wrong order
        # still differs from doing it in the right order, UNLESS
        # sampling_rate == 1.0 (recalibration is itself the identity).
        # Constructing the genuinely-degenerate, sampling_rate == 1.0
        # case here: recalibration AND isotonic are both the identity, so
        # order truly cannot matter -- this must be caught as vacuous.
        predictions = _correctly_ordered_predictions(
            sampling_rate=1.0, calibrator=_IdentityCalibrator()
        )
        artifact = _artifact(_IdentityCalibrator(), sampling_rate=1.0)
        with self.assertRaises(AssertionError) as ctx:
            preds_mod.assert_correction_order(predictions, artifact, sampling_rate=1.0)
        self.assertIn("REVERSE-order", str(ctx.exception))

    def test_genuine_downsampling_and_nontrivial_calibrator_is_not_vacuous(self):
        # The realistic case (rate < 1, a non-identity calibrator) must
        # NOT trip the vacuous-check guard -- i.e. assert_correction_order
        # passes cleanly, proving check 3 is satisfiable in practice, not
        # just theoretically avoidable.
        predictions = _correctly_ordered_predictions(sampling_rate=0.02)
        artifact = _artifact(_SquareRootCalibrator(), sampling_rate=0.02)
        preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)


# ---------------------------------------------------------------------------
# Check 4: calibrator must have been fit on validation, not test
# ---------------------------------------------------------------------------


class TestCheckFourFitSplit(unittest.TestCase):
    def test_fit_on_test_split_raises(self):
        predictions = _correctly_ordered_predictions()
        artifact = _artifact(_SquareRootCalibrator(), fit_split="season2_test")
        with self.assertRaises(AssertionError) as ctx:
            preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)
        self.assertIn("season2_val", str(ctx.exception))

    def test_fit_on_val_split_passes(self):
        predictions = _correctly_ordered_predictions()
        artifact = _artifact(_SquareRootCalibrator(), fit_split="season2_val")
        preds_mod.assert_correction_order(predictions, artifact, sampling_rate=0.02)


# ---------------------------------------------------------------------------
# attach_bidid_to_predictions() / _attach_bidid_positionally(): the
# alignment guard, tested as a pure function on hand-built frames.
# ---------------------------------------------------------------------------


class TestAttachBidIdPositionally(unittest.TestCase):
    def _predictions(self, n=3):
        return pd.DataFrame(
            {
                "y_true": [0, 1, 0][:n],
                "p_isotonic": [0.001, 0.01, 0.002][:n],
                "advertiser": ["1458", "1458", "3358"][:n],
                "timestamp": pd.to_datetime(
                    ["2013-06-12 00:00:01", "2013-06-12 00:00:02", "2013-06-12 00:00:03"][:n]
                ),
            }
        )

    def _matching_keyed(self, n=3):
        return pd.DataFrame(
            {
                "bidid": ["bidA", "bidB", "bidC"][:n],
                "advertiser": ["1458", "1458", "3358"][:n],
                "timestamp": pd.to_datetime(
                    ["2013-06-12 00:00:01", "2013-06-12 00:00:02", "2013-06-12 00:00:03"][:n]
                ),
            }
        )

    def test_happy_path_attaches_bidid_in_order(self):
        predictions = self._predictions()
        keyed = self._matching_keyed()
        out = preds_mod._attach_bidid_positionally(predictions, keyed)
        self.assertEqual(list(out["bidid"]), ["bidA", "bidB", "bidC"])
        # Original columns/order preserved.
        self.assertEqual(list(out["p_isotonic"]), list(predictions["p_isotonic"]))

    def test_row_count_mismatch_raises(self):
        predictions = self._predictions(n=3)
        keyed = self._matching_keyed(n=2)
        with self.assertRaises(ValueError) as ctx:
            preds_mod._attach_bidid_positionally(predictions, keyed)
        self.assertIn("row count mismatch", str(ctx.exception))

    def test_advertiser_mismatch_raises(self):
        predictions = self._predictions()
        keyed = self._matching_keyed()
        keyed = keyed.copy()
        keyed.loc[1, "advertiser"] = "9999"  # corrupt one row
        with self.assertRaises(ValueError) as ctx:
            preds_mod._attach_bidid_positionally(predictions, keyed)
        self.assertIn("advertiser", str(ctx.exception))

    def test_timestamp_mismatch_raises(self):
        predictions = self._predictions()
        keyed = self._matching_keyed()
        keyed = keyed.copy()
        keyed.loc[0, "timestamp"] = pd.Timestamp("2013-06-12 23:59:59")
        with self.assertRaises(ValueError) as ctx:
            preds_mod._attach_bidid_positionally(predictions, keyed)
        self.assertIn("timestamp", str(ctx.exception))

    def test_no_fuzzy_fallback_on_mismatch(self):
        # A subtler check: even if the SET of (advertiser, timestamp)
        # pairs matches but the ORDER differs, this must still raise --
        # there is no "sort and re-match" fallback.
        predictions = self._predictions()
        keyed = self._matching_keyed()
        shuffled = keyed.iloc[[1, 0, 2]].reset_index(drop=True)
        with self.assertRaises(ValueError):
            preds_mod._attach_bidid_positionally(predictions, shuffled)


# ---------------------------------------------------------------------------
# p_click_for_pool(): per-advertiser extraction aligned to pool order
# ---------------------------------------------------------------------------


def _make_pool(bidids, timestamps, advertiser="1458"):
    n = len(bidids)
    return replay.AuctionPool(
        advertiser=advertiser,
        date_range=("2013-06-12", "2013-06-12"),
        seasons=(2,),
        n_rows=n,
        excluded_payprice_gt_bidprice_count=0,
        excluded_payprice_gt_bidprice_share=0.0,
        excluded_payprice_zero_count=0,
        excluded_payprice_zero_share=0.0,
        timestamp=pd.Series(pd.to_datetime(timestamps), name="timestamp").reset_index(drop=True),
        bidid=pd.Series(bidids, name="bidid").reset_index(drop=True),
        click=np.zeros(n, dtype=np.int64),
        _payprice=np.zeros(n, dtype=np.float64),
        _feature_frame=pd.DataFrame({"slotwidth": np.arange(n)}),
    )


class TestPClickForPool(unittest.TestCase):
    def test_happy_path_aligns_to_pool_order(self):
        # Pool order is DELIBERATELY not the same as the predictions
        # frame's order, to prove alignment is by key, not position.
        pool = _make_pool(
            bidids=["b2", "b1", "b3"],
            timestamps=["2013-06-12 00:00:02", "2013-06-12 00:00:01", "2013-06-12 00:00:03"],
        )
        predictions_with_bidid = pd.DataFrame(
            {
                "bidid": ["b1", "b2", "b3"],
                "advertiser": ["1458", "1458", "1458"],
                "timestamp": pd.to_datetime(
                    ["2013-06-12 00:00:01", "2013-06-12 00:00:02", "2013-06-12 00:00:03"]
                ),
                "p_isotonic": [0.1, 0.2, 0.3],
            }
        )
        aligned = preds_mod.p_click_for_pool(pool, predictions_with_bidid)
        np.testing.assert_allclose(aligned, [0.2, 0.1, 0.3])  # pool order: b2, b1, b3

    def test_filters_to_requested_advertiser_only(self):
        pool = _make_pool(bidids=["b1"], timestamps=["2013-06-12 00:00:01"], advertiser="1458")
        predictions_with_bidid = pd.DataFrame(
            {
                "bidid": ["b1", "b1"],
                "advertiser": ["1458", "3358"],  # same bidid, different advertiser -- must not collide
                "timestamp": pd.to_datetime(["2013-06-12 00:00:01", "2013-06-12 00:00:01"]),
                "p_isotonic": [0.5, 0.9],
            }
        )
        aligned = preds_mod.p_click_for_pool(pool, predictions_with_bidid)
        np.testing.assert_allclose(aligned, [0.5])

    def test_missing_pool_row_raises(self):
        pool = _make_pool(bidids=["b1", "bMISSING"], timestamps=["2013-06-12 00:00:01", "2013-06-12 00:00:02"])
        predictions_with_bidid = pd.DataFrame(
            {
                "bidid": ["b1"],
                "advertiser": ["1458"],
                "timestamp": pd.to_datetime(["2013-06-12 00:00:01"]),
                "p_isotonic": [0.5],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            preds_mod.p_click_for_pool(pool, predictions_with_bidid)
        self.assertIn("no matching prediction", str(ctx.exception))

    def test_colliding_bidid_timestamp_key_is_resolved_by_averaging(self):
        # The rare, documented (bidid, timestamp) collision: two
        # predictions rows share the exact same key. Resolved by
        # averaging, not by arbitrarily picking one.
        pool = _make_pool(bidids=["bDUP"], timestamps=["2013-06-12 00:00:01"])
        predictions_with_bidid = pd.DataFrame(
            {
                "bidid": ["bDUP", "bDUP"],
                "advertiser": ["1458", "1458"],
                "timestamp": pd.to_datetime(["2013-06-12 00:00:01", "2013-06-12 00:00:01"]),
                "p_isotonic": [0.2, 0.6],
            }
        )
        aligned = preds_mod.p_click_for_pool(pool, predictions_with_bidid)
        np.testing.assert_allclose(aligned, [0.4])  # mean(0.2, 0.6)

    def test_no_predictions_for_advertiser_raises(self):
        pool = _make_pool(bidids=["b1"], timestamps=["2013-06-12 00:00:01"], advertiser="9999")
        predictions_with_bidid = pd.DataFrame(
            {
                "bidid": ["b1"],
                "advertiser": ["1458"],
                "timestamp": pd.to_datetime(["2013-06-12 00:00:01"]),
                "p_isotonic": [0.5],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            preds_mod.p_click_for_pool(pool, predictions_with_bidid)
        self.assertIn("no predictions rows found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
