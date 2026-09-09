"""Tests for `backend/src/simulation/strategies.py` -- the three baseline
bidding strategies (constant / random / linear-in-CTR). Covers
determinism/reproducibility, the shape/non-negativity contract, and
end-to-end integration with `replay.settle_auctions()`.

Deliberately stdlib `unittest`, same style as `test_replay.py`.
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

from backend.src.simulation import replay  # noqa: E402
from backend.src.simulation import strategies  # noqa: E402


def _make_pool(payprice, click, *, advertiser="1458"):
    n = len(payprice)
    return replay.AuctionPool(
        advertiser=advertiser,
        date_range=("2013-06-12", "2013-06-12"),
        seasons=(2,),
        n_rows=n,
        excluded_payprice_gt_bidprice_count=0,
        excluded_payprice_gt_bidprice_share=0.0,
        excluded_payprice_zero_count=0,
        excluded_payprice_zero_share=0.0,
        timestamp=pd.Series(
            pd.Timestamp("2013-06-12 00:00:00") + pd.to_timedelta(np.arange(n), unit="s"),
            name="timestamp",
        ).reset_index(drop=True),
        bidid=pd.Series([f"bid{i:04d}" for i in range(n)], name="bidid").reset_index(drop=True),
        click=np.asarray(click, dtype=np.int64),
        _payprice=np.asarray(payprice, dtype=np.float64),
        _feature_frame=pd.DataFrame({"slotwidth": np.arange(n)}),
    )


# ---------------------------------------------------------------------------
# constant_bid
# ---------------------------------------------------------------------------


class TestConstantBid(unittest.TestCase):
    def test_shape_and_value(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        bids = strategies.constant_bid(pool.bid_time_view(), amount=42.0)
        self.assertEqual(bids.shape, (3,))
        np.testing.assert_array_equal(bids, [42.0, 42.0, 42.0])

    def test_negative_amount_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.constant_bid(pool.bid_time_view(), amount=-1.0)

    def test_zero_amount_is_all_zero_non_negative(self):
        pool = _make_pool(payprice=[10.0, 20.0], click=[0, 0])
        bids = strategies.constant_bid(pool.bid_time_view(), amount=0.0)
        self.assertTrue(np.all(bids >= 0))
        np.testing.assert_array_equal(bids, [0.0, 0.0])

    def test_deterministic_repeat_call(self):
        pool = _make_pool(payprice=[10.0, 20.0], click=[0, 0])
        b1 = strategies.constant_bid(pool.bid_time_view(), amount=15.0)
        b2 = strategies.constant_bid(pool.bid_time_view(), amount=15.0)
        np.testing.assert_array_equal(b1, b2)

    def test_empty_pool(self):
        pool = _make_pool(payprice=[], click=[])
        bids = strategies.constant_bid(pool.bid_time_view(), amount=10.0)
        self.assertEqual(bids.shape, (0,))


# ---------------------------------------------------------------------------
# random_bid: determinism/reproducibility is the core requirement
# ---------------------------------------------------------------------------


class TestRandomBid(unittest.TestCase):
    def test_shape_and_bounds(self):
        pool = _make_pool(payprice=[10.0] * 100, click=[0] * 100)
        rng = np.random.default_rng(42)
        bids = strategies.random_bid(pool.bid_time_view(), low=5.0, high=50.0, rng=rng)
        self.assertEqual(bids.shape, (100,))
        self.assertTrue(np.all(bids >= 5.0))
        self.assertTrue(np.all(bids < 50.0))

    def test_same_seed_gives_identical_bids(self):
        pool = _make_pool(payprice=[10.0] * 20, click=[0] * 20)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        bids1 = strategies.random_bid(pool.bid_time_view(), low=0.0, high=100.0, rng=rng1)
        bids2 = strategies.random_bid(pool.bid_time_view(), low=0.0, high=100.0, rng=rng2)
        np.testing.assert_array_equal(bids1, bids2)

    def test_different_seed_gives_different_bids(self):
        pool = _make_pool(payprice=[10.0] * 20, click=[0] * 20)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        bids1 = strategies.random_bid(pool.bid_time_view(), low=0.0, high=100.0, rng=rng1)
        bids2 = strategies.random_bid(pool.bid_time_view(), low=0.0, high=100.0, rng=rng2)
        self.assertFalse(np.array_equal(bids1, bids2))

    def test_seed_sequence_reproduces_across_independent_calls(self):
        # Mirrors run_baselines.py's actual seeding convention:
        # np.random.default_rng([SEED, advertiser, index]).
        pool = _make_pool(payprice=[10.0] * 10, click=[0] * 10)
        rng_a = np.random.default_rng([42, 1458, 0])
        rng_b = np.random.default_rng([42, 1458, 0])
        bids_a = strategies.random_bid(pool.bid_time_view(), low=0.0, high=10.0, rng=rng_a)
        bids_b = strategies.random_bid(pool.bid_time_view(), low=0.0, high=10.0, rng=rng_b)
        np.testing.assert_array_equal(bids_a, bids_b)

    def test_negative_low_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.random_bid(pool.bid_time_view(), low=-1.0, high=10.0, rng=np.random.default_rng(0))

    def test_high_below_low_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.random_bid(pool.bid_time_view(), low=10.0, high=5.0, rng=np.random.default_rng(0))

    def test_degenerate_equal_range_is_constant(self):
        pool = _make_pool(payprice=[10.0] * 5, click=[0] * 5)
        bids = strategies.random_bid(pool.bid_time_view(), low=7.0, high=7.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(bids, [7.0] * 5)

    def test_empty_pool(self):
        pool = _make_pool(payprice=[], click=[])
        bids = strategies.random_bid(pool.bid_time_view(), low=0.0, high=10.0, rng=np.random.default_rng(0))
        self.assertEqual(bids.shape, (0,))


# ---------------------------------------------------------------------------
# linear_in_ctr_bid
# ---------------------------------------------------------------------------


class TestLinearInCtrBid(unittest.TestCase):
    def test_bid_equals_base_times_p_click(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        p_click = np.array([0.001, 0.002, 0.0005])
        bids = strategies.linear_in_ctr_bid(pool.bid_time_view(), base=100_000.0, p_click=p_click)
        np.testing.assert_allclose(bids, 100_000.0 * p_click)

    def test_deterministic_repeat_call(self):
        pool = _make_pool(payprice=[10.0, 20.0], click=[0, 0])
        p_click = np.array([0.001, 0.002])
        b1 = strategies.linear_in_ctr_bid(pool.bid_time_view(), base=5000.0, p_click=p_click)
        b2 = strategies.linear_in_ctr_bid(pool.bid_time_view(), base=5000.0, p_click=p_click)
        np.testing.assert_array_equal(b1, b2)

    def test_length_mismatch_raises(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        p_click = np.array([0.001, 0.002])  # length 2, pool has 3 rows
        with self.assertRaises(ValueError) as ctx:
            strategies.linear_in_ctr_bid(pool.bid_time_view(), base=1000.0, p_click=p_click)
        self.assertIn("(2,)", str(ctx.exception))
        self.assertIn("(3,)", str(ctx.exception))

    def test_negative_base_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.linear_in_ctr_bid(pool.bid_time_view(), base=-1.0, p_click=np.array([0.01]))

    def test_negative_p_click_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.linear_in_ctr_bid(pool.bid_time_view(), base=100.0, p_click=np.array([-0.01]))

    def test_nan_p_click_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            strategies.linear_in_ctr_bid(pool.bid_time_view(), base=100.0, p_click=np.array([np.nan]))

    def test_zero_p_click_gives_zero_bid_non_negative(self):
        pool = _make_pool(payprice=[10.0, 20.0], click=[0, 0])
        bids = strategies.linear_in_ctr_bid(pool.bid_time_view(), base=1000.0, p_click=np.array([0.0, 0.0]))
        np.testing.assert_array_equal(bids, [0.0, 0.0])
        self.assertTrue(np.all(bids >= 0))


# ---------------------------------------------------------------------------
# Integration: every strategy's output is a legal input to
# settle_auctions() (shape/non-negativity contract), and payprice is
# never reachable from the view handed to these functions.
# ---------------------------------------------------------------------------


class TestSettleAuctionsIntegration(unittest.TestCase):
    def setUp(self):
        self.pool = _make_pool(
            payprice=[10.0, 50.0, 5.0, 200.0, 30.0],
            click=[0, 1, 0, 0, 1],
        )

    def test_constant_bid_settles_without_error(self):
        bids = strategies.constant_bid(self.pool.bid_time_view(), amount=40.0)
        result = replay.settle_auctions(self.pool, bids, budget=1000.0)
        self.assertGreaterEqual(result.impressions_won, 0)
        self.assertLessEqual(result.total_spend, 1000.0)

    def test_random_bid_settles_without_error(self):
        rng = np.random.default_rng(3)
        bids = strategies.random_bid(self.pool.bid_time_view(), low=0.0, high=100.0, rng=rng)
        result = replay.settle_auctions(self.pool, bids, budget=1000.0)
        self.assertGreaterEqual(result.impressions_won, 0)
        self.assertLessEqual(result.total_spend, 1000.0)

    def test_linear_bid_settles_without_error(self):
        p_click = np.array([0.01, 0.5, 0.02, 0.9, 0.3])
        bids = strategies.linear_in_ctr_bid(self.pool.bid_time_view(), base=100.0, p_click=p_click)
        result = replay.settle_auctions(self.pool, bids, budget=1000.0)
        self.assertGreaterEqual(result.impressions_won, 0)
        self.assertLessEqual(result.total_spend, 1000.0)

    def test_strategies_never_see_payprice(self):
        # Structural check: the view handed to every strategy function
        # has no payprice column reachable, regardless of strategy.
        view = self.pool.bid_time_view()
        self.assertNotIn("payprice", view.columns)
        with self.assertRaises(KeyError):
            view["payprice"]
        # And bidding "as if" payprice were known (e.g. always winning by
        # exactly matching it) is not expressible through any strategy
        # function here -- linear_in_ctr_bid only ever multiplies a
        # caller-supplied p_click by a scalar base, never reads the view's
        # settlement-only data.
        bids = strategies.constant_bid(view, amount=1_000_000.0)
        self.assertTrue(np.all(bids == 1_000_000.0))  # not payprice-dependent at all


if __name__ == "__main__":
    unittest.main()
