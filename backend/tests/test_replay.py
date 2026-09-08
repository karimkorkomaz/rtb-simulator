"""Tests for `backend/src/simulation/replay.py` -- the auction replay
engine (second-price settlement + hard budget + the payprice guard).

Deliberately stdlib `unittest`, same style as the rest of `backend/tests/`
(see `test_feature_allowlist.py`'s module docstring for why: no pytest
dependency is installed in `backend/.venv`).

Most tests here use small, hand-built `AuctionPool` fixtures (constructed
directly via the dataclass, bypassing `load_auction_pool()`'s DuckDB read
entirely) so the expected numbers can be computed by hand and the suite
runs in milliseconds without the real ingested Parquet data. One test
class at the bottom (`TestLoadAuctionPoolRealData`) is gated with
`skipUnless` on the real data being present, matching
`test_dataset_row_order.py` / `test_feature_allowlist.py`'s convention --
it exists purely to sanity-check that `load_auction_pool()` loads
successfully against the real corpus, not to validate any simulation
result.
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

from backend.src.ingest import paths  # noqa: E402
from backend.src.simulation import replay  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_pool(
    payprice: list[float],
    click: list[int],
    *,
    timestamps: list[str] | None = None,
    bidids: list[str] | None = None,
    feature_frame: pd.DataFrame | None = None,
    advertiser: str = "1458",
) -> replay.AuctionPool:
    """Build a small, hand-controlled `AuctionPool` directly, bypassing
    `load_auction_pool()`'s DuckDB read. `payprice`/`click` fully determine
    the settlement-relevant state; `timestamps`/`bidids` default to an
    already-sorted, trivially-tiebroken sequence unless a test needs to
    exercise something specific about ordering.
    """
    n = len(payprice)
    assert len(click) == n
    if timestamps is None:
        timestamps = [f"2013-06-06 00:00:{i:02d}" for i in range(n)]
    if bidids is None:
        bidids = [f"bid{i:04d}" for i in range(n)]
    if feature_frame is None:
        feature_frame = pd.DataFrame({"slotwidth": np.arange(n)})
    return replay.AuctionPool(
        advertiser=advertiser,
        date_range=("2013-06-06", "2013-06-06"),
        seasons=(2,),
        n_rows=n,
        excluded_payprice_gt_bidprice_count=0,
        excluded_payprice_gt_bidprice_share=0.0,
        excluded_payprice_zero_count=0,
        excluded_payprice_zero_share=0.0,
        timestamp=pd.Series(pd.to_datetime(timestamps), name="timestamp").reset_index(drop=True),
        bidid=pd.Series(bidids, name="bidid").reset_index(drop=True),
        click=np.asarray(click, dtype=np.int64),
        _payprice=np.asarray(payprice, dtype=np.float64),
        _feature_frame=feature_frame.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Second-price settlement basics
# ---------------------------------------------------------------------------


class TestSecondPriceBoundary(unittest.TestCase):
    """A bid exactly equal to payprice wins and pays exactly that amount --
    the `>=` boundary, not `>`.
    """

    def test_bid_equal_to_payprice_wins_and_pays_payprice(self):
        pool = _make_pool(payprice=[100.0], click=[0])
        result = replay.settle_auctions(pool, bids=[100.0], budget=1000.0)
        self.assertEqual(result.impressions_won, 1)
        self.assertEqual(result.total_spend, 100.0)

    def test_bid_one_below_payprice_loses(self):
        pool = _make_pool(payprice=[100.0], click=[0])
        result = replay.settle_auctions(pool, bids=[99.0], budget=1000.0)
        self.assertEqual(result.impressions_won, 0)
        self.assertEqual(result.total_spend, 0.0)

    def test_bid_one_above_payprice_wins_and_pays_only_payprice(self):
        # Second-price: the winner pays the settlement price, not their bid.
        pool = _make_pool(payprice=[100.0], click=[0])
        result = replay.settle_auctions(pool, bids=[500.0], budget=1000.0)
        self.assertEqual(result.impressions_won, 1)
        self.assertEqual(result.total_spend, 100.0)  # NOT 500.0


# ---------------------------------------------------------------------------
# Zero-bid vector: no wins, no spend, no clicks, no division-by-zero
# ---------------------------------------------------------------------------


class TestZeroBids(unittest.TestCase):
    def setUp(self):
        self.pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[1, 0, 1])
        self.result = replay.settle_auctions(self.pool, bids=[0.0, 0.0, 0.0], budget=1000.0)

    def test_zero_wins(self):
        self.assertEqual(self.result.impressions_won, 0)

    def test_zero_spend(self):
        self.assertEqual(self.result.total_spend, 0.0)

    def test_zero_clicks(self):
        self.assertEqual(self.result.clicks_won, 0)

    def test_win_rate_is_zero_not_nan(self):
        # Both figures -- see settle_auctions() docstring "TWO WIN-RATE
        # FIGURES" -- are zero, not NaN, on a zero-bid vector (budget
        # never exhausted, so the two are also equal to each other).
        self.assertEqual(self.result.win_rate_bid_on, 0.0)
        self.assertFalse(np.isnan(self.result.win_rate_bid_on))
        self.assertEqual(self.result.win_rate_pool, 0.0)
        self.assertFalse(np.isnan(self.result.win_rate_pool))
        self.assertEqual(self.result.win_rate_bid_on, self.result.win_rate_pool)

    def test_effective_cpc_is_none_not_a_crash(self):
        # No division by zero -- effective CPC is undefined (None), not
        # inf/NaN, when there are zero clicks.
        self.assertIsNone(self.result.effective_cpc)

    def test_budget_not_exhausted(self):
        self.assertFalse(self.result.budget_exhausted)
        self.assertIsNone(self.result.exhaustion_index)
        self.assertIsNone(self.result.exhaustion_timestamp)

    def test_empty_pool_also_does_not_crash(self):
        # n_rows == 0 edge case: np.argmax on an empty over_budget array
        # must not be reached / must not raise.
        empty_pool = _make_pool(payprice=[], click=[])
        result = replay.settle_auctions(empty_pool, bids=[], budget=1000.0)
        self.assertEqual(result.impressions_won, 0)
        # Both win-rate figures are 0.0, not NaN, on an empty pool -- both
        # denominators are 0.
        self.assertEqual(result.win_rate_bid_on, 0.0)
        self.assertFalse(np.isnan(result.win_rate_bid_on))
        self.assertEqual(result.win_rate_pool, 0.0)
        self.assertFalse(np.isnan(result.win_rate_pool))
        self.assertIsNone(result.effective_cpc)
        self.assertFalse(result.budget_exhausted)


# ---------------------------------------------------------------------------
# Budget exhaustion: the core "hard stop, not skip-and-keep-shopping" test
# ---------------------------------------------------------------------------


class TestBudgetExhaustion(unittest.TestCase):
    """5 auctions, payprice = [10, 10, 10, 1, 10], budget = 25, bidder bids
    high enough to win every one of them if budget allowed.

    Hand-computed unconstrained cumulative spend: [10, 20, 30, 31, 41].
    First index where cumulative spend > 25 is index 2 (cum=30) -- so the
    bidder exhausts AT auction index 2. Auctions 2, 3, 4 are NOT bid on at
    all, even though auction 3 (payprice=1) would easily have fit under
    the remaining budget (25 - 20 = 5) if the engine kept shopping instead
    of stopping -- this is the specific "cherry-picking" behaviour the
    docstring says must NOT happen.
    """

    def setUp(self):
        self.payprice = [10.0, 10.0, 10.0, 1.0, 10.0]
        # Click on the row right after the stop point (index 3) -- must
        # NOT be counted, since that auction was never bid on.
        self.click = [0, 0, 0, 1, 0]
        self.pool = _make_pool(payprice=self.payprice, click=self.click)
        self.bids = [100.0, 100.0, 100.0, 100.0, 100.0]  # would win every auction unconstrained
        self.result = replay.settle_auctions(self.pool, bids=self.bids, budget=25.0)

    def test_budget_exhausted_flag(self):
        self.assertTrue(self.result.budget_exhausted)

    def test_exhaustion_index_is_first_auction_that_would_exceed_budget(self):
        self.assertEqual(self.result.exhaustion_index, 2)

    def test_exhaustion_timestamp_matches_pool(self):
        self.assertEqual(self.result.exhaustion_timestamp, self.pool.timestamp.iloc[2])

    def test_only_first_two_auctions_won(self):
        # Auctions 0, 1 won (fit under budget); auction 2 (the exhausting
        # bid) is voided; auctions 3, 4 are not bid on at all.
        self.assertEqual(self.result.impressions_won, 2)

    def test_impressions_bid_on_stops_at_exhaustion_index(self):
        self.assertEqual(self.result.impressions_bid_on, 2)

    def test_spend_never_exceeds_budget(self):
        self.assertLessEqual(self.result.total_spend, 25.0)
        self.assertEqual(self.result.total_spend, 20.0)  # 10 + 10, exactly

    def test_cheaper_later_auction_is_not_cherry_picked(self):
        # The core semantic assertion: auction index 3 (payprice=1.0, would
        # trivially fit under the remaining 5.0 of budget) is NOT won,
        # because the bidder had already stopped at index 2.
        self.assertEqual(self.result.impressions_won, 2)
        self.assertEqual(self.result.total_spend, 20.0)  # would be 21.0 if index 3 were cherry-picked

    def test_click_after_exhaustion_not_counted(self):
        # The click fixture is on index 3, which is past the stop point.
        self.assertEqual(self.result.clicks_won, 0)

    def test_win_rate_bid_on_computed_over_bid_on_auctions_only(self):
        # 2 wins / 2 bid-on auctions == 1.0, NOT 2/5 -- see settle_auctions()
        # docstring "TWO WIN-RATE FIGURES": win_rate_bid_on describes
        # performance while active.
        self.assertEqual(self.result.win_rate_bid_on, 1.0)

    def test_win_rate_pool_computed_over_full_pool_and_strictly_lower(self):
        # 2 wins / 5 pool rows == 0.4, and strictly LOWER than
        # win_rate_bid_on (1.0) precisely because the budget was
        # exhausted -- see settle_auctions() docstring.
        self.assertEqual(self.result.win_rate_pool, 2 / 5)
        self.assertLess(self.result.win_rate_pool, self.result.win_rate_bid_on)
        self.assertTrue(self.result.budget_exhausted)

    def test_spend_trajectory_flat_after_exhaustion(self):
        traj = self.result.spend_trajectory
        self.assertEqual(list(traj), [10.0, 20.0, 20.0, 20.0, 20.0])

    def test_budget_never_exhausted_when_it_easily_covers_everything(self):
        result = replay.settle_auctions(self.pool, bids=self.bids, budget=1_000_000.0)
        self.assertFalse(result.budget_exhausted)
        self.assertIsNone(result.exhaustion_index)
        self.assertEqual(result.impressions_bid_on, 5)
        self.assertEqual(result.impressions_won, 5)
        self.assertEqual(result.total_spend, sum(self.payprice))
        # Budget never exhausted -- the two win-rate figures are EQUAL
        # (impressions_bid_on == impressions_pool), see settle_auctions()
        # docstring "TWO WIN-RATE FIGURES".
        self.assertEqual(result.win_rate_bid_on, result.win_rate_pool)
        self.assertEqual(result.win_rate_bid_on, 1.0)


# ---------------------------------------------------------------------------
# Bid-vector alignment: fail loudly on mismatch
# ---------------------------------------------------------------------------


class TestBidVectorAlignment(unittest.TestCase):
    def test_too_short_bid_vector_raises(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        with self.assertRaises(ValueError):
            replay.settle_auctions(pool, bids=[10.0, 20.0], budget=1000.0)

    def test_too_long_bid_vector_raises(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        with self.assertRaises(ValueError):
            replay.settle_auctions(pool, bids=[10.0, 20.0, 30.0, 40.0], budget=1000.0)

    def test_error_message_names_the_mismatch(self):
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        with self.assertRaises(ValueError) as ctx:
            replay.settle_auctions(pool, bids=[10.0], budget=1000.0)
        message = str(ctx.exception)
        self.assertIn("(1,)", message)
        self.assertIn("(3,)", message)

    def test_no_silent_broadcast_of_scalar(self):
        # A bare scalar (shape ()) must not be silently broadcast to every
        # row -- it must raise exactly like any other length mismatch.
        pool = _make_pool(payprice=[10.0, 20.0, 30.0], click=[0, 0, 0])
        with self.assertRaises(ValueError):
            replay.settle_auctions(pool, bids=50.0, budget=1000.0)

    def test_negative_bid_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            replay.settle_auctions(pool, bids=[-5.0], budget=1000.0)

    def test_negative_budget_raises(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        with self.assertRaises(ValueError):
            replay.settle_auctions(pool, bids=[10.0], budget=-1.0)


# ---------------------------------------------------------------------------
# The payprice guard
# ---------------------------------------------------------------------------


class TestPayPriceGuard(unittest.TestCase):
    """`payprice` must be structurally unreachable from the bid-time view
    handed to bid-producing code. Column-set assertions alone are not
    enough (a future edit could re-add the column to the underlying frame
    without anyone noticing) -- these tests exercise the actual access
    paths (`.columns`, `[...]`) AND construct a `BidTimeView` directly
    around a frame that has payprice smuggled back into it, so a future
    regression that reintroduces payprice into the wrapped frame is still
    caught by the wrapper's own guard, not just by "the loader happens not
    to include it."
    """

    def test_bid_time_view_columns_excludes_payprice(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        view = pool.bid_time_view()
        self.assertNotIn("payprice", view.columns)
        self.assertNotIn("bidprice", view.columns)

    def test_bid_time_view_getitem_raises_for_payprice(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        view = pool.bid_time_view()
        with self.assertRaises(KeyError):
            view["payprice"]

    def test_bid_time_view_getitem_raises_for_bidprice(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        view = pool.bid_time_view()
        with self.assertRaises(KeyError):
            view["bidprice"]

    def test_bid_time_view_frame_excludes_payprice(self):
        pool = _make_pool(payprice=[10.0], click=[0])
        view = pool.bid_time_view()
        self.assertNotIn("payprice", view.frame().columns)

    def test_wrapper_blocks_payprice_even_if_smuggled_into_the_wrapped_frame(self):
        # This is the "make the test fail if someone adds payprice back
        # into that view" case: build a BidTimeView DIRECTLY around a raw
        # frame that DOES contain payprice (bypassing AuctionPool /
        # load_auction_pool entirely -- simulating exactly the regression
        # a future code change could introduce), and assert the guard
        # still holds.
        smuggled_frame = pd.DataFrame({"slotwidth": [1, 2], "payprice": [100.0, 200.0]})
        view = replay.BidTimeView(smuggled_frame)
        self.assertNotIn("payprice", view.columns)
        with self.assertRaises(KeyError):
            view["payprice"]
        self.assertNotIn("payprice", view.frame().columns)

    def test_wrapper_blocks_list_key_containing_payprice(self):
        smuggled_frame = pd.DataFrame({"slotwidth": [1], "payprice": [100.0]})
        view = replay.BidTimeView(smuggled_frame)
        with self.assertRaises(KeyError):
            view[["slotwidth", "payprice"]]

    def test_auction_pool_construction_rejects_a_feature_frame_containing_payprice(self):
        # Defence at the AuctionPool layer too, not only at BidTimeView:
        # constructing a pool whose _feature_frame carries payprice must
        # fail loudly (an assertion) rather than silently succeeding and
        # deferring the problem to whoever calls bid_time_view() later.
        bad_frame = pd.DataFrame({"slotwidth": [1], "payprice": [50.0]})
        with self.assertRaises(AssertionError):
            replay.AuctionPool(
                advertiser="1458",
                date_range=("2013-06-06", "2013-06-06"),
                seasons=(2,),
                n_rows=1,
                excluded_payprice_gt_bidprice_count=0,
                excluded_payprice_gt_bidprice_share=0.0,
                excluded_payprice_zero_count=0,
                excluded_payprice_zero_share=0.0,
                timestamp=pd.Series(pd.to_datetime(["2013-06-06"])),
                bidid=pd.Series(["bid0"]),
                click=np.asarray([0], dtype=np.int64),
                _payprice=np.asarray([50.0], dtype=np.float64),
                _feature_frame=bad_frame,
            )

    def test_settle_auctions_only_reads_payprice_from_the_private_field(self):
        # Sanity check that settlement outcomes are driven by
        # AuctionPool._payprice and NOT by anything in the (payprice-free)
        # bid-time view -- i.e. a strategy given only bid_time_view() has
        # no way to have peeked at the settlement price that produced
        # this result.
        pool = _make_pool(payprice=[10.0], click=[0])
        view = pool.bid_time_view()
        self.assertNotIn("payprice", view.columns)
        result = replay.settle_auctions(pool, bids=[10.0], budget=1000.0)
        self.assertEqual(result.total_spend, 10.0)  # only derivable from the private field


# ---------------------------------------------------------------------------
# Exclusion of unsimulatable rows: count/share correctness on small
# fixtures, reported SEPARATELY per rule, never merged -- see module
# docstring "ANOMALY EXCLUSION" and `_exclude_unsimulatable_rows()`.
# ---------------------------------------------------------------------------


class TestAnomalyExclusion(unittest.TestCase):
    def test_known_gt_bidprice_count_and_share(self):
        # 7 rows total, 3 of them payprice > bidprice, none payprice == 0.
        df = pd.DataFrame(
            {
                "_payprice": [10.0, 25.0, 5.0, 40.0, 8.0, 100.0, 3.0],
                "_bidprice": [20.0, 20.0, 5.0, 30.0, 8.0, 90.0, 30.0],
                # payprice > bidprice at rows 1 (25>20), 3 (40>30), 5 (100>90)
                # row 2 and row 4 are EQUAL (not anomalous: not strictly greater)
            }
        )
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 3)
        self.assertAlmostEqual(gt_share, 3 / 7)
        self.assertEqual(zero_count, 0)
        self.assertEqual(zero_share, 0.0)
        self.assertEqual(len(kept), 4)
        # The surviving rows are exactly the non-anomalous ones, in order.
        self.assertListEqual(list(kept["_payprice"]), [10.0, 5.0, 8.0, 3.0])

    def test_equal_payprice_and_bidprice_is_not_anomalous(self):
        # payprice == bidprice is a legitimate, exact second-price outcome
        # (the winner's bid exactly matches the settlement price) -- must
        # NOT be excluded, only strictly payprice > bidprice.
        df = pd.DataFrame({"_payprice": [50.0], "_bidprice": [50.0]})
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 0)
        self.assertEqual(gt_share, 0.0)
        self.assertEqual(zero_count, 0)
        self.assertEqual(zero_share, 0.0)
        self.assertEqual(len(kept), 1)

    def test_no_anomalies_gives_zero_share(self):
        df = pd.DataFrame({"_payprice": [1.0, 2.0, 3.0], "_bidprice": [10.0, 10.0, 10.0]})
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 0)
        self.assertEqual(gt_share, 0.0)
        self.assertEqual(zero_count, 0)
        self.assertEqual(zero_share, 0.0)
        self.assertEqual(len(kept), 3)

    def test_all_anomalies_gives_share_one(self):
        df = pd.DataFrame({"_payprice": [20.0, 30.0], "_bidprice": [10.0, 10.0]})
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 2)
        self.assertEqual(gt_share, 1.0)
        self.assertEqual(zero_count, 0)
        self.assertEqual(zero_share, 0.0)
        self.assertEqual(len(kept), 0)

    def test_empty_frame_does_not_divide_by_zero(self):
        df = pd.DataFrame({"_payprice": [], "_bidprice": []})
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 0)
        self.assertEqual(gt_share, 0.0)
        self.assertEqual(zero_count, 0)
        self.assertEqual(zero_share, 0.0)
        self.assertEqual(len(kept), 0)

    def test_known_zero_payprice_count_and_share(self):
        # 8 rows total: 2 payprice==0 (both with a positive, non-negative
        # bidprice, so NOT also caught by the gt_bidprice rule), 1
        # payprice>bidprice, 5 ordinary rows.
        df = pd.DataFrame(
            {
                "_payprice": [0.0, 10.0, 0.0, 5.0, 8.0, 40.0, 3.0, 12.0],
                "_bidprice": [50.0, 20.0, 30.0, 5.0, 8.0, 30.0, 30.0, 40.0],
                # zero rows: index 0, 2 (payprice==0, bidprice>0 -- not
                # also gt_bidprice-anomalous)
                # gt_bidprice-anomalous: index 5 (40>30)
            }
        )
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 1)
        self.assertAlmostEqual(gt_share, 1 / 8)
        self.assertEqual(zero_count, 2)
        self.assertAlmostEqual(zero_share, 2 / 8)
        # Neither count absorbs the other: 1 + 2 == 3 rows dropped, 5 kept.
        self.assertEqual(len(kept), 8 - 3)
        self.assertNotIn(0.0, list(kept["_payprice"]))
        self.assertListEqual(sorted(kept["_payprice"]), [3.0, 5.0, 8.0, 10.0, 12.0])

    def test_zero_payprice_rows_are_genuinely_gone_from_settled_pool(self):
        # The original motivating failure this exclusion fixes: a bid of
        # 0 legitimately "wins" a payprice==0 row under the `>=` rule.
        # After exclusion, such rows must not exist in the pool at all --
        # so a zero-bid vector wins nothing on them, because there is
        # nothing left to win.
        df = pd.DataFrame(
            {
                "_payprice": [0.0, 15.0, 0.0, 22.0],
                "_bidprice": [100.0, 20.0, 40.0, 30.0],
            }
        )
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(zero_count, 2)
        self.assertEqual(len(kept), 2)
        self.assertListEqual(list(kept["_payprice"]), [15.0, 22.0])

        # Feed the survivors into a real AuctionPool / settle_auctions run
        # and confirm a zero bid vector wins NOTHING -- the two
        # payprice==0 rows are simply absent, not present-but-losable.
        pool = _make_pool(payprice=list(kept["_payprice"]), click=[0, 0])
        result = replay.settle_auctions(pool, bids=[0.0, 0.0], budget=1000.0)
        self.assertEqual(result.impressions_won, 0)
        self.assertEqual(result.total_spend, 0.0)
        self.assertFalse(np.any(pool._payprice == 0.0))

    def test_disjointness_is_verified_not_assumed(self):
        # Exercise the disjointness assertion on the normal path: a
        # fixture with both rule types present but genuinely disjoint sets
        # must NOT raise, and must report each rule's count independently.
        df = pd.DataFrame(
            {
                "_payprice": [0.0, 25.0, 0.0, 9.0],
                "_bidprice": [10.0, 20.0, 5.0, 9.0],
                # index 0: payprice==0, bidprice=10 -- zero rule only
                # index 1: payprice=25 > bidprice=20 -- gt rule only
                # index 2: payprice==0, bidprice=5 -- zero rule only
                # index 3: payprice==bidprice==9 -- neither rule
            }
        )
        kept, gt_count, gt_share, zero_count, zero_share = replay._exclude_unsimulatable_rows(df)
        self.assertEqual(gt_count, 1)
        self.assertEqual(zero_count, 2)
        self.assertEqual(len(kept), 1)

    def test_overlapping_rule_matches_raise_loudly(self):
        # Construct the (structurally impossible under valid data)
        # overlapping case directly: a row with payprice==0 AND a
        # negative bidprice, so payprice > bidprice ALSO holds. This can
        # only happen if the data contains something invalid (a negative
        # bidprice) -- the function must fail loudly rather than
        # silently double-counting.
        df = pd.DataFrame({"_payprice": [0.0], "_bidprice": [-5.0]})
        with self.assertRaises(AssertionError):
            replay._exclude_unsimulatable_rows(df)


# ---------------------------------------------------------------------------
# AuctionPool field-alignment guard
# ---------------------------------------------------------------------------


class TestAuctionPoolAlignment(unittest.TestCase):
    def test_misaligned_click_array_raises(self):
        with self.assertRaises(ValueError):
            replay.AuctionPool(
                advertiser="1458",
                date_range=("2013-06-06", "2013-06-06"),
                seasons=(2,),
                n_rows=3,
                excluded_payprice_gt_bidprice_count=0,
                excluded_payprice_gt_bidprice_share=0.0,
                excluded_payprice_zero_count=0,
                excluded_payprice_zero_share=0.0,
                timestamp=pd.Series(pd.to_datetime(["2013-06-06"] * 3)),
                bidid=pd.Series(["a", "b", "c"]),
                click=np.asarray([0, 1], dtype=np.int64),  # length 2, not 3
                _payprice=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
                _feature_frame=pd.DataFrame({"x": [1, 2, 3]}),
            )


# ---------------------------------------------------------------------------
# load_auction_pool(): pure-Python validation (no real data needed)
# ---------------------------------------------------------------------------


class TestLoadAuctionPoolValidation(unittest.TestCase):
    def test_non_numeric_advertiser_raises(self):
        with self.assertRaises(ValueError):
            replay.load_auction_pool(
                "'; DROP TABLE impressions; --",
                ("2013-06-06", "2013-06-06"),
            )

    def test_empty_seasons_raises(self):
        with self.assertRaises(ValueError):
            replay.load_auction_pool(1458, ("2013-06-06", "2013-06-06"), seasons=())


# ---------------------------------------------------------------------------
# load_auction_pool() against real data -- sanity-check only that loading
# works, gated exactly like test_dataset_row_order.py / test_feature_allowlist.py
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    (paths.PROCESSED_ROOT / "impressions").exists(),
    "backend/data/processed/impressions is git-ignored and not present on "
    "a clean clone -- run `python -m src.ingest.ingest` (a real, non-sample "
    "run) from backend/ to generate it and enable this test.",
)
class TestLoadAuctionPoolRealData(unittest.TestCase):
    """Sanity-check only: does `load_auction_pool()` load successfully
    against the real corpus and produce a structurally sound pool? This is
    NOT a simulation-results test -- no bidding strategy is run here (out
    of scope for this module), just the loader itself.
    """

    @classmethod
    def setUpClass(cls):
        # Advertiser 1458, a single day, single season -- small and fast.
        cls.pool = replay.load_auction_pool(
            1458, ("2013-06-06", "2013-06-06"), seasons=(2,)
        )

    def test_pool_is_nonempty(self):
        self.assertGreater(self.pool.n_rows, 0)

    def test_advertiser_matches_request(self):
        self.assertEqual(self.pool.advertiser, "1458")

    def test_timestamps_are_sorted_ascending(self):
        ts = self.pool.timestamp
        self.assertTrue(ts.is_monotonic_increasing)

    def test_bid_time_view_excludes_payprice_and_bidprice(self):
        view = self.pool.bid_time_view()
        self.assertNotIn("payprice", view.columns)
        self.assertNotIn("bidprice", view.columns)

    def test_settle_auctions_runs_end_to_end_on_real_pool(self):
        # A trivial constant-zero bid vector, purely to confirm the whole
        # loader -> settlement pipeline runs without error on real data.
        # (No bidding strategy is being validated here -- out of scope.)
        #
        # `payprice == 0` rows are now excluded in `load_auction_pool()`
        # (see module docstring "ANOMALY EXCLUSION" and
        # `docs/analysis/simulation-results.md` S1) -- advertiser 1458 is
        # known to have carried 14 such rows before this change. With them
        # excluded, `pool._payprice` should contain no zero entries, so a
        # bid of 0 legitimately wins NOTHING (`0 >= payprice` is false for
        # every strictly-positive payprice) -- this is exactly the
        # motivating failure the exclusion fixes: a zero bid no longer
        # banks a "free" win.
        bids = np.zeros(self.pool.n_rows)
        result = replay.settle_auctions(self.pool, bids=bids, budget=1_000_000.0)
        self.assertEqual(result.total_spend, 0.0)
        self.assertEqual(result.clicks_won, 0)
        self.assertEqual(result.impressions_won, 0)
        self.assertFalse(np.any(self.pool._payprice == 0.0))

    def test_exclusion_metadata_is_present(self):
        self.assertGreaterEqual(self.pool.excluded_payprice_gt_bidprice_count, 0)
        self.assertGreaterEqual(self.pool.excluded_payprice_gt_bidprice_share, 0.0)
        self.assertLessEqual(self.pool.excluded_payprice_gt_bidprice_share, 1.0)
        self.assertGreaterEqual(self.pool.excluded_payprice_zero_count, 0)
        self.assertGreaterEqual(self.pool.excluded_payprice_zero_share, 0.0)
        self.assertLessEqual(self.pool.excluded_payprice_zero_share, 1.0)


if __name__ == "__main__":
    unittest.main()
