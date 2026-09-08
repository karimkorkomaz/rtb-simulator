"""
Auction replay engine (Phase 3): turns a strategy's proposed bids into
second-price auction outcomes against the historical iPinYou logs, one
advertiser at a time, under a hard budget constraint.

SCOPE: this module is ONLY the replay engine -- it loads a single
advertiser's auction pool, settles a caller-supplied bid vector against
it, and reports outcomes. It does not choose bids (constant/random/
linear-in-CTR bidders are a separate, later stage) and it does not touch
predicted click probabilities or their calibration corrections (also
separate, later concerns). Nothing here should import a bidding strategy
or a CTR model artifact.

THE COUNTERFACTUAL AND ITS LIMITS (state these explicitly, do not leave
them implicit -- an examiner will ask):

* **The logs contain only auctions iPinYou itself won.** Impressions the
  original bidder lost are absent entirely, so this engine cannot recover
  inventory that was never observed. Every strategy replayed here competes
  over the exact same won-auction pool; "more wins" never means "more
  inventory than iPinYou originally saw," only "a different split of the
  SAME won inventory."
* **The market is treated as static.** Settling a different bid against
  the recorded `payprice` assumes every other bidder in that auction would
  have behaved identically regardless of what we bid. Real second-price
  markets shift when a participant's behaviour changes -- this is the
  standard simplifying assumption used throughout the RTB literature
  (Zhang et al.'s iPinYou benchmark and the follow-on work built on it),
  named here explicitly rather than pretended away.
* **`bidprice` is a fixed data-collection constant, never a baseline or a
  feature.** It is iPinYou's own submitted bid, set as a campaign-level
  collection knob (see `ingest/schema.py`'s `NON_FEATURE_BID_COLUMNS`
  docstring), not a live bidding decision. This engine never reads it to
  choose or score a bid; it is excluded from `BidTimeView` by the exact
  same allowlist mechanism as `payprice` (see "THE PAYPRICE GUARD" below).

SETTLEMENT: second-price. `bid >= payprice` wins and pays exactly
`payprice`; `bid < payprice` loses and pays nothing. See
`settle_auctions()`.

ANOMALY EXCLUSION: two DISJOINT rules drop rows in `load_auction_pool()`,
BEFORE any settlement, because they cannot be replayed without either
corrupting the simulation or modelling an auction rule this project does
not use. Both are **simulation-only**: neither touches the CTR-modelling
path (`features/dataset.py` applies neither, and must not -- `payprice`
is already structurally excluded from the feature set there). Reported
SEPARATELY, never merged, on both `AuctionPool` and every
`SimulationResult` -- they rest on different justifications (see below and
`docs/analysis/simulation-results.md` S1, cited here rather than
restated), so a merged count would hide which basis removed how many rows:

  1. **`payprice > bidprice`**: 191,091 rows across the full corpus
     (1.241% of 15,395,258, see `docs/analysis/eda-findings.md` S1), which
     is impossible under second-price economics as documented for this
     dataset (payprice is "the highest bid from competitors ... auction
     winning price", capped by the winner's own bid by construction).
     Carried as `excluded_payprice_gt_bidprice_count` /
     `_share`.
  2. **`payprice == 0`**: 3,258 rows corpus-wide (0.0212%), excluded via
     TWO DIFFERENT arguments for two disjoint sub-populations (see
     `simulation-results.md` S1.8 for the full investigation -- cited, not
     restated, here):
       * 178 rows outside advertiser 2261 fail a DIRECT economic-
         impossibility test: 174/178 (97.8%) settle below their own
         `slotprice` floor -- the same class of violation as rule 1.
       * Advertiser 2261's 3,080 rows PASS that floor test (the floor is
         genuinely zero on this inventory) and are excluded instead on an
         INFERENTIAL daypart argument: a zero-rate spike in hours
         13:00-18:59, confined to three zero-floor QQlive slots on
         adexchange 3, decoupled from any movement in the surrounding
         non-zero clearing price on the same inventory.
     Both sub-populations are carried together as one
     `excluded_payprice_zero_count` / `_share` pair (the count/share is
     merged only WITHIN rule 2, never with rule 1); see
     `simulation-results.md` for the sub-population breakdown.

Both counts and both shares are carried on `AuctionPool` and re-surfaced on
every `SimulationResult`, so no downstream table can report a number
without also being able to report what was excluded to get it, and by
which rule.

SINGLE-ADVERTISER POOLS: `load_auction_pool()` takes exactly one
`advertiser` id. A live bidder never chooses between two different
advertisers' impressions -- each campaign bids on its own traffic with its
own budget -- so a pool spanning multiple advertisers would describe a
decision no real bidder makes. A caller wanting a cross-advertiser view
must run this engine once per advertiser and roll the resulting
`SimulationResult`s up itself, clearly labelled as a secondary view.

THE PAYPRICE GUARD: the only object handed to bid-producing code is
`AuctionPool.bid_time_view()`, a `BidTimeView`. `payprice` is structurally
unreachable from it, twice over:

    1. The wrapped frame is built from `ingest.schema.feature_columns()`
       -- the SAME allowlist CTR-model feature selection uses -- which
       excludes `payprice` (and `bidprice`, `logtype`, `keypage`)
       structurally, because none of them are members of `BID_COLUMNS`
       (a bid-log row is written before the auction settles -- see
       `schema.py`'s module docstring). This is reused, not
       reinvented: `load_auction_pool()` calls `schema.feature_columns()`
       exactly as `features/dataset.py` does.
    2. `BidTimeView` ADDITIONALLY refuses to return a column named
       `payprice` under `.columns`, `[...]`, or `.frame()` even if one
       somehow ended up in the frame it wraps -- defence in depth against
       a future bug that constructs a `BidTimeView` directly from a raw
       frame, bypassing `load_auction_pool()` entirely.

`payprice` lives ONLY on the private, settlement-only side of
`AuctionPool` (the `_payprice` field), read in exactly one place:
`settle_auctions()`.

BUDGET SEMANTICS: see `settle_auctions()`'s docstring for the precise,
deliberately-chosen "hard stop" (not "skip and keep shopping") behaviour.

DETERMINISTIC ORDER: `load_auction_pool()` sorts the pool by `timestamp`
ascending with a stable tiebreak on `bidid` ascending
(`kind="mergesort"` -- a genuinely stable sort, not pandas's unstable
default `quicksort`). This fixes both the settlement order (which matters
for budget exhaustion: an auction must not be silently reordered ahead of
an earlier one just because it happens to be cheaper) and the
spend-trajectory x-axis, so a run reproduces exactly given the same inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import duckdb
import numpy as np
import pandas as pd

from ..ingest import paths as ingest_paths
from ..ingest import schema

# ---------------------------------------------------------------------------
# The payprice guard
# ---------------------------------------------------------------------------

# Columns that exist only to settle an auction AFTER it has happened --
# never observable at bid time, and therefore never allowed to reach a
# bid-producing function through `BidTimeView`. `bidprice` is included too
# (belt-and-suspenders): it IS technically bid-time observable (see the
# module docstring's "THE COUNTERFACTUAL AND ITS LIMITS"), but this
# project's explicit modelling-hygiene rule is that it must never be used
# as a baseline or a feature -- see `ingest.schema.NON_FEATURE_BID_COLUMNS`,
# which is the actual, single source of truth this mirrors.
_SETTLEMENT_ONLY_COLUMNS = frozenset({"payprice", "bidprice"})


class BidTimeView:
    """Read-only wrapper around a bid-time-observable feature frame -- the
    ONLY object a bid-producing function should ever receive (see
    `AuctionPool.bid_time_view()` and the module docstring's "THE PAYPRICE
    GUARD").

    Structurally refuses to expose `payprice` (or `bidprice`) under any
    access path -- `.columns`, `[...]`, `.frame()` -- regardless of whether
    the wrapped frame happens to contain one of those columns. This is
    defence in depth ON TOP OF the frame already having them excluded
    structurally by `ingest.schema.feature_columns()` at construction time
    (see `load_auction_pool()`): even a future bug that builds a
    `BidTimeView` directly from a raw, unfiltered impressions frame still
    cannot leak either column through this class's API.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        object.__setattr__(self, "_frame", frame)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover - defensive
        raise AttributeError("BidTimeView is read-only; it cannot be mutated in place.")

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def columns(self) -> list[str]:
        return [c for c in self._frame.columns if c not in _SETTLEMENT_ONLY_COLUMNS]

    def __getitem__(self, key: Union[str, list, tuple]):
        keys = key if isinstance(key, (list, tuple)) else [key]
        forbidden = [k for k in keys if k in _SETTLEMENT_ONLY_COLUMNS]
        if forbidden:
            raise KeyError(
                f"BidTimeView: {forbidden!r} is not bid-time observable -- "
                "it is the settlement outcome (or a fixed data-collection "
                "constant that must never be used as a baseline/feature), "
                "unknown/disallowed at bid time, and is never exposed by "
                "this view. See AuctionPool._payprice / settle_auctions() "
                "for the only sanctioned place `payprice` is read."
            )
        return self._frame[key]

    def frame(self) -> pd.DataFrame:
        """A defensive COPY of the wrapped frame with any settlement-only
        column dropped (belt-and-suspenders -- see class docstring).
        Copied, not a view, so a caller mutating the result cannot corrupt
        the `AuctionPool`'s internal state.
        """
        cols = [c for c in self._frame.columns if c not in _SETTLEMENT_ONLY_COLUMNS]
        return self._frame[cols].copy()


# ---------------------------------------------------------------------------
# AuctionPool
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AuctionPool:
    """An immutable, single-advertiser, single-date-range set of historical
    auctions, ordered deterministically (timestamp asc, bidid tiebreak --
    see module docstring), ready to be bid on via `settle_auctions()`.

    `eq=False`: several fields are `pandas`/`numpy` containers, whose `==`
    returns an elementwise array rather than a bool -- the dataclass-
    generated `__eq__` would raise `ValueError` the first time anyone
    compared or hashed two pools (e.g. inside a test's `assertEqual`), so
    identity/no-`__eq__` is used instead, same reasoning as `SimulationResult`
    below.

    Fields prefixed `_` (currently only `_payprice`) are SETTLEMENT-ONLY:
    they exist on this object because `settle_auctions()` needs them, not
    because bid-producing code should read them. Bid-producing code must
    only ever receive `bid_time_view()`'s return value, never the pool
    itself -- see "THE PAYPRICE GUARD" in the module docstring.
    """

    advertiser: str
    date_range: tuple[str, str]
    seasons: tuple[int, ...]

    n_rows: int  # row count AFTER both exclusions below -- the length every other field must match

    # Two DISJOINT exclusion rules, reported separately -- see module
    # docstring "ANOMALY EXCLUSION". Both shares use the SAME denominator:
    # the row count LOADED before any exclusion (excluded_count + n_rows +
    # the other rule's excluded_count), so the two shares are addable and
    # directly comparable -- never merge them into one figure.
    excluded_payprice_gt_bidprice_count: int  # rows dropped for payprice > bidprice
    excluded_payprice_gt_bidprice_share: float
    excluded_payprice_zero_count: int  # rows dropped for payprice == 0 (two sub-populations, one rule -- see module docstring)
    excluded_payprice_zero_share: float

    timestamp: pd.Series  # ordering key, datetime64[us]; index 0..n_rows-1
    bidid: pd.Series  # tiebreak key, aligned to timestamp; index 0..n_rows-1
    click: np.ndarray  # 0/1 int array, aligned; whether the ORIGINAL iPinYou impression at this row was clicked

    _payprice: np.ndarray  # SETTLEMENT-ONLY. Read exactly once, in settle_auctions().
    _feature_frame: pd.DataFrame  # bid-time-observable columns only (schema.feature_columns()); wrapped by bid_time_view()

    def __post_init__(self) -> None:
        n = self.n_rows
        lengths = {
            "timestamp": len(self.timestamp),
            "bidid": len(self.bidid),
            "click": len(self.click),
            "_payprice": len(self._payprice),
            "_feature_frame": len(self._feature_frame),
        }
        misaligned = {name: length for name, length in lengths.items() if length != n}
        if misaligned:
            raise ValueError(
                f"AuctionPool: field(s) {misaligned!r} do not match "
                f"n_rows={n} -- every pool field must be aligned to the "
                "same row order and length."
            )
        # Enforced by construction, not by convention -- see "THE PAYPRICE
        # GUARD". If either of these ever fires it means
        # `load_auction_pool()` (or a hand-built pool, e.g. in a test) is
        # about to hand a payprice/bidprice-carrying frame to bidding code.
        assert "payprice" not in self._feature_frame.columns, (
            "AuctionPool: _feature_frame must never contain payprice -- "
            "see module docstring, 'THE PAYPRICE GUARD'."
        )
        assert "bidprice" not in self._feature_frame.columns, (
            "AuctionPool: _feature_frame must never contain bidprice -- "
            "it is a fixed data-collection constant, never a feature or a "
            "baseline (see ingest/schema.py NON_FEATURE_BID_COLUMNS)."
        )

    def __len__(self) -> int:
        return self.n_rows

    def bid_time_view(self) -> BidTimeView:
        """The ONLY sanctioned way for bid-producing code to see this
        pool's data -- see `BidTimeView` / module docstring."""
        return BidTimeView(self._feature_frame)


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """The outcome of settling one bid vector against one `AuctionPool`
    under one budget. See `settle_auctions()`.

    `eq=False`: `spend_trajectory` is a `pandas.Series` -- same reasoning
    as `AuctionPool.eq=False`.
    """

    advertiser: str
    date_range: tuple[str, str]
    budget: float

    impressions_pool: int  # AuctionPool.n_rows: total impressions in the actually-simulated pool, i.e. POST both exclusions (see module docstring "ANOMALY EXCLUSION") -- NOT the raw loaded row count
    impressions_bid_on: int  # impressions actually bid on before any budget-exhaustion stop (== impressions_pool if never exhausted)
    impressions_won: int

    # TWO win-rate figures, deliberately named so neither can be mistaken
    # for the other at a glance in a results table -- see settle_auctions()
    # docstring "TWO WIN-RATE FIGURES" for when they diverge and which to
    # use for cross-strategy comparison. Neither divides by zero: both are
    # 0.0 (not NaN/inf) if their denominator is 0.
    win_rate_bid_on: float  # impressions_won / impressions_bid_on -- performance while active; NOT comparable across strategies with different exhaustion times
    win_rate_pool: float  # impressions_won / impressions_pool -- comparable across ALL strategies over the same pool; <= win_rate_bid_on always, strictly < iff budget_exhausted

    clicks_won: int  # clicks on impressions this strategy actually won (see settle_auctions() docstring)

    total_spend: float  # always <= budget by construction
    effective_cpc: Optional[float]  # total_spend / clicks_won; None (not inf/NaN) if clicks_won == 0

    budget_exhausted: bool
    exhaustion_index: Optional[int]  # pool-order index of the first auction NOT bid on due to the budget stop; None if never exhausted
    exhaustion_timestamp: Optional[pd.Timestamp]  # AuctionPool.timestamp at exhaustion_index; None if never exhausted

    spend_trajectory: pd.Series  # cumulative spend, one entry per pool row, in pool order (RangeIndex 0..n-1); flat from exhaustion_index onward

    # Carried through from the AuctionPool this result was produced
    # against -- see module docstring "ANOMALY EXCLUSION" and
    # `AuctionPool`'s field comments for the shared-denominator convention.
    excluded_payprice_gt_bidprice_count: int
    excluded_payprice_gt_bidprice_share: float
    excluded_payprice_zero_count: int
    excluded_payprice_zero_share: float


# ---------------------------------------------------------------------------
# Exclusion of unsimulatable rows (pure, unit-testable independent of the
# DuckDB load)
# ---------------------------------------------------------------------------


def _exclude_unsimulatable_rows(
    df: pd.DataFrame,
    *,
    payprice_col: str = "_payprice",
    bidprice_col: str = "_bidprice",
) -> tuple[pd.DataFrame, int, float, int, float]:
    """Drop rows that cannot be replayed under this engine's second-price
    settlement rule -- TWO DISJOINT rules, applied together but reported
    SEPARATELY (never merged into one count/share) -- see module docstring
    "ANOMALY EXCLUSION" for the full rationale and
    `docs/analysis/simulation-results.md` S1 for the underlying
    investigation; both are cited here, not restated:

      1. `payprice > bidprice`.
      2. `payprice == 0`.

    Pure function of a plain DataFrame (independent of how it was loaded)
    so it can be unit-tested directly against small hand-built fixtures
    with known exclusion counts, without touching the real Parquet data or
    DuckDB.

    Both masks are computed against the SAME pre-exclusion frame, and are
    ASSERTED disjoint before anything is dropped, rather than assumed
    disjoint -- `payprice == 0` cannot exceed a non-negative `bidprice`, so
    overlap should be structurally impossible, but this is verified in
    code so the two counts can never silently double-count if the data
    ever turns out to contain something unexpected (e.g. a negative
    `bidprice`). Raises `AssertionError`, loudly, if it ever does overlap.

    Returns `(kept_df, gt_bidprice_count, gt_bidprice_share, zero_count,
    zero_share)`. BOTH shares use the SAME denominator: `len(df)`, the row
    count LOADED before any exclusion (matches
    `docs/analysis/eda-findings.md`'s "1.241% of 15,395,258" framing) --
    never the surviving/kept count. Because they share a denominator, the
    two shares are directly addable and comparable.
    """
    loaded_total = len(df)
    if loaded_total == 0:
        return df.copy(), 0, 0.0, 0, 0.0

    gt_bidprice_mask = df[payprice_col] > df[bidprice_col]
    zero_mask = df[payprice_col] == 0

    overlap = gt_bidprice_mask & zero_mask
    if overlap.any():
        raise AssertionError(
            f"_exclude_unsimulatable_rows(): {int(overlap.sum())} row(s) "
            "match BOTH the payprice>bidprice rule and the payprice==0 "
            "rule -- these two exclusion rules are supposed to be disjoint "
            "(payprice==0 cannot exceed a non-negative bidprice). Refusing "
            "to silently double-count; investigate the data before "
            "proceeding."
        )

    gt_bidprice_count = int(gt_bidprice_mask.sum())
    zero_count = int(zero_mask.sum())

    kept = df.loc[~(gt_bidprice_mask | zero_mask)].copy()

    gt_bidprice_share = gt_bidprice_count / loaded_total
    zero_share = zero_count / loaded_total
    return kept, gt_bidprice_count, gt_bidprice_share, zero_count, zero_share


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_auction_pool(
    advertiser: Union[int, str],
    date_range: tuple[str, str],
    *,
    seasons: Iterable[int] = (2, 3),
    processed_root: Optional[Path] = None,
) -> AuctionPool:
    """Load a single-advertiser, single-date-range `AuctionPool`.

    `date_range`: inclusive `(start, end)` `YYYY-MM-DD` strings, same
    convention as `modeling.split.SplitBoundaries` / `features.dataset`'s
    `date_range=` parameter.

    `seasons`: which ingested season(s) to pull the date range from.
    Defaults to both, but callers doing an in-time vs. out-of-time
    comparison should pass an explicit single season -- see
    `modeling/split.py`'s "season 3 is deliberately NOT pooled with season
    2" rationale, which applies here too: mixing season 2 and season 3
    rows into one pool silently blends two different four-months-apart
    advertiser campaigns unless the caller has actively decided that's
    what they want.

    Steps, in order (see module docstring for the WHY of each):
      1. Load impressions for `advertiser` + `date_range` (+ `seasons`),
         left-joined to `clicks` on `bidid` for the click label -- same
         "project only `bidid` from the clicks side" leak-safety as
         `features/dataset.py`'s label join (a clicks-table row only
         exists when a click happened, so joining any non-key clicks
         column back in would leak the label under an innocuous name).
      2. Drop unsimulatable rows: `payprice > bidprice` AND `payprice == 0`
         (`_exclude_unsimulatable_rows`, two disjoint rules, reported
         separately -- see module docstring "ANOMALY EXCLUSION").
      3. Sort deterministically: timestamp asc, bidid tiebreak, `mergesort`.
      4. Split into the settlement-only `payprice` array and the
         allowlisted bid-time feature frame (`schema.feature_columns()`).

    Raises `FileNotFoundError` if the processed impressions table hasn't
    been ingested. Raises `ValueError` for a non-numeric `advertiser`, an
    empty `seasons`, or a `(advertiser, seasons, date_range)` combination
    that loads zero rows (almost always a caller bug -- wrong id or date
    range -- rather than a legitimate empty pool).
    """
    advertiser_str = str(advertiser)
    if not advertiser_str.isdigit():
        raise ValueError(
            f"load_auction_pool(): advertiser={advertiser!r} must be a "
            "numeric iPinYou advertiser id (e.g. 1458) -- refusing to "
            "interpolate a non-numeric value into the SQL query."
        )

    season_list = [int(s) for s in seasons]
    if not season_list:
        raise ValueError("load_auction_pool(): `seasons` must be non-empty.")

    start, end = date_range

    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_dir = root / "impressions"
    if not imp_dir.exists():
        raise FileNotFoundError(
            f"{imp_dir} does not exist -- run ingestion first (a real, "
            "non-sample run: see backend/src/ingest/README.md), or pass "
            "processed_root= explicitly if you meant to read a different "
            "location."
        )
    imp_glob = str(imp_dir / "**" / "*.parquet")
    clk_glob = str(root / "clicks" / "**" / "*.parquet")

    con = duckdb.connect()
    schema_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{imp_glob}', hive_partitioning=1)"
        ).fetchall()
    ]
    feature_cols = schema.feature_columns(schema_cols)
    # Reused, not reinvented -- see module docstring "THE PAYPRICE GUARD".
    # Cheap, redundant-by-design assertion: schema.feature_columns() already
    # guarantees this structurally, but this is exactly the kind of
    # guarantee worth re-checking at the one call site that turns it into a
    # bid-time-visible frame.
    assert "payprice" not in feature_cols and "bidprice" not in feature_cols, (
        "load_auction_pool(): schema.feature_columns() admitted payprice "
        "and/or bidprice -- this must never happen; see ingest/schema.py."
    )
    feature_select = ", ".join(f'i."{c}"' for c in feature_cols)
    season_filter = "i.season IN (" + ", ".join(str(s) for s in season_list) + ")"

    # DATE/advertiser literals interpolated directly (not bound
    # parameters), same convention as `features/dataset.py`'s date_range
    # handling: `start`/`end` only ever come from `modeling.split` or a
    # caller passing fixed ISO strings, and `advertiser_str` is validated
    # `.isdigit()` immediately above, so this is not raw user input.
    query = f"""
        SELECT {feature_select},
               i."bidid" AS "_bidid",
               i."timestamp" AS "_timestamp",
               i."payprice" AS "_payprice",
               i."bidprice" AS "_bidprice",
               CASE WHEN c.bidid IS NULL THEN 0 ELSE 1 END AS "_click"
        FROM read_parquet('{imp_glob}', hive_partitioning=1) i
        LEFT JOIN (
            SELECT DISTINCT bidid FROM read_parquet('{clk_glob}', hive_partitioning=1)
        ) c USING (bidid)
        WHERE {season_filter}
          AND i.date BETWEEN DATE '{start}' AND DATE '{end}'
          AND i.advertiser = '{advertiser_str}'
    """
    df = con.execute(query).fetch_arrow_table().to_pandas()

    if len(df) == 0:
        raise ValueError(
            f"load_auction_pool(): zero impressions loaded for "
            f"advertiser={advertiser_str!r}, seasons={season_list!r}, "
            f"date_range={date_range!r} -- check the advertiser id and "
            "date range, and that ingestion has been run for these seasons."
        )

    (
        df,
        excluded_payprice_gt_bidprice_count,
        excluded_payprice_gt_bidprice_share,
        excluded_payprice_zero_count,
        excluded_payprice_zero_share,
    ) = _exclude_unsimulatable_rows(df)
    n_rows = len(df)

    # Deterministic order (see module docstring, "DETERMINISTIC ORDER"):
    # timestamp asc, bidid tiebreak, genuinely stable sort.
    df = df.sort_values(["_timestamp", "_bidid"], kind="mergesort").reset_index(drop=True)

    feature_frame = df[feature_cols].reset_index(drop=True)

    return AuctionPool(
        advertiser=advertiser_str,
        date_range=(start, end),
        seasons=tuple(season_list),
        n_rows=n_rows,
        excluded_payprice_gt_bidprice_count=excluded_payprice_gt_bidprice_count,
        excluded_payprice_gt_bidprice_share=excluded_payprice_gt_bidprice_share,
        excluded_payprice_zero_count=excluded_payprice_zero_count,
        excluded_payprice_zero_share=excluded_payprice_zero_share,
        timestamp=df["_timestamp"].reset_index(drop=True),
        bidid=df["_bidid"].reset_index(drop=True),
        click=df["_click"].to_numpy(dtype=np.int64),
        _payprice=df["_payprice"].to_numpy(dtype=np.float64),
        _feature_frame=feature_frame,
    )


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def settle_auctions(pool: AuctionPool, bids: Union[np.ndarray, pd.Series, list], budget: float) -> SimulationResult:
    """Vectorised second-price settlement of `bids` against `pool`, under a
    hard budget. No per-row Python loop -- every step below is a numpy
    array operation over the full pool at once.

    `bids` must be an array-like of length `len(pool)`, in POOL ORDER
    (i.e. aligned to `pool.timestamp` / `pool.bid_time_view()`, one bid per
    row, `bids[i]` is the bid for the i-th auction in pool order). A
    length mismatch raises `ValueError` rather than silently broadcasting
    a scalar or reindexing -- see the module's non-negotiable "fail loudly
    on mismatch" requirement. `payprice` must never have been used to
    produce `bids`; see `AuctionPool.bid_time_view()` for the only
    sanctioned bid-time data access. This function itself does not (and
    structurally cannot, from its own inputs) prevent a caller from
    proposing bids that consulted `payprice` some OTHER way; the guard is
    that `AuctionPool` never hands `payprice` to anything but this
    function.

    SECOND-PRICE SETTLEMENT: for auction `i`, `bids[i] >= payprice[i]`
    wins and pays exactly `payprice[i]`; `bids[i] < payprice[i]` loses and
    pays nothing. The `>=` boundary is a genuine win at exactly
    `payprice`, not a loss.

    BUDGET SEMANTICS -- read carefully, this is a deliberate modelling
    choice, not an implementation accident:

    Would-be wins are honoured strictly in pool order (timestamp asc,
    bidid tiebreak), accumulating spend as they go. The FIRST would-be win
    whose price would push cumulative spend over `budget` is where the
    bidder runs out: that specific win is VOIDED (not honoured, spend
    never exceeds budget), and -- critically -- EVERY auction from that
    point onward in pool order is treated as NOT BID ON AT ALL, not merely
    lost, regardless of whether a later auction would have been a loss
    anyway, or even an easily-affordable win that would still fit under
    the remaining budget. A live bidder that has exhausted its budget
    stops bidding for the rest of the period; it does not keep evaluating
    every subsequent auction and cherry-pick the ones it could still
    technically afford -- that "skip this one, keep shopping for a
    cheaper one" behaviour is not available to a real exhausted campaign,
    and letting the SIMULATOR do it would hand every budget-constrained
    strategy an advantage no live bidder actually has.

    Consequences of this choice, made explicit:
      * `total_spend` never exceeds `budget`, by construction.
      * A strictly cheaper win occurring after the exhaustion point is
        deliberately NOT captured -- it is lost potential the strategy
        forwent by spending on earlier, pricier wins first, exactly as a
        real campaign would.
      * A click on an impression occurring after the exhaustion point is
        NOT counted in `clicks_won` -- the strategy never served that
        impression, so it could never have earned that click.

    TWO WIN-RATE FIGURES -- named so neither can be mistaken for the other
    at a glance in a results table (a bare `win_rate` is exactly the name
    that invites the mix-up, so neither result field is bare):

      * `win_rate_bid_on = impressions_won / impressions_bid_on`. Computed
        over the auctions the strategy actually had a chance to bid on,
        not over the full pool -- an exhausted strategy's win rate
        describes how it performed WHILE IT WAS STILL ACTIVE, not diluted
        by a stretch of the period it was never bidding in at all. NOT
        directly comparable across two strategies with different
        exhaustion times: each is computed over a different denominator
        (`impressions_bid_on` shrinks the earlier a strategy exhausts its
        budget), so a higher `win_rate_bid_on` does not mean "won more of
        the same opportunity set."
      * `win_rate_pool = impressions_won / impressions_pool`, where
        `impressions_pool` is the full, post-exclusion, ACTUALLY-SIMULATED
        pool (`AuctionPool.n_rows` -- see its docstring; NOT the raw loaded
        row count before exclusion). Comparable across ALL strategies
        replayed against the same pool, exhausted or not, because the
        denominator never changes. USE THIS ONE for any cross-strategy
        comparison table.

      The two are EQUAL if and only if the budget was never exhausted
      (`impressions_bid_on == impressions_pool`, so the two ratios share a
      denominator); `win_rate_pool` is strictly LOWER than `win_rate_bid_on`
      whenever the budget WAS exhausted, because the same `impressions_won`
      numerator is then divided by a strictly larger denominator (the
      unreached tail of the pool still counts against it). Neither divides
      by zero: both are `0.0` (never NaN/inf) if their respective
      denominator is `0`.

    Raises `ValueError` if `len(bids) != len(pool)`, if any bid is
    negative, or if `budget` is negative.
    """
    n = pool.n_rows
    bids_arr = np.asarray(bids, dtype=float)
    if bids_arr.shape != (n,):
        raise ValueError(
            f"settle_auctions(): bids has shape {bids_arr.shape}, expected "
            f"({n},) to align exactly with the {n}-row AuctionPool for "
            f"advertiser {pool.advertiser!r} ({pool.date_range!r}). "
            "Refusing to broadcast or reindex -- supply exactly one bid "
            "per pool row, in pool order."
        )
    if n > 0 and np.any(bids_arr < 0):
        raise ValueError("settle_auctions(): bids must be non-negative.")
    if budget < 0:
        raise ValueError("settle_auctions(): budget must be non-negative.")

    payprice = pool._payprice  # SETTLEMENT-ONLY read -- see AuctionPool docstring

    win_unconstrained = bids_arr >= payprice
    cost_unconstrained = np.where(win_unconstrained, payprice, 0.0)
    cum_cost_unconstrained = np.cumsum(cost_unconstrained)

    over_budget = cum_cost_unconstrained > budget
    if over_budget.any():
        stop_index = int(np.argmax(over_budget))  # first True -- see BUDGET SEMANTICS above
        budget_exhausted = True
    else:
        stop_index = n
        budget_exhausted = False

    bid_on = np.arange(n) < stop_index
    win = win_unconstrained & bid_on
    spend_per_auction = np.where(win, payprice, 0.0)
    cumulative_spend = np.cumsum(spend_per_auction)

    total_spend = float(cumulative_spend[-1]) if n else 0.0
    impressions_won = int(win.sum())
    impressions_bid_on = int(stop_index)
    win_rate_bid_on = (impressions_won / impressions_bid_on) if impressions_bid_on else 0.0
    win_rate_pool = (impressions_won / n) if n else 0.0

    clicks_won = int(pool.click[win].sum()) if n else 0
    effective_cpc = (total_spend / clicks_won) if clicks_won else None

    exhaustion_index: Optional[int] = stop_index if budget_exhausted else None
    exhaustion_timestamp: Optional[pd.Timestamp] = (
        pool.timestamp.iloc[stop_index] if (budget_exhausted and n) else None
    )

    spend_trajectory = pd.Series(
        cumulative_spend, index=pd.RangeIndex(n), name="cumulative_spend"
    )

    return SimulationResult(
        advertiser=pool.advertiser,
        date_range=pool.date_range,
        budget=float(budget),
        impressions_pool=n,
        impressions_bid_on=impressions_bid_on,
        impressions_won=impressions_won,
        win_rate_bid_on=win_rate_bid_on,
        win_rate_pool=win_rate_pool,
        clicks_won=clicks_won,
        total_spend=total_spend,
        effective_cpc=effective_cpc,
        budget_exhausted=budget_exhausted,
        exhaustion_index=exhaustion_index,
        exhaustion_timestamp=exhaustion_timestamp,
        spend_trajectory=spend_trajectory,
        excluded_payprice_gt_bidprice_count=pool.excluded_payprice_gt_bidprice_count,
        excluded_payprice_gt_bidprice_share=pool.excluded_payprice_gt_bidprice_share,
        excluded_payprice_zero_count=pool.excluded_payprice_zero_count,
        excluded_payprice_zero_share=pool.excluded_payprice_zero_share,
    )
