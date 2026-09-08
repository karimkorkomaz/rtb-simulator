# Auction replay engine — build notes and open decisions

Phase 3, week 7. Scope of this build: the replay engine only. No bidding
strategies, no CTR-probability plumbing, no simulation runs, no results.
Results will go to `docs/analysis/simulation-results.md` once strategies exist.

## Files

| File | Lines | Purpose |
| --- | --- | --- |
| `backend/src/simulation/replay.py` | 731 | Loader, unsimulatable-row exclusion, settlement, budget constraint, payprice guard |
| `backend/src/simulation/__init__.py` | 9 | Package marker |
| `backend/tests/test_replay.py` | 656 | 54 tests |

Uncommitted as of 2026-09-08. Updated same day for two changes: (1) the
`payprice == 0` exclusion (see "The `payprice == 0` question — resolved"
below), (2) the pool-denominated win-rate figure (decision 2 below).

## Test status

Full suite, `backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -t .`:

```
Ran 104 tests in 96.069s

OK
```

104 pass — 54 in `test_replay.py` (5 added for this update: the
`payprice == 0` exclusion counts/share, its disjointness-from-the-other-rule
check, the overlap-fails-loudly case, the "zero rows are genuinely gone from
the settled pool" fixture, and the pool-denominator win-rate divergence
case), 50 pre-existing, no regressions.

## Public API

```python
load_auction_pool(advertiser, date_range, *, seasons=(2, 3), processed_root=None) -> AuctionPool
settle_auctions(pool, bids, budget) -> SimulationResult
AuctionPool.bid_time_view() -> BidTimeView
```

`SimulationResult` carries: `impressions_pool`, `impressions_bid_on`,
`impressions_won`, `win_rate_bid_on`, `win_rate_pool`, `clicks_won`,
`total_spend`, `effective_cpc`, `budget_exhausted`, `exhaustion_index`,
`exhaustion_timestamp`, `spend_trajectory`,
`excluded_payprice_gt_bidprice_count`, `excluded_payprice_gt_bidprice_share`,
`excluded_payprice_zero_count`, `excluded_payprice_zero_share`. The last four
are carried through unchanged from the `AuctionPool` a result was produced
against (see "Exclusion of unsimulatable rows" below) — the former
`anomaly_excluded_count`/`anomaly_excluded_share` names are retired; they
were ambiguous once a second exclusion rule existed. `win_rate` (bare) is
likewise retired in favour of the two explicitly-denominated fields — see
decision 2 below.

Settlement is fully vectorised — `cumsum` over the ordered pool, no per-row
Python loop. Pool order is timestamp ascending with a stable `bidid` tiebreak
(`kind="mergesort"`), so a run reproduces exactly.

## The payprice guard

`payprice` is read in exactly one place: `settle_auctions()`. Three layers keep
it away from anything that produces a bid.

1. The bid-time frame is built by calling `ingest.schema.feature_columns()` —
   the same allowlist CTR modelling uses, reused rather than reinvented. It
   excludes `payprice` and `bidprice` structurally.
2. `BidTimeView` independently refuses to expose either column via `.columns`,
   `[...]`, or `.frame()`, even if one is present in the frame it wraps.
3. `AuctionPool.__post_init__` asserts the feature frame carries neither.

The test that matters is
`test_wrapper_blocks_payprice_even_if_smuggled_into_the_wrapped_frame`: it
builds a `BidTimeView` directly around a frame that *does* contain `payprice`,
bypassing the loader entirely. Asserting on the column set alone would pass
even if the guard were removed; this fails if anyone reintroduces the column
downstream.

`bidprice` is excluded on the same footing — it is a fixed data-collection
constant, never a baseline to beat and never a feature.

## Exclusion of unsimulatable rows

Two DISJOINT rules, reported separately on `AuctionPool` /
`SimulationResult`, never merged into one count (see `replay.py`'s
`_exclude_unsimulatable_rows()` and its module docstring "ANOMALY
EXCLUSION" for the full rationale):

1. `payprice > bidprice` — **191,091 of 15,395,258** rows corpus-wide,
   matching `eda-findings.md` §1 exactly.
2. `payprice == 0` — **3,258 of 15,395,258** rows corpus-wide. See "The
   `payprice == 0` question — resolved" below for the argument (two
   different sub-populations, two different tests) and
   `docs/analysis/simulation-results.md` §1 for the full investigation.

Both are simulation-only and do not touch the CTR modelling path.
`_exclude_unsimulatable_rows()` computes both masks against the same
pre-exclusion frame and asserts they are disjoint before dropping anything
— verified in code, not assumed, even though `payprice == 0` cannot
structurally exceed a non-negative `bidprice`.

## Four decisions to ratify

These shape every comparison table, so they are worth settling before
strategies are built on top.

### 1. Budget is a hard stop, not "skip and keep shopping"

Would-be wins are honoured in pool order. The first one whose price would push
cumulative spend over budget is voided, and **every auction from that index
onward counts as not-bid-on**, not merely lost — including a cheaper later
auction that would still have fit under the remaining budget.

Rationale: a live campaign that has run out stops. It does not keep evaluating
every subsequent auction and pick off the ones it can still afford. Letting the
simulator do that would hand every budget-constrained strategy an advantage no
real bidder has. Tested by
`TestBudgetExhaustion.test_cheaper_later_auction_is_not_cherry_picked`.

Consequences: `total_spend` never exceeds budget by construction, and a click
after the exhaustion point is never counted — the strategy never served that
impression. Note `impressions_pool` (the denominator for decision 2 below) is
now the pool AFTER both exclusion rules — see "Exclusion of unsimulatable
rows" above — not the raw loaded row count.

### 2. Two win-rate figures, not one — resolved in favour of reporting both

Originally this engine reported a single `win_rate =
impressions_won / impressions_bid_on`. That denominator is still reported,
now named `win_rate_bid_on`, for the reason originally given: an exhausted
strategy's rate then describes how it performed while still active, not
diluted by a stretch of the period it was not bidding in at all.

But the trade-off flagged in the previous version of this note — win rates
across strategies with different exhaustion times are not directly
comparable, because they are computed over different denominators — is a
real problem for any cross-strategy comparison table, so it is now resolved
by adding a second figure rather than picking one: `win_rate_pool =
impressions_won / impressions_pool`, using the fixed, post-exclusion pool
size as denominator. `win_rate_pool` is comparable across every strategy
replayed against the same pool, exhausted or not, and is the one to use for
a cross-strategy comparison table. Both are named explicitly
(`win_rate_bid_on` / `win_rate_pool`) rather than leaving one bare, since a
bare `win_rate` was exactly the name that invited conflating the two.

The two are equal iff the budget was never exhausted (the two denominators
coincide); `win_rate_pool` is strictly lower than `win_rate_bid_on` whenever
it was, since the same numerator is then divided by a strictly larger
denominator. Neither divides by zero on an empty pool or an all-zero bid
vector — both are `0.0`, not NaN.

### 3. `exhaustion_timestamp` is the first voided auction

Not the last won auction — it marks the moment bidding stopped.

### 4. `spend_trajectory` is indexed by position

A `RangeIndex`, not timestamps. `pool.timestamp` sits alongside it, so plotting
against wall-clock time is available but requires a join.

## The `payprice == 0` question — resolved

This section previously flagged, as an open data question, that **3,258
impressions corpus-wide** have `payprice == 0` (0.021%), concentrated ~20x
above base rate on advertiser 2261 (3,080 of the 3,258, 0.45% of its
impressions), and that under the `>=` settlement rule a zero bid
legitimately wins these for zero spend — worth resolving before strategy
results are produced, since a strategy replayed over 2261 could otherwise
bank several thousand free impressions it did not earn.

**Resolved by the investigation in `docs/analysis/simulation-results.md`
§1** (cited here, not restated): all 3,258 rows are excluded from
simulation, via two separate arguments for two disjoint sub-populations —
178 rows outside advertiser 2261 fail a direct economic-impossibility test
(174/178, 97.8%, settle below their own `slotprice` floor); advertiser
2261's 3,080 rows pass that test (the floor is genuinely zero on their
inventory) and are excluded instead on an inferential daypart argument (a
zero-rate spike in hours 13:00–18:59 on three zero-floor QQlive slots on
adexchange 3, decoupled from any movement in the surrounding non-zero
clearing price). Zero click-positive rows are lost either way (§1.6: 0 of
3,258 carry a click).

**What the engine now does:** `load_auction_pool()` drops all `payprice ==
0` rows in the same step as the `payprice > bidprice` exclusion (see
"Exclusion of unsimulatable rows" above), reported as its own count/share
(`excluded_payprice_zero_count` / `_share`), never merged with the
`payprice > bidprice` figures. A bid of `0` therefore no longer wins
anything in the simulated pool — the rows it used to win for free are
simply absent. `TestAnomalyExclusion.test_zero_payprice_rows_are_genuinely_gone_from_settled_pool`
in `test_replay.py` exercises exactly this: it builds a small fixture with
`payprice == 0` rows, confirms they are dropped by
`_exclude_unsimulatable_rows()`, feeds the survivors through
`settle_auctions()` with an all-zero bid vector, and asserts zero wins and
zero spend.

## Deliberately not built

Per scope: constant / random / linear-in-CTR bidders, the downsampling
recalibration and isotonic composition, per-advertiser sweeps, budget sweeps,
and `docs/analysis/simulation-results.md`.
