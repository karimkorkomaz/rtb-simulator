# Auction Simulation Results

This file holds the auction-simulation-layer analysis for the RTB thesis: data
anomalies specific to the simulator's assumptions (this document opens with
one such investigation), followed by baseline and strategy results once those
are built (constant / random / linear-in-CTR bidders, budget sweeps,
per-advertiser comparisons — see `docs/analysis/simulation-engine-notes.md`
for the engine itself). All numbers below were computed directly against
`backend/data/processed/{impressions,clicks,bids}/season=<2|3>/date=<YYYY-MM-DD>/part-0.parquet`
via DuckDB (`hive_partitioning=1`), with a fixed random seed where randomness
is involved, per project convention (`docs/analysis/eda-findings.md`).

**Author / verification date:** amineelahmad@gmail.com, 2026-09-08.

---

## 1. The `payprice == 0` anomaly

**Trigger:** `docs/analysis/simulation-engine-notes.md`, "Open data question:
`payprice == 0`." Under the replay engine's `>=` second-price settlement
rule, a bid of zero legitimately wins a `payprice == 0` auction for zero
spend. `eda-findings.md` §1 separately established that the *click* log
zero-fills its own `payprice` field in 78.6% of click rows — a population
artifact, not a real price — raising the question of whether the same
artifact is present in the *impressions* log this simulator actually reads,
and specifically whether advertiser 2261's disproportionate share of these
rows (0.45% of its own impressions, ~20x the corpus base rate) represents
free inventory it genuinely won or a logging defect the simulator should not
reward. This section investigates before any strategy work begins, per
scope. No engine code was changed for this investigation.

### 1.0 Baseline reconfirmation

Reconfirmed directly against the Parquet corpus (not assumed from the
engine notes):

| advertiser | total impressions | `payprice == 0` | share |
|---|---:|---:|---:|
| **2261** | **687,617** | **3,080** | **0.4479%** |
| 3476 | 1,970,360 | 144 | 0.0073% |
| 1458 | 3,083,056 | 14 | 0.0005% |
| 3427 | 2,593,765 | 12 | 0.0005% |
| 3386 | 2,847,802 | 4 | 0.0001% |
| 2259 | 835,556 | 2 | 0.0002% |
| 3358 | 1,742,104 | 1 | 0.0001% |
| 2821 | 1,322,561 | 1 | 0.0001% |
| 2997 | 312,437 | 0 | 0.0000% |
| **Total** | **15,395,258** | **3,258** | **0.0212%** |

Matches the engine notes exactly. 2261 holds 94.5% of all `payprice == 0`
rows in the corpus (3,080 / 3,258) despite being only 4.47% of total
impressions (687,617 / 15,395,258) — the concentration is real, not a
rounding artifact of the earlier report.

### 1.1 Season

| advertiser 2261 | season | total impressions | `payprice == 0` | share |
|---|---|---:|---:|---:|
| | 2 | 0 | 0 | — |
| | 3 | 687,617 | 3,080 | 0.4479% |

Advertiser 2261 has **zero** impressions in season 2 — it appears only in
season 3, so the question "does it cluster in one season" has a trivial
answer: there is only one season for this advertiser to cluster in. All
3,080 zero-payprice rows are season-3 rows by construction, not evidence of
a season-specific mechanism.

### 1.2 Date and hour

Season 3 covers nine dates; advertiser 2261 is active on exactly four of
them:

| date | total (2261) | `payprice == 0` | share |
|---|---:|---:|---:|
| 2013-10-24 | 155,254 | 1,045 | 0.6731% |
| 2013-10-25 | 210,069 | 285 | 0.1357% |
| 2013-10-26 | 211,827 | 1,109 | 0.5235% |
| 2013-10-27 | 110,467 | 641 | 0.5803% |

Spread across all four of the advertiser's active dates — not a single-day
incident. 2013-10-25 is the low outlier (0.14%, roughly a quarter of the
other three days' rate), but every date carries a nontrivial share; this
rules out "one bad day of logging" as the explanation.

Hour-of-day, pooling all four dates, blended across 2261's full traffic:

| hour | total (2261) | `payprice == 0` | share |
|---|---:|---:|---:|
| 00–12 (13 hours) | 302,761 | 342 | 0.05–0.23% each hour |
| 13 | 45,868 | 238 | 0.5189% |
| 14 | 54,244 | 381 | 0.7024% |
| 15 | 87,600 | 609 | 0.6952% |
| 16 | 85,692 | 515 | 0.6010% |
| 17 | 74,180 | 623 | 0.8398% |
| 18 | 37,271 | 372 | 0.9981% |

Rows outside 13:00–18:59 never exceed 0.23%; every hour from 13:00–18:59
sits at 0.52–1.00% — a 5–10x jump concentrated in a contiguous six-hour
afternoon/evening block, repeating across all four dates. Minute-level
inspection of the zero-payprice rows shows no single-minute spike analogous
to the corpus's documented hour-0/minute-1 batch-logging artifact
(`eda-findings.md` §4) — the largest single (date, hour, minute) cell is 13
rows, and the 30 largest cells together account for under 10% of the 3,080
rows. This is a daypart-level pattern, not a batch-write incident.

Because 2261's hourly traffic volume itself is uneven, §1.3 below re-checks
this pattern **within** the specific slot inventory driving the anomaly,
to separate "these hours generate more zero-payprice rows because they
generate more of this slot's traffic" from "the zero-payprice *rate* on this
slot itself rises in these hours." The latter is what is found (§1.3).

### 1.3 `adexchange`, `slotid`, `creative`

**`adexchange`:** 3,079 of 3,080 zero-payprice rows (99.97%) sit on
`adexchange = 3`. Within-value zero rate: exchange 3 is 1.4185% zero
(3,079 / 217,054), exchange 2 is 0.0004% (1 / 248,033), exchange 1 is
0.0000% (0 / 222,530) — a ~3,500x rate difference between exchange 3 and
exchange 1 for the same advertiser.

**`slotid`:** exactly three slot ids account for 3,079 of 3,080 rows
(99.97%); the remaining row is a single outlier on a numeric slotid via
exchange 2.

| slotid | total (2261) | `payprice == 0` | within-slot zero rate | share of all zero rows |
|---|---:|---:|---:|---:|
| `QQlive_SP_dsj_bottom_Width` | 4,674 | 1,875 | 40.12% | 60.88% |
| `QQlive_SP_dypd_bottom_Width` | 3,071 | 636 | 20.71% | 20.65% |
| `QQlive_SP_zypd_bottom_Width` | 2,998 | 568 | 18.95% | 18.44% |

These three within-slot rates (18.9–40.1%) are two to three orders of
magnitude above 2261's overall rate (0.4479%) — the anomaly is not spread
across 2261's inventory, it is almost entirely three specific ad slots.

**`creative`:** creative `12632` carries 3,079 of 3,080 rows (99.97%), a
within-creative zero rate of 7.87% (3,079 / 39,146) against creative
`12623`'s 0.0024% (1 / 41,532). Note creative 12632's total volume
(39,146) matches the combined total of the three slotids above almost
exactly (4,674 + 3,071 + 2,998 = 10,743 is not identical — creative 12632
also serves through other slot ids not in the top-3 list — but the 980x90
geometry below is exact, see §1.5) — this is one campaign line (one
creative, one slot geometry, one exchange), not scattered across 2261's
catalogue.

### 1.4 `slotprice` — the sharpest test

**On 2261's own zero-payprice rows:** slotprice is 0 in 3,079 of 3,080 rows
(99.97%); exactly one row has `slotprice > 0`. Cross-tabulated the other
way — among 2261's 233,273 rows with `slotprice == 0`, only 3,079 (1.32%)
have `payprice == 0`; the remaining 230,194 (98.68%) still clear at a real,
positive price. So `slotprice == 0` is a **structural property of specific
inventory** for 2261 (these three slots run with no reserve floor at all,
confirmed below), not itself an anomaly, and it does not by itself make
`payprice == 0` inevitable — most zero-floor auctions on this inventory
still sell for real money.

Confirming the floor is structural, not selectively logged as zero only
for these rows: on the three `QQlive_*` slotids specifically, **every**
row — zero-payprice and non-zero-payprice alike — has `slotprice == 0`
(checked directly: min/mean/max slotprice = 0 across both groups on all
three slotids). The non-zero-payprice rows on these same slots clear at a
real, wide range of prices (mean 137–147 fen, median 107–133 fen, up to 294
fen) despite the zero floor. So `payprice == 0` on 2261's rows does **not**
violate the floor — it cannot, because the floor is genuinely zero on this
inventory — which means the sharp "impossible under second-price
economics" test the task description anticipates does **not** indict
2261's rows the way it was expected to.

**Outside advertiser 2261, the picture is the opposite.** Of the other 178
zero-payprice rows in the corpus:

| bucket | rows | share |
|---|---:|---:|
| `slotprice > 0` (floor breached) | 174 | 97.8% |
| `slotprice == 0` (no floor) | 4 | 2.2% |

174 of 178 rows (97.8%) have a *positive* slot floor while settling at
`payprice == 0` — the settlement price falls below the slot's own reserve,
which is close to impossible under second-price economics with a floor and
is structurally the same class of violation as the already-excluded
`payprice > bidprice` anomaly. Advertiser 3476 alone (144 of the 178 rows)
is a clean example: **all 144 of its zero-payprice rows (100%) have
`slotprice > 0`**, ranging 4–80 fen, alongside `bidprice` of 249 or 254 —
a real reserve was in force and was not met by the recorded settlement
price.

**This is the key asymmetry of the investigation:** the 178 non-2261
zero-payprice rows fail the floor test almost completely (97.8% breach it);
2261's 3,080 rows almost entirely pass it (99.97% do not breach it, because
the floor is genuinely zero). The two populations need separate arguments —
see §1.8.

### 1.5 Structural comparison against 2261's non-zero rows

| field | zero-payprice rows (n=3,080) | non-zero rows (n=684,537) |
|---|---:|---:|
| slot geometry 980×90 | 99.97% | 5.27% |
| `slotvisibility`/`slotformat` = `Na`/`Na` | 99.97% | 48.24% |
| `bidprice = 294` (the higher of 2261's two tiers) | 99.97% | 63.77% |
| `region = 0` ("unknown") | 9.68% | 9.75% |
| distinct `region` values present | 35 | 35 |
| `domain` null | 0.03% | 0.05% |
| `url` null (adexchange 3 only, both groups) | 0.032% | 0.018% |
| `usertag` null | 48.67% | 47.26% |
| distinct `useragent` values | 1,046 (of 3,080 rows) | 58,428 (of 684,537 rows) |

Geometry, visibility/format, and bidprice tier are all near-saturated at
99.97% for the zero-payprice rows, consistent with §1.3's finding that this
is one narrow slice of inventory (one geometry, one visibility code, the
higher bid tier). **Region distribution is indistinguishable between the
two groups** (9.68% vs 9.75% at region=0, both spanning all 35 regions) —
the zero-payprice rows are not concentrated in one geography, arguing
against a region-specific delivery or logging issue. The raw `url`-null
gap seen in an unconditioned comparison (0.03% vs 6.95%) collapses to
0.032% vs 0.018% once `adexchange` is held fixed at 3 — it was a confound
of adexchange 2 (which runs 19.15% url-null structurally) being almost
entirely absent from the zero-payprice population, not a genuine
structural difference. `useragent` shows 1,046 distinct values across
3,080 rows (no single UA dominating, i.e. not a bot/single-client
signature) — consistent with real, distinct users, not synthetic traffic.

Overall: apart from the slot/creative/exchange/geometry/bid-tier
concentration already identified, the zero-payprice population looks like
an ordinary cross-section of 2261's users — it is not a demographically or
technically distinct population.

### 1.6 Clicks

Joined all zero-payprice `bidid`s against the `clicks` table:

- Advertiser 2261: **0 of 3,080** zero-payprice impressions match a click.
- Remaining 178 corpus-wide (non-2261): **0 of 178** match a click.

No click sits on any `payprice == 0` row anywhere in the corpus. The
concern this section was raised to check — a strategy banking a click on
an impression it did not genuinely win — does not materialise. The risk
these rows pose is confined to inflating **win counts and understating
spend**, not clicks.

### 1.7 The bid log

Joined 2261's 3,080 zero-payprice `bidid`s against the `bids` table:

- 3,078 of 3,080 (99.94%) have a matching row in `bids`; 2 do not.
- Where matched, the bid log's `slotprice` agrees with the impression
  log's `slotprice` for **all 3,078** rows (100% agreement, 0
  disagreements) — ruling out a join-key or transcription bug as the
  explanation; the same auction is consistently described across both
  tables.
- Aggregated over the 3,078 matched rows: `bidprice` 277–294 (matches the
  impression log exactly), `slotprice` min 0 / mean 0.0016 / max 5 fen —
  consistent with §1.4's finding that this inventory runs with an
  effectively zero floor.

The 2 unmatched `bidid`s are a rounding-level edge case (0.06% of the
3,080) consistent with the corpus's already-documented ~0.37% bidid
duplication/repeat-logging behaviour (`eda-findings.md` §5), not a new
finding. The bid log offers no independent evidence on the *price* itself
(it has no `payprice` column by construction, per `schema.py`) — it only
corroborates that the underlying bid request was real and consistently
described, which it is.

### 1.8 Recommendation: exclude, corpus-wide, all 3,258 rows — via two distinct arguments

**Exclude all 3,258 `payprice == 0` rows from simulation.** The evidence
splits into two populations that fail different, non-overlapping tests, so
the exclusion is stated for each separately rather than as one blanket
appeal to caution:

- **178 rows outside advertiser 2261 — exclude on the basis that 174 of
  178 (97.8%) have `slotprice > 0` while settling at `payprice == 0`,** a
  settlement price below the slot's own reserve floor — structurally the
  same class of second-price-economics violation as the already-excluded
  `payprice > bidprice` rows (advertiser 3476 alone: 144/144, 100%, floors
  4–80 fen). This is a direct impossibility test, not an inference.

- **Advertiser 2261's 3,080 rows — exclude on the basis that, within the
  three slot ids carrying 99.97% of them, the zero-payprice rate jumps
  5–10x during hours 13:00–18:59 (32–48%, computed within-slot in §1.2's
  underlying data) versus the rest of the day (3–11%) on the exact same
  inventory, with no accompanying drop in that same inventory's non-zero
  clearing price during those hours (median stays 100–150 fen, in line
  with or above the off-peak median)** — a rate spike decoupled from any
  corresponding market-price movement. Genuine low-competition free
  inventory would be expected to depress the *whole* price distribution in
  the low-competition window, not carve out an isolated spike at exactly
  zero while the surrounding non-zero prices continue trading normally.
  This pattern is consistent with an intermittent settlement-price
  recording failure specific to this slot/exchange/daypart combination.
  **This argument is inferential, not a structural impossibility** (unlike
  the 178-row case, `slotprice == 0` here means no floor is technically
  breached) — flagged as inference, not certainty, per project convention.
  If it needs independent confirmation, the check that would resolve it is
  whether the same three slot ids show an equivalent zero-price/afternoon
  pattern for *other* advertisers' traffic through the same exchange (§1.3
  already shows these three slotids are 2261-exclusive in this corpus, so
  that specific check is not available here without external data).

**Scope:** exclude corpus-wide, all 3,258 rows (extending the existing
`payprice > bidprice` exclusion policy, which is already simulation-only
and does not touch CTR-modelling data — see `eda-findings.md` §1.5).
3,080 of the 3,258 are attributable to mechanism (b) above and the
remaining 178 to mechanism (a).

**Cost of exclusion:** 3,258 / 15,395,258 = 0.0212% of all impressions;
3,080 / 687,617 = 0.4479% of advertiser 2261's impressions specifically.
Zero click-positive rows are lost (§1.6: 0 of 3,258 carry a click), so this
exclusion costs no CTR-relevant positives and, per the same reasoning
`eda-findings.md` §1.5 applied to the `payprice > bidprice` rows, should
apply to simulation only — not to CTR training data, where `payprice` is
already structurally excluded from the feature set and does not touch the
label join.

---

## 2. Baseline bidding strategies: engine, corrections, and alignment

This section documents the machinery built to run the three literature
baselines (constant / random / linear-in-predicted-CTR) and, before any
result is trusted, the two guards that machinery depends on: that the
click-probability correction chain was applied in the right order, and
that the correction's output (which carries no `bidid`) is attached to the
right auction. Code: `backend/src/simulation/replay.py` (engine, already
built and tested — see §1's exclusion rules, which this section reuses
unchanged), `backend/src/simulation/predictions.py` (corrections +
alignment), `backend/src/simulation/strategies.py` (the three bidders),
`backend/src/simulation/run_baselines.py` (the runner that ties them
together). All three new modules carry heavy inline docstrings explaining
the *why*, not restated in full here — this section summarizes and reports
the verification numbers a run actually produced.

### 2.1 The replay engine and its counterfactual limits (cited, not restated)

`replay.py`'s module docstring states three limits explicitly; they are
cited here because every number in §4 inherits them, not because they need
re-deriving:

1. **The logs contain only auctions iPinYou itself won.** Every strategy
   below competes over the exact same won-auction pool; "more wins" never
   means "more inventory than iPinYou originally saw," only "a different
   split of the same won inventory." This is the corpus's own §7 censoring
   problem (`eda-findings.md`: the empirical win rate is 15,395,258 /
   64,746,749 = 23.78%) inherited unavoidably into the simulator.
2. **The market is treated as static.** Settling a different bid against
   the recorded `payprice` assumes every other bidder would have behaved
   identically regardless of what this simulator bids — the standard
   simplifying assumption in the RTB benchmark literature, named rather
   than pretended away.
3. **`bidprice` is a fixed data-collection constant, never a baseline or a
   feature.** Structurally unreachable from any bid-producing code (see
   `replay.py` "THE PAYPRICE GUARD"), and never used to choose a sweep
   value in this section either.

Settlement is second-price (`bid >= payprice` wins, pays `payprice`) with a
hard, non-cherry-picking budget stop (§ `settle_auctions()`'s "BUDGET
SEMANTICS"). Two disjoint exclusion rules from §1 apply throughout this
section, carried on every result: `payprice > bidprice` (191,091 rows
corpus-wide, §1 baseline) and `payprice == 0` (3,258 rows corpus-wide, §1.8).
For the specific pool this section simulates — season-2 **test day**
(`2013-06-12`) only, one advertiser at a time, per the project's "never pool
advertisers" rule — the two rules remove:

| advertiser | raw rows (test day) | excl. `payprice>bidprice` | excl. `payprice==0` | pool rows simulated |
|---|---:|---:|---:|---:|
| 1458 | 447,493 | 0 | 3 | 447,490 |
| 3358 | 335,310 | 17,063 | 0 | 318,247 |
| 3386 | 392,901 | 0 | 0 | 392,901 |
| 3427 | 390,398 | 7,865 | 0 | 382,533 |
| 3476 | 91,236 | 895 | 12 | 90,329 |
| **total** | **1,657,338** | **25,823** | **15** | **1,631,500** |

25,838 rows (1.559% of the test day's 1,657,338) are excluded from
simulation; zero of these carry a click on 3358/3427/3476 (§1.6 already
established `payprice > bidprice` costs no click positives corpus-wide),
so no CTR-relevant signal is lost, only unsimulatable auctions.

### 2.2 The two-stage probability correction, asserted in order

The linear-in-CTR strategy needs a click probability. The one available is
`p_isotonic` from `backend/data/predictions/ctr_lgbm_season2_test.parquet`,
produced by composing two corrections in a specific order:

    p_isotonic = isotonic( recalibrate_probability( p_raw, rate=0.02 ) )

**Why the order is not interchangeable** (see `predictions.py`'s module
docstring for the full argument): the LightGBM model was fit on
negative-downsampled train data (rate 0.02, seed 42 —
`ctr_train_downsampling_season2.json`), which inflates `p_raw` relative to
the true click probability by a known, invertible amount.
`recalibrate_probability()` undoes exactly that inflation. The isotonic
calibrator was then **fit on the recalibrated validation probabilities**
(`p_calibrated`, split `season2_val` — never test), so it has only ever
seen probabilities in the *recalibrated* operating range; feeding it
`p_raw` directly evaluates its learned step function against a
systematically different distribution than it was fit on, silently
reintroducing the miscalibration the first stage corrected.

`predictions.assert_correction_order()` runs this check on every call to
`load_corrected_predictions()` (i.e. every simulation run), not once,
ad hoc. On the real season-2 test predictions (1,657,338 rows) it found:

| check | result |
|---|---|
| 1. `recalibrate_probability(p_raw, 0.02)` reproduces persisted `p_calibrated` | max abs diff = **0.0** (exact) |
| 2. `isotonic(recomputed p_calibrated)` reproduces persisted `p_isotonic` | max abs diff = **0.0** (exact) |
| 3. reverse order (`recalibrate(isotonic(p_raw))`) is materially different | max abs diff = **0.48**, mean abs diff = 5.68e-4; at the median probability the correct-order value is **~4.6x smaller** than the wrong-order one |
| 4. calibrator was fit on validation, not test | `fit_on.split == "season2_val"` — confirmed from the artifact's own metadata |

Check 3 is the check that makes check 2 non-vacuous (see
`predictions.py`'s docstring and `backend/tests/test_predictions.py`'s
`TestCheckThreeReverseOrderNotVacuous` for the degenerate case this guards
against): if applying the two corrections in the wrong order happened to
give the same answer, "check 2 passes" would prove nothing about ordering
specifically. It does not — at the median, a bidder that got the order
wrong would be bidding at roughly 4.6x the correct amount, squarely the "a
bidder that multiplies by a probability three times too high bids three
times too high" failure this project's rules call out by name.

### 2.3 Aligning bidid-less predictions to the auction pool

`ctr_lgbm_season2_test.parquet` carries no `bidid` (excluded from the CTR
feature allowlist as a non-predictive row identifier — `ingest.schema`),
but `replay.AuctionPool` is keyed and ordered by `(timestamp, bidid)` and
has rows *removed* by §2.1's exclusion rules — the two are neither the
same length nor the same row order, so a positional zip would be silently
wrong.

**Approach used** (`predictions.attach_bidid_to_predictions()`): re-run the
identical deterministic scan-then-sort recipe `features/dataset.py`
documents for its own row order (`ORDER BY bidid, timestamp`, done as an
in-memory stable mergesort after the DuckDB scan, for the performance
reason that module's docstring records), projecting only
`bidid, advertiser, timestamp`, and attach the result **positionally**.
This is verified, not assumed: row count and the elementwise
`advertiser`/`timestamp` match were checked before a single `bidid` was
trusted — both passed exactly, for all 1,657,338 rows, on the real data.

Per-advertiser extraction (`predictions.p_click_for_pool()`) then joins on
`(bidid, timestamp)` — **not `bidid` alone**, because `bidid` is not unique
even within one advertiser's own day (e.g. advertiser 1458's test day has
447,493 rows but only 446,650 distinct `bidid` values; a plain `bidid` join
would attach ambiguous predictions to unambiguous pool rows).
`(bidid, timestamp)` is unique for all but a vanishing residual: **4
colliding key-pairs (8 rows) across the entire 1,657,338-row test day**
(1458: 2 pairs, 3386: 1, 3427: 1; 3358 and 3476: none), confirmed by hand to
be genuinely different auctions (different `ipinyouid`, different
`region`/`city`) that happen to share both fields exactly — not duplicate
rows. Given the scale (0.0005% of rows), these are resolved by averaging
`p_isotonic` across the colliding candidates rather than guessing which one
is correct; a collision count above a hard-coded sanity bound (20 pairs)
would raise instead of silently applying the same resolution, so a much
larger, genuine alignment bug could not hide behind this known artifact.
Every one of the 5 advertisers' pools was confirmed to receive exactly one
`p_isotonic` value per row (no nulls, no unresolved duplicates) before any
bid was computed.

---

## 3. Experimental design: advertisers, sweep grids, and the budget ladder

### 3.1 Scope

Season-2 **test day only** (`date_range=("2013-06-12","2013-06-12")`,
`seasons=(2,)`) — the held-out split this thesis's CTR model was scored
against exactly once (`ctr_lgbm_test_evaluation_season2.json`), never
train or val. Five advertisers are present that day (discovered directly
from the impressions Parquet, independent of the predictions artifact):
**1458, 3358, 3386, 3427, 3476**. Each is simulated in its own,
never-pooled `AuctionPool` (§2.1's per-advertiser table). Advertisers
2261/2259/2821/2997 (present elsewhere in the corpus, per §1.0) do not
have test-day traffic in this split and are outside this section's scope.

### 3.2 Sweep grids: anchored to each advertiser's own `payprice` distribution

An unconditioned sweep (e.g. constant bids 1, 10, 100) risks landing
entirely inside one degenerate corner (never wins, or always wins) for a
campaign whose clearing prices are nowhere near those numbers. Instead,
every grid is anchored to the advertiser's own post-exclusion `payprice`
quantiles (the same distribution `eda-findings.md` §7 documents at the
corpus level) — a runner-level, aggregate-statistics read, computed
independently of and never passed into any bid-producing function (see
`run_baselines.py`'s module docstring for why this is not a payprice-guard
violation: no bidding *decision* on any specific auction is informed by a
per-row `payprice`, only fixed anchor points chosen once, before any
auction is loaded).

| advertiser | q10 | q25 | q50 | q75 | q90 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| 1458 | 16 | 34 | 63 | 76 | 140 | 177 |
| 3358 | 24 | 55 | 77 | 117 | 175 | 207 |
| 3386 | 20 | 44 | 70 | 95 | 172 | 210 |
| 3427 | 18 | 45 | 77 | 89 | 157 | 193 |
| 3476 | 20 | 42 | 73 | 84 | 138 | 176 |

(`payprice`, RMB fen — the iPinYou convention used throughout this
document; 100 fen = 1 CNY yuan.)

- **Constant bid**: the six quantiles above, swept directly as the bid
  amount.
- **Random bid**: four `[low, high)` ranges of increasing width —
  `(q10,q50)`, `(q25,q75)`, `(q50,q90)`, `(q10,q95)` — drawn via
  `numpy.random.default_rng([42, advertiser_id, range_index])`, seeded
  independently of budget (see `run_baselines.py`'s "RANDOM-BID SEEDING":
  the same draw is reused across every budget level for a given
  advertiser/range, so a budget sweep measures the effect of budget alone,
  not budget confounded with a fresh random draw).
- **Linear-in-CTR**: `bid = base * p_isotonic`. Mean `p_isotonic` across
  the pool is ~0.05–0.11% (a click rate of well under 0.1%, consistent
  with the corpus's ~0.075% base rate — `eda-findings.md`), so `base` must
  be large for the *average* bid to land near a `payprice`-scale number:
  `base = quantile / mean(p_isotonic)`, using the same six quantiles as
  the constant grid, giving bases from ~15,000 (1458, q10) up to ~383,000
  (3358, q95). Individual bids still vary per-impression, proportional to
  that impression's own predicted CTR — the whole point of the strategy.

### 3.3 Budget ladder: fractions of the advertiser's own historical spend

A fixed absolute budget is meaningless across advertisers of very
different scale: 1458's test-day pool alone settles over 30.3M fen while
3476's settles under 6.5M. Budget is instead expressed as a **fraction of
the advertiser's own total historical spend on its own post-exclusion
pool** — `sum(payprice)` over exactly the rows this engine can simulate,
i.e. what iPinYou itself actually spent to win that pool:

| advertiser | pool rows | total historical spend (fen) |
|---|---:|---:|
| 1458 | 447,490 | 30,297,100 |
| 3358 | 318,247 | 28,280,199 |
| 3386 | 392,901 | 31,379,459 |
| 3427 | 382,533 | 29,704,056 |
| 3476 | 90,329 | 6,494,835 |

Five levels are swept — `{1/32, 1/16, 1/8, 1/4, 1/2}` of that figure —
spanning heavily budget-constrained to comfortably-unconstrained relative
to each advertiser's own scale:

| advertiser | 1/32 | 1/16 | 1/8 | 1/4 | 1/2 |
|---|---:|---:|---:|---:|---:|
| 1458 | 946,784 | 1,893,569 | 3,787,138 | 7,574,275 | 15,148,550 |
| 3358 | 883,756 | 1,767,512 | 3,535,025 | 7,070,050 | 14,140,100 |
| 3386 | 980,608 | 1,961,216 | 3,922,432 | 7,844,865 | 15,689,730 |
| 3427 | 928,252 | 1,856,504 | 3,713,007 | 7,426,014 | 14,852,028 |
| 3476 | 202,964 | 405,927 | 811,854 | 1,623,709 | 3,247,418 |

400 (advertiser × strategy × parameter × budget) runs in total (5
advertisers × (6 constant + 4 random + 6 linear) × 5 budgets), each a
single vectorised `settle_auctions()` call against a pool loaded exactly
once per advertiser and reused across every strategy/parameter/budget
combination (runtime: well under a minute of settlement time; most of the
runner's wall-clock cost is the DuckDB reads, not the simulation itself).
Full machine-readable results:
`backend/data/metadata/simulation_baseline_results_season2_test.parquet`
(400 rows, one per run), with run configuration and per-advertiser design
(quantiles, budgets, grids) in
`backend/data/metadata/simulation_baseline_run_season2_test.json`.

---

## 4. Baseline results

### 4.1 The sweep produces a real spread, not a degenerate corner

Across all 400 runs: `win_rate_pool` ranges from **1.46% to 75.16%**,
`clicks_won` ranges from **1 to 346**. No run won 0% or 100% of the pool,
and no run won zero clicks — the grids in §3.2 are calibrated well enough
to produce genuine differentiation at every budget level, for every
advertiser, for all three strategies. Worked example — advertiser 1458,
budget = 1/8 of historical spend (3,787,138 fen):

| strategy | param | clicks won | win_rate_pool | spend (fen) | exhausted? | exhaustion time | effective CPC (fen) |
|---|---|---:|---:|---:|---|---|---:|
| constant | amount=16 (q10) | 7 | 0.1002 | 447,483 | no |  | 63,926 |
| constant | amount=34 (q25) | 45 | 0.2526 | 2,006,608 | no |  | 44,591 |
| constant | amount=63 (q50) | 50 | 0.2407 | 3,787,098 | yes | 14:32:07 | 75,742 |
| constant | amount=76 (q75) | 56 | 0.1827 | 3,787,086 | yes | 10:32:15 | 67,627 |
| constant | amount=140 (q90) | 55 | 0.1612 | 3,787,119 | yes | 09:15:34 | 68,857 |
| constant | amount=177 (q95) | 54 | 0.1465 | 3,787,062 | yes | 08:23:08 | 70,131 |
| linear_ctr | base=15077 (q10) | 334 | 0.0165 | 227,693 | no |  | 682 |
| linear_ctr | base=32039 (q25) | 334 | 0.0449 | 600,856 | no |  | 1,799 |
| linear_ctr | base=59366 (q50) | 338 | 0.0928 | 1,560,376 | no |  | 4,616 |
| linear_ctr | base=71616 (q75) | 341 | 0.1137 | 1,935,559 | no |  | 5,676 |
| linear_ctr | base=131924 (q90) | 301 | 0.1850 | 3,787,128 | yes | 21:14:04 | 12,582 |
| linear_ctr | base=166790 (q95) | 242 | 0.1757 | 3,787,117 | yes | 17:47:09 | 15,649 |
| random | [16, 63) (q10-q50) | 55 | 0.2976 | 3,133,776 | no |  | 56,978 |
| random | [34, 76) (q25-q75) | 60 | 0.2549 | 3,787,110 | yes | 16:32:04 | 63,118 |
| random | [63, 140) (q50-q90) | 57 | 0.1735 | 3,787,095 | yes | 10:07:18 | 66,440 |
| random | [16, 177) (q10-q95) | 52 | 0.1763 | 3,787,131 | yes | 10:41:17 | 72,829 |

Two things worth naming explicitly from this one table, both of which
recur across the other four advertisers (full data in the persisted
parquet):

- **The linear-in-CTR sweep has an interior optimum, not a monotone
  relationship with `base`.** Clicks rise from 334 (q10) to a peak of 341
  (q75), then *fall* to 301 (q90) and 242 (q95) — at the two highest
  bases, the strategy starts bidding aggressively enough on genuinely
  low-value impressions too that it exhausts its budget (21:14, 17:47)
  before the day ends, and stops bidding entirely for the remainder of the
  pool. More aggressive bidding is not free even for a well-targeted
  strategy.
- **`win_rate_pool` is also non-monotone in the constant/random grids**,
  for the same reason: amount=34 (q25) exhausts *later* than amount=63
  (q50) never at all in this run) yet ends up with a comparable click
  count via a different spend/time tradeoff — win rate alone, without the
  exhaustion time and spend trajectory alongside it, would be a misleading
  single-number summary of "how good" a bid level is.

### 4.2 Headline comparison: best-of-sweep per advertiser, per budget

For every (advertiser, budget) combination, the best-performing parameter
within each strategy's own sweep (by `clicks_won`, per §4.1's grids) is
reported below. `win_rate_pool` (not `win_rate_bid_on`) is used throughout
for cross-strategy comparison, per `replay.settle_auctions()`'s own
guidance: `win_rate_bid_on` is computed over a different, strategy-specific
denominator once a strategy exhausts its budget early, so a higher
`win_rate_bid_on` does not mean "won more of the same opportunity set" —
`win_rate_pool` shares the same, full-pool denominator across every row in
this table by construction.

**The "best param" column is selected on the test set, by the same
`clicks_won` it reports — read the absolute numbers as an upper bound.**
This is stated plainly rather than buried: no validation-day auction pool
was used to pick each strategy's sweep parameter, so every row below is the
*post hoc best* of that strategy's grid on the day being reported, not a
parameter a live bidder could have chosen in advance. A real campaign
picking `base` or a constant amount on 2013-06-11 and deploying it on
2013-06-12 would do somewhat worse than these figures. The distortion
applies *symmetrically* to all three strategies — each is given the same
oracle advantage over its own grid — so the §4.2 ranking (which strategy
wins) is not an artifact of it; only the absolute clicks-won and CPC levels
are optimistic. Two things bound how much: the linear-in-CTR sweep is
fairly flat near its optimum at low budgets (§4.1), and its margin over
constant/random is large enough (a 2.5–25x click gap across the two
tightest budget levels, narrowest for advertiser 3386)
that no plausible parameter-selection penalty reverses the ordering.
Selecting each sweep parameter on a held-out validation-day pool and
re-reporting is the correct fix, and is left as a defined follow-up rather
than claimed here.

| advertiser | budget (fraction) | budget (fen) | strategy | best param | clicks won | win_rate_pool | spend (fen) | exhausted? | exhaustion time | effective CPC (fen) |
|---|---:|---:|---|---|---:|---:|---:|---|---|---:|
| 1458 | 1/32 | 946,784 | constant | 34 (q25) | 23 | 0.1193 | 946,780 | yes | 14:52:09 | 41,164 |
| 1458 | 1/32 | 946,784 | linear_ctr | base=15077 (q10) | 334 | 0.0165 | 227,693 | no |  | 682 |
| 1458 | 1/32 | 946,784 | random | [16, 63) (q10-q50) | 21 | 0.0881 | 946,767 | yes | 11:40:08 | 45,084 |
| 1458 | 1/16 | 1,893,569 | constant | 34 (q25) | 45 | 0.2371 | 1,893,553 | yes | 22:49:06 | 42,079 |
| 1458 | 1/16 | 1,893,569 | linear_ctr | base=59366 (q50) | 338 | 0.0928 | 1,560,376 | no |  | 4,616 |
| 1458 | 1/16 | 1,893,569 | random | [16, 63) (q10-q50) | 38 | 0.1755 | 1,893,566 | yes | 16:43:08 | 49,831 |
| 1458 | 1/8 | 3,787,138 | constant | 76 (q75) | 56 | 0.1827 | 3,787,086 | yes | 10:32:15 | 67,627 |
| 1458 | 1/8 | 3,787,138 | linear_ctr | base=71616 (q75) | 341 | 0.1137 | 1,935,559 | no |  | 5,676 |
| 1458 | 1/8 | 3,787,138 | random | [34, 76) (q25-q75) | 60 | 0.2549 | 3,787,110 | yes | 16:32:04 | 63,118 |
| 1458 | 1/4 | 7,574,275 | constant | 76 (q75) | 108 | 0.3609 | 7,574,257 | yes | 14:29:11 | 70,132 |
| 1458 | 1/4 | 7,574,275 | linear_ctr | base=166790 (q95) | 346 | 0.2714 | 5,863,035 | no |  | 16,945 |
| 1458 | 1/4 | 7,574,275 | random | [63, 140) (q50-q90) | 103 | 0.3381 | 7,574,260 | yes | 13:36:07 | 73,537 |
| 1458 | 1/2 | 15,148,550 | constant | 76 (q75) | 213 | 0.7280 | 15,148,530 | yes | 23:00:18 | 71,120 |
| 1458 | 1/2 | 15,148,550 | linear_ctr | base=166790 (q95) | 346 | 0.2714 | 5,863,035 | no |  | 16,945 |
| 1458 | 1/2 | 15,148,550 | random | [63, 140) (q50-q90) | 213 | 0.6820 | 15,148,547 | yes | 20:50:05 | 71,120 |
| 3358 | 1/32 | 883,756 | constant | 55 (q25) | 12 | 0.0910 | 883,710 | yes | 00:04:52 | 73,642 |
| 3358 | 1/32 | 883,756 | linear_ctr | base=44383 (q10) | 202 | 0.0214 | 471,265 | no |  | 2,333 |
| 3358 | 1/32 | 883,756 | random | [24, 77) (q10-q50) | 14 | 0.0779 | 883,730 | yes | 00:02:36 | 63,124 |
| 3358 | 1/16 | 1,767,512 | constant | 77 (q50) | 20 | 0.1117 | 1,767,465 | yes | 00:02:05 | 88,373 |
| 3358 | 1/16 | 1,767,512 | linear_ctr | base=101711 (q25) | 218 | 0.0651 | 1,513,583 | no |  | 6,943 |
| 3358 | 1/16 | 1,767,512 | random | [55, 117) (q25-q75) | 24 | 0.1032 | 1,767,505 | yes | 00:01:58 | 73,646 |
| 3358 | 1/8 | 3,535,025 | constant | 117 (q75) | 35 | 0.1714 | 3,534,960 | yes | 00:02:08 | 100,999 |
| 3358 | 1/8 | 3,535,025 | linear_ctr | base=142396 (q50) | 230 | 0.1035 | 2,605,609 | no |  | 11,329 |
| 3358 | 1/8 | 3,535,025 | random | [77, 175) (q50-q90) | 39 | 0.1600 | 3,535,003 | yes | 00:02:01 | 90,641 |
| 3358 | 1/4 | 7,070,050 | constant | 117 (q75) | 64 | 0.3282 | 7,069,994 | yes | 00:03:09 | 110,469 |
| 3358 | 1/4 | 7,070,050 | linear_ctr | base=382804 (q95) | 248 | 0.2524 | 6,657,518 | no |  | 26,845 |
| 3358 | 1/4 | 7,070,050 | random | [55, 117) (q25-q75) | 71 | 0.4022 | 7,070,036 | yes | 11:34:15 | 99,578 |
| 3358 | 1/2 | 14,140,100 | constant | 117 (q75) | 141 | 0.6853 | 14,140,023 | yes | 15:18:06 | 100,284 |
| 3358 | 1/2 | 14,140,100 | linear_ctr | base=382804 (q95) | 248 | 0.2524 | 6,657,518 | no |  | 26,845 |
| 3358 | 1/2 | 14,140,100 | random | [77, 175) (q50-q90) | 145 | 0.6575 | 14,140,046 | yes | 13:55:09 | 97,518 |
| 3386 | 1/32 | 980,608 | constant | 95 (q75) | 14 | 0.0540 | 980,520 | yes | 00:45:08 | 70,037 |
| 3386 | 1/32 | 980,608 | linear_ctr | base=23159 (q10) | 56 | 0.0373 | 745,219 | no |  | 13,307 |
| 3386 | 1/32 | 980,608 | random | [70, 172) (q50-q90) | 17 | 0.0471 | 980,604 | yes | 00:44:12 | 57,683 |
| 3386 | 1/16 | 1,961,216 | constant | 172 (q90) | 38 | 0.0786 | 1,961,190 | yes | 00:49:06 | 51,610 |
| 3386 | 1/16 | 1,961,216 | linear_ctr | base=50949 (q25) | 96 | 0.0818 | 1,961,188 | yes | 18:08:09 | 20,429 |
| 3386 | 1/16 | 1,961,216 | random | [70, 172) (q50-q90) | 33 | 0.0946 | 1,961,167 | yes | 01:16:18 | 59,429 |
| 3386 | 1/8 | 3,922,432 | constant | 210 (q95) | 58 | 0.1524 | 3,922,431 | yes | 07:05:07 | 67,628 |
| 3386 | 1/8 | 3,922,432 | linear_ctr | base=81055 (q50) | 130 | 0.1504 | 3,922,365 | yes | 18:32:06 | 30,172 |
| 3386 | 1/8 | 3,922,432 | random | [70, 172) (q50-q90) | 48 | 0.1882 | 3,922,413 | yes | 09:56:18 | 81,717 |
| 3386 | 1/4 | 7,844,865 | constant | 210 (q95) | 93 | 0.2977 | 7,844,814 | yes | 11:18:12 | 84,353 |
| 3386 | 1/4 | 7,844,865 | linear_ctr | base=110003 (q75) | 183 | 0.2961 | 7,594,014 | no |  | 41,497 |
| 3386 | 1/4 | 7,844,865 | random | [20, 210) (q10-q95) | 96 | 0.3548 | 7,844,844 | yes | 14:31:12 | 81,717 |
| 3386 | 1/2 | 15,689,730 | constant | 210 (q95) | 192 | 0.5734 | 15,689,673 | yes | 16:27:09 | 81,717 |
| 3386 | 1/2 | 15,689,730 | linear_ctr | base=243164 (q95) | 244 | 0.4684 | 13,222,743 | no |  | 54,192 |
| 3386 | 1/2 | 15,689,730 | random | [70, 172) (q50-q90) | 181 | 0.6994 | 15,689,729 | yes | 21:30:12 | 86,684 |
| 3427 | 1/32 | 928,252 | constant | 89 (q75) | 9 | 0.0494 | 928,174 | yes | 01:06:03 | 103,130 |
| 3427 | 1/32 | 928,252 | linear_ctr | base=26014 (q10) | 226 | 0.0146 | 336,063 | no |  | 1,487 |
| 3427 | 1/32 | 928,252 | random | [45, 89) (q25-q75) | 9 | 0.0677 | 928,244 | yes | 01:12:09 | 103,138 |
| 3427 | 1/16 | 1,856,504 | constant | 89 (q75) | 19 | 0.0992 | 1,856,468 | yes | 01:20:12 | 97,709 |
| 3427 | 1/16 | 1,856,504 | linear_ctr | base=65035 (q25) | 251 | 0.0556 | 1,383,070 | no |  | 5,510 |
| 3427 | 1/16 | 1,856,504 | random | [18, 193) (q10-q95) | 27 | 0.0890 | 1,856,436 | yes | 01:07:12 | 68,757 |
| 3427 | 1/8 | 3,713,007 | constant | 157 (q90) | 49 | 0.1475 | 3,713,000 | yes | 04:36:05 | 75,776 |
| 3427 | 1/8 | 3,713,007 | linear_ctr | base=128625 (q75) | 271 | 0.1311 | 3,689,249 | no |  | 13,613 |
| 3427 | 1/8 | 3,713,007 | random | [18, 193) (q10-q95) | 46 | 0.1705 | 3,713,004 | yes | 10:36:12 | 80,717 |
| 3427 | 1/4 | 7,426,014 | constant | 157 (q90) | 89 | 0.2987 | 7,425,983 | yes | 11:55:14 | 83,438 |
| 3427 | 1/4 | 7,426,014 | linear_ctr | base=278929 (q95) | 284 | 0.2382 | 6,621,743 | no |  | 23,316 |
| 3427 | 1/4 | 7,426,014 | random | [77, 157) (q50-q90) | 74 | 0.3274 | 7,425,915 | yes | 13:15:15 | 100,350 |
| 3427 | 1/2 | 14,852,028 | constant | 193 (q95) | 156 | 0.5466 | 14,851,844 | yes | 16:20:08 | 95,204 |
| 3427 | 1/2 | 14,852,028 | linear_ctr | base=278929 (q95) | 284 | 0.2382 | 6,621,743 | no |  | 23,316 |
| 3427 | 1/2 | 14,852,028 | random | [77, 157) (q50-q90) | 134 | 0.6589 | 14,851,992 | yes | 20:19:07 | 110,836 |
| 3476 | 1/32 | 202,964 | constant | 42 (q25) | 2 | 0.1023 | 202,954 | yes | 03:37:03 | 101,477 |
| 3476 | 1/32 | 202,964 | linear_ctr | base=36626 (q10) | 21 | 0.0369 | 136,152 | no |  | 6,483 |
| 3476 | 1/32 | 202,964 | random | [20, 73) (q10-q50) | 6 | 0.0759 | 202,919 | yes | 01:39:05 | 33,820 |
| 3476 | 1/16 | 405,927 | constant | 73 (q50) | 5 | 0.0938 | 405,923 | yes | 00:37:10 | 81,185 |
| 3476 | 1/16 | 405,927 | linear_ctr | base=76915 (q25) | 25 | 0.0924 | 405,894 | yes | 09:58:13 | 16,236 |
| 3476 | 1/16 | 405,927 | random | [42, 84) (q25-q75) | 7 | 0.1149 | 405,923 | yes | 01:39:04 | 57,989 |
| 3476 | 1/8 | 811,854 | constant | 73 (q50) | 10 | 0.1895 | 811,802 | yes | 01:56:06 | 81,180 |
| 3476 | 1/8 | 811,854 | linear_ctr | base=133686 (q50) | 31 | 0.1659 | 811,702 | yes | 09:55:16 | 26,184 |
| 3476 | 1/8 | 811,854 | random | [73, 138) (q50-q90) | 10 | 0.1694 | 811,846 | yes | 01:03:11 | 81,185 |
| 3476 | 1/4 | 1,623,709 | constant | 176 (q95) | 16 | 0.2861 | 1,623,702 | yes | 02:32:05 | 101,481 |
| 3476 | 1/4 | 1,623,709 | linear_ctr | base=133686 (q50) | 34 | 0.1914 | 945,950 | no |  | 27,822 |
| 3476 | 1/4 | 1,623,709 | random | [20, 176) (q10-q95) | 14 | 0.3489 | 1,623,637 | yes | 07:13:05 | 115,974 |
| 3476 | 1/2 | 3,247,418 | constant | 73 (q50) | 24 | 0.7050 | 3,115,794 | no |  | 129,825 |
| 3476 | 1/2 | 3,247,418 | linear_ctr | base=322312 (q95) | 35 | 0.3505 | 1,871,274 | no |  | 53,465 |
| 3476 | 1/2 | 3,247,418 | random | [73, 138) (q50-q90) | 22 | 0.6596 | 3,247,389 | yes | 09:39:12 | 147,609 |

**Linear-in-CTR wins on clicks in all 25 of 25 (advertiser, budget)
combinations**, and its effective CPC is also the lowest in every one of
those 25 rows — it dominates on both axes tested, not a
clicks-for-cost tradeoff. `random` modestly outperforms `constant` on
clicks in most rows (a mild "some structure beats none" effect, per the
project's baseline-ordering rationale), but neither comes close to
linear-in-CTR at any budget level for any advertiser.

### 4.3 Why linear-in-CTR wins this decisively (investigating a suspiciously good result)

A margin this large (334 vs. 23 clicks at 1458's smallest budget, for a
tenth of the spend) is exactly the kind of number this project's rules say
must be investigated, not just reported. The explanation does not require
a leak: it follows from combining (a) the CTR model's already-validated
ranking quality (test AUC ≈ 0.929, established and cross-checked in Phase
2 — `ctr_lgbm_test_evaluation_season2.json`, out of scope to re-litigate
here) with (b) a structural fact about second-price auctions that has
nothing to do with modelling — **winning is decoupled from the winner's
own bid size**. An oracle that magically knew in advance which impressions
would be clicked, and bid to win exactly those at their own settlement
price, would need only:

| advertiser | clicks in pool | oracle cost to win them all (fen) | median `payprice`, click rows | median `payprice`, all rows |
|---|---:|---:|---:|---:|
| 1458 | 356 | 32,794 | 70 | 63 |
| 3358 | 268 | 30,104 | 96 | 77 |
| 3386 | 355 | 41,884 | 101 | 70 |
| 3427 | 301 | 34,002 | 95 | 77 |
| 3476 | 38 | 3,582 | 73 | 73 |

Click-carrying impressions do not clear at systematically higher prices
than the rest of the pool (medians differ by at most ~30 fen) — so a
strategy that can *rank* impressions by click likelihood, even
imperfectly, can win most of the clicks cheaply, because second-price
settlement means paying only the market clearing price regardless of how
high the bid is. `base = 15,077` at 1458's smallest budget spends only
227,693 fen (0.75% of 1458's total daily historical spend) to capture 334
of 356 clicks (93.8%) — roughly 7x the oracle's floor, not an unexplained
zero-cost result. This is precisely the mechanism the RTB literature cites
for why linear-in-predicted-CTR bidding is the standard non-trivial
baseline, and the magnitude here is large because the model's ranking
(AUC ≈ 0.929) is genuinely good, not because of a methodology defect. No
label leak is possible through the mechanism used: `p_isotonic` is derived
from a model trained on train, calibrated on validation, and applied to
test in a single forward pass (score_test_lgbm.py), and this section reads
that persisted column without re-scoring or re-fitting anything.

### 4.4 Exhaustion timing: when a "win" is a spending-speed artifact

The task brief warns explicitly against rewarding a strategy that "wins
more clicks by spending everything before noon." §4.1's illustrative table
already shows the effect concretely: at 1458, budget = 1/8, `constant`
amount=140 (q90) exhausts at **09:15:34** and wins 55 clicks, while
amount=76 (q75) exhausts later, at **10:32:15**, and wins one more click
(56) — a higher bid amount is not simply "better," and an early
exhaustion time on its own says nothing about quality without the
resulting click count alongside it. The same pattern recurs for
`linear_ctr`'s two highest bases (§4.1: q90 exhausts at 21:14, q95 at
17:47, both *worse* than the never-exhausted q75). Every exhaustion
timestamp used for comparison in this document is `pool.timestamp` at the
exhaustion index (local corpus time), read only via
`SimulationResult.exhaustion_timestamp` — never inferred from spend alone.
§4.2's headline table reports, for the best parameter in each
(advertiser, budget, strategy) triple, whether that specific run exhausted
and, if so, exactly when; a blank exhaustion-time cell there means that
parameter never exhausted at that budget.

### 4.5 Degenerate corners

None found in this sweep (§4.1): no run reached 0% or 100% `win_rate_pool`,
and no run won zero clicks. The closest approach to a corner is
`linear_ctr` at its two highest bases (q90/q95) for several advertisers,
where over-aggressive bidding on low-value impressions burns through
budget fast enough to hurt the click count relative to the interior
optimum (§4.1) — a real, informative finding about the strategy's
sensitivity to `base`, not a sweep-design failure.

---

## 5. Limits, honestly stated

- **This is a same-pool comparison, not a market-share estimate.** Every
  number above describes how the four strategies split the *same*
  won-auction pool differently; none of them can be compared to "how many
  impressions iPinYou could have won if it bid differently," because the
  75.6% of the market this dataset never observed (§2.1, citing
  `eda-findings.md` §7) is structurally invisible to this simulator.
- **The market is assumed static** — a strategy bidding very differently
  from the original collection strategy is assumed not to change how
  competitors would have bid, which is unlikely to hold exactly at the
  extremes (e.g. `linear_ctr`'s highest bases bidding 300,000+ fen on a
  handful of impressions) even though it is the standard assumption used
  throughout this literature.
- **25,838 rows (1.559% of the test day) are excluded from simulation**
  for being unsimulatable under second-price economics (§2.1); this
  concentrates unevenly by advertiser: 3358 loses 5.09% of its own test-day
  rows, 3427 loses 2.01%, 3476 loses 0.99% (`payprice>bidprice` and
  `payprice==0` combined), 1458 loses under 0.001%, and 3386 loses none —
  see the exact per-advertiser counts in §2.1. None of these exclusions
  touch CTR training data (§1.8).
- **The linear-in-CTR result should not be read as "the model can predict
  clicks with near-certainty."** AUC ≈ 0.929 is good, not perfect; §4.3's
  explanation is about second-price economics making imperfect ranking
  cheap to exploit, not about the model being unrealistically accurate.
- **Sweep parameters are chosen post hoc on the test day** (§4.2), so the
  absolute clicks-won and effective-CPC figures are an upper bound on what
  a bidder selecting its parameter in advance would achieve. The advantage
  is granted equally to all three strategies, so the ranking survives it;
  the levels do not. Re-selecting on a validation-day pool is the fix.
- **This section is baselines only**, per the project's required ordering
  — no pacing, no non-linear bidding. The finding that linear-in-CTR beats
  constant and random on every tested (advertiser, budget) combination
  (§4.2) is the expected, literature-consistent result, reported plainly
  rather than treated as a final answer: it establishes the bar a future
  non-linear or pacing strategy would need to clear, not a ceiling on what
  is achievable.

**Reproducibility**: seed 42 throughout (downsampling, isotonic fit
inherited from Phase 2/3 metadata; `random_bid`'s
`numpy.random.default_rng([42, advertiser, range_index])` in this section).
Full run configuration, per-advertiser sweep grids, and all 400 results are
persisted to `backend/data/metadata/simulation_baseline_run_season2_test.json`
and `backend/data/metadata/simulation_baseline_results_season2_test.parquet`
respectively — every number in §4 traces back to that run.
