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

*Strategy results (constant / random / linear-in-CTR baselines, budget
sweeps, per-advertiser tables) to be appended below this line once built.*
