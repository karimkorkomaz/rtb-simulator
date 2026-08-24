# Exploratory Data Analysis Findings

Source data: `backend/data/processed/{bids,impressions,clicks,conversions}/season=<2|3>/date=<YYYY-MM-DD>/part-0.parquet`
(iPinYou seasons 2 and 3, ingested per `backend/src/ingest/README.md`). All numbers below were
computed directly against these Parquet files via DuckDB (`hive_partitioning=1`), not against the
raw bz2 logs, and not against any cached/rounded figure from `validation_report.json` unless
explicitly noted as cross-checking that report. Figures live in `reports/figures/` as matched
PNG (300 DPI) + SVG pairs.

**Author / verification date:** amineelahmad@gmail.com, 2026-08-24.

## Methodological caveat that applies to every section below

**No train/val/test split exists yet.** The planned split is temporal and explicit (see
`backend/src/ingest/README.md`, "Where train/val/test splitting and downsampling would slot in"
— e.g. train on days 1-5 of a season, validate on day 6, test on day 7, cut points configurable,
never random) but has not been implemented. Because of this, every analysis in this document runs
over the **full** ingested population (both seasons, all dates), which necessarily includes rows
that will eventually be assigned to a held-out test partition. This is a deliberate, acknowledged
violation of "never look at test data before modelling," made explicit here rather than silently
ignored, because there is currently no partition to exclude. **Findings that must be re-run once
the split lands:** class imbalance (§2, positive rate may shift if the held-out day(s) are
atypical), temporal patterns (§4, hour/day-of-week rates are pooled across seasons and would
change composition once test days are excluded), and feature-target CTR breakdowns (§6). The
payprice anomaly (§1) and feature cardinality/hashing analysis (§3) are structural/schema-level
findings that a split will not change, since they concern the full data-generating process, not a
train-set-specific estimate.

---

## 1. The `payprice > bidprice` anomaly

**Figure:** `reports/figures/part1_payprice_anomaly_concentration.png` / `.svg`

`validate.py` flags 191,091 impressions (1.241% of 15,395,258) where `payprice > bidprice`. In a
second-price auction the winner's settlement price can never exceed their own bid — the paper
this dataset comes from (Zhang et al., "Real-Time Bidding Benchmarking with iPinYou Dataset")
describes `payprice` explicitly as "the highest bid from competitors, also called market price and
auction winning price," i.e. a second-price outcome that is capped by the winner's own bid by
construction. These 191,091 rows are therefore structurally impossible under the stated auction
model. `known.data.bugs.txt` (shipped with the raw dataset) documents exactly one bug — season-1
conversion logs mislabelled `logtype=2` instead of `3` — and does **not** mention this anomaly;
season 1 is excluded from this pipeline entirely, so that documented bug is not the explanation
here, and no upstream documentation covers this specific issue. This is a finding derived
empirically from the data, not something citable to a known-bugs list.

### 1.1 Concentration: not spread uniformly, confined to 3 of 9 advertisers, 100% season 2

| advertiser | total impressions | anomalous | rate |
|---|---|---|---|
| 3358 | 1,742,104 | 84,412 | 4.85% |
| 3427 | 2,593,765 | 81,326 | 3.14% |
| 3476 | 1,970,360 | 25,353 | 1.29% |
| 1458, 2259, 2261, 2821, 2997, 3386 | 12,088,798 (combined) | 0 | 0.00% |

Split by season: **100% of the 191,091 anomalous rows are in season 2** (season-2 rate 1.56% of
12,237,087 impressions; season-3 rate exactly 0.00% of 3,158,171). Split by ad exchange: adexchange
1 carries the most (150,787 of 191,091, i.e. 3.24% of its 4,651,960 impressions), adexchange 2 has
31,969 (0.67%), adexchange 3 has 8,335 (0.16%), and adexchange 4 / null have zero. The
advertiser x adexchange cross-tab confirms the anomaly exists **only** at the intersection of
{3358, 3427, 3476} x {adexchange 1, 2, 3} — e.g. advertiser 3358 on adexchange 1 alone is 8.18%
anomalous (74,702 / 912,962), the single worst cell.

### 1.2 The clicks/conversions zero — verified, and it is a false negative, not a true absence

The raw `clicks` table's own `payprice > bidprice` check genuinely returns 0 (verified directly:
`SELECT count(*) FROM clk WHERE payprice > bidprice` = 0), and `clk.payprice = 0` for **78.6%**
of all 12,694 click rows (9,978/12,694) — a `clk`-native payprice of 0 can essentially never
exceed `bidprice` (min 227), so the clicks-table check is close to structurally guaranteed to read
zero regardless of the true auction outcome. This rules out explanation (a) — "anomalies are
concentrated in advertisers/exchanges that log few or no clicks" — clicks *do* exist for all three
anomalous advertisers (3358: 1,666 clicks; 3427: 2,179; 3476: 1,092). It confirms explanation (b):
**the `clk` table's own `payprice`/`bidprice` columns are not reliable evidence about the
underlying auction for that `bidid`.** Joining `clk` to `imp` on `bidid` (the same key `bidid`
identifies one underlying bid-request/auction across all four tables) and comparing:

- `bidprice` **always** agrees between the two tables for the same `bidid` (0 mismatches across
  12,724 joined rows).
- `payprice` disagrees in **9,974 of 12,724** joined rows (78.4%), and every sampled disagreement
  has `clk.payprice = 0` while `imp.payprice` holds the real settlement price. This is a data
  artifact in how the click log's payprice field is populated (frequently zero-filled), not a
  second, independent price observation.

Using the reliable source (`imp.payprice`/`imp.bidprice` for the matching `bidid`, which is what
`dataset.py`'s label join and any auction-simulation calibration would actually key off), **282 of
the 191,091 anomalous impression rows correspond to a clicked bidid**, and **39 correspond to a
converted bidid**. As a share of the positives that exist at all in the impressions table: 282 /
11,557 clicked rows = **2.44%**, and 39 / 935 converted rows = **4.17%**. So the anomaly is not
click-free — the clicks-table-only check simply cannot see it, and roughly 1 in 40 clicks (and
1 in 24 conversions) sits on a row this project has flagged as price-anomalous.

### 1.3 Magnitude: small and bounded, not a units or field-swap bug

On the 191,091 anomalous rows, `payprice - bidprice`: min +1, median +13, mean +16.41, max +40
(RMB fen). As a ratio, `payprice / bidprice` ranges from 1.004x to 1.176x (median 1.056x). A
1-unit-only overage would read as rounding noise; a >=10x overage would read as a field swap or
unit-mismatch bug. Neither is the case — this is a modest, tightly bounded overshoot (never more
than 17.6% above the logged bid), and the full overage histogram (Panel C of the figure) is
distinctly **spiky rather than smooth** (sharp peaks at +9 fen [43,407 rows], +6 [11,800], +21
[10,382], +24 [10,165], +33 [9,707]), consistent with a small number of discrete underlying causes
rather than continuous measurement noise.

### 1.4 Structure: overage rate is driven by which of several near-fixed `bidprice` "tiers" is logged

Each of the three anomalous advertisers logs **multiple, coexisting bidprice values on every single
day** (not a single constant, unlike most of the other 6 advertisers): 3358 uses {227, 238, 241},
3427 uses {227, 238, 241}, 3476 uses {238, 249, 254} — all three values appear on all days, ruling
out a simple "the collection bid changed over time" explanation. The anomaly rate is strongly
concentrated in each advertiser's **lowest** tier: at `bidprice=227`, 3358 is anomalous 8.18% of
the time and 3427 is anomalous 8.29% of the time — roughly 5-40x the rate at that same advertiser's
higher tiers (3358 at `bidprice=241`: 0.09%; 3427 at `bidprice=241`: 0.76%). This is the clearest
structural signature in the data: whichever of the 2-3 near-fixed values happens to be logged as
`bidprice` for a given request does not reliably bound the settlement price for that request, and
the lower the logged tier, the more often it is breached. This is consistent with the project's own
`bidprice` documentation (`schema.py` / ingest README): `bidprice` here is "a fixed data-collection
strategy... not a live bidding decision," i.e. closer to a periodically-set collection knob than a
per-request ceiling strictly enforced by the exchange. Slot dimensions among the anomalous rows are
dominated by standard sizes (300x250: 65% of anomalous rows; 950x90: 14%; 728x90: 7%; 336x280: 6%)
— unremarkable, not evidence of a separate slot-level cause. **This mechanism (which of several
near-fixed bidprice tiers gets logged, vs. what the exchange actually cleared against) is a
plausible explanation consistent with all observed structure, but it was not independently
confirmed against an iPinYou source describing the collection pipeline's internals — flagged as
inference, not certainty.**

### 1.5 Recommendation: exclude for auction-simulation calibration, keep for CTR modelling

These two downstream uses have different requirements, and the right decision differs between them:

- **Auction simulation (bid-price / market-price calibration): exclude the 191,091 rows.** A
  second-price simulator calibrated on `payprice` needs `payprice <= bidprice` to hold as an
  invariant of the auction model it is reproducing; rows that violate it cannot be fit without
  either corrupting the calibration or requiring the simulator to special-case "sometimes you pay
  more than you bid," which is not the auction being modelled. **Cost of exclusion:** 191,091 rows
  / 15,395,258 impressions = **1.241%** of all impressions, but this cost is not evenly spread — it
  removes **4.85% of advertiser 3358's impressions, 3.14% of advertiser 3427's, and 1.29% of
  advertiser 3476's**, while costing the other six advertisers nothing. A simulator trained
  per-advertiser (or validated per-advertiser) should be aware that these three campaigns lose a
  non-trivial slice of their price data. In positives-lost terms specifically for calibration
  purposes: this exclusion is about *price* data, not click labels, so it does not directly cost
  CTR-model positives — but if the simulator's evaluation harness also filters impressions (e.g. to
  jointly assess "would this bid have won, and would it have been clicked"), the same 282
  click-bearing and 39 conversion-bearing rows would be dropped from that joint evaluation.
- **CTR modelling: do not exclude.** `payprice` and `bidprice` are both already structurally
  excluded from the CTR feature set (`schema.NON_FEATURE_BID_COLUMNS`, `feature_columns()`), and
  the click label is constructed via a key-only join on `bidid` against the `clicks` table
  (`dataset.py`) that never touches `payprice`/`bidprice` at all. The anomaly is therefore
  invisible to, and does not corrupt, either the CTR features or the CTR label. Excluding these
  rows from CTR training would only shrink the training set by 191,091 rows (1.241%) and — more
  importantly, since positives are the scarce resource — **discard 282 of the dataset's 11,557
  click-positive rows (2.44%) and 39 of its 935 conversion-positive rows (4.17%)** for no
  corresponding benefit. Given how rare positives already are (§2 below), that trade is not worth
  it: keep these rows for CTR modelling.

No pipeline change is made here — this is a recorded recommendation with evidence, per project
convention; `validate.py`'s flag remains a flag, not a filter.

---

## 2. Class imbalance

**Figure:** `reports/figures/part2_class_imbalance.png` / `.svg`

Using the exact label-construction logic in `dataset.py` (impressions LEFT JOIN'd against
`SELECT DISTINCT bidid FROM clicks`, label = 1 iff a match exists): out of **15,395,258**
impression rows, **11,557** carry `click = 1` — a positive rate of **0.0751%**, i.e. roughly
1 click per 1,332 impressions. (For reference, the raw click-table row count is 12,694 — larger
than 11,557 because some `bidid`s appear more than once in the `clicks` table, up to 17 times for
one `bidid`; the label-join logic correctly collapses these to a single positive per matched
impression row.) Conversions are rarer still: 935 positive rows out of 15,395,258, **0.0061%**, or
roughly 1 conversion per 16,465 impressions.

**Why accuracy is a useless metric here:** a trivial classifier that always predicts "no click"
achieves 99.9249% accuracy while producing zero ranking information and being useless for bid
shading or budget allocation — accuracy cannot distinguish this trivial model from a genuinely
predictive one, because at this imbalance the negative class dominates the score almost entirely
regardless of how well positives are ranked. **What is used instead:**

- **AUC-ROC** measures ranking quality (probability a random positive is scored above a random
  negative) independent of the classification threshold and independent of the class prior, so it
  is not degenerate under 1,332:1 imbalance the way accuracy is.
  - **PR-AUC (precision-recall AUC)** is reported alongside AUC-ROC specifically because ROC can
  look deceptively good under extreme imbalance (the false-positive rate denominator is dominated
  by the huge negative class); PR-AUC is sensitive to exactly the quantity that matters
  operationally here — of the impressions the model ranks highest, how many are actually clicks.
- **Log loss** (cross-entropy) is used because RTB bidding needs a *calibrated probability*, not
  just a ranking — the bid price itself is typically a function of predicted CTR x value, so a
  well-ranked but poorly-calibrated model (e.g. one that outputs 0.5 for everything it ranks
  "high") would systematically mis-price bids even with good AUC.

This also motivates negative downsampling as a training-time technique (documented as a planned,
not-yet-implemented step in `backend/src/ingest/README.md`) — but any downsampling ratio must be
recorded in `backend/data/metadata/` so predicted probabilities can be recalibrated back to the
true 0.0751% prior, and must never be applied to whatever is designated the test split.

---

## 3. Feature cardinality

**Figure:** `reports/figures/part2_feature_cardinality.png` / `.svg`

Distinct-value counts computed directly on the `impressions` table (all rows; see the
methodological caveat above), restricted to `schema.feature_columns()`'s allowlist (i.e.
`bidid`, `bidprice`, `logtype`, `payprice`, `keypage` correctly excluded as non-features or
structurally unobservable at bid time).

**Fixed-vocabulary fields (`schema.LOW_CARDINALITY_CATEGORY_FIELDS`), kept as plain strings, no
hashing applied — confirmed appropriately small:** `slotformat` 4, `adexchange` 4,
`slotvisibility` 11, `advertiser` 9, `region` 35, `city` 370. These are cheap to one-hot or
integer-encode directly; hashing them would only destroy information for no computational benefit,
consistent with the ingest-time design choice.

**High-cardinality fields (`schema.HIGH_CARDINALITY_HASH_FIELDS`), pre-hash (raw string) vs.
post-hash (distinct values of the `<field>_hash` companion column, `HASH_BUCKETS = 2^20 =
1,048,576`):**

| field | pre-hash distinct | post-hash distinct buckets used | collision rate |
|---|---|---|---|
| creative | 131 | 131 | 0% |
| domain | 51,319 | 50,106 | 2.4% |
| slotid | 180,694 | 165,913 | 8.2% |
| useragent | 593,496 | 453,346 | 23.6% |
| IP | 704,964 | 513,304 | 27.2% |
| url | 3,271,943 | 1,002,023 | **69.4%** |
| ipinyouid | 12,966,032 | 1,048,570 | **91.9%** |

The post-hash bucket counts match the birthday-paradox expectation for a random hash with no
implementation defect (expected distinct buckets filled = `m*(1-exp(-n/m))` for `n` raw values into
`m = 1,048,576` buckets; every field above matches this formula to within ~0.3%), so this is not a
hashing-implementation bug — it is the mathematically expected consequence of `HASH_BUCKETS`
being fixed at 2^20 regardless of a field's true cardinality. **`url` and `ipinyouid` are the two
fields where this matters:** `url` has 3.27M raw distinct values against a 1.05M-bucket space, so
69.4% of distinct URLs collide with at least one other URL in the same bucket; `ipinyouid` has
12.97M raw distinct values (near-1:1 with the 15.4M rows — most users are seen once), so its hash
column saturates the *entire* bucket space (1,048,570 of 1,048,576 possible buckets occupied) and
carries almost no residual per-user identity signal — on average, ~12.4 distinct real users share
every bucket. **Implication for modelling:** `useragent_hash`/`IP_hash`/`slotid_hash`/`domain_hash`
are reasonable, low-collision categorical features as-is; `url_hash` should be treated as a lossy,
coarse feature (not a URL identifier); `ipinyouid_hash` in particular should not be relied on as a
per-user identity feature at this bucket count — a user-level model (e.g. frequency capping,
sequential/session features) would need either a larger hash space, a different encoding (e.g. the
raw `ipinyouid` string with a proper open-vocabulary embedding table), or acceptance that
`ipinyouid_hash` at 2^20 buckets is closer to noise than identity for this dataset's user volume.
This is exactly the kind of finding the hashing design in `schema.py` invites: `HASH_BUCKETS = 2^20`
was documented as a fixed, non-frequency-based convenience choice at ingestion, and this analysis
shows it is well-sized for `domain`/`slotid`/`creative`, marginal for `useragent`/`IP`, and
undersized for `url` and (severely) `ipinyouid`.

Other fields: `slotwidth` 21 distinct, `slotheight` 14 distinct, 29 distinct `(slotwidth,
slotheight)` combinations (standard IAB-style banner sizes), `slotprice` 290 distinct values,
`usertag` (list-valued) 68 distinct tag ids with an average of 5.67 tags per non-null row (20.55%
of rows have no usertag at all — see missingness, §5), and `urlid` is **always** null (0 distinct
non-null values across 15,395,258 rows) — confirmed consistent with the ingest README's own note
that `urlid` was observed literally `"null"` on every row checked; not a new finding, cross-checked
against existing documentation.

---

## 4. Temporal patterns

**Figure:** `reports/figures/part2_temporal_patterns.png` / `.svg`

Traffic volume by hour-of-day (Beijing local time, both seasons pooled) is highly uneven: a trough
of 117,733 impressions/hour at 04:00 versus an organic peak of 981,670 impressions/hour at 22:00
— an **8.3x swing**. (Hour 0 nominally shows the single highest count, 1,152,929, but this is a
verified logging artifact, not organic traffic — see the callout below — and is excluded from the
"organic peak" figure.) Click-through rate also varies by hour, roughly 2x: from a low of 0.0492%
at 09:00 to a high of 0.0990% at 18:00, broadly higher in the evening (17:00-23:00) than overnight
(01:00-09:00). **This is the empirical justification for budget pacing:** a bidder spending its
daily budget at a uniform rate would, on this traffic pattern, exhaust its budget well before the
higher-volume, higher-CTR evening hours if pacing is not hour-aware — naive uniform spending burns
budget disproportionately during the 02:00-09:00 trough where both volume and CTR are at their
lowest, leaving nothing for the 17:00-23:00 window where both are highest.

Day-of-week volume and CTR were also computed (pooled across the two seasons' 7 + 9 = 16 total
distinct dates, which cover only **two partial weeks** — season 2 is exactly one full week
[2013-06-06 Thursday through 2013-06-12 Wednesday], season 3 is nine consecutive days spanning
parts of two weeks [2013-10-19 Saturday through 2013-10-27 Sunday]). Monday shows the highest pooled volume
(2,769,110 impressions) but this is because one of the only two Mondays in the data
(2013-06-10) happened to be the single busiest day of season 2 (1,920,370 impressions on its own)
— **with only 1-2 observations per day-of-week, this breakdown should be read as descriptive, not
as a robust weekly seasonality estimate**, and should be re-examined once more calendar coverage
is available (or treated cautiously given the dataset's inherent 16-day limit).

**Verified anomaly — hour-0 logging-burst artifact:** the raw hour-of-day aggregation initially
showed hour 0 (00:00-00:59) as the single busiest hour, 1,152,929 impressions, nearly 3x hour 1's
379,883. Per the project's "verify anything surprising before writing it up" rule, this was
checked at minute granularity: **minute 00:01 alone accounts for 409,689 of the 1,152,929 hour-0
impressions (35.5%)** — a number wildly out of proportion to any other single minute in the hour
(next highest: minute 2 with 153,561; typical minutes elsewhere run 3,000-25,000). This pattern
recurs across dates within a season and is not attributable to a single day. This is consistent
with a batch-logging or day-rollover artifact in the source iPinYou logging infrastructure (e.g. a
buffered writer flushing at midnight) rather than genuine bid traffic, and is **not** an artifact
of this project's own ingestion pipeline (ingestion performs no timestamp transformation; row
counts were verified to match the raw files exactly, see `ingestion_row_counts.json`). It is
flagged rather than corrected, per project convention (no silent filtering); the figure marks hour
0 distinctly and the "organic peak" figure used above (981,670 at 22:00) already excludes it. A
pacing model trained naively on raw hourly counts should be aware this artifact would otherwise
bias it toward over-provisioning budget at the start of each day.

---

## 5. Missingness and anomalies

**Figure:** `reports/figures/part2_missingness.png` / `.svg`

Among genuine bid-time feature fields (impressions table, `schema.feature_columns()` allowlist),
non-zero null rates are: `usertag` 20.55%, `domain` 5.59%, `url` 3.54%, `adexchange` 2.03%,
`useragent` 0.01%. All other allowlisted fields (`region`, `city`, `slotid`, `slotwidth`,
`slotheight`, `slotvisibility`, `slotformat`, `slotprice`, `creative`, `advertiser`) are 0.00%
null. `urlid` is 100% null (a documented, expected degenerate constant — see §3 above — not a
missingness concern, since it carries no information whether present or absent). `region`/`city`
code `0` ("unknown") is a legitimate category, not a null marker, per the ingest pipeline's own
documented, verified design (`region.en.txt`/`city.en.txt`); it is correctly *not* nulled out and
is *not* counted above.

**Duplicate rows — a genuine, verified data-quality note, not a pipeline bug:** the impressions
table has 15,395,258 total rows but only 15,338,148 distinct `bidid` values — **57,110 excess
rows (0.37%) sit on a `bidid` that appears more than once.** The single most-repeated `bidid`
appears 26 times. Inspecting one such group directly: all 26 rows share identical `payprice`,
`bidprice`, `advertiser`, `slotid`, and `creative`, but have **distinct timestamps**, spaced
seconds to minutes apart (e.g. 18:14:20.504 through 18:23:34.451 on 2013-10-25) — i.e. the same
underlying auction/bid identifier was logged as a separate impression event multiple times, not
that the ingestion pipeline introduced exact duplicate rows. This looks like re-delivery/retry
logging on iPinYou's side rather than a random data-corruption artifact (the consistent price/slot/
creative across repeats rules out "different auctions coincidentally sharing a bidid"). It is not
large enough (0.37%) to materially affect the aggregate findings in this document, but a modelling
stage that treats `bidid` as a natural row key (e.g. any per-`bidid` deduplication logic, including
the label join in `dataset.py`) should be aware that ~57K impression rows are not 1:1 with unique
auctions. No exact full-row duplicate check across all ~20 columns was completed for this document
— an initial attempt via `GROUP BY ALL` over the full wide row (including free-text `useragent`/
`url` columns) did not complete in a reasonable time on the available hardware and was abandoned in
favour of the `bidid`-based check above, which is the more actionable signal for downstream
`bidid`-keyed joins in any case.

---

## 6. Feature-target relationships (CTR by categorical field)

**Figure:** `reports/figures/part2_ctr_by_feature.png` / `.svg`

**By advertiser:** CTR ranges from 0.030% (advertiser 2261) to 0.444% (advertiser 2997) — roughly
a 15x spread; excluding 2997 (see verified outlier note below), the range across the remaining 8
advertisers is 0.030%-0.080%, still a ~2.7x spread. **By ad exchange:** 0.052% (exchange 2) to
0.081% (exchange 1) among the three main exchanges (excluding the `null`-adexchange traffic, which
is entirely advertiser 2997's, see below). **By slot visibility:** a clear ordering with
`FirstView` at 0.192% far above every other position (`SecondView` 0.054%, `ThirdView` 0.033%,
`FourthView` 0.018%, `FifthView` 0.000% on 6,462 impressions), and season-2's numeric code `1`
(0.124%) similarly elevated relative to codes `0`/`2`/`255` (0.049%-0.073%) — consistent with `1`
being season 2's "FirstView"-equivalent code. All three fields carry real, usable CTR signal for a
model, well above what would be expected from noise alone given the sample sizes involved (the
smallest visibility bucket, `FourthView`, still has 112,873 impressions).

**Verified outlier — advertiser 2997's 0.444% CTR (5.6x the next-highest advertiser):** per the
project's "verify anything surprising" rule, this was checked directly rather than reported as-is.
Advertiser 2997 is a small, structurally distinct campaign: it ran for only 4 days (2013-10-23
through 2013-10-26, season 3 only), covers just 312,437 impressions (the smallest of the 9
advertisers, vs. 687,617-3,083,056 for the others), and **100% of its traffic has `adexchange =
NULL`** — it is the entire explanation for the `null` category in the ad-exchange breakdown, and
does not appear on any of the three standard exchanges (1, 2, 3) at all. Its 1,386 clicks come from
1,378 distinct users (verified: not a duplicate-logging or click-spam artifact — the join contains
no anomalous many-to-one inflation), so the elevated rate is a genuine property of this campaign's
traffic, not a computation bug. The most likely explanation is that this is a direct/managed
placement (not standard RTB exchange inventory) with a fundamentally different, higher-intent
audience — but this was not independently confirmed against an external source, so it is reported
as a verified-but-unexplained structural outlier: **any advertiser-level CTR comparison or
advertiser-conditioned model evaluation should treat advertiser 2997 as a distinct traffic
population, not pool it with the other 8 exchange-sourced advertisers without accounting for the
confound.**

---

## 7. Win price distribution and the censoring problem

**Figure:** `reports/figures/part2_payprice_censoring.png` / `.svg`

**The censoring problem, stated explicitly:** `payprice` is recorded only in the `impressions`
table, i.e. only for auctions this bidder **won**. The `bids` table (64,746,749 rows) records every
bid request regardless of outcome; the `impressions` table (15,395,258 rows) is the subset that
won. The empirical win rate is **15,395,258 / 64,746,749 = 23.78%** — meaning **any distribution
fit on `payprice` describes only the cheapest ~24% of the demand curve this bidder faced**: the
76.22% of auctions this bidder lost are disproportionately the ones where the true market price
exceeded what this bidder was willing/able to pay, so naively fitting a market-price model on
`payprice` alone will systematically **understate** true market price, and there is no direct way
to validate that model against the losing 76.22% (their true clearing price is never observed in
this dataset at all — only that this bidder's offer fell short of it). This is a hard constraint on
simulator design: a bid-landscape / win-rate model needs to be fit jointly using both the full
`bids` population (for "how often would a given bid price have won") and the `impressions`-only
`payprice` distribution (for "what would it have cost"), not `payprice` in isolation, and any
reported win-rate curve derived purely from `impressions` will be circular (a model trained only on
`payprice` <= observed `bidprice` cannot recover the shape of the price distribution above the
bidder's own historical bids).

**Distribution shape (won auctions only, `payprice` in RMB fen, 0-300 range observed):** mode = 70
fen (887,425 impressions at exactly that price — the single most common settlement price by a wide
margin), median 69, mean 80.28 (matches `validation_report.json`'s cached figure, cross-checked and
confirmed consistent), p95 = 219. Only 3,258 impressions (0.0212%) have `payprice = 0` (essentially
free/remnant wins), so the "floor" of the distribution is not degenerate. By advertiser, median
`payprice` ranges from 41 fen (advertiser 2997, the small non-exchange campaign from §6) to 77 fen
(advertiser 3358, notably one of the three advertisers implicated in the §1 anomaly) — advertiser-
level heterogeneity in win price is real and any simulator calibrated globally (pooling all 9
advertisers) would misrepresent any single campaign's actual price environment.

---

## Summary of recommendations and open items for the modelling stage

1. **Auction simulator:** exclude the 191,091 `payprice > bidprice` rows from price-calibration
   data (cost: 1.241% of impressions, concentrated in 3 of 9 advertisers up to 4.85% of one
   campaign's rows); do not exclude them from CTR training (cost there would be 191,091 rows and,
   more importantly, 282 click positives + 39 conversion positives for no benefit).
2. **Auction simulator:** must account for censoring — `payprice` is a winner-only sample (23.78%
   win rate); fit against `bids` + `impressions` jointly, not `impressions.payprice` alone.
3. **CTR model:** expect and design for extreme imbalance (0.0751% click rate); use AUC-ROC +
   PR-AUC + log loss, not accuracy; if downsampling negatives, record the exact ratio in
   `backend/data/metadata/` and never apply it to the test split.
4. **Feature encoding:** the current `HASH_BUCKETS = 2^20` hashing is well-sized for `domain` /
   `slotid` / `creative`, marginal for `useragent` / `IP`, and undersized for `url` (69.4%
   collision) and severely undersized for `ipinyouid` (91.9% collision, bucket space fully
   saturated) — a user-level feature relying on `ipinyouid_hash` as an identity proxy should not be
   trusted at this bucket count.
5. **Budget pacing:** traffic and CTR both vary substantially by hour (8.3x volume swing, ~2x CTR
   swing across 24 hours) — naive uniform-rate spending is empirically unjustified; note the
   hour-0 logging artifact (independently verified, not organic) when building any hour-keyed
   historical-rate model.
6. **Advertiser 2997** should be treated as a structurally distinct population (non-exchange
   traffic, small 4-day campaign, 5.6x higher CTR) in any cross-advertiser analysis or evaluation.
7. **Re-run once the temporal train/val/test split lands:** §2 (class imbalance), §4 (temporal
   patterns), and §6 (feature-target CTR) all currently pool data that will eventually include a
   held-out test partition; §1 and §3 are structural/schema findings that should not change.

---

*Future sections should be appended below this line, following the same structure (figure path(s)
first, then numbered findings with actual figures cited in prose, not just qualitative claims).*
