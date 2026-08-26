# CTR Model Results: Evaluation Protocol and Logistic-Regression Baseline

Source data: `backend/data/processed/{impressions,clicks}/season=2/date=YYYY-MM-DD/part-0.parquet`
(iPinYou season 2, ingested per `backend/src/ingest/README.md`). Features and label are produced
exclusively through `backend/src/features/dataset.py::load_impression_features()` (the sanctioned
loader -- allowlist derived structurally from `schema.BID_COLUMNS`, label via a key-only join
against `clicks` that projects only `bidid`). Splitting, downsampling, evaluation, and the
logistic-regression baseline itself live under `backend/src/modeling/`.

**Author / verification date:** amineelahmad@gmail.com, 2026-08-24.

---

## Evaluation protocol

**1. Split temporally, never randomly.** Season 2 has 7 ingested dates (2013-06-06..2013-06-12).
Per `backend/src/ingest/README.md`'s specified shape (train days 1-5 / validate day 6 / test day
7), the boundaries are named constants in `backend/src/modeling/split.py`
(`SEASON_2_TRAIN_START/END`, `SEASON_2_VAL_DATE`, `SEASON_2_TEST_DATE`), not literals scattered
through code:

| split | dates | 
|---|---|
| train | 2013-06-06 .. 2013-06-10 (5 days) |
| val | 2013-06-11 |
| test | 2013-06-12 |

Val and test are strictly after every train date (`test_val_and_test_are_strictly_after_train` in
`backend/tests/test_split_boundaries.py`), and the three splits cover the seven ingested dates
exactly once each, no gap and no overlap (also tested). **Season 3** (2013-10-19..2013-10-27, 9
dates, different campaigns, four months later) is reserved as a separate out-of-time evaluation
block and is **not** pooled with season 2 here -- `split.get_split_boundaries(3)` raises
`NotImplementedError` rather than guessing a shape for it; season-3 evaluation is future work.

**2. Metrics are AUC and log loss, never accuracy.** The positive (click) rate is ~0.07-0.09% (see
counts below). A trivial "always predict no click" classifier scores >99.9% accuracy while
carrying zero ranking information and being useless for bid shading or budget allocation --
accuracy cannot distinguish that trivial model from a genuinely useful one at this imbalance. AUC
measures ranking/discrimination; log loss measures the quality of the predicted probabilities
themselves, which matters directly here because predictions feed a bidding function.

**3. Downsampling ratio recorded, recalibration applied before any probability is used as a
probability.** Negatives in the TRAIN split only are downsampled at a fixed, persisted rate (see
below); validation and test always keep the true, undownsampled class distribution. The evaluation
harness (`backend/src/modeling/evaluate.py::evaluate()`) applies the recalibration correction
(`backend/src/modeling/downsample.py::recalibrate_probability()`) to raw model output **before**
computing AUC, log loss, or calibration error -- never on raw downsampled-scale probabilities.

**4. Calibration is reported alongside AUC and log loss.** Expected Calibration Error (ECE) with
**quantile (equal-count) bins**, not fixed equal-width bins on `[0, 1]`: at a ~0.08% true positive
rate, almost every predicted probability lands below 0.01, so equal-width bins would put >99% of
rows in a single bin and say nothing about calibration in the range that matters. `n_bins=10`
throughout. A reliability curve (log-log, since every quantity is a small positive fraction) is
plotted for both val and test -- see `reports/figures/part3_lr_calibration_{val,test}.{png,svg}`.

**5. Baselines before complexity.** This report covers the logistic-regression hashed-feature
baseline only. No gradient-boosting model has been trained or evaluated in this phase -- LightGBM
is explicitly out of scope here and will be scored by the exact same harness in a later phase, so
its result has something concrete to beat.

**6. Test touched once.** All hyperparameter selection (the `C` grid below) happens against
validation. `backend/src/modeling/train_lr.py --touch-test` performs exactly one test evaluation
per invocation and refuses to overwrite an existing test-evaluation record
(`backend/data/metadata/ctr_lr_test_evaluation_season2.json`) unless `--overwrite` is passed
explicitly. Test was evaluated once for the results below.

**7. Fixed seed everywhere.** `RANDOM_SEED = 42` (`backend/src/modeling/train_lr.py`) seeds the
negative-downsampling RNG and `LogisticRegression(random_state=...)`. Row order out of the feature
loader is also made deterministic (see "A reproducibility bug found and fixed," below) so that a
fixed seed actually reproduces the same sampled rows, not just the same sample *count*, across
runs.

---

## Verified per-split counts and positive rates

Cross-checked two independent ways: once via `modeling.split.load_split()` (the loader path
training/evaluation actually use) and once via a standalone SQL query
(`modeling.split.verify_split_counts()`) that re-implements the same join independently. Both
agree exactly.

| split | dates | impressions | clicks | positive rate |
|---|---|---:|---:|---:|
| train | 06-06..06-10 | 8,834,027 | 6,080 | 0.0688% |
| val | 06-11 | 1,745,722 | 1,392 | 0.0797% |
| test | 06-12 | 1,657,338 | 1,369 | 0.0826% |
| **season-2 total** | 06-06..06-12 | 12,237,087 | 8,841 | 0.0723% |

Impression counts match the task brief's per-date figures exactly (and
`ingestion_row_counts.json`); click counts here are the **deduplicated, per-impression** join count
(`load_impression_features()`'s label logic: `LEFT JOIN (SELECT DISTINCT bidid FROM clicks)`), which
is smaller than the raw `clicks`-table row count given in the brief (e.g. 1,289 raw rows on
06-06 vs. 1,162 here) because a small number of `bidid`s appear more than once in the raw clicks
table (see `docs/analysis/eda-findings.md` §2) -- the label-join logic correctly collapses these to
one positive per matched impression row, which is the number that matters for modelling.

**Train positive rate vs. the pooled EDA figure.** The train split's positive rate (0.0688%) is
materially lower than both the season-2-pooled rate (0.0723%, computed above) and the
full-population (season 2 + 3) pooled figure in `docs/analysis/eda-findings.md` §2 (0.0751%) --
about 8-9% relative lower than the full-population figure. This is expected and exactly the
caveat `eda-findings.md` flags in its own methodological note ("no split existed yet... positive
rate may shift if the held-out day(s) are atypical"): 06-10 and 06-11 (partly in val) and 06-12
(test) have visibly higher per-day click counts (1,546 / 1,392 / 1,369) than the earlier training
days (1,162 / 1,059 / 1,167 / 1,146), so excluding val/test from the rate used for training
mechanically pulls the observed train rate down. This directly informed **not** blindly reusing
the pooled EDA rate anywhere downstream (e.g. as a sanity prior) -- every rate used in modelling
is recomputed per-split.

---

## Negative downsampling

Applied to the **train split only** (never val/test). Persisted to
`backend/data/metadata/ctr_train_downsampling_season2.json`:

| field | value |
|---|---|
| `negative_sampling_rate` (fraction of true negatives kept) | **0.02** |
| `seed` | 42 |
| `n_positive` (kept in full) | 6,080 |
| `n_negative_true` | 8,827,947 |
| `n_negative_sampled` | 176,559 |
| `positive_rate_before` | 0.0688% |
| `positive_rate_after` | 3.329% |

**Recalibration.** Given a rate `w` (fraction of negatives kept) and a raw model probability `p`
fit on the downsampled distribution, the true-population probability is recovered via

```
q = p / (p + (1 - p) / w)
```

(derivation and citation to He et al. 2014 "Practical Lessons from Predicting Clicks on Ads" in
`backend/src/modeling/downsample.py`'s module docstring). Implemented as
`recalibrate_probability()`, unit-tested against a hand-derived example
(`backend/tests/test_downsample.py::test_matches_hand_derived_example`) and for the `w=1`
(no-op), monotonicity, and scalar/Series-preservation properties. The evaluation harness applies
this correction to every raw prediction before computing any metric.

---

## Logistic-regression baseline: features

Full rationale in `backend/src/modeling/features_lr.py`'s module docstring; summary:

- **Hashing trick** (`sklearn.feature_extraction.FeatureHasher`, `input_type="string"`,
  `alternate_sign=True`) over `"field=value"` tokens, one shared **`HASH_DIM = 2**18` (262,144)
  bucket space** across all categorical/multi-valued fields -- a deliberately smaller,
  differently-scoped space than the ingestion-time per-field `2**20` convention (see
  `docs/analysis/eda-findings.md` §3 for that convention's own collision-rate table).
- **Categorical fields (13):** `region, city, adexchange, slotvisibility, slotformat, advertiser`
  (low cardinality) and `domain, url, slotid, creative, useragent, IP, ipinyouid` (high
  cardinality) -- the raw string columns, not the ingestion-time `<field>_hash` companions.
- **`usertag` (multi-valued):** multi-hot over **all** tags seen in train (43 distinct tag ids,
  verified directly against the train split) -- no frequency threshold, since the vocabulary is
  already small; a null/empty usertag list contributes no tokens (absence, not a positive "no
  tag" signal).
- **`hour` (derived from `timestamp`):** an unordered `hour=H` token (0-23), not a numeric/ordinal
  feature. **Hour 0 is a known batch-logging artifact, not a real traffic peak**
  (`docs/analysis/eda-findings.md` §4) -- kept as its own category (dropping it would lose real
  rows), but no feature here assumes it represents genuine peak organic traffic, and no
  day-of-week/calendar-date feature was built at all (train spans only 5 calendar days, each a
  distinct weekday, so a day-of-week feature would be indistinguishable from "which training
  date" and would not generalize to val/test's different calendar dates).
- **Numeric (3):** `slotwidth, slotheight, slotprice`, standardized (`StandardScaler` fit on train
  only, reused unfit-again for val/test), concatenated as a small dense block alongside the hashed
  sparse block.
- **Excluded:** `urlid` (100% null, zero information), raw `timestamp` (superseded by `hour`), and
  the six ingestion-time `<field>_hash` companion columns (superseded by this module's own hashing
  pass over the corresponding raw string).

---

## Leakage investigation

The full-data validation AUC (0.930) is high enough to warrant investigation before reporting, per
protocol ("an unexpectedly high AUC is a bug until proven otherwise"). Two checks:

**1. Feature ablation** (2M-row train sample / 800K-row val sample, same pipeline, `C=1.0`):

| feature set | AUC |
|---|---:|
| full field set | 0.810 |
| drop `ipinyouid` | 0.812 |
| drop `ipinyouid`, `useragent`, `IP` (all identity-ish fields) | 0.809 |
| drop **all** high-cardinality fields (`domain, url, slotid, creative, useragent, IP, ipinyouid`) | 0.743 |

Removing user/device identity fields (`ipinyouid`, `useragent`, `IP`) changes AUC by less than
0.003 -- ruling out "the model is just memorizing which user/device clicked" as the driver.
Removing **all** high-cardinality fields (including placement/campaign identifiers `domain`,
`slotid`, `creative`, `url`) drops AUC by ~0.07. This isolates the signal to
**placement/campaign-level heterogeneity** (some ad slots, creatives, and domains have
structurally different baseline CTR), not user-identity leakage.

**2. Per-category CTR check** (train split, `creative`/`slotid`/`domain`, categories with >500
rows): the highest observed category-level CTR was ~3.7% (`slotid`), ~2.5% (`domain`), and ~1.4%
(`creative`) -- 20-50x the base rate, but nowhere near a deterministic 100%, ruling out a
degenerate/near-perfectly-separating column.

**Conclusion:** no leakage found. The AUC is legitimately driven by real placement/campaign
heterogeneity: the ablation above shows removing high-cardinality placement fields
(`domain`/`slotid`/`creative`/`url`/etc.) costs ~0.07 AUC while removing identity-ish fields
(`ipinyouid`/`useragent`/`IP`) costs <0.003, and the per-category CTR check found no
near-deterministic column (highest observed category CTR ~3.7%, far from 100%). This is this
repo's own evidence, not an external benchmark comparison -- no claim is made here about how this
number compares to other iPinYou-based results elsewhere.

**This does not mean the pooled AUC is the right number to report as "the" model quality,
though.** Pooled AUC additionally gets ranking credit for separating advertisers (who have
structurally different base click rates) from each other, not only for separating clickers from
non-clickers within one advertiser's traffic -- which is what a bidder actually experiences, since
a live bidder never chooses between two different advertisers' impressions. See "Per-advertiser
AUC (primary metric)" below: the honest within-advertiser figure (row-weighted mean 0.9132) sits
~0.0096 below the pooled 0.9228 reported in Results -- a modest correction in aggregate, but one
that conceals a wide spread across individual advertisers (0.9858 down to 0.7412).

---

## Results

Selection metric for `C` (L2 regularization strength; `sklearn.linear_model.LogisticRegression`,
`solver="lbfgs"`, `max_iter=1000`, `random_state=42`) is **validation log loss** -- the metric most
directly tied to the probabilities a bidding function would consume. Full grid
(`backend/data/metadata/ctr_lr_runs.jsonl`):

| C | val AUC | val log loss | val ECE |
|---:|---:|---:|---:|
| 0.01 | 0.9174 | 0.0049512 | 0.000609 |
| **0.1** | **0.9299** | **0.0045260** | 0.000346 |
| 1.0 | 0.9270 | 0.0045384 | 0.000142 |
| 10.0 | 0.9026 | 0.0054895 | 0.000360 |

`C=0.1` selected. Final numbers (`backend/data/metadata/ctr_lr_test_evaluation_season2.json`):

| model | split | n | positives | AUC | log loss | ECE (10 quantile bins) |
|---|---|---:|---:|---:|---:|---:|
| `lr_hashed_C=0.1` | season2_val | 1,745,722 | 1,392 | 0.9299 | 0.0045260 | 0.000346 |
| `lr_hashed_C=0.1` | season2_test (touched once) | 1,657,338 | 1,369 | 0.9228 | 0.0048598 | 0.000352 |

**Caveat -- the pooled AUC above overstates within-advertiser ranking ability.** `advertiser` is
itself one of the 13 hashed categorical fields the model sees directly, and the 5 advertisers
present in season-2 test have somewhat different base click rates (0.042%-0.092%, see
"Per-advertiser AUC (primary metric)" below). Pooled AUC gets partial credit simply for ranking a
higher-base-rate advertiser's rows above a lower-base-rate advertiser's rows -- a distinction a
live bidder never actually makes, since each advertiser bids independently on its own traffic. The
row-weighted mean per-advertiser test AUC is **0.9132** (simple/unweighted mean **0.9131**) vs.
**0.9228** pooled -- a gap of **0.0096**, which is the share of the pooled figure attributable to
cross-advertiser separability rather than genuine within-advertiser click-vs-no-click ranking.

That gap is real and in the expected direction, but it is deliberately *not* claimed here to be
large: it is a correction to the headline figure, not an explanation of why that figure is high.
Four of the five advertisers sit in a narrow 0.080%-0.092% base-rate band, so there is limited
cross-advertiser base-rate spread for pooled AUC to exploit in the first place, and a correction of
~0.01 is consistent with that. The more consequential thing the pooled number hides is
**dispersion, not inflation**: within-advertiser AUC ranges from 0.9858 (advertiser 1458) down to
**0.7412** (advertiser 3386, 392,901 rows -- 24% of test volume), and no single pooled figure
represents both. Full breakdown in "Per-advertiser AUC (primary metric)" below.

Val -> test AUC drops by 0.0071 and log loss rises slightly (0.0045260 -> 0.0048598) -- a small,
expected generalization gap for a model selected on val, not a red flag.

**Reliability (test, 10 quantile bins; full table in
`ctr_lr_test_evaluation_season2.json["reliability_table"]`, plotted in
`reports/figures/part3_lr_calibration_test.png`):**

| bin | n | mean predicted | mean actual | abs gap |
|---:|---:|---:|---:|---:|
| 1 | 165,734 | 0.0000290 | 0.0000420 | 0.0000130 |
| 2 | 165,734 | 0.0000800 | 0.0000600 | 0.0000200 |
| 3 | 165,735 | 0.0001330 | 0.0000660 | 0.0000660 |
| 4 | 165,732 | 0.0001910 | 0.0000480 | 0.0001420 |
| 5 | 165,734 | 0.0002610 | 0.0000840 | 0.0001770 |
| 6 | 165,734 | 0.0003490 | 0.0002110 | 0.0001380 |
| 7 | 165,733 | 0.0004620 | 0.0002230 | 0.0002380 |
| 8 | 165,734 | 0.0006260 | 0.0003500 | 0.0002760 |
| 9 | 165,734 | 0.0009140 | 0.0005970 | 0.0003160 |
| 10 | 165,734 | 0.0044450 | 0.0065770 | 0.0021310 |

Aggregate calibration is good (ECE ~0.00035, points hug the y=x diagonal in
`part3_lr_calibration_{val,test}.png`) after the downsampling correction is applied -- direct
evidence the recalibration formula is working, not just theoretically correct.

**Calibration caveat: a long tail of individually overconfident predictions.** Aggregate ECE
looks good, but inspecting the highest-confidence individual test predictions surfaces a real
risk for the downstream bidding function: the 5 highest recalibrated probabilities on test are
all ~0.9999-1.0 (essentially certain), but **4 of those 5 are actually non-clicks** -- the model
is assigning near-total confidence to a handful of specific, rarely-seen
`(domain, slotid, creative, advertiser, adexchange)` combinations that happened to have a
disproportionately high click rate in the small, downsampled training sample for that exact
hashed combination (a classic small-sample/rare-category overfitting artifact, made worse by
hash-space sharing across 14 fields at `2**18` buckets). Bin 10 in the reliability table above
(`prob_max` in the underlying data reaches ~0.9998) already hints at this: it is the widest,
least-representative bin.

Quantified over the full test set (recalibrated probabilities):

| threshold | n rows | true positives among them | implied precision |
|---|---:|---:|---:|
| p > 0.01 | 4,384 | 596 | 13.6% |
| p > 0.1 | 1,046 | 230 | 22.0% |
| p > 0.9 | 59 | (not separately counted) | -- |
| max | 1 row at p = 0.9998 | -- | -- |

The bulk of elevated-probability rows (thousands, at the 0.01-0.1 range) carry real, strongly
informative lift over the 0.08% base rate (13-22% precision is a 170-280x lift) -- this is the
"aggregate calibration is good" signal the ECE/reliability numbers above are measuring, and it
does not show up as a large ECE hit because so few rows are affected. But the most extreme tail
(59 rows above p=0.9, out of 1,657,338) behaves differently: the top-5 sample above shows 80% of
the *most* confident predictions are wrong, indicating that right at the extreme end the model is
overfitting to specific rare hashed combinations rather than generalizing. A bidding function
computing `bid = value * p_click` would drastically overbid on exactly these ~59 rare-combination
auctions. **Not fixed here** -- flagged as a concrete follow-up (stronger L2, a frequency-based
floor/shrinkage for rare hashed combinations, or explicit prediction clipping) before any
LR-baseline probability is wired into the actual bidding simulator.

---

## Per-advertiser AUC (primary metric) and relative calibration

**Source.** Both tables below are computed from `backend/data/predictions/ctr_lr_baseline_season2_test.parquet`
(1,657,338 rows, one per season-2-test impression, columns `y_true`, `p_raw`, `p_calibrated`,
`advertiser`, `timestamp`), produced by `backend/src/modeling/score_test.py`. That script loads the
already-fitted artifact (`ctr_lr_baseline_season2.joblib`) and the saved `StandardScaler` -- it
never refits anything -- runs one forward pass over season-2 test, and asserts its pooled AUC/log
loss reproduce `ctr_lr_test_evaluation_season2.json`'s recorded numbers to within `1e-9` before
writing anything to disk (they did, exactly: `auc=0.9228246833`, `log_loss=0.0048598082`, matching
the Results section above). This is analysis of an already-recorded, single-touch test evaluation,
not a second test-set decision -- no model selection or tuning happens here.

### Per-advertiser AUC (primary metric)

Pooled AUC is inflated by advertiser separability (see the Results caveat and the Leakage
investigation section above) -- per-advertiser AUC, computed by grouping the persisted test
predictions by `advertiser` and scoring `roc_auc_score(y_true, p_calibrated)` within each group
separately, is what a bidder actually experiences, since a live bidding decision is always made
*within* one advertiser's campaign, never across two different advertisers' traffic.

| advertiser | n | positives | positive rate | AUC |
|---|---:|---:|---:|---:|
| 1458 | 447,493 | 356 | 0.0796% | 0.9858 |
| 3386 | 392,901 | 355 | 0.0904% | 0.7412 |
| 3427 | 390,398 | 313 | 0.0802% | 0.9739 |
| 3358 | 335,310 | 307 | 0.0916% | 0.9458 |
| 3476 | 91,236 | 38 | 0.0417% | 0.9187 |

All 5 advertisers present in season-2 test have both positive and negative rows, so none produced
an undefined (n/a) AUC here. Had any advertiser subset had zero positives or zero negatives, AUC is
mathematically undefined for that subset and would be reported as `n/a` with the specific reason
(`"zero positives"` / `"zero negatives"`) -- never silently substituted with 0.5, and never
silently dropped from the table.

- **Row-weighted mean:** 0.9132 (`sum(auc_i * n_i) / sum(n_i)`)
- **Simple (unweighted) mean:** 0.9131 (`mean(auc_i)` across the 5 advertisers)
- **Pooled AUC (Results section, all advertisers scored together):** 0.9228
- **Gap:** pooled exceeds the per-advertiser mean by **~0.0096** -- a modest but real correction

**Which is the honest headline number.** The row-weighted mean (0.9132) is treated as the headline
here, since it reflects the actual mix of bidding volume a deployed system would see (advertiser
1458, the largest at 447K rows, and advertiser 3476, the smallest at 91K rows, should not count
equally toward "how well does this model rank clicks for the traffic it actually serves"). The
simple mean is reported alongside as a diagnostic, not the headline, but it is what surfaces
advertiser 3386's weak 0.7412 AUC without letting the two largest, highest-AUC advertisers
(1458, 3427) wash it out through sheer row count -- both means happen to be nearly identical here
(0.9131 vs. 0.9132) only because the four large advertisers are fairly close in size; that would
not hold in general. Either way, **both per-advertiser means sit ~0.01 AUC below the pooled
figure**, and advertiser 3386 in particular performs far worse in isolation (0.7412) than the
pooled number would suggest -- a bidder running campaigns for advertiser 3386 specifically would
see meaningfully worse ranking quality than the headline 0.9228 implies.

### Relative calibration: predicted/observed ratio by decile

The existing reliability table (Results section) reports **absolute** gaps between mean predicted
and mean actual click rate per quantile bin, which are all small in absolute terms (~1e-5 to
~2e-3) at this ~0.08% base rate and look reassuring on their own. The table below adds a
**ratio** column (`mean_predicted / mean_actual`) computed over the same 10 quantile bins (via
`modeling.evaluate.expected_calibration_error`, reused as-is -- no separate binner was written) on
the persisted, recalibrated test predictions, which exposes multiplicative miscalibration the
absolute-gap view hides:

| bin | n | mean predicted | mean actual | ratio (pred/obs) |
|---:|---:|---:|---:|---:|
| 1 | 165,734 | 0.0000294 | 0.0000422 | 0.70 |
| 2 | 165,734 | 0.0000804 | 0.0000603 | 1.33 |
| 3 | 165,735 | 0.0001327 | 0.0000664 | 2.00 |
| 4 | 165,732 | 0.0001905 | 0.0000483 | 3.95 |
| 5 | 165,734 | 0.0002612 | 0.0000845 | 3.09 |
| 6 | 165,734 | 0.0003487 | 0.0002112 | 1.65 |
| 7 | 165,733 | 0.0004616 | 0.0002233 | 2.07 |
| 8 | 165,734 | 0.0006258 | 0.0003500 | 1.79 |
| 9 | 165,734 | 0.0009138 | 0.0005973 | 1.53 |
| 10 | 165,734 | 0.0044455 | 0.0065768 | 0.68 |

No bin in this table has zero observed clicks, so no divide-by-zero guard was triggered in
practice; `score_test`'s downstream analysis treats a zero-observed-click bin as an undefined
(`None`) ratio rather than `inf` or a silently dropped row, matching the per-advertiser table's
n/a convention.

**Reading the ratio column.** Bins 3-5 (predicted probability roughly 0.013%-0.026%, covering
~30% of test rows) **over-predict by 2-4x** -- the model's predicted click probability in this
range is two to four times the click rate actually observed for those rows. A bidding function
computing `bid = value * p_click` would overbid by the same 2-4x factor for this whole slice of
traffic, not just for the small overconfident tail already flagged in the Results section's
"Calibration caveat" above -- this is a much larger, more consequential slice of rows than that
59-row extreme tail. Bins 6-9 also over-predict, more mildly (1.5-2x). The two extreme bins move in
the **opposite** direction: bin 1 (lowest decile) and bin 10 (highest decile) both
**under-predict** (ratio ~0.68-0.70) -- the model is somewhat too conservative at both ends of its
own predicted-probability range. No bin sits close enough to a ratio of 1.0 to be called
well-calibrated in this multiplicative sense; bin 2 (1.33) is the closest.

This refines, rather than contradicts, the Results section's "aggregate calibration is good"
conclusion: ECE (~0.00035) is a row-count-weighted average of *absolute* gaps, which stays small
here specifically because the base rate is tiny -- the same data, viewed as a ratio, shows the
model is off by a factor of 2-4x for a third of test rows. Both statements are true simultaneously;
a downstream bidding function should be evaluated (and, if needed, further corrected) against the
ratio view, not just ECE, before its output is trusted as a probability multiplier.

---

## A reproducibility bug found and fixed

While producing the numbers above, re-running the full pipeline twice with an identical seed
selected a **different** `C` on val (`1.0` vs. `0.1`, both near-tied on log loss). Root cause:
`features/dataset.py`'s query had no row-order guarantee, and DuckDB's parallelized Parquet scan
materialized the same set of rows in a different order across separate calls. Since
`modeling.downsample.downsample_negatives()` samples negatives via a seeded RNG indexed
positionally into the materialized DataFrame, an unstable row order meant a fixed seed no longer
picked the same *actual* rows across runs, even though it picked the same *count*. Fixed by
carrying `bidid`/`timestamp` through as hidden columns and sorting on them, once, in pandas after
materializing (a SQL-level `ORDER BY` was tried first and measured to be prohibitively slow -- 20+
minutes and still running on the full ~8.8M-row/20-column train table on this machine, vs. ~11-30s
for the pandas approach used instead). Regression-tested in
`backend/tests/test_dataset_row_order.py`. All numbers in this report come from the run made
*after* this fix.

---

## Files

- `backend/src/features/dataset.py` -- sanctioned feature loader, extended with `date_range=` (for
  split pushdown) and deterministic row ordering.
- `backend/src/modeling/split.py` -- named split boundary constants, `load_split()`,
  `verify_split_counts()`.
- `backend/src/modeling/downsample.py` -- `downsample_negatives()`, `recalibrate_probability()`.
- `backend/src/modeling/evaluate.py` -- shared harness (`evaluate()`, ECE, reliability plotting);
  reused as-is by any future model, including LightGBM.
- `backend/src/modeling/features_lr.py` -- hashed-feature encoding for the LR baseline.
- `backend/src/modeling/train_lr.py` -- training/selection/test-touch orchestration script.
- `backend/tests/test_split_boundaries.py`, `test_downsample.py`, `test_dataset_row_order.py`.
- `backend/data/metadata/ctr_train_downsampling_season2.json`,
  `ctr_lr_runs.jsonl`, `ctr_lr_test_evaluation_season2.json` -- generated, traceable run records.
- `backend/data/models/ctr_lr_baseline_season2.joblib` -- fitted model + scaler + encoder config.
- `reports/figures/part3_lr_calibration_val.{png,svg}`, `part3_lr_calibration_test.{png,svg}`.
- `backend/src/modeling/score_test.py` -- loads the fitted artifact (never retrains), reproduces
  the single-touch test evaluation exactly (asserted), and persists per-row predictions to
  `backend/data/predictions/ctr_lr_baseline_season2_test.parquet` (+ JSON sidecar) so the
  per-advertiser-AUC and decile-ratio analyses above never need to re-score.
- `backend/requirements.txt` -- added `scikit-learn`, `scipy`, `joblib`, `threadpoolctl`,
  `narwhals` (scikit-learn's dependency closure); file re-saved in its original UTF-16LE/CRLF
  encoding, not silently converted to UTF-8.

## Explicitly out of scope here

- LightGBM / any gradient-boosting model -- next phase.
- Season 3 out-of-time evaluation -- `split.get_split_boundaries(3)` deliberately raises; wiring
  it up is future work, not silently pooled with season 2.
- Fixing the long-tail overconfidence issue above (documented, not patched).

---

## Placement-overlap analysis: seen vs. unseen `(domain, slotid, creative)` triples

**Question.** How much of the LR baseline's test-set discrimination depends on having seen the
exact `(domain, slotid, creative)` placement triple during training? This is pure analysis over
already-persisted artifacts -- no model or scaler was refit or re-scored to produce it.

**Source and alignment.** `backend/data/predictions/ctr_lr_baseline_season2_test.parquet` (1,657,338
rows) does not carry `domain`/`slotid`/`creative`, so the raw season-2 test split was reloaded via
`split.load_split(2, "test")` (`row_limit=None`, the only case `load_impression_features()`
guarantees a deterministic row order for -- see `backend/tests/test_dataset_row_order.py`) and
aligned to the predictions parquet **positionally**. That alignment was verified empirically in the
same run, not merely assumed: `y_true`, `advertiser`, and `timestamp` were checked element-wise
across all 1,657,338 rows and matched exactly (`backend/src/modeling/seen_unseen_analysis.py::
_assert_test_split_aligns_with_predictions`). Had any row mismatched, the script would have stopped
rather than falling back to a fuzzy or key-based join; no such fallback exists.

**Null-handling policy.** Verified directly against season-2 train: `domain` has real (pandas `NaN`)
missing values (75,383 of 1,657,338 test rows, 4.5484%); `slotid` and `creative` have zero nulls and
zero literal `"null"`-string values anywhere in season 2. A missing `domain` is mapped to the same
sentinel token the LR baseline itself already uses for missing categoricals
(`features_lr.NULL_TOKEN`, `"__null__"`) before the triple key is built. This is a deliberate,
stated choice: two rows with a missing `domain` and identical `slotid`/`creative` are treated as the
**same** triple (a missing-domain placement can be "seen" if a train row shares its
`slotid`/`creative` and also has a missing domain), not silently collapsed without comment.

**Two distinct "seen in train" definitions, reported separately.** The natural reading of "the train
split" is the full season-2 train split (8,834,027 rows, 243,999 distinct triples) -- this is the
headline figure below. The LR baseline, however, was actually fit on the negatively-downsampled
train set (2% of true negatives kept, seed 42 -- `ctr_train_downsampling_season2.json`), which
contains only 25,166 distinct triples (~10.3% of the full-train figure), reconstructed here via
`downsample.downsample_negatives()` at the exact recorded rate/seed (resampling, not retraining) and
verified to reproduce the recorded `n_positive`/`n_negative_true`/`n_negative_sampled` exactly. The
two definitions are reported side by side and never combined into one number.

### Row and click share

| definition of "seen" | bucket | n | % of test rows | positives | % of test clicks | positive rate | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| full train (243,999 triples) | seen | 1,616,702 | 97.5481% | 1,342 | 98.0278% | 0.0830% | 0.9226 |
| full train (243,999 triples) | unseen | 40,636 | 2.4519% | 27 | 1.9722% | 0.0664% | 0.9393 |
| downsampled train (25,166 triples) | seen | 1,500,531 | 90.5386% | 1,222 | 89.2622% | 0.0814% | 0.9191 |
| downsampled train (25,166 triples) | unseen | 156,807 | 9.4614% | 147 | 10.7378% | 0.0937% | 0.9544 |

Sanity checks (both definitions): seen-n + unseen-n = 1,657,338 and seen-positives + unseen-positives
= 1,369, exactly, in both rows. No bucket above has zero positives or zero negatives, so no AUC here
is `n/a`; had one occurred it would be reported as `n/a` with the specific reason
(`"zero positives"` / `"zero negatives"`), the same convention as the per-advertiser table above --
never a silent 0.5 or a dropped row.

**Headline number.** 97.5481% of season-2 test rows (98.0278% of test clicks) have a
`(domain, slotid, creative)` triple that also appears somewhere in the full season-2 train split.
Under the definition of "seen" that actually matches what the fitted model was trained on
(downsampled train), that fraction drops to 90.5386% of rows (89.2622% of clicks) -- 9.4614% of test
rows, and 10.7378% of test clicks, involve a placement triple the model's fit never saw a single
example of, even though the same triple may well appear in the full (undownsampled) train split.

### Interpretation

**Seen and unseen buckets are not directly comparable.** They differ in both size (97.5% vs. 2.5% of
rows under the full-train definition; 90.5% vs. 9.5% under the downsampled-train definition) and
base rate (0.0830% vs. 0.0664% seen/unseen under full train; 0.0814% vs. 0.0937% under downsampled
train) -- an AUC difference between them reflects both discrimination and these compositional
differences, not discrimination alone.

**The seen-bucket AUC sits close to, not below, the pooled AUC.** Seen (full-train) is 0.9226 vs. the
pooled test AUC of 0.9228 -- a 0.0002 gap, effectively noise. Seen (downsampled-train) is 0.9191 vs.
0.9228 pooled -- a 0.0037 gap, smaller than the 0.0096 pooled-vs-per-advertiser gap already
documented in "Per-advertiser AUC (primary metric)" above. Both are consistent with, and smaller
than, that already-documented separability artifact (pooled AUC getting some credit for
distinguishing between subgroups a live bidder never actually has to distinguish between) -- this
section is not reporting a new instance of a large effect, only a small one in the same direction as
the seen bucket goes.

**The unseen bucket's AUC is higher than the seen bucket's, in both definitions -- the opposite of
the generalization concern this analysis set out to check for.** Full-train: unseen 0.9393 vs. seen
0.9226 (unseen higher by 0.0167). Downsampled-train: unseen 0.9544 vs. seen 0.9191 (unseen higher by
0.0352). Had the seen-bucket AUC substantially exceeded the unseen-bucket AUC, that would have been a
deployment-relevant generalization concern (poor ranking on new inventory); that is not what these
numbers show. Two caveats on this specific result, stated plainly rather than treated as a clean
finding:

1. The full-train unseen bucket has only 27 positive examples (out of 40,636 rows) -- an AUC
   computed from 27 clicks carries substantial sampling variance, and 0.9393 should be read as a
   noisy point estimate, not a precise measurement. Quantified below: under this definition the
   difference is **not statistically distinguishable from zero**.
2. The downsampled-train unseen bucket is more populous (147 positives, 156,807 rows) and shows the
   same direction (unseen AUC higher), which weakens a pure small-sample explanation for that case,
   but the underlying reason is not investigated further here. One plausible contributor, not
   verified: `domain`/`slotid`/`creative` are hashed into a shared `2**18`-bucket space alongside 11
   other fields (`features_lr.py`), and ranking for a given row also draws on `region`, `city`,
   `adexchange`, `advertiser`, `hour`, and `usertag` tokens that are typically shared between seen
   and unseen placements -- consistent with the "Leakage investigation" ablation above, which found
   that removing all high-cardinality placement fields costs ~0.07 AUC in aggregate, i.e. most of the
   model's discrimination does not depend on memorizing any single exact placement triple. This is
   offered as a plausible explanation, not a demonstrated one.

**Sampling variance, quantified (Hanley-McNeil standard errors).** Caveats 1 and 2 above are
resolved here rather than left qualitative. Standard errors use the Hanley & McNeil (1982)
analytic form for the variance of a single AUC (`Q1 = A/(2-A)`, `Q2 = 2A^2/(1+A)`), and the
seen-vs-unseen difference is tested by treating the two buckets as independent samples
(`SE_diff = sqrt(SE_seen^2 + SE_unseen^2)`), which they are, since the buckets partition the test
rows:

| definition | bucket | positives | AUC | SE | 95% CI |
|---|---|---:|---:|---:|---|
| full train | seen | 1,342 | 0.9226 | 0.0051 | [0.9127, 0.9325] |
| full train | unseen | 27 | 0.9393 | 0.0320 | [0.8766, 1.0020] |
| downsampled train | seen | 1,222 | 0.9191 | 0.0054 | [0.9085, 0.9297] |
| downsampled train | unseen | 147 | 0.9544 | 0.0120 | [0.9308, 0.9780] |

| definition | unseen - seen | SE | 95% CI | z | p |
|---|---:|---:|---|---:|---:|
| full train | +0.0167 | 0.0324 | [-0.0468, +0.0802] | 0.52 | 0.61 |
| downsampled train | +0.0353 | 0.0132 | [+0.0094, +0.0611] | 2.68 | 0.007 |

**The two definitions give genuinely different verdicts, and only one of them supports a claim.**
Under the full-train definition the +0.0167 difference has a 95% CI comfortably spanning zero
(p = 0.61) -- with 27 positives it is indistinguishable from noise, and no directional claim
should be made from it. Under the downsampled-train definition -- the one that matches the rows the
model was actually fit on -- the +0.0353 difference excludes zero (p = 0.007). So the "unseen ranks
at least as well as seen" observation is statistically supported only under the
model-fit-matching definition, and the unseen bucket's CI upper bound of 1.0020 under the
full-train definition (an impossible AUC, an artifact of the normal approximation at small
positive counts) is itself a reminder of how little that 27-positive estimate constrains.

Note also that these two rows are not independent tests of one hypothesis: they are two views of
the same 1,657,338 test rows under different definitions of "seen", and the downsampled-train
"unseen" bucket contains many rows whose triple *does* appear in full train. The p-values are
reported per-definition and are not corrected for multiplicity, which is appropriate here only
because no selection between them is being performed -- the downsampled-train definition was
designated the model-matching one in advance, not chosen because it reached significance.

**Net assessment.** This analysis does not surface a cold-start ranking degradation for placements
absent from training. Under the model-fit-matching (downsampled-train) definition the unseen bucket
ranks *better* than the seen bucket by 0.0353 AUC (95% CI [+0.0094, +0.0611], p = 0.007); under the
full-train definition the apparent +0.0167 difference is not distinguishable from noise (p = 0.61)
and supports no directional claim.
It also does not change the headline test AUC (0.9228, "Results" above) or the per-advertiser
correction (0.9132 row-weighted mean, "Per-advertiser AUC" above): both of those already account for
all test rows, seen and unseen alike.

**Files.** Analysis script: `backend/src/modeling/seen_unseen_analysis.py` (loads persisted
artifacts only; fits nothing). Companion output:
`backend/data/predictions/ctr_lr_baseline_season2_test_seen_flags.parquet` (1,657,338 rows, columns
`seen_in_full_train` and `seen_in_downsampled_train`, positionally aligned with
`ctr_lr_baseline_season2_test.parquet` -- not merged into that file or its sidecar, so this analysis
never needs to be recomputed from raw splits again). Full numeric results also written to
`backend/data/metadata/ctr_seen_unseen_placement_analysis_season2.json`.
