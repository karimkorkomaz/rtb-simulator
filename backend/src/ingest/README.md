# iPinYou raw -> Parquet ingestion pipeline

Turns the raw iPinYou season-2 and season-3 training logs
(`backend/data/raw/ipinyou.contest.dataset/training2nd/`,
`.../training3rd/`) into partitioned Parquet under `backend/data/processed/`.
Ingestion only — no train/val/test splitting, no downsampling (see
"Where splitting/downsampling would slot in" below).

Only `training2nd/` and `training3rd/` are used. `training1st` is skipped
(different, incompatible schema — no `usertag` column, and a documented
`logtype` bug in its conversion logs, see `known.data.bugs.txt`). The
`testing*` directories are skipped entirely — they are leaderboard/held-out
data, not needed for building a research training set.

## Dataset scale (from the last full run)

| table | rows |
|---|---|
| bids | 64,746,749 |
| impressions | 15,395,258 |
| clicks | 12,694 |
| conversions | 1,033 |
| **total** | **80,155,734** |

`rows_in == rows_out` for every one of the 61 raw files (see
`backend/data/metadata/ingestion_row_counts.json`) — no silent row drops.
Raw input: 4.02 GB compressed (bz2) across `training2nd/` + `training3rd/`.
Output: 8.52 GB Parquet (snappy) under `backend/data/processed/`. Full run
(all 61 files, `--workers 6` on an 8-core machine): 985.9s (~16.4 min)
wall-clock; peak aggregate memory across all 6 worker processes stayed
under 3 GB (see "Parallelism" below).

## How to re-run

From `backend/`, using the project venv:

```
backend\.venv\Scripts\python.exe -m src.ingest.ingest              # full run, sequential
backend\.venv\Scripts\python.exe -m src.ingest.ingest --workers 6  # full run, parallel (~16 min vs. ~70-90 min)
backend\.venv\Scripts\python.exe -m src.ingest.ingest --sample     # ~15-30s dev subset (20k rows/file)
backend\.venv\Scripts\python.exe -m src.ingest.ingest --season 3 --file-type bid
backend\.venv\Scripts\python.exe -m src.ingest.validate            # post-ingestion checks
backend\.venv\Scripts\python.exe -m src.ingest.validate --sample
```

Idempotent: files are skipped if their output already exists; pass
`--overwrite` to force a redo. There is no random component in ingestion
itself (nothing is sampled or shuffled), so there's no seed to fix here —
re-running against the same raw inputs reproduces the same Parquet files.

## The empirically discovered bid-log schema

The contest's own schema page (`http://contest.ipinyou.com/data.shtml`) is
dead, so the schema below was derived directly from the raw data: decompress
a few rows (`bzcat training2nd/bid.20130606.txt.bz2 | head`), count
tab-separated fields, and diff against the known 24-column imp/clk/conv
layout field by field.

**imp / clk / conv logs (24 columns, confirmed identical layout across all
three, and across both seasons):**

```
1  bidid            9  adexchange       17 slotformat       
2  timestamp        10 domain           18 slotprice
3  logtype          11 url              19 creative
4  ipinyouid        12 urlid            20 bidprice
5  useragent        13 slotid           21 payprice
6  IP               14 slotwidth        22 keypage
7  region           15 slotheight       23 advertiser
8  city             16 slotvisibility   24 usertag
```

`logtype` is 1 for imp, 2 for clk, 3 for conv, and was verified to match the
source file for every season-2/season-3 file spot-checked. (The one
documented exception, in `known.data.bugs.txt`, is season-1 conversion logs
being mislabelled `logtype=2` — not applicable here, since season 1 is
excluded from this pipeline. `ingest.py` asserts this invariant per chunk
and raises loudly if it's ever violated, rather than trusting it silently.)

**bid logs (21 columns) — determined empirically by diffing against the
24-column layout above:**

```
1  bidid            9  adexchange       17 slotprice
2  timestamp        10 domain           18 creative
3  ipinyouid        11 url              19 bidprice
4  useragent        12 urlid            20 advertiser
5  IP               13 slotid           21 usertag
6  region           14 slotwidth
7  city              15 slotheight
8  adexchange...    16 slotvisibility
```

(see `schema.py::BID_COLUMNS` for the exact, executable version — it is
built programmatically as "the 24-column list minus `{logtype, payprice,
keypage}`", so the two schemas can't drift apart via a copy-paste typo.)

**The three columns absent from bid logs are exactly `logtype`, `payprice`,
and `keypage`.** This makes sense structurally: a bid-log row is written
when the bid request is sent, before any auction is won — there is no
winning price yet (`payprice`), no landing/key page redirect yet
(`keypage`), and no need for a `logtype` discriminator column since every
row in a bid file is implicitly a "bid" event (unlike imp/clk/conv, which
share one file format and need `logtype` to tell rows apart). The remaining
21 columns are in the same relative order as their counterparts in the
24-column layout.

**Row counts, per source file type** (used to confirm the diff was found
correctly, and to sanity-check `EXPECTED_LOGTYPE`): checked directly against
`training2nd/imp.20130606.txt.bz2` — 1,821,350 rows, `urlid` literally
`"null"` on every single row, `keypage` populated with exactly 5 distinct
hashed values (one per season-2 advertiser campaign).

## Null-marker quirk: `urlid` is `"null"` in imp/clk/conv but empty string in bid logs

Same "field not populated" condition, two different raw encodings for it,
depending on file type:

- imp/clk/conv logs: the literal 4-character string `"null"`.
- bid logs: an empty string (nothing between two tab characters).

Both are normalized to a real null (`pd.NA` -> Arrow null) at ingestion —
see `schema.EMPTY_STRING_IS_NULL_COLUMNS` and `transform._normalize_nulls`.
This is exactly the kind of silent-drift risk a schema-validation step
exists to catch: treat only one of the two encodings as null, and `urlid`'s
apparent null rate would differ by file type for no data reason.

## Null-marker non-quirk: `region`/`city` code `0` is a real category, not null

`region.en.txt` and `city.en.txt` both explicitly define code `0` as
`"unknown"` — a legitimate, meaningful category value (roughly 5–10% of
rows in the files checked), not a missing-value placeholder. The pipeline
deliberately does **not** null this out. This is the other half of the "null
markers are `null` or `0` depending on column" hint in the task brief: `0`
is a null-like marker for some fields elsewhere in ad-tech logs in general,
but empirically, for *this* dataset's `region`/`city` fields specifically,
it is not — nulling it would be a silent, undocumented change to a real
value.

## Schema drift between seasons: `slotvisibility` / `slotformat`

Season 2 encodes these as small integer codes (`slotvisibility` in
`{0,1,2,5,255}`, `slotformat` in `{0,1,5}`). Season 3 encodes them as string
labels (`slotvisibility` in `{FirstView, SecondView, ThirdView, FourthView,
FifthView, OtherView, Na}`, `slotformat` always `"Na"` in the files
checked). Both are kept as **string** columns in the Parquet output (not
cast to int) specifically so this doesn't silently break/coerce for season
3 — casting to int would either fail loudly (good, but then season-3 data
couldn't be ingested at all) or silently truncate the string labels to null
(bad). A downstream consumer that wants season-2-style integer codes for
`slotvisibility` needs to write an explicit, documented season-aware mapping
— that is a modelling-stage decision, not something this ingestion stage
should decide unilaterally.

## `bidprice`: fixed data-collection strategy, not a real bidding decision

**This is the single most important methodological caveat in this
pipeline.** From the dataset README, section 2:

> "For each season, we have run five advertiser campaigns to get these logs
> with a fixed relatively high-price bidding strategy (with exceptions on
> some campaigns in the third season), which is for the purpose of getting
> enough impressions and their paying prices and is different from that of
> our internal live bidding algorithms."

`bidprice` is therefore a data-collection artefact — a knob turned up high
so enough impressions/paying-prices would be observed — not a modelled
bidding decision, and not iPinYou's real live-bidding-algorithm output.
Consequences:

- `bidprice` **must not** be used as a model feature.
- `bidprice` **must not** be used as a baseline bidding policy to benchmark
  against (it isn't a policy; it's closer to a fixed collection cost).
- `bidprice` **is** retained in the Parquet output — dropping it would lose
  the ability to reconstruct exactly what was collected (e.g. computing win
  rate at the fixed price, or checking the README's season-3-exceptions
  claim, see below).

### How this is actually enforced: an allowlist derived from `BID_COLUMNS`, not a denylist

The mechanism is `schema.feature_columns()`, the single sanctioned way to go
from "columns present in a Parquet table" to "candidate CTR-model
features." It is **not** a hand-maintained list of forbidden names (an
earlier version of this pipeline used a `FEATURE_DENYLIST` set for this,
which has been removed — a denylist means every *future* column is a
feature by default, and exact-name matching is trivially defeated by a
rename or a join suffix like `payprice_1`). The rule now is:

> A column is a candidate CTR feature iff it is present in `BID_COLUMNS`
> (the columns genuinely knowable at bid-request time — see "The
> empirically discovered bid-log schema" above), minus
> `schema.NON_FEATURE_BID_COLUMNS`, an explicit, documented exception set.

This makes the exclusion of `logtype`, `payprice`, and `keypage` **structural**:
they are absent from `BID_COLUMNS` itself (a bid-log row is written before
the auction resolves, so none of the three exist yet — see above), so
nobody has to remember to list them anywhere, and they cannot silently
reappear as features without first being added to `BID_COLUMNS`, which is a
much louder, more visible change than editing an exclusion list.

`bidprice`, by contrast, genuinely **is** present in the bid log — iPinYou
submits its own bid before the auction runs, so it's known at bid-request
time and does *not* fall out of `BID_COLUMNS` structurally. It therefore
needs, and gets, an **explicit** exclusion:
`schema.NON_FEATURE_BID_COLUMNS = {"bidid", "bidprice"}` (`bidid` is a row
identifier with no predictive content; `bidprice` is the fixed
data-collection strategy described above — a collection knob, not a live
bidding decision). `feature_columns()` also recognises the Hive partition
keys (`season`, `date`) as known-but-not-features, and admits a `<base>_hash`
companion column (e.g. `ipinyouid_hash`) iff `<base>` is admitted — which is
what makes `keypage_hash` excluded automatically too, with no name-based
special case, since `keypage` isn't in `BID_COLUMNS`. Any column
`feature_columns()` doesn't recognise at all (an unlisted new field, a
join-suffixed rename like `payprice_1`, a manual rename like `pay_price`)
raises immediately rather than being silently dropped or silently admitted
— see `backend/src/ingest/schema.py` and
`backend/tests/test_feature_allowlist.py` for the enforced behaviour and its
test coverage. `backend/src/features/dataset.py` is the loader that calls
`feature_columns()` internally with no `columns=`/`select_all` bypass; any
future modelling code that does something like
`df.columns.difference({"click"})` instead of using that loader (or calling
`feature_columns()` directly) is doing it wrong, and reviewers should flag it.

**One caveat, so this isn't overclaimed:** `bidprice` is exactly recoverable
from `(advertiser, adexchange)` in this dataset — the season-3-exceptions
analysis below shows there is exactly one distinct `bidprice` per
`(season, advertiser, adexchange)` group, zero exceptions, across all 9
training campaigns — and both `advertiser` and `adexchange` are legitimate,
admitted bid-time features. So excluding `bidprice` itself is **modelling
hygiene, not an information-theoretic guarantee**: nothing stops a model
from reconstructing the same signal via `advertiser`/`adexchange`, which is
expected and fine (those two are real bid-time features in their own
right) — what the exclusion prevents is the *appearance* of using a
per-request bidding decision as a feature when no such decision exists in
this data.

### Season-3 exceptions: what "variable `bidprice`" actually means in this data

The README's parenthetical — "with exceptions on some campaigns in the
third season" — was checked against the **full** ingested `bids` table
(64,746,749 rows, all 9 campaigns, both seasons) via `validate.py`'s
`bidprice_by_advertiser` breakdown (`GROUP BY season, advertiser`,
`COUNT(DISTINCT bidprice)`) and a follow-up ad-hoc query grouping by
`(season, advertiser, adexchange)`. Findings (numbers below are from
`backend/data/metadata/validation_report.json`, reproducible by re-running
`ingest.py` then `validate.py`):

| season | advertiser | rows (bid log) | distinct `bidprice` values | range |
|---|---|---|---|---|
| 2 | 1458 | 14,701,496 | 1 | 300 |
| 2 | 3358 | 3,751,016 | 3 | 227–241 |
| 2 | 3386 | 14,091,931 | 1 | 300 |
| 2 | 3427 | 14,032,619 | 3 | 227–241 |
| 2 | 3476 | 6,712,268 | 3 | 238–254 |
| 3 | 2259 | 2,987,731 | 2 | 277–294 |
| 3 | 2261 | 2,159,708 | 2 | 277–294 |
| 3 | 2821 | 5,292,053 | 2 | 277–294 |
| 3 | 2997 | 1,017,927 | 1 | 277 |

At face value this looks like it supports the README: some campaigns (in
**both** seasons, not just season 3 — `3358`/`3427`/`3476` in season 2 also
have 2–3 distinct values) have a non-constant `bidprice`. But grouping one
level finer, by `(season, advertiser, adexchange)`, resolves this
completely:

```
season  advertiser  adexchange  n            n_distinct(bidprice)
2       3427        1           1,133,686    1  (bidprice = 227)
2       3427        2           7,133,508    1  (bidprice = 238)
2       3427        3           5,765,425    1  (bidprice = 241)
3       2259        1             718,695    1  (bidprice = 294)
3       2259        2           1,028,787    1  (bidprice = 277)
3       2259        3           1,240,249    1  (bidprice = 294)
```

**Across every one of the 9 campaigns in both seasons, `COUNT(DISTINCT
bidprice)` within a `(season, advertiser, adexchange)` group is exactly 1
— zero exceptions, zero variance.** `bidprice` is not a per-auction
decision anywhere in this dataset; it is a deterministic lookup table keyed
by `(advertiser, adexchange)`. Campaigns that show 2–3 distinct values at
the advertiser level (in both season 2 and season 3) are simply running on
2–3 different ad exchanges, each with its own fixed price — advertiser
`2821` (season 3) even spans a 4th ad exchange not seen in season 2's data,
still with one fixed price per exchange, no exceptions.

**Practical implication for downstream analysis:** the README's "with
exceptions on some campaigns in the third season" does not manifest, in
this specific field, as genuine per-request bid variability for any of the
9 training campaigns — the fixed-strategy assumption holds exactly, once
you condition on ad exchange. Do not assume a single global constant per
advertiser (it can differ by exchange), but do not treat any campaign here
as having a "real" bidding policy either — `bidprice` remains structurally
denylisted as a feature/baseline for the reasons above, since even a
per-exchange-fixed price is still a collection artefact, not a modelled
decision. (This finding is about the `bidprice` field specifically in the
season-2/season-3 **training** data ingested here; it does not rule out
the README's claim applying to campaigns or data outside this pipeline's
scope, e.g. leaderboard/testing data, which is intentionally excluded.)

**Related anomaly, also surfaced:** advertiser `2997` (season 3) has `NULL`
`adexchange` on **100%** of its rows in both the `bids` and `impressions`
tables (1,017,927 and 312,437 rows respectively) — not a pipeline bug
(consistent across both independently-sourced tables), but a genuine
property of this campaign's raw logs. Not imputed or dropped; carried
through as a real null, and worth knowing before running any
exchange-conditioned analysis that includes this advertiser.

## High-cardinality categorical handling

Fields with effectively unbounded cardinality (`useragent`, `IP`, `domain`,
`url`, `slotid`, `creative`, `keypage`, `ipinyouid`) get a companion
`<field>_hash` column (`int64`, values in `[0, 2**20)`, deterministic
blake2-style hash via `pandas.util.hash_array` with a fixed seed —
`schema.HASH_SEED` / `schema.HASH_BUCKETS`), **in addition to** the raw
string column, which is always kept intact.

**Why hashing instead of frequency-based bucketing at this stage:**
frequency-based bucketing (group rare values into an "other" bucket) needs a
full pass over the corpus to know each value's count before deciding what's
rare — expensive here, since files are read in a single streaming pass
specifically to avoid materializing them in memory (see "Memory strategy"
below). A stable, seeded hash needs no such pass: every chunk can be hashed
independently, and the result is reproducible across re-runs. The tradeoff:
hashing can silently collide unrelated rare values into the same bucket, and
loses "this is a top-N popular value" information that frequency bucketing
would preserve. Since this is the *ingestion* stage, not feature engineering
for one specific model, the hash is a convenience companion column — a
later modelling stage that wants proper frequency-based grouping can compute
the true distribution cheaply with a DuckDB `GROUP BY` directly over these
Parquet files (no pandas full-load needed) and bucket however it likes,
using the raw column that's still there.

Low-cardinality, fixed-vocabulary fields (`region`, `city`, `adexchange`,
`slotvisibility`, `slotformat`, `advertiser`, `logtype`) are kept as plain
string columns without hashing — their dictionaries are small and lossless,
so hashing them would only destroy information for no benefit.

## Output layout

```
backend/data/processed/
  bids/season={2,3}/date=YYYY-MM-DD/part-0.parquet
  impressions/season={2,3}/date=YYYY-MM-DD/part-0.parquet
  clicks/season={2,3}/date=YYYY-MM-DD/part-0.parquet
  conversions/season={2,3}/date=YYYY-MM-DD/part-0.parquet
```

**Log type is the top-level partition**, not folded into date, because bids
vs. impressions/clicks/conversions are structurally different tables (21 vs.
24 columns; bid-log row counts are 1–2 orders of magnitude larger) — mixing
them into one dataset would mean either padding bid rows with null
`payprice`/`keypage`/`logtype` (misleading — those aren't "missing", they
structurally don't exist for a bid) or losing the distinction entirely.

**Season is its own partition level** (not folded into the date string)
because season 2 and season 3 differ in ways that matter for filtering: the
`slotvisibility`/`slotformat` schema drift above, and the advertiser-set
change between seasons (different campaigns run in season 2 vs. season 3,
see the `bidprice`-by-advertiser table above). Partitioning this way means
a query or downstream script can trivially do `WHERE season = 2` via
Hive-style partition pruning, without parsing dates or advertiser IDs.

**Date is the leaf partition**, matching the natural one-file-per-day
granularity of the raw data, and making the pipeline embarrassingly
resumable/idempotent: each raw file maps to exactly one output file, so
`--overwrite`-free re-runs just skip whatever's already there.

Each `date=` directory currently holds a single `part-0.parquet` (the raw
files are already date-partitioned 1:1, so there was no reason to split
further); the fixed `part-0` naming leaves room to add `part-1` etc. later
without a directory-layout change if a file ever needs splitting.

## Memory strategy

Each raw file is read via `pandas.read_csv(..., compression="bz2",
chunksize=300_000)`. Every chunk is transformed (`transform.py`, all
vectorized pandas/pyarrow operations, no per-row Python loops except a tiny
list-parse for `usertag`) into a `pyarrow.Table` with a schema declared
*before* any data is read (`transform.output_schema`), and appended to a
single open `pyarrow.parquet.ParquetWriter` as a new row group. Peak memory
is therefore O(chunk size), independent of file size — this is what makes it
safe to process season-2 bid files (250–625MB compressed, decompressing to
several GB of text, tens of millions of rows) on a 16GB machine. The
alternative (`pandas.read_csv` without `chunksize`, or DuckDB's `read_csv`)
was not used because: (a) a full in-memory load defeats the whole point at
this file size, and (b) DuckDB 1.5.5's `read_csv` does not support `bz2`
compression at all (`Unrecognized file compression type "bz2"` — verified
directly against this exact dataset before deciding on the pandas-chunked
approach; DuckDB is still used downstream in `validate.py`, where it reads
the *Parquet* output directly off disk with projection/aggregation pushdown,
which is exactly the workload it's good at).

**`data/interim/` stays empty by design.** The chunked-streaming approach
above reads directly from the raw `.bz2` files and writes straight to the
final Parquet output — there is no intermediate decompressed-text or
partially-transformed file ever written to disk, so there is nothing to
place in `data/interim/` or clean up after the fact. (Verified: this was
tested with an intentional hard-kill mid-run during development — see
`.parquet.tmp` handling below — and confirmed no stray files were left
under `data/interim/`.)

Output is written to a `.parquet.tmp` file and atomically `replace()`d onto
the canonical path only after the whole source file has been fully written
and the row-count in/out invariant has been checked — so a crash mid-file
never leaves a corrupt or partial file at the canonical output path for a
later run (or a human) to mistake for a complete one.

## Validation (`validate.py`)

Runs after ingestion, queries the Parquet output directly via DuckDB
(pushdown, not a pandas full-load):

- total row count per table (bids/impressions/clicks/conversions)
- null rate per column, single-pass
- `slotprice`/`bidprice`/`payprice` distributions (min/max/p50/p95/mean)
- **anomaly checks, surfaced not silently fixed:** `payprice > bidprice`
  (should not happen in a second-price auction — see finding below),
  non-positive `slotwidth`/`slotheight`
- `bidprice` distinct-value count per `(season, advertiser)` — the evidence
  base for the fixed-strategy investigation above (refined further, outside
  `validate.py`, by an ad-hoc `(season, advertiser, adexchange)` grouping)

Writes `backend/data/metadata/validation_report.json` and exits non-zero if
any hard anomaly count is non-zero, so it can be wired into a re-run check
rather than requiring someone to remember to open the JSON.

**Anomaly found and surfaced (not fixed) during development:** 191,091 of
15,395,258 impression rows (1.24%) have `payprice > bidprice`, which should
be structurally impossible in a second-price auction (you can't be forced
to pay more than your own bid). This is preserved as-is in the Parquet
output and flagged by `validate.py` rather than clipped/dropped — see
`validation_report.json` (`impressions.anomalies.payprice_greater_than_bidprice`)
and treat it as a known data-quality caveat for any auction-simulation code
built on top of this data. 1.24% is a small fraction of rows, but "small"
is not "zero" — this pipeline's job is to surface it, not decide for the
modelling stage whether/how to handle it (e.g. whether it reflects
currency-conversion rounding, a logging race between the bid and win
events, or something else is a modelling-stage investigation, not an
ingestion-stage fix).

## Where train/val/test splitting and downsampling would slot in

Deliberately **not** built in this task (ingestion only). `backend/src/features/dataset.py`
(`load_impression_features()` / `impression_feature_relation()`) exists as
the sanctioned loader that turns the ingested `impressions` table into
`(X, y)` — features via `schema.feature_columns()`, label via a
key-only join against `clicks` (see that module's docstring for why the
join projects *only* `bidid` from the clicks side) — but it does not itself
split or downsample; it hands back the full (season-filtered) row set. For
the next stage:

- **Temporal split, never random:** `timestamp` is already a proper
  `datetime64[us]` column in the Parquet output, partitioned by
  `season`/`date`, so a downstream script can pick explicit cut points
  (e.g. train on days 1–5 of a season, validate on day 6, test on day 7)
  directly via partition pruning, with the cut points as an explicit,
  configurable argument — not inferred, not random.
- **Downsampling:** would apply to negatives in the `impressions` table
  (joined against `clicks`/`conversions` on `bidid` to get the positive
  label) for a CTR-modelling stage, and the exact sampling rate must be
  written to `backend/data/metadata/` (not just implied by row counts) so a
  modelling stage can recalibrate predicted probabilities. **Never** applied
  to whatever is designated the test split — it must reflect the true
  (heavily imbalanced) class distribution.
- Both stages should write to new paths under `backend/data/processed/` or
  a new `backend/data/processed/splits/` — never back into the ingestion
  output, preserving this Parquet layer as a stable, re-derivable base.
