"""
Column definitions for the iPinYou season-2/season-3 training logs.

Everything here was derived empirically from the raw files themselves
(`bzcat training2nd/bid.20130606.txt.bz2 | head` etc.), not from the dead
contest.ipinyou.com schema page. See README.md in this directory for the
full derivation. Summary of what was found:

* imp / clk / conv logs all share ONE 24-column layout. `logtype` (column 3,
  1-indexed) is 1 for imp, 2 for clk, 3 for conv, and matches the source
  file in every season-2/season-3 file checked (the README's documented bug
  where season-1 conversion logs are mislabelled logtype=2 does not apply
  here — verified against training2nd/conv.* and training3rd/conv.*).

* bid logs have 21 columns. They are the SAME 24-column layout with three
  columns removed: `logtype`, `payprice`, and `keypage`. This makes sense —
  a bid log row is written before an auction is won, so there is no winning
  price and no key-page redirect yet, and the log type is implicitly "bid"
  for every row in the file (no need for a discriminator column).

* One additional, easy-to-miss quirk: the `urlid` field is the empty string
  in bid logs but the literal string "null" in imp/clk/conv logs, for what
  is the same "field not populated" condition. Both are normalized to a
  real null at ingestion (see `NULL_LITERALS` below) — this is exactly the
  kind of null-marker drift the pipeline is supposed to catch rather than
  silently propagate.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Column layout (empirically confirmed, see docstring above and README.md)
# ---------------------------------------------------------------------------

# The 24-column layout shared by imp / clk / conv logs, in file order.
IMP_CLK_CONV_COLUMNS: list[str] = [
    "bidid",           # 1  unique id for the bid request that led to this row
    "timestamp",       # 2  YYYYMMDDHHMMSSfff, local (Beijing) time, ms precision
    "logtype",         # 3  1=impression, 2=click, 3=conversion
    "ipinyouid",       # 4  hashed user id (cookie-based)
    "useragent",       # 5  raw browser UA string (free text: may contain , ( ) ;)
    "IP",               # 6  IPv4 with last octet masked, e.g. "1.2.3.*"
    "region",          # 7  region code, see region.en.txt (0 = "unknown", a
                       #    real category, NOT a null marker)
    "city",            # 8  city code, see city.en.txt (0 = "unknown", ditto)
    "adexchange",      # 9  ad exchange id (observed: 1, 2, 3 only)
    "domain",          # 10 hashed publisher domain
    "url",             # 11 hashed page URL
    "urlid",           # 12 anonymized url id; observed always "null" in these files
    "slotid",          # 13 ad slot id
    "slotwidth",       # 14 ad slot width, px
    "slotheight",      # 15 ad slot height, px
    "slotvisibility",  # 16 season2: numeric code (0/1/2/5/255); season3: string
                       #    label ("FirstView".."Na"). Genuine schema drift
                       #    between seasons -> kept as string, not int.
    "slotformat",      # 17 season2: numeric code (0/1/5); season3: "Na" only.
                       #    Same drift as slotvisibility -> kept as string.
    "slotprice",       # 18 floor/reserve price for the slot, in RMB fen (cent)
    "creative",        # 19 creative id
    "bidprice",        # 20 iPinYou's own submitted bid, RMB fen. FIXED
                       #    data-collection strategy in most campaigns, NOT a
                       #    live bidding decision -- see NON_FEATURE_BID_COLUMNS below.
    "payprice",        # 21 winning/settlement price, RMB fen -- the real
                       #    auction-economics signal this dataset is used for
    "keypage",         # 22 landing/key page id, "null" if not applicable
    "advertiser",      # 23 advertiser/campaign id
    "usertag",         # 24 comma-separated user profile tag ids, or "null"
]

# The 21-column bid-log layout: IMP_CLK_CONV_COLUMNS with logtype, payprice
# and keypage removed, order otherwise unchanged. Built programmatically
# rather than typed out a second time so the two schemas cannot drift apart
# by a typo.
_BID_LOG_DROPPED = {"logtype", "payprice", "keypage"}
BID_COLUMNS: list[str] = [c for c in IMP_CLK_CONV_COLUMNS if c not in _BID_LOG_DROPPED]

assert len(IMP_CLK_CONV_COLUMNS) == 24
assert len(BID_COLUMNS) == 21

# logtype value each source file type is expected to contain, for imp/clk/conv.
# Validated against the actual data (not assumed) during `validate.py`.
EXPECTED_LOGTYPE = {"imp": 1, "clk": 2, "conv": 3}

# File-type -> which column layout it uses.
COLUMNS_BY_FILE_TYPE = {
    "bid": BID_COLUMNS,
    "imp": IMP_CLK_CONV_COLUMNS,
    "clk": IMP_CLK_CONV_COLUMNS,
    "conv": IMP_CLK_CONV_COLUMNS,
}

# ---------------------------------------------------------------------------
# Null-marker normalization
# ---------------------------------------------------------------------------

# Literal strings observed in the raw data standing in for "no value", which
# get normalized to a real null (pd.NA / pyarrow null) at ingestion. Notably
# this does NOT include "0" for region/city (that is a legitimate "unknown"
# *category*, confirmed against region.en.txt / city.en.txt, both of which
# define code 0 as "unknown" -- nulling it out would silently destroy a real,
# meaningful value). It also does not include empty-string handling for the
# `urlid` bid-log quirk, which is not a marker at all but an empty field --
# handled directly as `.replace("", NULL)` for that one column, see ingest.py.
NULL_LITERALS = {"null"}

# Columns where an empty string is ALSO a null (mainly the bid-log `urlid`
# quirk described in the module docstring, plus `usertag`/`keypage` should
# a truly empty field ever appear rather than the literal "null").
EMPTY_STRING_IS_NULL_COLUMNS = {"urlid", "keypage", "usertag", "domain", "url"}

# ---------------------------------------------------------------------------
# High-cardinality categorical handling
#
# WHY hashing rather than frequency-based bucketing: frequency-based
# bucketing needs a full pass over the corpus to know each value's count
# before it can decide what's "rare" -- expensive here since these files are
# read in a single streaming pass specifically to avoid materializing them
# in memory. A stable hash (blake2b, truncated, seeded) needs no such pass:
# every chunk can be hashed independently and the result is reproducible
# and stable across re-runs and across season2/season3 (same hash space).
# The tradeoff: hashing can collide unrelated rare values into the same
# bucket, and (unlike frequency bucketing) it does not preserve "this is a
# top-N popular value" information. Given this stage is ingestion (not
# feature engineering for a specific model), we treat the hash as a
# convenience companion column, and keep the raw string column fully intact
# alongside it so a later modelling stage can still do frequency-based
# grouping properly if it wants to, using the real distribution computed
# from the Parquet files (a cheap DuckDB GROUP BY, not a pandas full-load).
#
# Threshold: any field whose cardinality is unbounded / user- or
# content-generated (i.e. scales with row count, not a fixed vocabulary)
# gets a hashed companion column. Fields with a small fixed vocabulary
# (region, city, adexchange, slotvisibility, slotformat, advertiser,
# logtype) are left as plain categorical strings -- their dictionaries are
# cheap and lossless, hashing them would only destroy information.
#
# NOTE: this section is declared BEFORE the feature allowlist below because
# `feature_columns()` needs HIGH_CARDINALITY_HASH_FIELDS to recognise
# `<base>_hash` companion columns generically (see there).
# ---------------------------------------------------------------------------

HIGH_CARDINALITY_HASH_FIELDS = [
    "useragent", "IP", "domain", "url", "slotid", "creative", "keypage",
    "ipinyouid",
]
HASH_BUCKETS = 2 ** 20  # ~1.05M buckets; collision-rate documented in README.md
HASH_SEED = 20130606  # fixed for reproducibility (also: season2 day 1, cute)

LOW_CARDINALITY_CATEGORY_FIELDS = [
    "region", "city", "adexchange", "slotvisibility", "slotformat",
    "advertiser", "logtype",
]

# ---------------------------------------------------------------------------
# Feature allowlist -- the structural enforcement that a column must be
# knowable at bid-request time (and not otherwise excluded for a distinct,
# documented reason) before `feature_columns()` will hand it out as a
# candidate CTR-model feature.
#
# RULE: a column is a candidate CTR feature iff it is a member of
# BID_COLUMNS (the columns present in the bid log -- see the module
# docstring and `_BID_LOG_DROPPED` above), minus NON_FEATURE_BID_COLUMNS
# below. This is an ALLOWLIST *derived from* BID_COLUMNS -- not a
# hand-maintained second list of names, and not a denylist. That
# distinction is the entire point of this section:
#
#   * `logtype`, `payprice`, `keypage` are excluded STRUCTURALLY. They are
#     simply absent from BID_COLUMNS -- a bid-log row is written when the
#     bid request is sent, before any auction resolves, so none of the
#     three exist yet at that point (see the module docstring). Nobody has
#     to remember to list them here, and a future edit cannot silently
#     re-admit them without also re-adding them to BID_COLUMNS itself,
#     which is a much louder, more visible change (it would also affect
#     ingest.py/transform.py's bid-log handling, not just this function).
#     In particular: `payprice` is excluded here because it is not
#     observable at bid-request time for ANY model scoring a bid request --
#     not because "it is the label" for some particular modelling task.
#     A win-price-modelling script has its own, task-specific reasons to
#     treat `payprice` as its target; that is a decision for that script to
#     make explicitly, not something this shared allowlist should encode
#     by baking in one task's framing.
#
#   * `bidprice`, by contrast, genuinely IS present in the bid log -- it is
#     known at bid-request time (iPinYou submits its own bid before the
#     auction runs) -- so it does NOT fall out of BID_COLUMNS structurally.
#     It needs an EXPLICIT exclusion, with its own distinct rationale (see
#     NON_FEATURE_BID_COLUMNS below): the dataset README documents that
#     campaigns were run "with a fixed relatively high-price bidding
#     strategy ... for the purpose of getting enough impressions and their
#     paying prices", different from iPinYou's live bidding algorithm --
#     i.e. bidprice is a fixed data-collection knob set by the logging
#     campaign, not a live per-request bidding decision. Training a CTR
#     model on it would mean learning against a collection artefact, not a
#     real bidding signal. (It is exactly recoverable from
#     `(advertiser, adexchange)` in this dataset -- see README.md -- so its
#     exclusion is modelling hygiene, not an information-theoretic
#     guarantee: nothing stops a caller from using `advertiser` and
#     `adexchange`, both legitimately bid-time features, to reconstruct it.)
#
# `bidprice` is NOT dropped from the Parquet output -- it's needed to
# reconstruct exactly what was collected (e.g. to compute win rate at the
# fixed price) -- but it must never be silently picked up by a "select all
# columns as features" step downstream. Any such step MUST go through
# `feature_columns()` below rather than doing
# `df.columns.difference({"click"})` or similar.
# ---------------------------------------------------------------------------

NON_FEATURE_BID_COLUMNS = frozenset({
    "bidid",     # row identifier, not predictive
    "bidprice",  # fixed data-collection strategy, present at bid-request
                 # time but excluded on modelling-hygiene grounds -- see the
                 # block comment above for the full rationale
})

# The Hive-style partition keys baked into the output directory structure
# (`season=.../date=...`, see ingest.py::_output_path). These surface as
# columns when the Parquet dataset is read back via DuckDB/pyarrow
# partition discovery, so `feature_columns()` must recognise them
# explicitly as "known, and known not to be a feature" rather than letting
# them trip the fail-closed check below -- they are metadata about *where*
# a row was written, not something known at bid-request time in the sense
# BID_COLUMNS is.
PARTITION_KEY_COLUMNS = frozenset({"season", "date"})

# Derived once at import time: BID_COLUMNS minus the explicit non-feature
# set. This -- not FEATURE_DENYLIST, which no longer exists -- is the
# actual allowlist. Recomputing it from BID_COLUMNS means a future change
# to BID_COLUMNS (e.g. if the bid-log schema is ever found to include a
# column that was previously missed) automatically flows through here
# without a second edit.
_ALLOWED_FEATURE_COLUMNS = frozenset(BID_COLUMNS) - NON_FEATURE_BID_COLUMNS


def feature_columns(all_columns: list[str]) -> list[str]:
    """The ONLY sanctioned way to go from "columns present in the Parquet
    output" to "candidate CTR-model features" in this codebase.

    A column is admitted iff it is in BID_COLUMNS (i.e. genuinely knowable
    at bid-request time) and not in NON_FEATURE_BID_COLUMNS (excluded for
    an explicit, documented, non-structural reason). A `<base>_hash`
    companion column (see HIGH_CARDINALITY_HASH_FIELDS / transform.py) is
    admitted iff `<base>` would be admitted -- e.g. `keypage_hash` is
    excluded automatically because `keypage` is not in BID_COLUMNS, with no
    name-based special case required.

    Callers are expected to pass the FULL column list of whatever table
    they read (e.g. the `impressions` table, which -- unlike the bid log --
    genuinely does contain `logtype`/`payprice`/`keypage`, since those are
    only absent from bid-log rows). Those three are recognised as legitimate
    columns of the known iPinYou schema (IMP_CLK_CONV_COLUMNS) that simply
    fall outside BID_COLUMNS, and are therefore excluded silently -- exactly
    like `keypage_hash` above, this requires no name-based listing anywhere:
    they are structurally absent from BID_COLUMNS and structurally present
    in IMP_CLK_CONV_COLUMNS, and that's the entire test.

    Fails CLOSED, not open: any input column that is not a member of the
    known universe -- IMP_CLK_CONV_COLUMNS, a `<base>_hash` companion of a
    HIGH_CARDINALITY_HASH_FIELDS entry, or a PARTITION_KEY_COLUMNS entry --
    raises immediately, rather than being silently dropped (which would
    hide a real bug -- e.g. a typo'd column name never making it into the
    feature set) or silently kept (which would recreate the exact failure
    mode this replaces: a denylist where every new/renamed/joined column is
    a feature by default). This is what rejects join-suffix or rename bugs
    such as `payprice_1`, `timestamp_1`, or a manually renamed `pay_price`
    -- none of those exact strings are members of IMP_CLK_CONV_COLUMNS, so
    all three raise rather than silently passing through. Adding any
    genuinely new column to the pipeline therefore forces a conscious,
    reviewable decision: it belongs in IMP_CLK_CONV_COLUMNS/BID_COLUMNS (if
    it's a real column of the source schema); or it belongs in
    NON_FEATURE_BID_COLUMNS / PARTITION_KEY_COLUMNS with a documented
    reason; or it is not allowed to come out of this function at all.
    """
    features: list[str] = []
    unrecognised: list[str] = []
    for col in all_columns:
        if col in PARTITION_KEY_COLUMNS:
            continue  # metadata about where a row was written, not a feature

        base = col[: -len("_hash")] if col.endswith("_hash") else None
        if base is not None and base in HIGH_CARDINALITY_HASH_FIELDS:
            if base in _ALLOWED_FEATURE_COLUMNS:
                features.append(col)
            # else: `base` itself is not an admissible feature (e.g.
            # `keypage` is absent from BID_COLUMNS -- a post-auction
            # field), so its hash companion `keypage_hash` is excluded the
            # same way. This branch is generic over ANY `<base>_hash`
            # column, not a `keypage_hash`-specific special case.
            continue

        if col in NON_FEATURE_BID_COLUMNS:
            continue  # explicit exclusion, e.g. bidprice -- see block comment above

        if col in _ALLOWED_FEATURE_COLUMNS:
            features.append(col)
            continue

        if col in IMP_CLK_CONV_COLUMNS:
            # A real column of the source schema (logtype/payprice/keypage
            # are the only ones that land here) that is structurally absent
            # from BID_COLUMNS -- excluded silently, same mechanism as
            # above, no name-based listing required. See module docstring.
            continue

        unrecognised.append(col)

    if unrecognised:
        raise ValueError(
            f"feature_columns(): unrecognised column(s) {sorted(unrecognised)!r} -- "
            "refusing to guess whether they're safe to use as CTR-model "
            "features. Add each one to schema.IMP_CLK_CONV_COLUMNS/"
            "BID_COLUMNS (if it's a genuine column of the source schema) "
            "or to schema.NON_FEATURE_BID_COLUMNS / "
            "schema.PARTITION_KEY_COLUMNS (with a documented, reviewable "
            "reason) before it can be selected here."
        )
    return features
