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
                       #    live bidding decision -- see FEATURE DENYLIST below.
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
# Feature allowlist / denylist -- the structural enforcement of the
# methodological constraint that `bidprice` must never be used as a feature
# or a baseline policy.
#
# WHY: the dataset README states the campaigns were run "with a fixed
# relatively high-price bidding strategy ... for the purpose of getting
# enough impressions and their paying prices", explicitly different from
# iPinYou's live bidding algorithm. bidprice is therefore a data-collection
# artefact, not a modelled decision -- training on it or benchmarking a
# policy against it would be learning/measuring against a constant (or a
# collection knob), not a real bidding strategy.
#
# It is NOT dropped from the Parquet output -- it's needed to reconstruct
# exactly what was collected (e.g. to compute win rate at the fixed price,
# or to identify the season-3 exceptions below) -- but it must never be
# silently picked up by a "select all columns as features" step downstream.
# Any such step MUST go through `feature_columns()` below rather than doing
# `df.columns.difference({"target"})` or similar.
# ---------------------------------------------------------------------------

FEATURE_DENYLIST = frozenset({
    "bidprice",  # fixed data-collection strategy, see docstring above -- not
                 # a real bidding decision, must not be a feature or baseline
    "bidid",     # row identifier, not predictive
    "payprice",  # this is the auction outcome / label for win-price modelling,
                 # not an input feature -- including it would leak the target
})


def feature_columns(all_columns: list[str]) -> list[str]:
    """Return `all_columns` with the denylisted columns structurally removed.

    This is the ONLY sanctioned way to go from "all columns in the Parquet
    file" to "candidate model features" in this codebase. Downstream
    modelling code should import and call this rather than re-deriving the
    same exclusion list, so `bidprice` cannot be accidentally reintroduced
    by a future "just select everything" shortcut.
    """
    return [c for c in all_columns if c not in FEATURE_DENYLIST]


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
