"""Hashed-feature encoding for the logistic-regression CTR baseline.

**Design.** Every categorical/high-cardinality field is turned into a
`"field=value"` string token and hashed via `sklearn.feature_extraction.
FeatureHasher` (`input_type="string"`, `alternate_sign=True`) into one
shared `HASH_DIM`-wide sparse space -- the textbook "hashing trick" used
for hashed-feature CTR baselines (Weinberger et al., 2009; the same idea
Vowpal Wabbit's `-b` flag implements). `alternate_sign=True` randomly
flips the sign of a token's contribution based on a second hash, which
keeps hash collisions from being a purely additive bias (two colliding
tokens partially cancel rather than always stacking) -- the standard
mitigation, not a bespoke one.

`HASH_DIM = 2**18` (262,144) is a **deliberate, smaller** choice than the
ingestion-time `schema.HASH_BUCKETS = 2**20` per-field convention:
ingestion hashes each high-cardinality field into its own 2**20-bucket
space (see `docs/analysis/eda-findings.md` §3 for the resulting
collision-rate table -- up to 91.9% for `ipinyouid`); here, ALL fields
share ONE combined 2**18-bucket space, which is a different collision
regime (more distinct tokens compete for buckets, but there are also far
fewer tokens overall once low-cardinality fields and a single categorical
value per field per row are accounted for -- see the run log for the
observed non-zero-column count at this dimension). 2**18 was chosen as a
size that keeps the resulting sparse design matrix and
`sklearn.linear_model.LogisticRegression` fit tractable in memory/time on
a full downsampled-train run while still being large relative to this
dataset's actual field cardinalities (the largest single field,
`ipinyouid`, has ~13M raw distinct values dataset-wide, but only as many
*distinct users active on a single train day* actually appear as tokens
for any one row -- ~1.7-8.8M rows per split, well under 2**18 individually
per field, though the SHARED space across 14 fields means real collision
pressure -- an explicit trade-off, not a claim of zero collisions).

**Fields used, and why:**

- Low-cardinality categoricals (`region`, `city`, `adexchange`,
  `slotvisibility`, `slotformat`, `advertiser`) -- included as `field=value`
  tokens. Hashing a small, fixed vocabulary "wastes" some of the hash
  space's collision-avoidance benefit relative to a plain one-hot/ordinal
  encoding, but keeps the encoding pipeline uniform (one code path for
  every categorical field, not a special case per field) and is standard
  practice for a from-scratch hashed baseline.
- High-cardinality categoricals (`domain`, `url`, `slotid`, `creative`,
  `useragent`, `IP`, `ipinyouid`) -- included as `field=value` tokens over
  the RAW string column, not the ingestion-time `<field>_hash` companion
  columns. Using our own hashing pass (rather than re-hashing an
  already-hashed integer) keeps the LR baseline's hash dimension and
  collision behaviour fully independent of, and traceable to, this
  module -- not entangled with the ingestion stage's separate, larger,
  per-field hash space.
- `usertag` (multi-valued: a list of tag ids per row, already parsed to
  `INTEGER[]` at ingestion -- see `schema.py`) -- **multi-hot, over ALL
  tags seen, no frequency threshold.** Verified directly against the
  season-2 train split (`2013-06-06..2013-06-10`): only **43 distinct tag
  ids** appear (see the run log this module's caller writes) -- small
  enough that a frequency cutoff would only discard signal for no
  computational benefit; contrast with `domain`/`url`/`slotid` above,
  where a threshold-free hashing approach is used precisely because a
  full-vocabulary encoding is NOT tractable. Each tag contributes its own
  `usertag=<id>` token; a null/empty usertag list contributes no tokens
  for this field (not a special null token), which is the correct
  behaviour for a multi-hot encoding (absence of any tag, not a positive
  "no tag" signal).
- `hour` (derived from `timestamp`, NOT `timestamp` itself, which is
  dropped as a raw feature) -- included as an unordered `hour=H` token
  (0-23), not a numeric/ordinal feature, so the model can learn an
  arbitrary per-hour effect rather than assuming CTR is linear/smooth in
  the hour number. **Hour 0 is a known batch-logging artifact, not a real
  traffic peak** (see `docs/analysis/eda-findings.md` §4) -- it is still
  included as its own category (dropping it would lose real rows'
  information), but no feature here is built on the assumption that hour
  0 represents genuine peak organic traffic.
- Numeric (`slotwidth`, `slotheight`, `slotprice`) -- standardized
  (`sklearn.preprocessing.StandardScaler`, fit on the TRAIN split only,
  after downsampling, and reused unfit-again for val/test) and
  concatenated as a small dense block alongside the hashed sparse block,
  rather than hashed as categorical tokens -- these are genuinely
  quantitative (auction floor price, slot pixel dimensions), and
  standardizing preserves that instead of discretizing it away.

**Fields deliberately excluded, and why:**

- `urlid` -- 100% null in this dataset (confirmed at ingestion and again
  in EDA, §3/§5) -- zero information, not worth a hash slot.
- `timestamp` (raw) -- superseded by the derived `hour` token above; using
  the raw timestamp directly would effectively memorize exact rows, not
  generalize.
- `<field>_hash` ingestion-time companion columns (`ipinyouid_hash`,
  `useragent_hash`, `IP_hash`, `domain_hash`, `url_hash`, `slotid_hash`,
  `creative_hash`) -- superseded by this module's own hashing pass over
  the corresponding raw string column (see above); including both would
  double-count the same underlying signal under two different hash
  geometries for no benefit.
- Day-of-week / calendar-date features -- NOT built. The season-2 train
  split spans exactly 5 calendar dates (2013-06-06..06-10), each a
  distinct weekday; a day-of-week feature computed from only 5 training
  days is indistinguishable, per weekday, from "which specific training
  date" -- i.e. it would let the model memorize a per-training-day
  baseline rate rather than learn a genuinely generalizable weekday
  effect, and val/test (different calendar dates entirely) could then
  only be hurt or spuriously matched by it. `hour` does not have this
  problem (every train day contributes all 24 hours).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

HASH_DIM = 2 ** 18  # see module docstring for the rationale
NULL_TOKEN = "__null__"

CATEGORICAL_FIELDS = [
    "region", "city", "adexchange", "slotvisibility", "slotformat", "advertiser",
    "domain", "url", "slotid", "creative", "useragent", "IP", "ipinyouid",
]
NUMERIC_FIELDS = ["slotwidth", "slotheight", "slotprice"]

# Every admitted feature column this module is aware of and either uses or
# deliberately drops -- kept here (rather than implicitly relying on
# `dataset.load_impression_features()`'s column order) so a future column
# added to the allowlist causes a loud, explicit failure in
# `assert_known_columns()` below instead of silently being ignored.
_HANDLED_OR_DROPPED_COLUMNS = frozenset(
    CATEGORICAL_FIELDS
    + NUMERIC_FIELDS
    + ["usertag", "timestamp", "urlid"]
    + [f"{base}_hash" for base in ("ipinyouid", "useragent", "IP", "domain", "url", "slotid", "creative")]
)


def assert_known_columns(columns: list[str]) -> None:
    """Fail loudly if `load_impression_features()` ever hands back a
    column this module doesn't know to either use or explicitly drop --
    the same "fail closed on the unexpected" philosophy as
    `schema.feature_columns()`, one layer up: a silently-ignored new
    feature column would be a much quieter, harder-to-notice bug than an
    exception here.
    """
    unknown = sorted(set(columns) - _HANDLED_OR_DROPPED_COLUMNS)
    if unknown:
        raise ValueError(
            f"features_lr: column(s) {unknown!r} are not handled by "
            "features_lr.py (neither used as a feature nor explicitly "
            "listed as dropped) -- add them to CATEGORICAL_FIELDS, "
            "NUMERIC_FIELDS, or the dropped-columns list in the module "
            "docstring before proceeding."
        )


def _field_tokens(X: pd.DataFrame, field: str) -> np.ndarray:
    """Vectorized `"field=value"` string array for one categorical column,
    with an explicit null token (missingness is itself informative for
    some of these fields -- see eda-findings.md §5 -- so it gets its own
    category rather than being silently dropped).
    """
    s = X[field]
    filled = s.astype(object).where(s.notna(), NULL_TOKEN)
    return (f"{field}=" + filled.astype(str)).to_numpy()


def _hour_tokens(X: pd.DataFrame) -> np.ndarray:
    hour = pd.to_datetime(X["timestamp"]).dt.hour
    return ("hour=" + hour.astype(str)).to_numpy()


def _usertag_token_lists(X: pd.DataFrame) -> np.ndarray:
    def to_tokens(tags):
        if tags is None:
            return []
        try:
            if len(tags) == 0:
                return []
        except TypeError:
            return []
        # `set()` de-duplicates in case the same tag id ever appears twice
        # in one row's list -- keeps the multi-hot encoding a presence
        # indicator (0/1 per tag), not a count.
        return [f"usertag={t}" for t in set(tags)]

    return X["usertag"].apply(to_tokens).to_numpy()


def build_token_lists(
    X: pd.DataFrame,
    *,
    categorical_fields: Optional[list[str]] = None,
) -> list[list[str]]:
    """One list of hash-ready string tokens per row, covering every
    categorical/multi-valued field described in the module docstring.

    `categorical_fields=None` (the default) uses the module-level
    `CATEGORICAL_FIELDS` list -- the behaviour every existing caller
    (`train_lr.py`, `score_test.py`) already depends on, unchanged. Passing
    an explicit (shorter) list restricts tokenization to those fields only
    -- used by the full-scale feature-ablation runs (see
    `modeling.ablation_fullscale`) so a variant can drop, e.g., `ipinyouid`
    without touching this module's default path at all. `hour` and
    `usertag` are NOT gated by this parameter -- every ablation variant in
    this codebase keeps them; only `CATEGORICAL_FIELDS` membership is
    restricted.
    """
    fields = CATEGORICAL_FIELDS if categorical_fields is None else categorical_fields
    n = len(X)
    per_field_tokens = [_field_tokens(X, f) for f in fields]
    per_field_tokens.append(_hour_tokens(X))
    usertag_tokens = _usertag_token_lists(X)

    token_lists: list[list[str]] = []
    for i in range(n):
        row = [col[i] for col in per_field_tokens]
        row.extend(usertag_tokens[i])
        token_lists.append(row)
    return token_lists


def encode_features(
    X: pd.DataFrame,
    *,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
    hash_dim: int = HASH_DIM,
    categorical_fields: Optional[list[str]] = None,
) -> tuple[sparse.csr_matrix, StandardScaler]:
    """`X` (admitted feature columns from `load_impression_features()`) ->
    a sparse `(n_rows, hash_dim + len(NUMERIC_FIELDS))` design matrix.

    `fit_scaler=True` fits a fresh `StandardScaler` on `X`'s numeric
    columns (use this ONLY for the training split) and returns it
    alongside the encoded matrix; `fit_scaler=False` requires `scaler=`
    to be passed in (the one fit on train) and reuses it unchanged --
    val/test must never fit their own scaler, or the "scaled relative to
    what" question becomes split-dependent, silently.

    `categorical_fields=None` (the default) leaves this function's
    behaviour byte-for-byte identical to before this parameter existed --
    it tokenizes the module-level `CATEGORICAL_FIELDS`, exactly as
    `train_lr.py`/`score_test.py` require. Passing an explicit list
    restricts tokenization to only those fields (see
    `build_token_lists()`), for feature-ablation runs only.
    """
    assert_known_columns(list(X.columns))

    token_lists = build_token_lists(X, categorical_fields=categorical_fields)
    hasher = FeatureHasher(n_features=hash_dim, input_type="string", alternate_sign=True)
    X_cat = hasher.transform(token_lists)

    numeric_raw = X[NUMERIC_FIELDS].fillna(0.0).to_numpy(dtype=np.float64)
    if fit_scaler:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_raw)
    else:
        if scaler is None:
            raise ValueError(
                "encode_features(): scaler=... must be provided when "
                "fit_scaler=False (fit it once on the train split, then "
                "reuse it for val/test -- see module docstring)."
            )
        numeric_scaled = scaler.transform(numeric_raw)
    X_num = sparse.csr_matrix(numeric_scaled)

    X_full = sparse.hstack([X_cat, X_num], format="csr")
    return X_full, scaler
