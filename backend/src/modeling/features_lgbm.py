"""Native-categorical feature encoding for the LightGBM CTR model.

**Field-inclusion set is identical to the LR baseline's, on purpose.** This
module uses exactly `features_lr.CATEGORICAL_FIELDS` (the same 13 fields:
`region, city, adexchange, slotvisibility, slotformat, advertiser, domain,
url, slotid, creative, useragent, IP, ipinyouid`) plus a derived `hour`
field, the same `NUMERIC_FIELDS` (`slotwidth, slotheight, slotprice`), and
the same multi-valued `usertag` field -- see `features_lr.py`'s module
docstring for the full field-inclusion rationale (which columns are
dropped and why: `urlid` 100%-null, raw `timestamp` superseded by `hour`,
the `<field>_hash` ingestion companions superseded by using the raw string
columns directly, no day-of-week feature). Keeping the field set identical
means a val/test AUC or log-loss difference between the LR baseline and
this model is attributable to the MODEL (linear + hashing trick vs.
gradient-boosted trees + native categorical splitting), not to one model
seeing information the other didn't.

**Why native categorical splitting instead of the hashing trick.**
LightGBM can split directly on a categorical feature via an optimal
(for binary-classification objectives) partition search over the
feature's levels, ordered by the training-objective-relevant statistic
(essentially, sorted by the per-level target mean for a binary target --
see Fisher, 1958, "On Grouping for Maximum Homogeneity", and the LightGBM
docs "Optimal Split for Categorical Features"), rather than needing to be
converted to a numeric/one-hot representation at all. Trade-offs relative
to `features_lr.py`'s shared-hash-space `FeatureHasher` encoding:

- **Buys:** no hash collisions (two distinct category values can never
  share a "bucket" and be forced to compete for the same weight the way
  two hash-colliding tokens do in the LR baseline's shared `2**18`-bucket
  space); LightGBM's per-node categorical split search is computed to be
  *optimal* for the current tree/node objective, not a fixed one-hot/hash
  encoding decided before any tree is grown.
- **Costs:** a categorical feature's split search cost scales with its
  cardinality (bounded here -- see `MIN_FREQUENCY` below); a
  high-cardinality field with many singleton-ish levels is a real
  overfitting risk for tree-based models specifically, because a single
  rare level can end up isolated in its own leaf and memorize the handful
  of rows that share it, in a way the LR baseline's linear model (one
  weight per hash bucket, shared and regularized across many colliding
  tokens) is structurally less prone to. This is exactly why this module
  imposes a minimum-frequency threshold on high-cardinality fields (see
  below) rather than handing LightGBM raw, unbounded-cardinality columns.

**Category mapping is fit on the downsampled TRAIN split only, exactly
the discipline `features_lr.py`'s `StandardScaler` follows.** A
`CategoryEncoderState` (see `fit_category_encoder()`) is fit once, on the
already-downsampled train `(X, y)` -- rare-level bucketing thresholds, the
per-field category vocabulary (`pandas.CategoricalDtype`), the `usertag`
multi-hot vocabulary -- and reused, unrefit, for val/test via
`encode_features(..., state=..., fit=False)`. This is the leakage-critical
part of this module: fitting the category vocabulary on val/test (even
just to decide "what counts as rare") would leak information about the
held-out distribution into training-time feature definitions.

**How an unseen category is handled, deterministically.** For each
categorical field, `fit_category_encoder()` fixes a
`pandas.CategoricalDtype` (an explicit, ordered list of allowed category
values) from the TRAIN split. Encoding any split (train, val, or test)
then does `values.astype(that_fixed_dtype)`: any value present in the
fixed category list keeps its value; any value NOT in that list --
whether never seen anywhere in train, or a genuinely new value at
val/test time -- becomes `NaN` (pandas' standard behaviour for
`Categorical.astype()` against a value outside the given categories).
LightGBM treats `NaN` in a categorical column as a missing value and
routes it down whichever branch was determined optimal for missing values
during training (`use_missing=True` is LightGBM's default; not overridden
here) -- i.e. **unseen category values are always treated as missing,
never silently coerced to some arbitrary existing category**, and this
behaviour is identical across every field and every split.

This is deliberately a DIFFERENT mechanism from the rare-value bucketing
below: a value that appeared in train, but rarely, is explicitly folded
into a real, explicit `__rare__` category (still part of the fixed
category list, so it participates in LightGBM's split search as its own
level); a value that never appeared in train at all becomes `NaN`
(missing) instead. Conflating "rare-but-seen" with "never-seen" would
throw away the (weak, but real) signal that a rare-but-seen category
carries -- namely a below-threshold-but-nonzero, still-in-corpus click
rate estimate; wiring both to `NaN` would lose that.

**High-cardinality fields get a documented minimum-frequency threshold;
low-cardinality fields do not.** `HIGH_CARDINALITY_FIELDS = ["domain",
"url", "slotid", "creative", "useragent", "IP", "ipinyouid"]` (the same
7 fields `features_lr.py`'s docstring calls out as high-cardinality, and
the same 7 fields `schema.HIGH_CARDINALITY_HASH_FIELDS` flags at
ingestion, minus `keypage`, which is not an admitted feature at all --
see `schema.NON_FEATURE_BID_COLUMNS`). `MIN_FREQUENCY = 5`: any value of
a high-cardinality field occurring FEWER than 5 times in the downsampled
train split (~182.6K rows; see `modeling.downsample`) is folded into an
explicit `"__rare__"` category for that field, rather than kept as its
own level. Rationale: at `MIN_FREQUENCY = 5`, LightGBM's per-node
categorical split search never has to evaluate a candidate level backed
by fewer than 5 downsampled-train rows, which bounds the single-leaf
memorization risk described above for the sparsest tail of each field's
vocabulary, while keeping every level with even a modest amount of
same-value repetition available to the tree as its own split candidate.
The exact resulting per-field cardinality (post-thresholding, i.e. how
many distinct categories -- including `__rare__` where applicable -- each
field ends up with) is written to this module's caller's run-log record
(see `train_lgbm.py`) and reported in `docs/analysis/ctr-results.md`, not
just asserted here. Low-cardinality fields (`region, city, adexchange,
slotvisibility, slotformat, advertiser`) get NO thresholding -- every
value seen in train becomes its own category -- because (per
`features_lr.py`'s docstring and `schema.LOW_CARDINALITY_CATEGORY_FIELDS`)
these are small, fixed, dense vocabularies where a frequency threshold
would only destroy information for no tractability benefit.

**`hour`** (derived from `timestamp`, exactly as in `features_lr.py` --
raw `timestamp` itself is dropped, not used as a feature) is encoded as a
`pandas.CategoricalDtype` fixed to `list(range(24))` (not "whichever
hours were observed in train"): the value range is a structural property
of the derivation (`pandas.Series.dt.hour`), not something that should be
data-dependent, so it's hardcoded rather than fit. **Hour 0 is a known
batch-logging artifact, not a real traffic peak** (see
`docs/analysis/eda-findings.md` §4 and `features_lr.py`'s docstring) --
it is still included as its own category (dropping it would lose real
rows' information), but nothing here treats it as genuine peak organic
traffic.

**`usertag` (multi-valued) stays multi-hot, explicit binary columns
instead of hashed tokens** (LightGBM has no native multi-valued-category
split mechanism the way it does for single-valued categoricals, so the
hashing-trick multi-hot approach `features_lr.py` uses doesn't carry over
as-is): one `usertag=<id>` binary (0/1) column per DISTINCT tag id
observed in the downsampled TRAIN split (**43 distinct tag ids**, the
same figure verified against season-2 train in `features_lr.py`'s
docstring -- re-verified here directly against the downsampled train
split this module actually fits against, since downsampling could in
principle drop the only row(s) carrying a rare tag; it did not: all 43
persist). A tag id that never appears in the downsampled train split
gets no column at all -- at val/test time, rows carrying only such
unseen tags simply have all 43 known-tag indicator columns at 0, which
is the direct multi-hot analogue of "unseen category -> missing"
(the feature quietly contributes no signal, rather than erroring or
being coerced into an existing tag's column).

**Numeric fields (`slotwidth, slotheight, slotprice`) are passed through
RAW, unscaled -- deliberately, unlike `features_lr.py`'s
`StandardScaler`.** Tree-based split-finding only cares about the RANK
order of a numeric feature's values at each candidate split point, not
their scale (splitting on `slotprice > 250` finds exactly the same
partitions as splitting on `standardized_slotprice > k` for the
corresponding `k`); standardizing would not change a single split
LightGBM chooses, so it is skipped entirely here. Missing numeric values
are left as `NaN` (not filled with `0.0`, unlike `features_lr.py`'s
`fillna(0.0)` before scaling): LightGBM's native missing-value handling
(`use_missing=True`, the default, not overridden here) learns which
branch a missing numeric value should follow from the training
objective, whereas filling with `0.0` would conflate genuine missingness
with an actual, meaningful `0` value (`slotprice == 0` is a real,
observed floor price, not a missing-value marker -- overwriting missing
`slotprice` with `0.0` would make it indistinguishable from that).

**Fields deliberately excluded** -- identical list to `features_lr.py`:
`urlid` (100% null), raw `timestamp` (superseded by `hour`), and the six
`<field>_hash` ingestion-time companion columns (superseded by using the
corresponding raw string column directly, exactly as the LR baseline
already does). See `features_lr.py`'s module docstring for the full
rationale on each.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import numpy as np
import pandas as pd

from . import features_lr

CATEGORICAL_FIELDS = list(features_lr.CATEGORICAL_FIELDS)  # identical 13 fields, see docstring
NUMERIC_FIELDS = list(features_lr.NUMERIC_FIELDS)  # slotwidth, slotheight, slotprice
NULL_TOKEN = features_lr.NULL_TOKEN  # "__null__" -- same sentinel as the LR baseline

RARE_TOKEN = "__rare__"
MIN_FREQUENCY = 5  # occurrences in the DOWNSAMPLED train split -- see module docstring

HIGH_CARDINALITY_FIELDS = ["domain", "url", "slotid", "creative", "useragent", "IP", "ipinyouid"]
LOW_CARDINALITY_FIELDS = [f for f in CATEGORICAL_FIELDS if f not in HIGH_CARDINALITY_FIELDS]

HOUR_FIELD = "hour"
HOUR_CATEGORIES = list(range(24))  # fixed by construction (pandas .dt.hour range), not data-fit

# Every admitted feature column this module is aware of and either uses or
# deliberately drops -- same "fail closed on the unexpected" philosophy as
# `features_lr.assert_known_columns()`.
_HANDLED_OR_DROPPED_COLUMNS = frozenset(
    CATEGORICAL_FIELDS
    + NUMERIC_FIELDS
    + ["usertag", "timestamp", "urlid"]
    + [f"{base}_hash" for base in ("ipinyouid", "useragent", "IP", "domain", "url", "slotid", "creative")]
)


def assert_known_columns(columns: list[str]) -> None:
    """Fail loudly if `load_impression_features()` ever hands back a
    column this module doesn't know to either use or explicitly drop --
    mirrors `features_lr.assert_known_columns()` exactly (same handled/
    dropped column set, since the field-inclusion decision is shared).
    """
    unknown = sorted(set(columns) - _HANDLED_OR_DROPPED_COLUMNS)
    if unknown:
        raise ValueError(
            f"features_lgbm: column(s) {unknown!r} are not handled by "
            "features_lgbm.py (neither used as a feature nor explicitly "
            "listed as dropped) -- add them to CATEGORICAL_FIELDS, "
            "NUMERIC_FIELDS, or the dropped-columns list in the module "
            "docstring before proceeding."
        )


@dataclass(frozen=True)
class CategoryEncoderState:
    """Everything fit on the downsampled TRAIN split and reused, unrefit,
    for val/test -- the LightGBM-encoder analogue of `features_lr.py`'s
    fitted `StandardScaler`. See module docstring for exactly how each
    piece is used at encode time.
    """

    categories: dict          # field -> ordered list of allowed category values (train-fit)
    rare_values: dict         # high-cardinality field -> set of raw values folded into "__rare__"
    cardinality: dict         # field -> len(categories[field]), post-thresholding -- for reporting
    usertag_vocab: list       # sorted distinct tag ids seen in downsampled train (43, verified)
    min_frequency: int
    categorical_fields: list = dataclass_field(default_factory=lambda: list(CATEGORICAL_FIELDS))
    numeric_fields: list = dataclass_field(default_factory=lambda: list(NUMERIC_FIELDS))

    def to_summary_dict(self) -> dict:
        """Compact, JSON-serializable summary (no raw category lists --
        those can be large) for run-log records: per-field cardinality
        after thresholding, and how many values got folded to rare.
        """
        return {
            "min_frequency": self.min_frequency,
            "usertag_vocab_size": len(self.usertag_vocab),
            "per_field_cardinality": dict(self.cardinality),
            "per_field_n_rare_values_folded": {
                f: len(self.rare_values.get(f, ())) for f in HIGH_CARDINALITY_FIELDS
            },
        }


def _field_string_values(X: pd.DataFrame, field: str) -> pd.Series:
    """`field`'s raw column as a string Series with an explicit null
    token (same missingness-is-informative treatment as
    `features_lr._field_tokens()`, minus the `"field="` prefix -- not
    needed here since each field gets its own native categorical column
    rather than sharing one hashed token space).
    """
    s = X[field]
    return s.astype(object).where(s.notna(), NULL_TOKEN).astype(str)


def _hour_values(X: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(X["timestamp"]).dt.hour.astype("Int64")


def _usertag_lists(X: pd.DataFrame) -> pd.Series:
    def to_list(tags):
        if tags is None:
            return []
        try:
            if len(tags) == 0:
                return []
        except TypeError:
            return []
        return sorted({str(t) for t in tags})

    return X["usertag"].apply(to_list)


def fit_category_encoder(X: pd.DataFrame, *, min_frequency: int = MIN_FREQUENCY) -> CategoryEncoderState:
    """Fit a `CategoryEncoderState` on `X` -- call this ONLY on the
    downsampled TRAIN split (see module docstring's leakage-critical
    section). Never call this on val/test.
    """
    assert_known_columns(list(X.columns))

    categories: dict = {}
    rare_values: dict = {}
    cardinality: dict = {}

    for f in CATEGORICAL_FIELDS:
        counts = _field_string_values(X, f).value_counts()
        if f in HIGH_CARDINALITY_FIELDS:
            frequent_mask = counts >= min_frequency
            frequent = sorted(counts.index[frequent_mask])
            rare = set(counts.index[~frequent_mask])
            cats = frequent + ([RARE_TOKEN] if rare else [])
        else:
            frequent = sorted(counts.index)
            rare = set()
            cats = frequent
        categories[f] = cats
        rare_values[f] = rare
        cardinality[f] = len(cats)

    categories[HOUR_FIELD] = list(HOUR_CATEGORIES)
    cardinality[HOUR_FIELD] = len(HOUR_CATEGORIES)

    usertag_lists = _usertag_lists(X)
    vocab = sorted({t for tags in usertag_lists for t in tags})

    return CategoryEncoderState(
        categories=categories,
        rare_values=rare_values,
        cardinality=cardinality,
        usertag_vocab=vocab,
        min_frequency=min_frequency,
    )


def encode_features(
    X: pd.DataFrame,
    *,
    state: Optional[CategoryEncoderState] = None,
    fit: bool = False,
    min_frequency: int = MIN_FREQUENCY,
) -> tuple[pd.DataFrame, CategoryEncoderState, list[str]]:
    """`X` (admitted feature columns from `load_impression_features()`) ->
    `(X_encoded, state, categorical_feature_names)`.

    `X_encoded` is a `pandas.DataFrame` ready to hand to
    `lightgbm.Dataset(..., categorical_feature=categorical_feature_names)`
    or `lightgbm.LGBMClassifier.fit(..., categorical_feature=...)`:
    every field in `categorical_feature_names` has `pandas` `category`
    dtype (native LightGBM categorical handling); every numeric field is
    `float64`, unscaled, `NaN`-preserving; every `usertag=<id>` column is
    `0`/`1` (`int8`).

    `fit=True` fits a fresh `CategoryEncoderState` on `X` (use ONLY for
    the downsampled TRAIN split) and returns it; `fit=False` requires
    `state=` (the one fit on train) and reuses it unchanged -- val/test
    must never fit their own category vocabulary, mirroring
    `features_lr.encode_features()`'s `scaler`/`fit_scaler` discipline
    exactly.
    """
    assert_known_columns(list(X.columns))

    if fit:
        state = fit_category_encoder(X, min_frequency=min_frequency)
    elif state is None:
        raise ValueError(
            "encode_features(): state=... must be provided when fit=False "
            "(fit it once on the downsampled train split, then reuse it "
            "for val/test -- see module docstring)."
        )

    out: dict = {}

    for f in CATEGORICAL_FIELDS:
        values = _field_string_values(X, f)
        if f in HIGH_CARDINALITY_FIELDS:
            rare = state.rare_values.get(f, set())
            if rare:
                values = values.where(~values.isin(rare), RARE_TOKEN)
        dtype = pd.CategoricalDtype(categories=state.categories[f])
        out[f] = values.astype(dtype)

    hour_dtype = pd.CategoricalDtype(categories=state.categories[HOUR_FIELD])
    out[HOUR_FIELD] = _hour_values(X).astype("object").astype(hour_dtype)

    usertag_lists = _usertag_lists(X)
    tag_sets = usertag_lists.apply(set)
    for tag in state.usertag_vocab:
        out[f"usertag={tag}"] = tag_sets.apply(lambda s, t=tag: 1 if t in s else 0).astype("int8")

    for f in NUMERIC_FIELDS:
        out[f] = X[f].astype(np.float64)  # NaN preserved -- see module docstring

    categorical_feature_names = CATEGORICAL_FIELDS + [HOUR_FIELD]
    X_encoded = pd.DataFrame(out, index=X.index)
    return X_encoded, state, categorical_feature_names
