"""
Per-chunk transform: raw tab-separated strings -> a typed, null-normalized
pyarrow Table with an explicit, fixed schema.

Everything here operates on one in-memory chunk (default 300k rows) at a
time, so a single call never holds more than a small, bounded slice of a
multi-GB file in memory. All hashing/casting is vectorized (pandas/pyarrow
C paths), not row-by-row Python loops -- that matters at this scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.util import hash_array

from . import schema

# Numeric measure columns and their Arrow type. These are the columns that
# are genuinely quantitative (prices, pixel dimensions) as opposed to
# small-vocabulary codes (region/city/adexchange/... -- see
# schema.LOW_CARDINALITY_CATEGORY_FIELDS, kept as strings/dictionaries).
NUMERIC_COLUMNS = {
    "slotwidth": pa.int32(),
    "slotheight": pa.int32(),
    "slotprice": pa.int32(),
    "bidprice": pa.int32(),
    "payprice": pa.int32(),
}

STRING_COLUMNS_ARROW_TYPE = pa.string()


def _normalize_nulls(chunk: pd.DataFrame) -> pd.DataFrame:
    """Replace the raw data's null markers with real (pandas) NA.

    Two distinct conventions are normalized here, deliberately, rather than
    silently: the literal string "null" (used across imp/clk/conv logs and
    for `usertag`/`keypage`/`urlid`), and the empty string (used ONLY by
    the bid-log `urlid` field for what is the same "not populated"
    condition -- see schema.py docstring). Region/city code "0" is NOT
    touched here: it is a legitimate "unknown" category per region.en.txt /
    city.en.txt, not a missing-value marker, and nulling it would be a
    silent, undocumented data change of exactly the kind this pipeline is
    supposed to prevent.
    """
    out = chunk.copy()
    for col in out.columns:
        if col in schema.EMPTY_STRING_IS_NULL_COLUMNS:
            out[col] = out[col].replace("", pd.NA)
        out[col] = out[col].replace("null", pd.NA)
    return out


def _hash_bucket(series: pd.Series) -> pd.Series:
    """Deterministic hash-bucket a high-cardinality string column.

    Nulls stay null (rather than hashing to a spurious "null" bucket) so a
    downstream reader can distinguish "this value hashed to bucket 0" from
    "this value was missing".
    """
    values = series.to_numpy(dtype=object)
    hashed = hash_array(values, hash_key=str(schema.HASH_SEED).ljust(16, "0")[:16],
                         categorize=False)
    bucketed = hashed % schema.HASH_BUCKETS  # numpy uint64 array
    mask = series.isna().to_numpy()
    result = pd.array(bucketed.astype("int64"), dtype="Int64")
    result[mask] = pd.NA
    return result


def _parse_usertag(series: pd.Series) -> pa.Array:
    """usertag is a comma-separated list of tag ids, or absent (null after
    normalization). Parsed into a proper Arrow list<int32> column rather
    than left as an opaque string, so downstream code doesn't have to
    re-parse a CSV-inside-a-TSV-cell for every query.
    """
    def split_row(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return None
        return [int(x) for x in str(v).split(",") if x]

    parsed = series.apply(split_row)
    return pa.array(parsed, type=pa.list_(pa.int32()))


def transform_chunk(chunk: pd.DataFrame, file_type: str) -> pa.Table:
    """Transform one raw chunk (all-string dtype, as read from CSV) into a
    typed Arrow Table matching this file_type's fixed output schema.

    Returns a Table so the caller can write it straight to a
    pyarrow.parquet.ParquetWriter without any further conversion, and so the
    schema is identical (by construction) across every chunk of every file
    of the same file_type -- required for ParquetWriter's row-group append
    to work at all.
    """
    chunk = _normalize_nulls(chunk)

    columns: dict[str, pa.Array] = {}

    # bidid: identifier, not analyzed, plain string.
    columns["bidid"] = pa.array(chunk["bidid"], type=pa.string())

    # timestamp: YYYYMMDDHHMMSSfff -> proper datetime. Confirmed empirically
    # (see README.md) that Python/pandas %f zero-pads on the right, so a
    # 3-digit millisecond suffix like "008" parses as 8ms, not 800ms/8us --
    # exactly the semantics we want.
    ts = pd.to_datetime(chunk["timestamp"], format="%Y%m%d%H%M%S%f", errors="raise")
    columns["timestamp"] = pa.array(ts, type=pa.timestamp("us"))

    # logtype only exists in imp/clk/conv; for bid logs it is not present in
    # the raw data (see schema.py) -- we do NOT synthesize one, because
    # doing so would blur the empirically-verified fact that bid logs have
    # no logtype column, which is exactly the kind of thing a schema
    # validation step should be able to catch if it were ever wrong.
    if "logtype" in chunk.columns:
        columns["logtype"] = pa.array(chunk["logtype"].astype("string"), type=pa.string())

    columns["ipinyouid"] = pa.array(chunk["ipinyouid"], type=pa.string())
    columns["ipinyouid_hash"] = pa.array(_hash_bucket(chunk["ipinyouid"]), type=pa.int64())

    for col in ("useragent", "IP", "domain", "url", "slotid", "creative"):
        columns[col] = pa.array(chunk[col], type=pa.string())
        columns[f"{col}_hash"] = pa.array(_hash_bucket(chunk[col]), type=pa.int64())

    if "keypage" in chunk.columns:
        columns["keypage"] = pa.array(chunk["keypage"], type=pa.string())
        columns["keypage_hash"] = pa.array(_hash_bucket(chunk["keypage"]), type=pa.int64())

    # urlid: observed to be always-null in every file checked; kept as a
    # plain nullable string rather than dropped, so a future file that
    # *does* populate it doesn't get silently discarded.
    columns["urlid"] = pa.array(chunk["urlid"], type=pa.string())

    for col in schema.LOW_CARDINALITY_CATEGORY_FIELDS:
        if col == "logtype":
            continue  # handled above
        if col in chunk.columns:
            columns[col] = pa.array(chunk[col], type=pa.string())

    for col, arrow_type in NUMERIC_COLUMNS.items():
        if col in chunk.columns:
            # errors="raise" deliberately: a non-numeric value in a price or
            # dimension column is an anomaly, not something to coerce away.
            # Per project rules, anomalies must be surfaced (crash loudly
            # here) and handled with an explicit, recorded decision -- never
            # silently NaN'd out. Legitimate nulls (already normalized to
            # pd.NA in _normalize_nulls) pass through pd.to_numeric fine.
            numeric = pd.to_numeric(chunk[col], errors="raise")
            columns[col] = pa.array(numeric, type=arrow_type)

    if "usertag" in chunk.columns:
        columns["usertag"] = _parse_usertag(chunk["usertag"])

    return pa.table(columns)


def output_schema(file_type: str) -> pa.Schema:
    """The fixed Arrow schema every chunk of this file_type must produce.

    Declared explicitly (not inferred) so ParquetWriter can be opened before
    the first chunk is transformed and so every subsequent chunk is
    guaranteed to match it -- a mismatch raises immediately instead of
    silently coercing or dropping data.
    """
    fields = [
        pa.field("bidid", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
    ]
    if file_type != "bid":
        fields.append(pa.field("logtype", pa.string()))
    fields.append(pa.field("ipinyouid", pa.string()))
    fields.append(pa.field("ipinyouid_hash", pa.int64()))
    for col in ("useragent", "IP", "domain", "url", "slotid", "creative"):
        fields.append(pa.field(col, pa.string()))
        fields.append(pa.field(f"{col}_hash", pa.int64()))
    if file_type != "bid":
        fields.append(pa.field("keypage", pa.string()))
        fields.append(pa.field("keypage_hash", pa.int64()))
    fields.append(pa.field("urlid", pa.string()))
    for col in schema.LOW_CARDINALITY_CATEGORY_FIELDS:
        if col == "logtype":
            continue
        fields.append(pa.field(col, pa.string()))
    fields.append(pa.field("slotwidth", pa.int32()))
    fields.append(pa.field("slotheight", pa.int32()))
    fields.append(pa.field("slotprice", pa.int32()))
    fields.append(pa.field("bidprice", pa.int32()))
    if file_type != "bid":
        fields.append(pa.field("payprice", pa.int32()))
    fields.append(pa.field("usertag", pa.list_(pa.int32())))
    return pa.schema(fields)
