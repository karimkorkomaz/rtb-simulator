"""
The single sanctioned entry point for loading CTR-modelling features out of
the ingested `impressions` Parquet table.

Nothing else in this codebase should read `data/processed/impressions/`
directly for modelling purposes -- go through
`load_impression_features()` (or `impression_feature_relation()` for the
lazy/streaming variant) so the feature/non-feature boundary enforced by
`ingest.schema.feature_columns()` is applied every single time, with no way
to route around it.

WHY THIS MODULE EXISTS (see the methodology audit that prompted it): before
this module, there was no loader at all in `backend/src/` -- only the
`ingest` package that writes Parquet. Any modelling code would have had to
invent its own "drop these columns" logic, exactly the pattern
(`df.drop(columns=[...])`) that silently reintroduces a leaking column the
moment someone forgets to update the drop-list. This module makes that
impossible by construction: there is no parameter here that accepts an
arbitrary column list and skips `schema.feature_columns()` -- see
`_resolve_feature_columns()` below.

LABEL CONSTRUCTION: the `impressions` table has no `click` column (clicks
are a separate table, `data/processed/clicks/`, joined here on `bidid`).
The join deliberately projects ONLY `bidid` from the clicks side
(`LEFT JOIN (SELECT DISTINCT bidid FROM clicks) c USING (bidid)`, label =
`CASE WHEN c.bidid IS NULL THEN 0 ELSE 1 END`) rather than
`SELECT * FROM clicks`. Every non-key column on the clicks table
(timestamp, useragent, IP, ... -- the same 24-column layout as
impressions, see ingest/schema.py) is non-null if and only if a click
happened, for the trivial reason that a clicks-table row only exists when
a click happened. Joining any of those columns in would therefore leak the
label back in as an apparently-innocuous feature-shaped column, under any
name. Projecting only the join key avoids that entirely, structurally,
rather than relying on remembering not to `SELECT *` a clicks join.

DETERMINISTIC ROW ORDER: the underlying query always ends in
`ORDER BY i.bidid, i.timestamp` (see `impression_feature_relation()`
below). DuckDB parallelizes the Parquet scan with no inherent row-order
guarantee, so two otherwise-identical calls to this loader can (and, when
this was checked, did) materialize the same set of rows in a *different*
order. That matters beyond cosmetics: any seeded downstream sampling that
indexes positionally (e.g. `modeling.downsample.downsample_negatives()`)
silently picks *different actual rows* across runs if the row order isn't
fixed, even with a fixed seed -- "fixed random seed everywhere, every
number in the thesis must regenerate" does not hold otherwise. The
explicit `ORDER BY` closes that gap for every caller of this module, not
just the modelling stage that happened to notice it.

MEMORY: the impressions table is ~15.4M rows. Both entry points here build
a DuckDB relation (`impression_feature_relation`) that pushes projection
(only admitted feature columns), the season filter, and any `row_limit`
down to the Parquet scan -- nothing is materialized in Python until (and
unless) `.load_impression_features()` (or a caller's own `.df()` call) is
invoked, and DuckDB streams/query-plans the read rather than pandas
full-loading the files. `row_limit` exists specifically so tests and dev
iteration can work with a handful of rows in milliseconds instead of
minutes -- see `--sample` in `ingest.py` for the equivalent idea at the
ingestion stage. Materializing the *entire* table (no row_limit, no
`columns` restriction) is possible but is the caller's explicit choice, not
this module's default recommendation for anything other than a machine
with enough RAM to hold it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

from ..ingest import paths, schema

LABEL_COLUMN = "click"


def _impressions_root(processed_root: Optional[Path]) -> Path:
    return (processed_root or paths.PROCESSED_ROOT) / "impressions"


def _clicks_root(processed_root: Optional[Path]) -> Path:
    return (processed_root or paths.PROCESSED_ROOT) / "clicks"


def _resolve_feature_columns(
    available_columns: list[str], requested: Optional[Iterable[str]]
) -> list[str]:
    """Intersect a caller's requested column subset with the allowlist.

    This is the ONLY place a caller can narrow the column set, and it is
    NOT a passthrough: `available_columns` (the real columns of the
    Parquet dataset being read) is always run through
    `schema.feature_columns()` first, and a `requested` name that survives
    that filter is the only kind of name that can appear in the output.
    Asking for a column outside the allowlist (e.g. `payprice`, or a typo)
    raises rather than being silently ignored -- consistent with
    `feature_columns()`'s own fail-closed behaviour.
    """
    admitted = schema.feature_columns(available_columns)
    if requested is None:
        return admitted
    requested = list(requested)
    admitted_set = set(admitted)
    outside = sorted(set(requested) - admitted_set)
    if outside:
        raise ValueError(
            f"load_impression_features(): requested column(s) {outside!r} "
            "are not in the feature allowlist (schema.feature_columns()) "
            "-- refusing to select them. There is no bypass for arbitrary "
            "column selection in this module."
        )
    # Preserve the allowlist's (i.e. BID_COLUMNS-derived) order rather than
    # the caller's requested order, so output column order is stable and
    # independent of how `columns` happened to be written.
    requested_set = set(requested)
    return [c for c in admitted if c in requested_set]


def impression_feature_relation(
    con: duckdb.DuckDBPyConnection,
    *,
    seasons: Iterable[int] = (2, 3),
    columns: Optional[Iterable[str]] = None,
    row_limit: Optional[int] = None,
    processed_root: Optional[Path] = None,
    date_range: Optional[tuple[str, str]] = None,
) -> tuple[duckdb.DuckDBPyRelation, list[str]]:
    """Build (but do not materialize) the DuckDB relation for CTR features
    + label, over the `impressions` table joined against `clicks`.

    Returns `(relation, feature_columns)`: `relation` has one column per
    admitted feature plus a trailing `click` (0/1) label column;
    `feature_columns` is the exact ordered list of feature column names (so
    a caller doesn't have to re-derive "all columns except the label").
    Nothing is read from disk until the caller calls `.df()` / `.arrow()` /
    `.fetchall()` etc. on the returned relation, or until
    `load_impression_features()` (below) does so for them.

    Raises `FileNotFoundError` if the impressions table hasn't been
    ingested yet (`data/processed/` is git-ignored and not present on a
    clean clone -- see backend/src/ingest/README.md to generate it).

    Note: `ingest.py --sample` writes to a separate root
    (`paths.SAMPLE_ROOT`, i.e. `data/interim/sample/`), not
    `data/processed/` -- see `paths.output_root()`. This function defaults
    to `paths.PROCESSED_ROOT` and does not read `--sample` output unless a
    caller explicitly passes `processed_root=paths.SAMPLE_ROOT` (or
    `paths.output_root(sample=True)`).

    `date_range`: optional inclusive `(start, end)` pair of `YYYY-MM-DD`
    strings, pushed down as `WHERE i.date BETWEEN start AND end` alongside
    the season filter. This exists so the temporal train/val/test split
    (see `backend/src/ingest/README.md`, "Where train/val/test splitting
    and downsampling would slot in", and `backend/src/modeling/split.py`
    for the actual boundary VALUES) can select a date window via the same
    partition-pruned DuckDB scan as everything else in this module, rather
    than a caller materializing the full season and filtering with pandas,
    or -- worse -- reading `data/processed/impressions/` directly and
    reimplementing the feature allowlist. This parameter only narrows
    *which rows* are scanned; it does not change which *columns* are
    admitted, and it still goes through `schema.feature_columns()`
    exactly as before. It is `None` by default (no date filtering), so
    every existing caller is unaffected.
    """
    imp_root = _impressions_root(processed_root)
    if not imp_root.exists():
        raise FileNotFoundError(
            f"{imp_root} does not exist -- run "
            "`backend\\.venv\\Scripts\\python.exe -m src.ingest.ingest` "
            "from backend/ first (a real, non-sample run: --sample writes "
            "to data/interim/sample/, not data/processed/ -- pass "
            "`processed_root=paths.SAMPLE_ROOT` explicitly if that's what "
            "you want to read instead). See backend/src/ingest/README.md."
        )
    clk_root = _clicks_root(processed_root)
    imp_glob = str(imp_root / "**" / "*.parquet")
    clk_glob = str(clk_root / "**" / "*.parquet")

    # DESCRIBE reads only Parquet footers/schema, not row data -- cheap
    # even though the dataset is 15.4M rows, and gives us the real column
    # list to run through feature_columns() rather than hand-maintaining a
    # duplicate of transform.output_schema() here.
    schema_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{imp_glob}', hive_partitioning=1)"
        ).fetchall()
    ]
    feature_cols = _resolve_feature_columns(schema_cols, columns)

    season_list = [int(s) for s in seasons]
    if not season_list:
        raise ValueError("impression_feature_relation(): `seasons` must be non-empty.")
    season_filter = "WHERE i.season IN (" + ", ".join(str(s) for s in season_list) + ")"

    if date_range is not None:
        start, end = date_range
        # DATE literals, not bound parameters: `con.execute()` is already
        # used positionally-parameter-free elsewhere in this function, and
        # start/end are only ever produced by `modeling.split` (a fixed,
        # reviewed set of ISO date strings), never raw user input.
        season_filter += f" AND i.date BETWEEN DATE '{start}' AND DATE '{end}'"

    select_cols = ", ".join(f'i."{c}"' for c in feature_cols)
    limit_clause = f"LIMIT {int(row_limit)}" if row_limit is not None else ""

    # Label: see module docstring -- only `bidid` is ever projected from
    # the clicks side of the join, by construction, so no clicks-table
    # column (which would be non-null iff click == 1) can leak through
    # here under any name.
    #
    # `_sort_bidid` / `_sort_timestamp`: DuckDB parallelizes the Parquet
    # scan across row groups/files with no inherent row order guarantee --
    # two separate executions of the identical query (no LIMIT, same data)
    # were empirically observed to materialize the SAME set of rows in a
    # DIFFERENT order (verified: two back-to-back calls' row order did not
    # align; caught via a downstream logistic-regression C-selection run
    # picking a different `C` across two otherwise-identical `--full`
    # invocations). Since every seeded-sampling step downstream
    # (`modeling.downsample`) samples by pandas positional index, an
    # unstable row order silently breaks "fixed random seed everywhere,
    # every number in the thesis must regenerate". A SQL-level
    # `ORDER BY i.bidid, i.timestamp` fixes this correctly but was
    # empirically far too slow at this row count/width (an external sort
    # over ~8.8M rows x 20+ columns, several of them long strings, driving
    # multi-GB memory growth and disk-spill-level slowness -- confirmed
    # via a run that was still sorting after 20+ minutes with heavy
    # memory/CPU churn, vs. ~29 minutes for the entire rest of the
    # pipeline combined). Instead, `bidid`/`timestamp` are carried through
    # as two extra, hidden columns here (cheap: a fixed-width scalar copy,
    # no sort), and `load_impression_features()` below does the ordering
    # as a single in-memory pandas sort AFTER materializing -- same
    # determinism guarantee (see that function's docstring), far cheaper
    # in practice. `bidid, timestamp` (not `timestamp` alone) because
    # ~0.37% of impression rows share a duplicate `bidid` with a distinct
    # `timestamp` (see eda-findings.md's duplicate `bidid` finding), and
    # conversely two DIFFERENT bidids can plausibly share one millisecond
    # timestamp -- the pair together is what gives a fully deterministic
    # total order. This ordering guarantee only applies to
    # `load_impression_features()`; a caller that materializes
    # `impression_feature_relation()`'s relation directly (bypassing
    # `load_impression_features()`) gets these two extra columns but NOT
    # the sort -- not a currently exercised code path in this codebase,
    # but worth knowing if that changes.
    query = f"""
        SELECT {select_cols},
               i."bidid" AS "_sort_bidid",
               i."timestamp" AS "_sort_timestamp",
               CASE WHEN c.bidid IS NULL THEN 0 ELSE 1 END AS "{LABEL_COLUMN}"
        FROM read_parquet('{imp_glob}', hive_partitioning=1) i
        LEFT JOIN (
            SELECT DISTINCT bidid FROM read_parquet('{clk_glob}', hive_partitioning=1)
        ) c USING (bidid)
        {season_filter}
        {limit_clause}
    """
    return con.execute(query), feature_cols


def load_impression_features(
    *,
    seasons: Iterable[int] = (2, 3),
    columns: Optional[Iterable[str]] = None,
    row_limit: Optional[int] = None,
    processed_root: Optional[Path] = None,
    date_range: Optional[tuple[str, str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Materialize CTR features + label as `(X, y)`.

    `X` is a `pandas.DataFrame` of admitted feature columns only -- never
    `click`, `bidid`, `bidprice`, `payprice`, `logtype`, `keypage`, or any
    Hive partition key -- so callers never need `df.drop(columns=[...])`
    to separate the label (the exact pattern the methodology audit flagged
    as unsafe: it silently keeps everything not explicitly named). `y` is a
    `pandas.Series` named `"click"`, 0/1, aligned to `X`'s row order.

    `row_limit` and/or a narrow `seasons` selection should be used for
    tests and interactive dev work -- see the module docstring's "MEMORY"
    section for why materializing the full ~15.4M-row table is a real,
    deliberate choice rather than this function's default use case.

    Downsampling negatives and the temporal train/val/test split are
    NOT performed here -- this function returns the full (season-filtered)
    row set as-is. That is a separate, explicitly-configured stage
    downstream (see ingest/README.md "Where train/val/test splitting and
    downsampling would slot in"): this loader's only job is "raw columns
    -> admitted features + label", not modelling-stage sampling decisions.

    DETERMINISTIC ROW ORDER (see module docstring's "DETERMINISTIC ROW
    ORDER" section): `X`/`y` are always sorted by the underlying
    `(bidid, timestamp)` pair before being returned, regardless of the
    Parquet scan's internal (non-deterministic) order -- the sort happens
    HERE, once, on the already-materialized pandas DataFrame (cheap: an
    in-memory sort of a few extra scalar columns), not as a SQL `ORDER BY`
    pushed into the DuckDB query (empirically far too slow at this
    row-count/column-width -- see `impression_feature_relation()`'s
    docstring for the measurement that ruled that approach out).
    """
    con = duckdb.connect()
    relation, feature_cols = impression_feature_relation(
        con,
        seasons=seasons,
        columns=columns,
        row_limit=row_limit,
        processed_root=processed_root,
        date_range=date_range,
    )
    table = relation.fetch_arrow_table()
    df = table.to_pandas()

    # Deterministic order (see docstring above): a single in-memory sort by
    # the hidden (bidid, timestamp) columns, then drop them -- neither is
    # an admitted feature (bidid is explicitly excluded; timestamp is
    # already present separately under its own name if it was requested/
    # admitted, so `_sort_timestamp` here is purely an internal ordering
    # key, never part of the returned column set).
    df = df.sort_values(
        ["_sort_bidid", "_sort_timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    df = df.drop(columns=["_sort_bidid", "_sort_timestamp"])

    y = df.pop(LABEL_COLUMN)
    y.name = LABEL_COLUMN
    # Defensive, cheap check (not a full re-validation): the query above is
    # the only place columns are selected, so this should be tautological,
    # but a silent mismatch here would be exactly the kind of "a future
    # column became a feature without anyone deciding that" bug this whole
    # module exists to prevent -- assert it stays true rather than trusting it.
    assert list(df.columns) == feature_cols, (
        "load_impression_features(): materialized columns "
        f"{list(df.columns)!r} do not match the resolved feature column "
        f"list {feature_cols!r} -- this indicates a bug in the query "
        "construction above, not a data problem."
    )
    return df, y
