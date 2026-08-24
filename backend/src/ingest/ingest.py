"""
Raw iPinYou bz2 logs -> partitioned Parquet under data/processed/.

Single-command, reproducible entry point:

    backend\\.venv\\Scripts\\python.exe -m src.ingest.ingest
    backend\\.venv\\Scripts\\python.exe -m src.ingest.ingest --sample   # fast dev subset
    backend\\.venv\\Scripts\\python.exe -m src.ingest.ingest --season 3 --file-type imp

Run from the `backend/` directory (so `src` resolves as a package). No
random component is involved in ingestion itself (that only enters at the
train/val/test split and downsampling stage, deliberately NOT built here --
see README.md "Where splitting/downsampling would slot in"), so there is no
seed to fix for this stage; re-running is byte-for-byte idempotent given the
same raw inputs, and files are skipped unless --overwrite is passed.

Memory strategy: each raw file is read in fixed-size chunks via
pandas.read_csv(..., compression="bz2", chunksize=...), transformed
in-memory (transform.transform_chunk), and appended as a new Parquet
row-group via a single open pyarrow.parquet.ParquetWriter per file. Nothing
ever holds a whole file, let alone the whole dataset, in memory -- peak
memory per *file* is O(chunksize), independent of file size. This is why
season-2 bid files (250-625MB compressed, decompressing to several GB of
text) are safe to process on a 16GB machine.

Parallelism (--workers, default 1): raw bz2 decompression is single-threaded
and CPU-bound (Python's bz2 module, same library pandas uses under the
hood), and it dominates wall-clock time for the large season-2 bid files --
one 444MB file alone takes several minutes single-threaded. Since every raw
file maps to exactly one independent output file (see "Output layout" in
README.md), files are trivially parallelizable across a process pool with
zero shared state: each worker owns its own reader, its own transform
calls, and its own ParquetWriter, so nothing needs locking or merging.
Memory scales linearly with --workers (each worker independently holds
O(chunksize) at a time), so the flag exists rather than defaulting to
"use all cores": on a 16GB machine, --workers 4 with the default chunksize
comfortably fits; pushing workers higher without lowering --chunksize is
the user's tradeoff to make explicitly, not this script's default.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import paths, schema, transform

FILENAME_RE = re.compile(r"^(bid|imp|clk|conv)\.(\d{8})\.txt\.bz2$")


def discover_files(seasons: list[int], file_types: list[str]) -> list[dict]:
    """Enumerate raw files to ingest. Missing dates/types (e.g. season3 has
    only 5 conv dates against 10 bid dates) are simply absent from the glob
    result -- there is nothing to special-case, which is the point: the
    pipeline must not assume every date has every file type.
    """
    files = []
    for season in seasons:
        season_dir = paths.SEASON_DIRS[season]
        for fp in sorted(season_dir.glob("*.txt.bz2")):
            m = FILENAME_RE.match(fp.name)
            if not m:
                continue
            file_type, date_str = m.groups()
            if file_type not in file_types:
                continue
            files.append({
                "path": fp, "season": season, "file_type": file_type, "date": date_str,
            })
    return files


def _output_path(file_type: str, season: int, date_str: str) -> Path:
    date_fmt = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    # Partitioned by log type (top level -- these are structurally different
    # tables, not just a "date" split), then by season, then by date. Season
    # is its own partition level (not folded into date) because season2 and
    # season3 differ in schema drift (slotvisibility/slotformat encoding)
    # and in the bidprice-fixed-strategy assumption (season3 has documented
    # exceptions, see README.md) -- keeping season explicit in the path
    # makes it trivial to filter it out/in with a DuckDB/pyarrow predicate
    # without parsing dates.
    out_dir = (paths.PROCESSED_ROOT / paths.FILE_TYPE_NAMES[file_type]
               / f"season={season}" / f"date={date_fmt}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "part-0.parquet"


def ingest_file(meta: dict, chunksize: int, sample_rows: int | None,
                 overwrite: bool) -> dict:
    out_path = _output_path(meta["file_type"], meta["season"], meta["date"])
    if out_path.exists() and not overwrite:
        return {
            "path": str(meta["path"]), "out_path": str(out_path), "skipped": True,
            "rows_in": None, "rows_out": None,
        }

    file_type = meta["file_type"]
    names = schema.COLUMNS_BY_FILE_TYPE[file_type]
    out_schema = transform.output_schema(file_type)

    reader = pd.read_csv(
        meta["path"], sep="\t", header=None, names=names, dtype=str,
        compression="bz2", chunksize=chunksize,
        keep_default_na=False,   # we do our own null normalization (schema.py) --
                                  # pandas' default NA sniffing would also treat
                                  # things like the string "NA" (a real region
                                  # code prefix? no, but e.g. a real hashed id
                                  # that happens to collide) as missing, which
                                  # is exactly the kind of silent behavior we
                                  # want to avoid for a hashed-id column.
        quoting=csv.QUOTE_NONE,   # required: useragent contains raw commas/#/
                                  # quotes; the README explicitly warns about
                                  # this for R readers, same issue applies here.
        nrows=sample_rows,
    )

    tmp_path = out_path.with_suffix(".parquet.tmp")
    writer = None
    rows_in = 0
    rows_out = 0
    try:
        for chunk in reader:
            rows_in += len(chunk)

            if "logtype" in chunk.columns:
                expected = str(schema.EXPECTED_LOGTYPE[file_type])
                bad = chunk["logtype"] != expected
                if bad.any():
                    bad_values = chunk.loc[bad, "logtype"].unique().tolist()
                    raise ValueError(
                        f"{meta['path']}: expected logtype=={expected} for all "
                        f"rows (file type {file_type!r}) but found {bad_values}. "
                        "Refusing to silently relabel or drop -- this needs a "
                        "recorded decision (see known.data.bugs.txt for the "
                        "documented season-1 logtype bug; this file is not "
                        "season1, so this is unexpected)."
                    )

            table = transform.transform_chunk(chunk, file_type)
            assert table.schema.equals(out_schema), (
                f"{meta['path']}: chunk schema drifted from the declared "
                f"output schema -- refusing to write a Parquet file with an "
                f"inconsistent schema across row groups.\n"
                f"got: {table.schema}\nexpected: {out_schema}"
            )

            if writer is None:
                writer = pq.ParquetWriter(tmp_path, out_schema, compression="snappy")
            writer.write_table(table)
            rows_out += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if rows_in != rows_out:
        # Should be structurally impossible given the transform never
        # filters rows, but this is the row-count tripwire the project
        # standards call for -- if it ever fires, that's a real bug to fix,
        # not a warning to log past.
        tmp_path.unlink(missing_ok=True)
        raise AssertionError(
            f"{meta['path']}: rows_in={rows_in} != rows_out={rows_out}; "
            "a silent row drop occurred during transform."
        )

    tmp_path.replace(out_path)  # atomic on the same filesystem: no reader
                                 # ever sees a partially-written file at the
                                 # canonical path.
    return {
        "path": str(meta["path"]), "out_path": str(out_path), "skipped": False,
        "season": meta["season"], "file_type": file_type, "date": meta["date"],
        "rows_in": rows_in, "rows_out": rows_out,
    }


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, nargs="+", choices=[2, 3], default=[2, 3])
    parser.add_argument("--file-type", nargs="+", choices=list(paths.FILE_TYPES),
                         default=list(paths.FILE_TYPES))
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample", action="store_true",
                         help="Process only the first 20,000 rows of every "
                              "matched file, for fast (~seconds) dev iteration.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of raw files to process in parallel "
                              "(separate OS processes, one ParquetWriter "
                              "each -- see module docstring 'Parallelism' "
                              "for the memory tradeoff). Default 1 "
                              "(sequential, matches the documented O(chunksize) "
                              "memory bound exactly).")
    args = parser.parse_args(argv)

    sample_rows = 20_000 if args.sample else None

    files = discover_files(args.season, args.file_type)
    if not files:
        print("No matching raw files found.", file=sys.stderr)
        return 1

    # Largest files first when running in parallel, so a big season-2 bid
    # file starts immediately rather than being scheduled last and leaving
    # workers idle at the tail of the run (classic bin-packing heuristic).
    if args.workers > 1:
        files = sorted(files, key=lambda m: m["path"].stat().st_size, reverse=True)

    paths.METADATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Schema manifest, written every run (cheap, always current) so
    # `data/metadata/` documents the exact column layout/dtypes without a
    # human having to open schema.py -- required by the "metadata/ ...
    # schema" layout convention.
    schema_manifest = {
        "bid_columns": schema.BID_COLUMNS,
        "imp_clk_conv_columns": schema.IMP_CLK_CONV_COLUMNS,
        "columns_dropped_from_bid_log": sorted(schema._BID_LOG_DROPPED),
        "expected_logtype": schema.EXPECTED_LOGTYPE,
        "null_literals": sorted(schema.NULL_LITERALS),
        "empty_string_is_null_columns": sorted(schema.EMPTY_STRING_IS_NULL_COLUMNS),
        "feature_denylist": sorted(schema.FEATURE_DENYLIST),
        "feature_denylist_rationale": (
            "bidprice: fixed data-collection strategy per README (with "
            "season-3 exceptions, see validation_report.json "
            "bidprice_by_advertiser); bidid: row identifier, not "
            "predictive; payprice: this is the win-price label, not an "
            "input feature."
        ),
        "high_cardinality_hash_fields": schema.HIGH_CARDINALITY_HASH_FIELDS,
        "hash_buckets": schema.HASH_BUCKETS,
        "hash_seed": schema.HASH_SEED,
        "low_cardinality_category_fields": schema.LOW_CARDINALITY_CATEGORY_FIELDS,
        "output_schema_by_file_type": {
            ft: [f"{f.name}:{f.type}" for f in transform.output_schema(ft)]
            for ft in paths.FILE_TYPES
        },
    }
    (paths.METADATA_ROOT / "schema_manifest.json").write_text(
        json.dumps(schema_manifest, indent=2, default=str)
    )

    log_rows = []
    t0 = time.time()

    if args.workers <= 1:
        for i, meta in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {meta['path'].name} "
                  f"(season={meta['season']}, type={meta['file_type']})...", flush=True)
            result = ingest_file(meta, args.chunksize, sample_rows, args.overwrite)
            log_rows.append(result)
            if result["skipped"]:
                print("  skipped (output exists; use --overwrite to redo)")
            else:
                print(f"  rows_in={result['rows_in']}  rows_out={result['rows_out']}  "
                      f"-> {result['out_path']}")
    else:
        print(f"Ingesting {len(files)} files with {args.workers} worker processes...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(ingest_file, meta, args.chunksize, sample_rows, args.overwrite): meta
                for meta in files
            }
            for i, fut in enumerate(as_completed(futures), 1):
                meta = futures[fut]
                result = fut.result()  # re-raises any worker exception here,
                                        # in the main process -- a single bad
                                        # file still fails the whole run loudly
                                        # rather than being swallowed silently.
                log_rows.append(result)
                status = "skipped" if result["skipped"] else (
                    f"rows_in={result['rows_in']} rows_out={result['rows_out']}")
                print(f"[{i}/{len(files)}] {meta['path'].name} "
                      f"(season={meta['season']}, type={meta['file_type']}): {status}",
                      flush=True)

    elapsed = time.time() - t0
    suffix = "_sample" if args.sample else ""
    log_path = paths.METADATA_ROOT / f"ingestion_row_counts{suffix}.json"
    log_path.write_text(json.dumps(log_rows, indent=2, default=str))

    total_rows_in = sum(r["rows_in"] for r in log_rows if r["rows_in"] is not None)
    total_rows_out = sum(r["rows_out"] for r in log_rows if r["rows_out"] is not None)
    footprint = _dir_size_bytes(paths.PROCESSED_ROOT) if paths.PROCESSED_ROOT.exists() else 0

    print(f"\nDone in {elapsed:.1f}s. rows_in={total_rows_in} rows_out={total_rows_out}")
    print(f"data/processed footprint: {footprint / 1e9:.2f} GB")
    print(f"Row-count log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
