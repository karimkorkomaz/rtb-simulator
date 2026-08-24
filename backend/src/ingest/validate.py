"""
Post-ingestion validation over the Parquet output.

Everything here queries the Parquet files directly with DuckDB (no pandas
`.read_parquet()` full-load) -- DuckDB pushes projection/aggregation down to
the Parquet files on disk, so this scales to the full dataset without
materializing it. Run after `ingest.py`:

    backend\\.venv\\Scripts\\python.exe -m src.ingest.validate
    backend\\.venv\\Scripts\\python.exe -m src.ingest.validate --sample

This step deliberately does not "fix" anything it finds -- e.g. it will
report rows where payprice > bidprice (which should not happen in a
second-price auction) as a flagged count, not silently clip or drop them.
Per project rules, anomalies are surfaced with a recorded decision, not
silently handled.
"""
from __future__ import annotations

import argparse
import json

import duckdb

from . import paths

# Columns present on every table (bid/imp/clk/conv all have these).
COMMON_COLUMNS = [
    "bidid", "timestamp", "ipinyouid", "useragent", "IP", "region", "city",
    "adexchange", "domain", "url", "urlid", "slotid", "slotwidth",
    "slotheight", "slotvisibility", "slotformat", "slotprice", "creative",
    "bidprice", "advertiser", "usertag",
]
# imp/clk/conv-only columns.
WON_ONLY_COLUMNS = ["logtype", "payprice", "keypage"]


def _dataset_glob(file_type_dir: str, sample: bool) -> str:
    root = paths.PROCESSED_ROOT / file_type_dir
    return str(root / "**" / "*.parquet")


def validate_dataset(con: duckdb.DuckDBPyConnection, file_type_dir: str, sample: bool) -> dict:
    root = paths.PROCESSED_ROOT / file_type_dir
    if not root.exists():
        return {"file_type_dir": file_type_dir, "exists": False}

    glob_pattern = _dataset_glob(file_type_dir, sample)
    has_won_cols = file_type_dir != "bids"
    columns = COMMON_COLUMNS + (WON_ONLY_COLUMNS if has_won_cols else [])

    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}', hive_partitioning=1)"
    ).fetchone()[0]

    # Per-column null rate in one pass (DuckDB computes all aggregates in a
    # single scan of the Parquet files).
    null_exprs = ", ".join(
        f"SUM(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END) AS null_{i}"
        for i, c in enumerate(columns)
    )
    null_row = con.execute(
        f"SELECT {null_exprs} FROM read_parquet('{glob_pattern}', hive_partitioning=1)"
    ).fetchone()
    null_rates = {
        c: (null_row[i] / total_rows if total_rows else None)
        for i, c in enumerate(columns)
    }

    price_cols = ["slotprice", "bidprice"] + (["payprice"] if has_won_cols else [])
    price_stats = {}
    for col in price_cols:
        row = con.execute(f"""
            SELECT MIN("{col}"), MAX("{col}"),
                   approx_quantile("{col}", 0.5) AS p50,
                   approx_quantile("{col}", 0.95) AS p95,
                   AVG("{col}"),
                   SUM(CASE WHEN "{col}" < 0 THEN 1 ELSE 0 END) AS n_negative
            FROM read_parquet('{glob_pattern}', hive_partitioning=1)
        """).fetchone()
        price_stats[col] = {
            "min": row[0], "max": row[1], "p50": row[2], "p95": row[3],
            "mean": row[4], "n_negative": row[5],
        }

    anomalies = {}
    if has_won_cols:
        # In a second-price auction the winner should never pay more than
        # their own bid. Flag (do not fix) any violation.
        n_pay_gt_bid = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{glob_pattern}', hive_partitioning=1)
            WHERE "payprice" > "bidprice"
        """).fetchone()[0]
        anomalies["payprice_greater_than_bidprice"] = n_pay_gt_bid

    n_bad_dims = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{glob_pattern}', hive_partitioning=1)
        WHERE "slotwidth" <= 0 OR "slotheight" <= 0
    """).fetchone()[0]
    anomalies["nonpositive_slot_dimensions"] = n_bad_dims

    # bidprice-per-advertiser distribution -- this is the direct check for
    # the "fixed bidding strategy, with exceptions in season 3" caveat from
    # the README. A campaign with a single dominant bidprice value is
    # consistent with the fixed-strategy claim; a campaign with many
    # distinct bidprice values (n_distinct large relative to n rows) is a
    # season-3-style exception and should be flagged prominently.
    bidprice_by_advertiser = con.execute(f"""
        SELECT season, advertiser, COUNT(*) AS n,
               COUNT(DISTINCT "bidprice") AS n_distinct_bidprice,
               MIN("bidprice") AS min_bidprice, MAX("bidprice") AS max_bidprice
        FROM read_parquet('{glob_pattern}', hive_partitioning=1)
        GROUP BY season, advertiser
        ORDER BY season, advertiser
    """).fetchdf().to_dict(orient="records")

    return {
        "file_type_dir": file_type_dir,
        "exists": True,
        "total_rows": total_rows,
        "null_rates": null_rates,
        "price_stats": price_stats,
        "anomalies": anomalies,
        "bidprice_by_advertiser": bidprice_by_advertiser,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args(argv)

    con = duckdb.connect()
    report = {}
    for file_type_dir in ("bids", "impressions", "clicks", "conversions"):
        print(f"Validating {file_type_dir}...", flush=True)
        report[file_type_dir] = validate_dataset(con, file_type_dir, args.sample)

    paths.METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = "_sample" if args.sample else ""
    out_path = paths.METADATA_ROOT / f"validation_report{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nValidation report written to {out_path}")

    # Fail loudly (non-zero exit) if any hard anomaly was found, so this can
    # be wired into a re-run/CI check rather than requiring a human to
    # remember to open the JSON.
    hard_failure = False
    for ft, r in report.items():
        if not r.get("exists"):
            continue
        for k, v in r.get("anomalies", {}).items():
            if v:
                print(f"ANOMALY [{ft}] {k}: {v} rows")
                hard_failure = True
    return 1 if hard_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
