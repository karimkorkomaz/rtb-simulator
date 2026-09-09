"""
Phase 3 baseline runner: constant / random / linear-in-CTR bidding
strategies, replayed per advertiser (never pooled -- see `replay.py`'s
"SINGLE-ADVERTISER POOLS"), at multiple budget levels, against the
season-2 test-day auction pool (`date_range=("2013-06-12","2013-06-12")`,
`seasons=(2,)` -- the held-out test day, never train or val).

SCOPE AND ORDER: only the three literature baselines from
`docs/analysis/simulation-results.md`'s "Baselines, in this order" -- no
non-linear bidding, no pacing. Each of constant/random/linear-in-CTR is
swept across a data-driven parameter grid (see `build_sweep_grids()`) at
each of several budget levels (see `BUDGET_FRACTIONS`), for every
advertiser present in the season-2 test day.

WHY THE SWEEP GRIDS ARE DATA-DRIVEN, NOT ARBITRARY: an unconditioned
sweep (e.g. constant bids 1, 10, 100, 1000) risks landing entirely inside
one degenerate corner (never wins, or always wins) for a campaign whose
`payprice` distribution is nowhere near those numbers, producing an
uninformative table. Instead, `_advertiser_payprice_quantiles()` reads
the ACTUAL post-exclusion `payprice` distribution for the advertiser
(the same distribution documented in `docs/analysis/eda-findings.md`
S7), directly via DuckDB, and anchors every grid to it:

  * `constant_bid` amounts = the payprice deciles/quartiles themselves
    (so roughly `q`% of the pool should be winnable at the `q`-th
    quantile bid, by construction of what a quantile means).
  * `random_bid` ranges = `[low_quantile, high_quantile)` pairs spanning
    increasingly wide slices of the same distribution.
  * `linear_in_ctr_bid` bases = `quantile / mean(p_click)`, so the
    *average* bid across the pool lands near that same quantile even
    though any individual bid is scaled by that impression's own
    predicted CTR (see module docstring rationale in `strategies.py`).

IMPORTANT: this DuckDB read of `payprice` is a RUNNER-level, aggregate,
grid-design decision, computed independently of, and never passed into,
any `strategies.py` bid-producing function -- it never touches a
`BidTimeView`, and this module never reads `AuctionPool._payprice`
(the settlement-only field) for anything other than this one aggregate
statistic, entirely OUTSIDE `settle_auctions()`. This is a deliberate,
narrow exception to "read `_payprice` only in `settle_auctions()`",
justified because: (a) it is domain-level statistical design work an
analyst does before running an experiment, exactly analogous to how
`docs/analysis/eda-findings.md` studies the win-price distribution
before any strategy exists, and (b) no bidding DECISION (a specific bid
placed on a specific auction) is ever informed by a per-row payprice --
only fixed, pool-level anchor points chosen once, before any auction is
even loaded, are threaded through.

BUDGET LADDER: fractions of the advertiser's own total historical spend
on its own (post-exclusion) pool -- i.e. `sum(payprice)` over the exact
rows this engine can simulate, computed the same way as the quantiles
above. A fixed absolute budget (e.g. "10,000 fen for every advertiser")
would be meaningless: advertiser 1458's test-day pool alone settles over
30 million fen, while 3476's settles under 6.5 million -- a shared
absolute number would be trivially unconstrained for one and
immediately exhausted for the other. Expressing budget as a FRACTION of
each advertiser's own historical spend puts every advertiser's sweep on
the same relative footing (see `BUDGET_FRACTIONS`).

RANDOM-BID SEEDING: seeded by `(SEED, advertiser, range_index)` --
deliberately NOT including the budget level. A bidding policy's PROPOSED
bid for a given auction does not (and should not) depend on how much
budget happens to be left when this simulator is asked to also test a
different budget level -- `settle_auctions()`'s hard-stop budget
semantics are what varies with budget, not the underlying bid vector.
Reusing the exact same bid vector across every budget level for a given
`(advertiser, range)` isolates the effect of budget from the effect of
randomness in the draw, and also means the (comparatively expensive)
random draw happens once per `(advertiser, range)`, not once per
`(advertiser, range, budget)`.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.simulation.run_baselines
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from ..ingest import paths as ingest_paths
from . import predictions as preds_mod
from . import strategies
from .replay import AuctionPool, SimulationResult, load_auction_pool, settle_auctions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"

RESULTS_PARQUET_PATH = _METADATA_ROOT / "simulation_baseline_results_season2_test.parquet"
RUN_METADATA_PATH = _METADATA_ROOT / "simulation_baseline_run_season2_test.json"

SEASON = 2
DATE_RANGE: tuple[str, str] = ("2013-06-12", "2013-06-12")
SEED = 42

# At least four, per the project's "sweep budget levels" requirement --
# five here, spanning heavily-constrained to near-unconstrained relative
# to the advertiser's own historical spend on this pool (see module
# docstring "BUDGET LADDER").
BUDGET_FRACTIONS: list[float] = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]

# Anchor quantiles used for BOTH the constant-bid grid and the
# linear-in-CTR base grid -- see module docstring "WHY THE SWEEP GRIDS
# ARE DATA-DRIVEN".
QUANTILE_ANCHORS: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# (low_quantile, high_quantile) pairs for the random-bid range grid --
# increasingly wide, all anchored to the same observed distribution.
RANDOM_RANGE_ANCHOR_PAIRS: list[tuple[float, float]] = [
    (0.10, 0.50),
    (0.25, 0.75),
    (0.50, 0.90),
    (0.10, 0.95),
]

_HOURS_PER_DAY = 24


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Advertiser discovery
# ---------------------------------------------------------------------------


def discover_advertisers(
    *,
    seasons: tuple[int, ...] = (SEASON,),
    date_range: tuple[str, str] = DATE_RANGE,
    processed_root: Optional[Path] = None,
) -> list[str]:
    """All advertisers present in the given season/date range, sorted.
    A plain `DISTINCT advertiser` scan -- independent of the predictions
    parquet, so advertiser discovery does not implicitly depend on the
    CTR-modelling artifact having been generated for exactly this set.
    """
    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_glob = str((root / "impressions") / "**" / "*.parquet")
    season_filter = "season IN (" + ", ".join(str(int(s)) for s in seasons) + ")"
    start, end = date_range
    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT advertiser
        FROM read_parquet('{imp_glob}', hive_partitioning=1)
        WHERE {season_filter} AND date BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY advertiser
    """
    return [str(row[0]) for row in con.execute(query).fetchall()]


# ---------------------------------------------------------------------------
# Data-driven grid design (aggregate-only reads of payprice -- see module
# docstring for why this is not a payprice-guard violation)
# ---------------------------------------------------------------------------


def _advertiser_payprice_quantiles(
    advertiser: str,
    *,
    seasons: tuple[int, ...] = (SEASON,),
    date_range: tuple[str, str] = DATE_RANGE,
    processed_root: Optional[Path] = None,
    quantiles: list[float] = QUANTILE_ANCHORS,
) -> dict[float, float]:
    """Quantiles of `payprice` over the SAME post-exclusion population
    `load_auction_pool()` simulates (`payprice > bidprice` and
    `payprice == 0` both excluded, matching `replay._exclude_unsimulatable_rows`
    exactly) -- an aggregate-statistics-only read, see module docstring.
    """
    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_glob = str((root / "impressions") / "**" / "*.parquet")
    season_filter = "season IN (" + ", ".join(str(int(s)) for s in seasons) + ")"
    start, end = date_range
    con = duckdb.connect()
    quantile_select = ", ".join(
        f"quantile_cont(payprice, {q}) AS q_{int(q * 1000)}" for q in quantiles
    )
    query = f"""
        SELECT {quantile_select}
        FROM read_parquet('{imp_glob}', hive_partitioning=1)
        WHERE {season_filter}
          AND date BETWEEN DATE '{start}' AND DATE '{end}'
          AND advertiser = '{advertiser}'
          AND NOT (payprice > bidprice)
          AND NOT (payprice = 0)
    """
    row = con.execute(query).fetchone()
    return {q: float(v) for q, v in zip(quantiles, row)}


def _advertiser_total_historical_spend(
    advertiser: str,
    *,
    seasons: tuple[int, ...] = (SEASON,),
    date_range: tuple[str, str] = DATE_RANGE,
    processed_root: Optional[Path] = None,
) -> float:
    """`sum(payprice)` over the same post-exclusion population -- what
    iPinYou itself actually spent to win exactly this pool. Used ONLY to
    scale the budget ladder (see module docstring "BUDGET LADDER"), never
    as a per-row signal.
    """
    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_glob = str((root / "impressions") / "**" / "*.parquet")
    season_filter = "season IN (" + ", ".join(str(int(s)) for s in seasons) + ")"
    start, end = date_range
    con = duckdb.connect()
    query = f"""
        SELECT SUM(payprice)
        FROM read_parquet('{imp_glob}', hive_partitioning=1)
        WHERE {season_filter}
          AND date BETWEEN DATE '{start}' AND DATE '{end}'
          AND advertiser = '{advertiser}'
          AND NOT (payprice > bidprice)
          AND NOT (payprice = 0)
    """
    (total,) = con.execute(query).fetchone()
    return float(total)


def build_sweep_grids(quantiles: dict[float, float], mean_p_click: float) -> dict:
    """Assemble the three strategies' parameter grids from the advertiser's
    own `payprice` quantiles (see module docstring). Returns a dict with
    keys `constant_amounts`, `random_ranges`, `linear_bases`, each a list
    of (value_or_tuple, provenance_label) pairs for traceability.
    """
    constant_amounts = [(quantiles[q], f"q{int(q * 100)}") for q in QUANTILE_ANCHORS]
    random_ranges = [
        (quantiles[lo], quantiles[hi], f"q{int(lo * 100)}-q{int(hi * 100)}")
        for lo, hi in RANDOM_RANGE_ANCHOR_PAIRS
    ]
    linear_bases = [
        (quantiles[q] / mean_p_click, f"q{int(q * 100)}") for q in QUANTILE_ANCHORS
    ]
    return {
        "constant_amounts": constant_amounts,
        "random_ranges": random_ranges,
        "linear_bases": linear_bases,
    }


# ---------------------------------------------------------------------------
# Spend trajectory summary (compact, for the report -- not the full
# per-row array, which would be prohibitively large across 400 runs)
# ---------------------------------------------------------------------------


def _hourly_spend_fractions(pool: AuctionPool, result: SimulationResult) -> list[float]:
    """Cumulative spend as a fraction of `result.budget`, sampled at the
    END of each hour-of-day 0..23 in `pool.timestamp` (local Beijing time,
    per the corpus convention). One compact 24-length summary per run,
    used for the report's "spend trajectory over the period" requirement
    without persisting a full per-row array for every one of the ~400
    runs in this sweep.
    """
    if pool.n_rows == 0 or result.budget <= 0:
        return [0.0] * _HOURS_PER_DAY
    hours = pool.timestamp.dt.hour.to_numpy()
    traj = result.spend_trajectory.to_numpy()
    out = []
    last_seen = 0.0
    for h in range(_HOURS_PER_DAY):
        idx = np.where(hours == h)[0]
        if len(idx) == 0:
            out.append(last_seen)
            continue
        last_seen = float(traj[idx[-1]] / result.budget)
        out.append(last_seen)
    return out


# ---------------------------------------------------------------------------
# Per-run record assembly
# ---------------------------------------------------------------------------


def _record(
    *,
    advertiser: str,
    strategy: str,
    params: dict,
    budget_fraction: float,
    pool: AuctionPool,
    result: SimulationResult,
) -> dict:
    return {
        "advertiser": advertiser,
        "strategy": strategy,
        "params": json.dumps(params, sort_keys=True),
        "budget_fraction_of_historical_spend": budget_fraction,
        "budget": result.budget,
        "impressions_pool": result.impressions_pool,
        "impressions_bid_on": result.impressions_bid_on,
        "impressions_won": result.impressions_won,
        "win_rate_bid_on": result.win_rate_bid_on,
        "win_rate_pool": result.win_rate_pool,
        "clicks_won": result.clicks_won,
        "total_spend": result.total_spend,
        "effective_cpc": result.effective_cpc,
        "budget_exhausted": result.budget_exhausted,
        "exhaustion_index": result.exhaustion_index,
        "exhaustion_timestamp": (
            result.exhaustion_timestamp.isoformat()
            if result.exhaustion_timestamp is not None
            else None
        ),
        "excluded_payprice_gt_bidprice_count": result.excluded_payprice_gt_bidprice_count,
        "excluded_payprice_gt_bidprice_share": result.excluded_payprice_gt_bidprice_share,
        "excluded_payprice_zero_count": result.excluded_payprice_zero_count,
        "excluded_payprice_zero_share": result.excluded_payprice_zero_share,
        "hourly_spend_fraction": _hourly_spend_fractions(pool, result),
    }


# ---------------------------------------------------------------------------
# Per-advertiser run
# ---------------------------------------------------------------------------


def run_advertiser(
    advertiser: str,
    corrected: preds_mod.CorrectedPredictions,
    *,
    processed_root: Optional[Path] = None,
) -> tuple[list[dict], dict]:
    """Run every (strategy x parameter x budget) combination for one
    advertiser. Loads the `AuctionPool` exactly ONCE and reuses it across
    every strategy/parameter/budget combination (see module's runtime
    requirement) -- `settle_auctions()` is a cheap, vectorised call per
    combination on top of the one load.

    Returns `(records, design)`: `records` is one dict per
    (strategy, parameter, budget) combination (see `_record()`); `design`
    captures the grid/ladder actually used for this advertiser, for the
    run metadata sidecar.
    """
    pool = load_auction_pool(advertiser, DATE_RANGE, seasons=(SEASON,), processed_root=processed_root)
    view = pool.bid_time_view()
    p_click = preds_mod.p_click_for_pool(pool, corrected.predictions_with_bidid)
    mean_p_click = float(np.mean(p_click)) if pool.n_rows else 0.0

    quantiles = _advertiser_payprice_quantiles(advertiser, processed_root=processed_root)
    total_historical_spend = _advertiser_total_historical_spend(advertiser, processed_root=processed_root)
    budgets = [(frac, frac * total_historical_spend) for frac in BUDGET_FRACTIONS]

    grids = build_sweep_grids(quantiles, mean_p_click)

    records: list[dict] = []

    # Bid vectors are computed ONCE per (strategy, parameter) -- reused
    # across every budget level (see module docstring "RANDOM-BID SEEDING"
    # for why this is the methodologically correct choice, not just an
    # optimisation).
    constant_bid_vectors = {
        label: strategies.constant_bid(view, amount=amount)
        for amount, label in grids["constant_amounts"]
    }
    random_bid_vectors = {}
    for idx, (low, high, label) in enumerate(grids["random_ranges"]):
        rng = np.random.default_rng([SEED, int(advertiser), idx])
        random_bid_vectors[label] = (strategies.random_bid(view, low=low, high=high, rng=rng), low, high)
    linear_bid_vectors = {
        label: strategies.linear_in_ctr_bid(view, base=base, p_click=p_click)
        for base, label in grids["linear_bases"]
    }

    for budget_fraction, budget in budgets:
        for amount, label in grids["constant_amounts"]:
            bids = constant_bid_vectors[label]
            result = settle_auctions(pool, bids, budget)
            records.append(
                _record(
                    advertiser=advertiser,
                    strategy="constant",
                    params={"amount": amount, "quantile_label": label},
                    budget_fraction=budget_fraction,
                    pool=pool,
                    result=result,
                )
            )
        for idx, (low, high, label) in enumerate(grids["random_ranges"]):
            bids, low_v, high_v = random_bid_vectors[label]
            result = settle_auctions(pool, bids, budget)
            records.append(
                _record(
                    advertiser=advertiser,
                    strategy="random",
                    params={"low": low_v, "high": high_v, "range_label": label, "seed": [SEED, int(advertiser), idx]},
                    budget_fraction=budget_fraction,
                    pool=pool,
                    result=result,
                )
            )
        for base, label in grids["linear_bases"]:
            bids = linear_bid_vectors[label]
            result = settle_auctions(pool, bids, budget)
            records.append(
                _record(
                    advertiser=advertiser,
                    strategy="linear_ctr",
                    params={"base": base, "quantile_label": label},
                    budget_fraction=budget_fraction,
                    pool=pool,
                    result=result,
                )
            )

    design = {
        "advertiser": advertiser,
        "n_rows_pool": pool.n_rows,
        "excluded_payprice_gt_bidprice_count": pool.excluded_payprice_gt_bidprice_count,
        "excluded_payprice_gt_bidprice_share": pool.excluded_payprice_gt_bidprice_share,
        "excluded_payprice_zero_count": pool.excluded_payprice_zero_count,
        "excluded_payprice_zero_share": pool.excluded_payprice_zero_share,
        "mean_p_click": mean_p_click,
        "payprice_quantiles": {str(q): v for q, v in quantiles.items()},
        "total_historical_spend": total_historical_spend,
        "budgets": [{"fraction": f, "value": b} for f, b in budgets],
        "constant_amounts": [{"amount": a, "label": lbl} for a, lbl in grids["constant_amounts"]],
        "random_ranges": [{"low": lo, "high": hi, "label": lbl} for lo, hi, lbl in grids["random_ranges"]],
        "linear_bases": [{"base": b, "label": lbl} for b, lbl in grids["linear_bases"]],
    }
    return records, design


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print("[run_baselines] loading + verifying corrected predictions (correction-order asserted) ...")
    corrected = preds_mod.load_corrected_predictions(date_range=DATE_RANGE, seasons=(SEASON,))
    print(
        f"[run_baselines] OK: sampling_rate={corrected.sampling_rate}, "
        f"calibrator_fit_split={corrected.calibrator_fit_split!r}"
    )

    advertisers = discover_advertisers()
    print(f"[run_baselines] advertisers in season {SEASON} test day {DATE_RANGE}: {advertisers}")

    all_records: list[dict] = []
    designs: dict[str, dict] = {}
    for advertiser in advertisers:
        print(f"[run_baselines] running advertiser {advertiser} ...")
        records, design = run_advertiser(advertiser, corrected)
        all_records.extend(records)
        designs[advertiser] = design
        print(f"[run_baselines]   {len(records)} (strategy x parameter x budget) runs")

    results_df = pd.DataFrame.from_records(all_records)
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(RESULTS_PARQUET_PATH, index=False)
    print(f"[run_baselines] wrote {len(results_df):,} rows -> {RESULTS_PARQUET_PATH}")

    run_metadata = {
        "generated_at": _now_iso(),
        "season": SEASON,
        "date_range": list(DATE_RANGE),
        "seed": SEED,
        "budget_fractions": BUDGET_FRACTIONS,
        "quantile_anchors": QUANTILE_ANCHORS,
        "random_range_anchor_pairs": RANDOM_RANGE_ANCHOR_PAIRS,
        "advertisers": advertisers,
        "predictions": {
            "source_predictions_path": corrected.source_predictions_path,
            "source_calibrator_path": corrected.source_calibrator_path,
            "sampling_rate": corrected.sampling_rate,
            "calibrator_fit_split": corrected.calibrator_fit_split,
        },
        "per_advertiser_design": designs,
        "results_parquet": str(RESULTS_PARQUET_PATH),
        "n_runs": len(all_records),
    }
    with open(RUN_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"[run_baselines] wrote run metadata -> {RUN_METADATA_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
