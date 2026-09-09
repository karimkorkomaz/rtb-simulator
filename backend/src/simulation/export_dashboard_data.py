"""
Phase 4 data export: turns the persisted Phase 3 sweep
(`simulation_baseline_results_season2_test.parquet`, 400 rows =
5 advertisers x 5 budgets x 16 (strategy, parameter) combinations) into
ONE static JSON file the React dashboard imports directly -- no fetch, no
server, no client-side recomputation (see `docs/analysis/simulation-results.md`
and the Phase 4 task brief).

SCOPE: this module only reads already-computed results and reshapes them.
It never re-runs `settle_auctions()`, never re-scores a CTR model, and
never touches `payprice`/`bidprice` -- all of that is Phase 3's job
(`run_baselines.py`, `replay.py`, `strategies.py`), out of scope here.

FULL-SWEEP EXPORT (current shape): every one of the 400 raw
(advertiser, budget, strategy, PARAMETER) rows is exported -- nothing is
collapsed or dropped. Within each (advertiser, budget) "cell",
`strategies.<name>` is an ORDERED ARRAY of that strategy's sweep points
(6 for `constant`, 4 for `random`, 6 for `linear_ctr`), ordered by the
sweep parameter ascending (see `STRATEGY_LABEL_ORDER` below), so a line
chart over the array reads left-to-right without the caller needing to
re-sort. Each point additionally carries `is_best_by_clicks`, marking
whichever ONE point per (advertiser, budget, strategy) has the highest
`clicks_won` in that strategy's own sweep -- exactly the same selection
rule and tie-break (pandas `groupby(...).idxmax()`, which resolves ties
by first occurrence in the sweep's own grid order) used by an earlier
version of this script that exported ONLY that point, and verified,
programmatically, to reproduce all 75 rows of
`docs/analysis/simulation-results.md` S4.2's headline table exactly. The
dashboard can therefore render both the full parameter curve and the
"headline" peak from this one dataset.

This IS a transformation of the raw data (reshaping + an ordering + a
derived boolean flag, no rows dropped) -- per the task brief's honesty
constraint, it is done here, in a reviewable, re-runnable script, not
silently inside a React component.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.simulation.export_dashboard_data

AMBIGUITIES / DECISIONS A REVIEWER SHOULD CHECK (flagged, not buried):

1. **Which win-rate column.** `win_rate_pool` (not `win_rate_bid_on`) is
   exported for cross-strategy comparison, per `replay.SimulationResult`'s
   own field comment and `simulation-results.md` S4.2's explicit
   justification: it shares the same full-pool denominator across every
   strategy, so it stays comparable even once a strategy exhausts its
   budget early. `win_rate_bid_on` is NOT exported at all (out of scope
   per the task brief's four required metrics).
2. **Units.** All monetary figures (`budget`, `total_spend`,
   `effective_cpc`) are RMB fen (1/100 CNY yuan) -- the raw iPinYou
   collection unit, per `ingest/schema.py`'s column comments
   (`payprice`, `slotprice`, `bidprice` are all documented "RMB fen").
   No currency conversion is applied anywhere in this pipeline; this
   script does not introduce one either.
3. **Full sweep, not best-of-sweep.** Every raw (advertiser, budget,
   strategy, parameter) row is exported -- 400 leaf records total, not
   collapsed to one per strategy. `is_best_by_clicks` marks the same
   "best" point an earlier version of this script exported exclusively
   (see module docstring above), so the headline table and a full
   parameter-sensitivity chart can both be built from one file.
4. **Sweep-point ordering.** `constant` and `linear_ctr` points are
   ordered by their own numeric parameter (`amount` / `base`) ascending,
   which is also exactly the quantile order (q10 < q25 < ... < q95) by
   construction (`run_baselines.QUANTILE_ANCHORS`). `random` points do
   NOT have a single scalar parameter (each is a `[low, high)` pair, and
   neither `low` nor the range width is monotonic across
   `run_baselines.RANDOM_RANGE_ANCHOR_PAIRS` -- e.g. the widest range,
   q10-q95, has the SAME low end as the narrowest, q10-q50), so `random`
   points are instead ordered by the fixed grid-definition order those
   pairs were swept in (`RANDOM_RANGE_LABEL_ORDER` below, copied from
   `run_baselines.RANDOM_RANGE_ANCHOR_PAIRS`) -- "increasingly wide" per
   that module's own docstring, not a numeric axis. `meta.sweep_parameters`
   records this explicitly per strategy so the UI does not have to
   guess what its x-axis means.
5. **Exhaustion time.** Exported as an explicit
   `{"exhausted": bool, "timestamp": str | null}` pair (never a bare
   nullable timestamp) precisely so "never exhausted" cannot be confused
   with "exhaustion timestamp missing/unknown" -- both would otherwise
   serialize to `null`. `timestamp`, when present, is
   `SimulationResult.exhaustion_timestamp` verbatim (ISO 8601, local
   Beijing corpus time, `2013-06-12`), never inferred from spend alone.
6. **Spend trajectory resolution.** The persisted parquet does NOT carry
   a per-auction `spend_trajectory` (see `run_baselines.py`'s
   `_hourly_spend_fractions()` docstring: a full per-row array "would be
   prohibitively large across 400 runs" and was deliberately never
   persisted). The finest-grained trajectory that exists is
   `hourly_spend_fraction`: 24 values, one per hour-of-day 0-23, each the
   CUMULATIVE spend as a FRACTION of that run's budget at the end of that
   hour. This script converts fraction -> RMB fen by multiplying by the
   run's `budget` (rounded to 2 decimal places) so the chart can plot an
   actual currency axis instead of a unitless fraction. No further
   downsampling is applied or possible -- 24 points is already the
   persisted resolution; this is recorded in `meta.spend_trajectory`
   rather than left implicit. Now exported on ALL 400 leaf records
   (previously only the 75 best-of-sweep ones) -- this is the single
   largest driver of the export's size; see the module-level size report
   printed by `main()`.
7. **NaN/Infinity handling.** `effective_cpc` is `None` (not `NaN`/`inf`)
   whenever `clicks_won == 0` -- inherited directly from
   `replay.settle_auctions()`'s own contract (see its docstring / the
   `SimulationResult.effective_cpc` field comment). No run in the full
   400-row sweep actually needs the `None` case (S4.1 of
   `simulation-results.md` establishes `clicks_won` ranges 1-346 across
   the full sweep, never 0), but the guard is applied unconditionally
   rather than assumed. `json.dumps(..., allow_nan=False)` is used so any
   future NaN/Infinity would raise loudly at export time instead of
   silently producing invalid JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

RESULTS_PARQUET_PATH = _METADATA_ROOT / "simulation_baseline_results_season2_test.parquet"
RUN_METADATA_PATH = _METADATA_ROOT / "simulation_baseline_run_season2_test.json"
OUTPUT_JSON_PATH = _FRONTEND_SRC / "simulationData.json"

STRATEGIES: list[str] = ["constant", "random", "linear_ctr"]

# Fraction -> display label. Matches `run_baselines.BUDGET_FRACTIONS`
# exactly; kept as an explicit lookup (not computed via
# `fractions.Fraction`) so the label is exactly the human-readable form
# used throughout `simulation-results.md` ("1/32", not "0.03125" or
# "1/32.0000001" from float round-trip noise).
_BUDGET_FRACTION_LABELS: dict[float, str] = {
    1 / 32: "1/32",
    1 / 16: "1/16",
    1 / 8: "1/8",
    1 / 4: "1/4",
    1 / 2: "1/2",
}

_HOURS_PER_DAY = 24

# Canonical sweep-point ordering per strategy -- see module docstring
# point 4. `constant`/`linear_ctr` labels double as a numeric-ascending
# order (q10 < q25 < ... < q95 in the underlying amount/base value, by
# construction of `run_baselines.QUANTILE_ANCHORS`); `random`'s labels
# are the fixed grid-definition order copied from
# `run_baselines.RANDOM_RANGE_ANCHOR_PAIRS` (NOT a numeric axis -- see
# point 4 for why neither `low` nor range width is monotonic there).
QUANTILE_LABEL_ORDER: list[str] = ["q10", "q25", "q50", "q75", "q90", "q95"]
RANDOM_RANGE_LABEL_ORDER: list[str] = ["q10-q50", "q25-q75", "q50-q90", "q10-q95"]

STRATEGY_LABEL_FIELD: dict[str, str] = {
    "constant": "quantile_label",
    "random": "range_label",
    "linear_ctr": "quantile_label",
}
STRATEGY_LABEL_ORDER: dict[str, list[str]] = {
    "constant": QUANTILE_LABEL_ORDER,
    "random": RANDOM_RANGE_LABEL_ORDER,
    "linear_ctr": QUANTILE_LABEL_ORDER,
}
STRATEGY_PARAMETER_NAME: dict[str, str] = {
    "constant": "amount",
    "random": "range",
    "linear_ctr": "base",
}


def _budget_label(fraction: float) -> str:
    for known_fraction, label in _BUDGET_FRACTION_LABELS.items():
        if abs(fraction - known_fraction) < 1e-9:
            return label
    raise ValueError(
        f"_budget_label(): {fraction!r} does not match any known budget "
        f"fraction {sorted(_BUDGET_FRACTION_LABELS)!r} -- refusing to "
        "invent a label for an unrecognised budget level."
    )


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    """Round a float to `ndigits`, passing through `None`, and refusing to
    silently emit NaN/Infinity (raises instead) -- see module docstring
    point 7."""
    if value is None:
        return None
    f = float(value)
    if not np.isfinite(f):
        raise ValueError(f"_round_or_none(): non-finite value {f!r} would corrupt the JSON export.")
    return round(f, ndigits)


def _spend_trajectory_fen(hourly_spend_fraction: list[float], budget: float) -> list[float]:
    """`hourly_spend_fraction` (cumulative spend / budget, one value per
    hour-of-day 0-23) -> cumulative spend in RMB fen, rounded to 2
    decimal places -- see module docstring point 6."""
    return [round(float(frac) * float(budget), 2) for frac in hourly_spend_fraction]


def _param_label(strategy: str, params: dict) -> str:
    field = STRATEGY_LABEL_FIELD[strategy]
    if field not in params:
        raise ValueError(f"_param_label(): strategy={strategy!r} params={params!r} missing {field!r}.")
    return params[field]


def _param_sort_key(strategy: str, params: dict) -> int:
    label = _param_label(strategy, params)
    order = STRATEGY_LABEL_ORDER[strategy]
    if label not in order:
        raise ValueError(
            f"_param_sort_key(): strategy={strategy!r} label={label!r} not in expected "
            f"order {order!r} -- the sweep grid has changed; update STRATEGY_LABEL_ORDER."
        )
    return order.index(label)


def _best_indices(df: pd.DataFrame) -> set:
    """Index labels of the highest-`clicks_won` row within each
    (advertiser, budget_fraction, strategy) group -- the exact same
    selection rule and tie-break (pandas `idxmax`: first occurrence in
    original/grid order on ties) as the earlier best-of-sweep-only export,
    now used only to set `is_best_by_clicks` rather than to filter rows.
    """
    group_cols = ["advertiser", "budget_fraction_of_historical_spend", "strategy"]
    idx = df.groupby(group_cols)["clicks_won"].idxmax()
    expected_n = df["advertiser"].nunique() * df["budget_fraction_of_historical_spend"].nunique() * len(STRATEGIES)
    if len(idx) != expected_n:
        raise ValueError(
            f"_best_indices(): expected {expected_n} groups (advertisers x budgets x "
            f"strategies), got {len(idx)} -- the sweep grid is not fully rectangular; "
            "investigate before exporting."
        )
    return set(idx.tolist())


def _sweep_point_record(row: pd.Series, *, is_best: bool) -> dict:
    params = json.loads(row["params"])
    exhausted = bool(row["budget_exhausted"])
    exhaustion_timestamp = row["exhaustion_timestamp"]
    # Parquet round-trips a missing timestamp as NaN (float) or None
    # depending on pandas' string-column inference -- normalise both to
    # None explicitly rather than letting either leak into the JSON.
    if exhaustion_timestamp is None or (isinstance(exhaustion_timestamp, float) and pd.isna(exhaustion_timestamp)):
        exhaustion_timestamp = None
    if exhausted and exhaustion_timestamp is None:
        raise ValueError("_sweep_point_record(): budget_exhausted=True but exhaustion_timestamp is missing.")
    if not exhausted and exhaustion_timestamp is not None:
        raise ValueError("_sweep_point_record(): budget_exhausted=False but exhaustion_timestamp is present.")

    clicks_won = int(row["clicks_won"])
    effective_cpc = row["effective_cpc"]
    if clicks_won == 0:
        # Contract per replay.SimulationResult.effective_cpc: None, never
        # NaN/inf, when clicks_won == 0. No run in the full 400-row sweep
        # actually hits this (see module docstring point 7), but the
        # guard is applied unconditionally rather than assumed.
        effective_cpc = None
    elif isinstance(effective_cpc, float) and not np.isfinite(effective_cpc):
        raise ValueError(f"_sweep_point_record(): non-finite effective_cpc with clicks_won={clicks_won}.")

    hourly_fraction = list(row["hourly_spend_fraction"])
    if len(hourly_fraction) != _HOURS_PER_DAY:
        raise ValueError(
            f"_sweep_point_record(): expected {_HOURS_PER_DAY} hourly_spend_fraction points, "
            f"got {len(hourly_fraction)}."
        )

    return {
        "selected_param": params,
        "is_best_by_clicks": bool(is_best),
        "clicks_won": clicks_won,
        "win_rate_pool": _round_or_none(row["win_rate_pool"], 4),
        "spend_fen": _round_or_none(row["total_spend"], 2),
        "effective_cpc_fen": _round_or_none(effective_cpc, 2),
        "exhaustion": {
            "exhausted": exhausted,
            "timestamp": exhaustion_timestamp,
        },
        "spend_trajectory_fen": _spend_trajectory_fen(hourly_fraction, float(row["budget"])),
    }


def build_export(df: pd.DataFrame, run_metadata: dict) -> dict:
    best_idx = _best_indices(df)

    advertisers = sorted(df["advertiser"].unique().tolist(), key=int)
    budget_fractions = sorted(df["budget_fraction_of_historical_spend"].unique().tolist())

    n_points_per_strategy = {s: len(STRATEGY_LABEL_ORDER[s]) for s in STRATEGIES}

    cells: list[dict] = []
    for advertiser in advertisers:
        for budget_fraction in budget_fractions:
            cell_rows = df[
                (df["advertiser"] == advertiser)
                & (df["budget_fraction_of_historical_spend"] == budget_fraction)
            ]
            budget_values = cell_rows["budget"].unique()
            if len(budget_values) != 1:
                raise ValueError(
                    f"build_export(): advertiser={advertiser!r} budget_fraction={budget_fraction!r} "
                    f"has {len(budget_values)} distinct budget (fen) values across strategies, expected 1."
                )

            strategies_out: dict[str, list[dict]] = {}
            for strategy in STRATEGIES:
                strat_rows = cell_rows[cell_rows["strategy"] == strategy]
                expected_n = n_points_per_strategy[strategy]
                if len(strat_rows) != expected_n:
                    raise ValueError(
                        f"build_export(): advertiser={advertiser!r} budget_fraction={budget_fraction!r} "
                        f"strategy={strategy!r}: expected {expected_n} sweep points, got {len(strat_rows)}."
                    )
                # Order ascending by the strategy's own sweep parameter --
                # see module docstring point 4 / STRATEGY_LABEL_ORDER.
                ordered = sorted(
                    strat_rows.itertuples(index=True),
                    key=lambda t: _param_sort_key(strategy, json.loads(t.params)),
                )
                seen_labels = set()
                points = []
                for t in ordered:
                    row = strat_rows.loc[t.Index]
                    label = _param_label(strategy, json.loads(row["params"]))
                    if label in seen_labels:
                        raise ValueError(
                            f"build_export(): duplicate sweep-point label {label!r} for "
                            f"advertiser={advertiser!r} budget_fraction={budget_fraction!r} "
                            f"strategy={strategy!r}."
                        )
                    seen_labels.add(label)
                    points.append(_sweep_point_record(row, is_best=(t.Index in best_idx)))
                if seen_labels != set(STRATEGY_LABEL_ORDER[strategy]):
                    raise ValueError(
                        f"build_export(): advertiser={advertiser!r} budget_fraction={budget_fraction!r} "
                        f"strategy={strategy!r}: sweep labels {sorted(seen_labels)!r} do not match "
                        f"expected {sorted(STRATEGY_LABEL_ORDER[strategy])!r}."
                    )
                n_best = sum(1 for p in points if p["is_best_by_clicks"])
                if n_best != 1:
                    raise ValueError(
                        f"build_export(): advertiser={advertiser!r} budget_fraction={budget_fraction!r} "
                        f"strategy={strategy!r}: expected exactly 1 is_best_by_clicks point, got {n_best}."
                    )
                strategies_out[strategy] = points

            cells.append(
                {
                    "advertiser": advertiser,
                    "budget_fraction": budget_fraction,
                    "budget_label": _budget_label(budget_fraction),
                    "budget_fen": _round_or_none(float(budget_values[0]), 2),
                    "strategies": strategies_out,
                }
            )

    n_leaf_records = sum(
        len(points) for cell in cells for points in cell["strategies"].values()
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    meta = {
        "generated_at": generated_at,
        "generator_script": "backend/src/simulation/export_dashboard_data.py",
        "source_files": {
            "results_parquet": "backend/data/metadata/simulation_baseline_results_season2_test.parquet",
            "run_metadata_json": "backend/data/metadata/simulation_baseline_run_season2_test.json",
        },
        "season": run_metadata.get("season"),
        "date_range": run_metadata.get("date_range"),
        "advertisers": advertisers,
        "budget_fractions": budget_fractions,
        "budget_labels": {str(f): _budget_label(f) for f in budget_fractions},
        "strategies": STRATEGIES,
        "n_advertisers": len(advertisers),
        "n_budgets": len(budget_fractions),
        "n_strategies": len(STRATEGIES),
        "n_cells": len(cells),
        "n_points_per_strategy": n_points_per_strategy,
        "n_leaf_records": n_leaf_records,
        "n_raw_sweep_runs": int(len(df)),
        "units": {
            "spend_fen": "RMB fen (1/100 CNY yuan) -- the raw iPinYou collection unit; no currency conversion applied anywhere in this pipeline.",
            "budget_fen": "RMB fen -- a fraction of the advertiser's own historical spend on its post-exclusion test-day pool; see run_metadata per-advertiser design.",
            "effective_cpc_fen": "RMB fen per click; null (never NaN/Infinity) when clicks_won == 0.",
            "win_rate_pool": "fraction in [0, 1]: impressions_won / impressions_pool (full-pool denominator, comparable across strategies -- see selection_methodology).",
            "clicks_won": "count of clicks on impressions this strategy actually won, over the 2013-06-12 season-2 test day.",
            "exhaustion.timestamp": "ISO 8601 local Beijing corpus time on 2013-06-12; null iff exhaustion.exhausted is false.",
            "spend_trajectory_fen": "cumulative spend in RMB fen at the end of each hour-of-day 0-23 (24 points); derived from the persisted hourly_spend_fraction * budget -- see spend_trajectory block below.",
        },
        "selection_methodology": (
            "The FULL parameter sweep is exported: every (advertiser, "
            "budget, strategy, parameter) combination in the raw 400-row "
            "results parquet appears as one point in `strategies.<name>` "
            "(6 constant amounts / 4 random ranges / 6 linear-in-CTR "
            "bases, all anchored to the advertiser's own payprice "
            "quantiles -- see run_baselines.py). Within each "
            "(advertiser, budget, strategy) group, exactly ONE point is "
            "flagged `is_best_by_clicks: true` -- the parameter with the "
            "highest `clicks_won` in that strategy's own sweep, selected "
            "via the same rule (pandas groupby(...).idxmax(), which "
            "breaks ties by first occurrence in the sweep's own grid "
            "order) used previously to build "
            "docs/analysis/simulation-results.md S4.2's headline table; "
            "this has been verified programmatically to reproduce all 75 "
            "rows of that table exactly. A dashboard can render the full "
            "parameter curve (all points, ordered -- see "
            "sweep_parameters below) and mark or isolate the peak "
            "(`is_best_by_clicks`) from this one dataset."
        ),
        "caveat_post_hoc_selection": (
            "Sweep parameters are selected on the SAME test day being "
            "reported (no held-out validation-day pool was used to pick "
            "each strategy's parameter in advance) -- absolute clicks-won "
            "and effective-CPC figures are therefore an UPPER BOUND on "
            "what a bidder choosing its parameter ahead of time would "
            "achieve. This applies to the `is_best_by_clicks` point "
            "specifically (it is the post hoc best of that day's sweep); "
            "the same oracle advantage is given to all three strategies, "
            "so the cross-strategy ranking is not an artifact of it -- "
            "only the absolute levels are optimistic. See "
            "docs/analysis/simulation-results.md S4.2 and S5."
        ),
        "sweep_parameters": {
            "constant": {
                "parameter_name": "amount",
                "parameter_units": "RMB fen (flat bid on every auction)",
                "label_field": "quantile_label",
                "label_order": QUANTILE_LABEL_ORDER,
                "ordering_basis": "ascending by the numeric `amount` value, which is monotonic with quantile (q10 < q25 < ... < q95) by construction.",
            },
            "random": {
                "parameter_name": "range",
                "parameter_units": "RMB fen (bid ~ Uniform[low, high))",
                "label_field": "range_label",
                "label_order": RANDOM_RANGE_LABEL_ORDER,
                "components": ["low", "high"],
                "ordering_basis": (
                    "fixed grid-definition order (NOT a numeric axis -- "
                    "neither `low` nor range width is monotonic across "
                    "these 4 pairs; e.g. q10-q95's low end equals "
                    "q10-q50's). Treat label_order as categorical x-axis "
                    "ticks, not a continuous scale."
                ),
            },
            "linear_ctr": {
                "parameter_name": "base",
                "parameter_units": "RMB fen (bid = base * predicted CTR)",
                "label_field": "quantile_label",
                "label_order": QUANTILE_LABEL_ORDER,
                "ordering_basis": "ascending by the numeric `base` value, which is monotonic with quantile (q10 < q25 < ... < q95) by construction.",
            },
        },
        "spend_trajectory": {
            "resolution": "hourly (24 points per run: hour-of-day 0-23, local Beijing time), the finest grain persisted by the simulation run -- no per-auction trajectory was saved (see run_baselines.py:_hourly_spend_fractions).",
            "derivation": "spend_trajectory_fen[h] = hourly_spend_fraction[h] * budget_fen, rounded to 2 decimal places.",
            "downsampling_applied": False,
            "exported_on": "every leaf record (all 400 sweep points, not only is_best_by_clicks ones).",
        },
        "exclusion_rules_note": (
            "Every strategy is replayed over the SAME post-exclusion "
            "auction pool per advertiser (payprice > bidprice and "
            "payprice == 0 rows dropped before any settlement, per "
            "replay.py's ANOMALY EXCLUSION); this export does not carry "
            "the per-advertiser exclusion counts -- see "
            "simulation_baseline_run_season2_test.json / "
            "docs/analysis/simulation-results.md S1-S2 for those."
        ),
    }

    return {"meta": meta, "cells": cells}


def main() -> int:
    print(f"[export_dashboard_data] reading {RESULTS_PARQUET_PATH} ...")
    df = pd.read_parquet(RESULTS_PARQUET_PATH)
    print(f"[export_dashboard_data]   {len(df):,} rows")

    print(f"[export_dashboard_data] reading {RUN_METADATA_PATH} ...")
    with open(RUN_METADATA_PATH, "r", encoding="utf-8") as f:
        run_metadata = json.load(f)

    export = build_export(df, run_metadata)

    _FRONTEND_SRC.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: refuse to silently write NaN/Infinity into the
    # JSON (see module docstring point 7) -- would raise ValueError here
    # instead of producing an invalid file a JS JSON.parse can't read.
    text = json.dumps(export, indent=2, allow_nan=False, sort_keys=False)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    size_bytes = len(text.encode("utf-8"))
    print(
        f"[export_dashboard_data] wrote {export['meta']['n_cells']} cells "
        f"({export['meta']['n_advertisers']} advertisers x "
        f"{export['meta']['n_budgets']} budgets), "
        f"{export['meta']['n_leaf_records']} leaf sweep-point records "
        f"-> {OUTPUT_JSON_PATH}"
    )
    print(f"[export_dashboard_data] size: {size_bytes:,} bytes ({size_bytes / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
