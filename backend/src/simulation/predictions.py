"""
Loads the fully-corrected click-probability column (`p_isotonic`) that
Phase 3 bidding strategies are allowed to use, asserts the two-stage
correction chain was applied in the right order, and aligns those
predictions -- which carry no `bidid` -- onto an `AuctionPool`'s row order.

SCOPE: this module produces `p_click` values for bid-producing code. It
never reads `payprice`/`bidprice` and never touches `replay.AuctionPool`'s
settlement-only fields -- see `replay.py`'s module docstring "THE PAYPRICE
GUARD" for the discipline this mirrors. Nothing here settles an auction.

---------------------------------------------------------------------------
WHY THE CORRECTION ORDER IS NOT INTERCHANGEABLE
---------------------------------------------------------------------------

The LightGBM CTR model was fit on a TRAIN split whose negatives were
downsampled (`modeling.downsample`, rate=0.02, seed=42 --
`ctr_train_downsampling_season2.json`). A model fit on downsampled data
systematically over-predicts the positive class on the true distribution
-- `p_raw` is the model's output on that downsampled-fit model, and is
inflated relative to the true click probability by a known, invertible
amount. `modeling.downsample.recalibrate_probability(p_raw, rate)` undoes
exactly that inflation, nothing else: it is the analytic inverse of the
downsampling prior shift and requires no fitted parameters beyond `rate`.

The isotonic calibrator (`calibrate_isotonic_lgbm.py`) was THEN fit on
`p_calibrated` -- i.e. it was fit to correct residual miscalibration in
the ALREADY-recalibrated probability, on the validation split. It has
never seen a raw, still-downsampling-inflated probability at fit time, so
its learned step function is calibrated against the wrong operating range
if fed `p_raw` directly: the isotonic map would be evaluated far outside
(or in a systematically shifted part of) the domain it was fit on,
silently reintroducing the very miscalibration Phase 2 corrected.

Composing them in the correct order --
`p_isotonic = isotonic(recalibrate_probability(p_raw, rate))` -- is
therefore not a convention but a hard requirement: applying only one step,
or swapping the order, changes bid sizes by roughly the downsampling
inflation factor (up to `1/rate` = 50x at the low-probability end for
`rate=0.02`), which is exactly the "a bidder that multiplies by a
probability three times too high bids three times too high" failure this
project's rules call out explicitly. `assert_correction_order()` below
proves, on every run, that the persisted `p_isotonic` column really was
produced by the correct-order composition and that the reverse order is
detectably different (not a vacuous check -- see its docstring).

---------------------------------------------------------------------------
WHY POSITIONAL BIDID ALIGNMENT (not a fuzzy join)
---------------------------------------------------------------------------

`ctr_lgbm_season2_test.parquet` (produced by `score_test_lgbm.py`, via
`modeling.split.load_split(2, "test")` ->
`features.dataset.load_impression_features()`) carries no `bidid` column
-- `bidid` is deliberately excluded from the CTR feature allowlist
(`ingest.schema.NON_FEATURE_BID_COLUMNS`: "row identifier, not
predictive"). But `replay.AuctionPool` is keyed and ordered by
`(timestamp, bidid)` and has rows REMOVED by the two anomaly-exclusion
rules (`replay._exclude_unsimulatable_rows`), so the predictions parquet
and a pool are neither the same length nor the same row order -- a
positional zip between the two would be silently wrong.

The fix used here: `features/dataset.py::load_impression_features()`
documents its own deterministic materialize-then-sort discipline exactly
(`ORDER BY` pushed down as an in-memory pandas
`sort_values(["_sort_bidid", "_sort_timestamp"], kind="mergesort")`, AFTER
the DuckDB scan, precisely because a SQL-level `ORDER BY` at this width
was too slow -- see that module's docstring). `attach_bidid_to_predictions()`
below re-runs the identical scan-then-sort recipe, projecting
`bidid, advertiser, timestamp` only, and attaches the result POSITIONALLY
to the (whole test day, whole test day = one call, one sort, same
determinism) predictions parquet. This is verified, not assumed: row
count and the elementwise `advertiser`/`timestamp` match are asserted
before a single `bidid` is trusted (see `attach_bidid_to_predictions()`).

RESIDUAL LIMITATION, NAMED EXPLICITLY: `(bidid, timestamp)` is not a
perfectly unique key corpus-wide -- a vanishingly small number of rows
(8 rows / 4 colliding pairs across the entire 1,657,338-row test day, see
`_MAX_EXPECTED_KEY_COLLISION_PAIRS` below) share an identical `bidid` AND
millisecond `timestamp` while being genuinely different auctions (verified
by hand: different `ipinyouid`, different `region`/`city`). A stable sort
preserves relative order among exactly-tied keys according to whichever
order the two rows arrived from DuckDB's (documented, non-deterministic)
parallel Parquet scan -- see `features/dataset.py`'s own docstring for the
same caveat applied to ITS sort. This means the positional attach could,
in principle, swap `bidid` between two such colliding rows without
tripping the advertiser/timestamp elementwise check (both fields are
identical between the colliding pair by construction). Given the
population affected is 8 rows out of 1,657,338 (0.0005%), this is treated
as a bounded, reported residual risk rather than grounds to redesign the
alignment: `p_click_for_pool()` additionally detects any
`(bidid, timestamp)` collision WITHIN one advertiser's slice and resolves
it by averaging `p_isotonic` across the colliding rows (assigning that
average to every pool row landing on that key) rather than guessing which
one is "correct" -- see that function's docstring. A collision count above
`_MAX_EXPECTED_KEY_COLLISION_PAIRS` raises, on the assumption that a much
larger number would indicate a genuine alignment bug rather than this
known, tiny data artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import joblib
import numpy as np
import pandas as pd

from ..ingest import paths as ingest_paths
from ..modeling.downsample import recalibrate_probability

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREDICTIONS_ROOT = _REPO_ROOT / "backend" / "data" / "predictions"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"

TEST_PREDICTIONS_PARQUET = _PREDICTIONS_ROOT / "ctr_lgbm_season2_test.parquet"
ISOTONIC_CALIBRATOR_PATH = _MODELS_ROOT / "ctr_lgbm_season2_isotonic_calibrator.joblib"
DOWNSAMPLE_METADATA_PATH = _METADATA_ROOT / "ctr_train_downsampling_season2.json"

SEASON2_TEST_DATE_RANGE: tuple[str, str] = ("2013-06-12", "2013-06-12")

# Empirically confirmed (see module docstring "RESIDUAL LIMITATION"): 4
# colliding (bidid, timestamp) key-pairs (8 rows) across the whole
# 1,657,338-row season-2 test day. Set generously above that so a genuine
# regression (e.g. a much bigger collision count from a data or query
# change) still raises loudly rather than being silently absorbed by the
# same averaging fallback.
_MAX_EXPECTED_KEY_COLLISION_PAIRS = 20

_TOL = 1e-9
# Threshold for the "reverse order must be MATERIALLY different" guard in
# assert_correction_order() -- see its docstring. Chosen well above _TOL
# (which is a floating-point-noise tolerance) and well below the scale of
# the actual observed reverse-order discrepancy, so a genuine near-identity
# coincidence would still be caught.
_REVERSE_ORDER_MIN_DIVERGENCE = 1e-4


# ---------------------------------------------------------------------------
# Loading the raw ingredients
# ---------------------------------------------------------------------------


def load_test_predictions(path: Path = TEST_PREDICTIONS_PARQUET) -> pd.DataFrame:
    """Load the persisted, already-scored season-2 test predictions.

    Never re-scores the model -- this is purely a Parquet read of an
    artifact produced by `modeling.score_test_lgbm`. Raises
    `FileNotFoundError` if it hasn't been generated yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run "
            "`backend\\.venv\\Scripts\\python.exe -m src.modeling.score_test_lgbm` "
            "(and `-m src.modeling.calibrate_isotonic_lgbm` for `p_isotonic`) first."
        )
    return pd.read_parquet(path)


def load_isotonic_calibrator_artifact(path: Path = ISOTONIC_CALIBRATOR_PATH) -> dict:
    """Load the fitted isotonic calibrator artifact dict (see
    `calibrate_isotonic_lgbm.py`'s `calibrator_artifact` for its shape:
    `calibrator`, `fit_on`, `sampling_rate`, ...). Never fits a calibrator
    -- this is a `joblib.load` of an existing artifact.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run "
            "`backend\\.venv\\Scripts\\python.exe -m src.modeling.calibrate_isotonic_lgbm` first."
        )
    return joblib.load(path)


def load_downsample_metadata(path: Path = DOWNSAMPLE_METADATA_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Requirement 1: assert the correction order, every run
# ---------------------------------------------------------------------------


def assert_correction_order(
    predictions: pd.DataFrame,
    calibrator_artifact: dict,
    *,
    sampling_rate: float,
    tol: float = _TOL,
    min_reverse_divergence: float = _REVERSE_ORDER_MIN_DIVERGENCE,
) -> None:
    """Assert that `predictions['p_isotonic']` really was produced by
    `isotonic(recalibrate_probability(p_raw, sampling_rate))`, in that
    order, and that the reverse order would have produced something
    materially different -- see module docstring "WHY THE CORRECTION
    ORDER IS NOT INTERCHANGEABLE". Raises `AssertionError` loudly on any
    failure; this function has no return value because a passing call is
    the only acceptable outcome for downstream bidding code to proceed.

    Four checks, in order:

      1. `recalibrate_probability(p_raw, sampling_rate)` reproduces the
         persisted `p_calibrated` column to within `tol` -- confirms the
         FIRST correction stage matches what was actually persisted (not
         a different rate, not skipped).
      2. Applying the loaded isotonic calibrator to that recomputed
         `p_calibrated` reproduces the persisted `p_isotonic` column to
         within `tol` -- confirms the SECOND stage was applied to the
         output of the first, not to `p_raw` directly.
      3. NOT VACUOUS: applying isotonic to `p_raw` directly and THEN
         recalibrating (the wrong order) must differ from the persisted
         `p_isotonic` by more than `min_reverse_divergence` (max absolute
         difference) -- if checks 1+2 passed AND the wrong order gave the
         same answer, check 2 would prove nothing about ordering
         specifically. This guards against exactly that degenerate
         coincidence.
      4. The calibrator was fit on the VALIDATION split, never test --
         read from the artifact's own `fit_on.split` metadata, not
         inferred.
    """
    p_raw = predictions["p_raw"].to_numpy(dtype=np.float64)
    p_calibrated_persisted = predictions["p_calibrated"].to_numpy(dtype=np.float64)
    p_isotonic_persisted = predictions["p_isotonic"].to_numpy(dtype=np.float64)
    calibrator = calibrator_artifact["calibrator"]

    # --- Check 1: downsampling recalibration reproduces p_calibrated ----
    recomputed_calibrated = recalibrate_probability(p_raw, sampling_rate)
    max_diff_1 = float(np.max(np.abs(recomputed_calibrated - p_calibrated_persisted)))
    if max_diff_1 > tol:
        raise AssertionError(
            "assert_correction_order(): recalibrate_probability(p_raw, "
            f"sampling_rate={sampling_rate!r}) does NOT reproduce the "
            f"persisted p_calibrated column (max abs diff={max_diff_1!r} > "
            f"tol={tol!r}). Either the wrong sampling_rate was passed, or "
            "p_calibrated on disk was not produced by this recalibration -- "
            "refusing to proceed with an unverified correction chain."
        )

    # --- Check 2: isotonic(recomputed p_calibrated) reproduces p_isotonic
    recomputed_isotonic_correct_order = calibrator.predict(recomputed_calibrated)
    max_diff_2 = float(np.max(np.abs(recomputed_isotonic_correct_order - p_isotonic_persisted)))
    if max_diff_2 > tol:
        raise AssertionError(
            "assert_correction_order(): isotonic calibrator applied to the "
            "recomputed p_calibrated does NOT reproduce the persisted "
            f"p_isotonic column (max abs diff={max_diff_2!r} > tol={tol!r}). "
            "The isotonic calibrator on disk may not be the one that "
            "produced this predictions parquet, or the correction chain "
            "was broken between the two stages."
        )

    # --- Check 3: reverse order is DETECTABLY different (not vacuous) ---
    wrong_order_isotonic_first = calibrator.predict(p_raw)
    wrong_order_result = recalibrate_probability(wrong_order_isotonic_first, sampling_rate)
    max_diff_reverse = float(np.max(np.abs(wrong_order_result - p_isotonic_persisted)))
    if max_diff_reverse <= min_reverse_divergence:
        raise AssertionError(
            "assert_correction_order(): the REVERSE-order composition "
            "(isotonic(p_raw) then recalibrate) produced a result "
            f"indistinguishable from the persisted p_isotonic (max abs "
            f"diff={max_diff_reverse!r} <= {min_reverse_divergence!r}). "
            "This would make check 2 above vacuous -- the two orderings "
            "must be genuinely different corrections for that check to "
            "mean anything. Investigate before trusting the ordering "
            "assertion at all."
        )

    # --- Check 4: fit on validation, not test ---------------------------
    fit_split = calibrator_artifact.get("fit_on", {}).get("split")
    if fit_split != "season2_val":
        raise AssertionError(
            "assert_correction_order(): isotonic calibrator artifact's "
            f"fit_on.split={fit_split!r}, expected 'season2_val' -- "
            "refusing to bid using a calibrator that may have been fit on "
            "(or leaked) the test split it is about to be applied to."
        )


# ---------------------------------------------------------------------------
# Requirement 2: aligning bidid-less predictions onto the pool
# ---------------------------------------------------------------------------


def _query_bidid_advertiser_timestamp(
    *,
    seasons: tuple[int, ...],
    date_range: tuple[str, str],
    processed_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Re-run the SAME scan-then-sort recipe as
    `features.dataset.load_impression_features()` (see that module's
    "DETERMINISTIC ROW ORDER" docstring section), but projecting only
    `bidid, advertiser, timestamp` -- `bidid` is excluded from the CTR
    feature allowlist, so it never comes back from
    `load_impression_features()` itself, and this is the one place it is
    fetched, for alignment purposes only, never as a model feature.
    """
    root = processed_root or ingest_paths.PROCESSED_ROOT
    imp_dir = root / "impressions"
    if not imp_dir.exists():
        raise FileNotFoundError(f"{imp_dir} does not exist -- run ingestion first.")
    imp_glob = str(imp_dir / "**" / "*.parquet")

    season_filter = "i.season IN (" + ", ".join(str(int(s)) for s in seasons) + ")"
    start, end = date_range
    con = duckdb.connect()
    query = f"""
        SELECT i."bidid" AS "bidid", i."advertiser" AS "advertiser", i."timestamp" AS "timestamp"
        FROM read_parquet('{imp_glob}', hive_partitioning=1) i
        WHERE {season_filter}
          AND i.date BETWEEN DATE '{start}' AND DATE '{end}'
    """
    df = con.execute(query).fetch_arrow_table().to_pandas()
    # Same stable sort, same keys, same tiebreak convention as
    # load_impression_features()'s internal (_sort_bidid, _sort_timestamp)
    # sort -- see that module's docstring for why this must be an
    # in-memory mergesort, not a SQL ORDER BY.
    df = df.sort_values(["bidid", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def _attach_bidid_positionally(predictions: pd.DataFrame, keyed: pd.DataFrame) -> pd.DataFrame:
    """Pure function doing the actual verify-then-attach work of
    `attach_bidid_to_predictions()` below, split out so it can be
    unit-tested against small hand-built frames without a real DuckDB /
    Parquet read (same "pure, unit-testable independent of the DuckDB
    load" pattern as `replay._exclude_unsimulatable_rows`).

    VERIFIED, NOT ASSUMED: raises `ValueError` immediately if `len(keyed)
    != len(predictions)`, or if `keyed`'s `advertiser`/`timestamp` columns
    do not match `predictions`'s own columns elementwise. No fuzzy join,
    no reindexing, no fallback -- a mismatch here means the two datasets
    are not the same materialization and must be investigated, not
    silently reconciled.

    Returns a copy of `predictions` with `bidid` inserted as the first
    column; all original columns and row order are preserved unchanged.
    """
    if len(keyed) != len(predictions):
        raise ValueError(
            "attach_bidid_to_predictions(): row count mismatch -- "
            f"re-queried {len(keyed):,} rows but predictions has "
            f"{len(predictions):,} rows. Refusing to attach bidid "
            "positionally onto a differently-shaped frame; investigate "
            "before proceeding (wrong seasons/date_range? predictions "
            "parquet regenerated from a different source?)."
        )

    pred_advertiser = predictions["advertiser"].astype(str).to_numpy()
    keyed_advertiser = keyed["advertiser"].astype(str).to_numpy()
    advertiser_mismatch = pred_advertiser != keyed_advertiser
    if advertiser_mismatch.any():
        n_bad = int(advertiser_mismatch.sum())
        raise ValueError(
            f"attach_bidid_to_predictions(): {n_bad:,} row(s) have a "
            "mismatched 'advertiser' between the re-queried frame and "
            "predictions at the same position -- the two are not in the "
            "same row order. Refusing to fall back to a fuzzy join; "
            "investigate the sort/query discrepancy."
        )

    pred_timestamp = predictions["timestamp"].to_numpy()
    keyed_timestamp = keyed["timestamp"].to_numpy()
    timestamp_mismatch = pred_timestamp != keyed_timestamp
    if timestamp_mismatch.any():
        n_bad = int(timestamp_mismatch.sum())
        raise ValueError(
            f"attach_bidid_to_predictions(): {n_bad:,} row(s) have a "
            "mismatched 'timestamp' between the re-queried frame and "
            "predictions at the same position -- the two are not in the "
            "same row order. Refusing to fall back to a fuzzy join; "
            "investigate the sort/query discrepancy."
        )

    out = predictions.copy()
    out.insert(0, "bidid", keyed["bidid"].to_numpy())
    return out


def attach_bidid_to_predictions(
    predictions: pd.DataFrame,
    *,
    seasons: tuple[int, ...] = (2,),
    date_range: tuple[str, str] = SEASON2_TEST_DATE_RANGE,
    processed_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Attach a `bidid` column to `predictions` (which has none) by
    re-running the identical deterministic scan-then-sort query used to
    produce it, and joining POSITIONALLY -- see module docstring "WHY
    POSITIONAL BIDID ALIGNMENT". Verification is done by
    `_attach_bidid_positionally()` (see there for the exact checks).
    """
    keyed = _query_bidid_advertiser_timestamp(
        seasons=seasons, date_range=date_range, processed_root=processed_root
    )
    return _attach_bidid_positionally(predictions, keyed)


def p_click_for_pool(pool, predictions_with_bidid: pd.DataFrame) -> np.ndarray:
    """Extract `p_isotonic`, aligned to `pool`'s row order, for exactly
    `pool.advertiser`. `pool` is a `replay.AuctionPool`; only its
    (public) `advertiser`/`bidid`/`timestamp` fields are read here --
    never `_payprice` (that field is not even passed in: only whatever
    `pool` object the caller has is used, and this function never imports
    or references `replay._payprice`-shaped data).

    Join key is `(bidid, timestamp)`, NOT `bidid` alone -- `bidid` is not
    unique within a single advertiser's own day (see module docstring:
    ~0.2% of rows share a duplicate `bidid` with a DIFFERENT timestamp on
    this test day), so `bidid` alone would attach ambiguous predictions to
    unambiguous pool rows. `(bidid, timestamp)` is unique for all but a
    vanishing number of rows (see module docstring "RESIDUAL LIMITATION");
    those few colliding keys are resolved by averaging `p_isotonic` across
    the colliding candidates (never by guessing which one is "correct"),
    and the resolution is only applied to genuinely-colliding keys, never
    silently to the rest of the join.

    Raises `ValueError` if any pool row fails to get exactly one
    (possibly averaged-over-a-collision) `p_isotonic` value, or if the
    number of colliding keys exceeds `_MAX_EXPECTED_KEY_COLLISION_PAIRS`
    (a sanity bound against a much larger, real alignment bug hiding
    behind the same code path as the known tiny artifact).
    """
    advertiser = str(pool.advertiser)
    sub = predictions_with_bidid[
        predictions_with_bidid["advertiser"].astype(str) == advertiser
    ].copy()
    if len(sub) == 0:
        raise ValueError(
            f"p_click_for_pool(): no predictions rows found for "
            f"advertiser={advertiser!r} -- check that predictions_with_bidid "
            "covers this advertiser's date range."
        )

    sub_key = pd.MultiIndex.from_arrays([sub["bidid"].to_numpy(), sub["timestamp"].to_numpy()])
    dup_mask = sub_key.duplicated(keep=False)
    n_collision_pairs = 0
    if dup_mask.any():
        collided = sub.loc[dup_mask].copy()
        collided_key = pd.MultiIndex.from_arrays(
            [collided["bidid"].to_numpy(), collided["timestamp"].to_numpy()]
        )
        n_collision_pairs = collided_key.nunique()
        if n_collision_pairs > _MAX_EXPECTED_KEY_COLLISION_PAIRS:
            raise ValueError(
                f"p_click_for_pool(): {n_collision_pairs} colliding "
                "(bidid, timestamp) key-pairs found for advertiser "
                f"{advertiser!r}, exceeding the expected bound "
                f"({_MAX_EXPECTED_KEY_COLLISION_PAIRS}) for the known, "
                "tiny corpus artifact documented in this module's "
                "docstring -- this looks like a real alignment bug, not "
                "that artifact. Investigate before proceeding."
            )
        # Resolve each colliding key by averaging p_isotonic across its
        # candidates -- see docstring. Non-colliding rows are untouched.
        averaged = (
            collided.groupby(["bidid", "timestamp"], as_index=False)["p_isotonic"].mean()
        )
        sub = sub.loc[~dup_mask].copy()
        sub = pd.concat([sub, averaged.assign(advertiser=advertiser)], ignore_index=True, sort=False)

    sub = sub.drop_duplicates(subset=["bidid", "timestamp"], keep="first")
    lookup = sub.set_index(["bidid", "timestamp"])["p_isotonic"]
    if lookup.index.duplicated().any():  # pragma: no cover - defensive, should be impossible here
        raise AssertionError(
            "p_click_for_pool(): lookup index still has duplicates after "
            "collision resolution -- this should be impossible; investigate."
        )

    pool_key = pd.MultiIndex.from_arrays([pool.bidid.to_numpy(), pool.timestamp.to_numpy()])
    missing = ~pool_key.isin(lookup.index)
    if missing.any():
        n_missing = int(missing.sum())
        raise ValueError(
            f"p_click_for_pool(): {n_missing:,} of {len(pool):,} pool rows "
            f"for advertiser={advertiser!r} have no matching prediction -- "
            "every pool row must get exactly one p_isotonic value before "
            "bidding. Refusing to proceed with a partially-aligned bid "
            "vector."
        )

    aligned = lookup.reindex(pool_key).to_numpy(dtype=np.float64)
    if len(aligned) != len(pool):
        raise AssertionError(
            f"p_click_for_pool(): aligned length {len(aligned)} != pool "
            f"length {len(pool)} -- alignment invariant violated."
        )
    if np.isnan(aligned).any():
        raise AssertionError(
            "p_click_for_pool(): aligned p_isotonic contains NaN after a "
            "supposedly complete join -- investigate before bidding."
        )
    return aligned


# ---------------------------------------------------------------------------
# Convenience: load + verify + return the full, ready-to-use table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectedPredictions:
    """The verified, bidid-attached predictions table plus the metadata
    needed to trace it back to a specific model/calibrator/sampling-rate
    combination -- persisted into every simulation run's metadata sidecar.
    """

    predictions_with_bidid: pd.DataFrame
    sampling_rate: float
    calibrator_fit_split: str
    source_predictions_path: str
    source_calibrator_path: str


def load_corrected_predictions(
    *,
    predictions_path: Path = TEST_PREDICTIONS_PARQUET,
    calibrator_path: Path = ISOTONIC_CALIBRATOR_PATH,
    downsample_metadata_path: Path = DOWNSAMPLE_METADATA_PATH,
    seasons: tuple[int, ...] = (2,),
    date_range: tuple[str, str] = SEASON2_TEST_DATE_RANGE,
    processed_root: Optional[Path] = None,
) -> CorrectedPredictions:
    """The single sanctioned entry point for Phase 3 bidding code to
    obtain a verified `p_isotonic` column, aligned to `bidid`.

    Runs, in order: load predictions -> load calibrator artifact -> load
    downsampling metadata -> cross-check the calibrator's own
    `sampling_rate` against the downsampling metadata's
    `negative_sampling_rate` (they must agree; a mismatch means the
    calibrator was fit against a different recalibration than the one
    about to be asserted below) -> `assert_correction_order()` -> attach
    `bidid` -> return.
    """
    predictions = load_test_predictions(predictions_path)
    calibrator_artifact = load_isotonic_calibrator_artifact(calibrator_path)
    downsample_metadata = load_downsample_metadata(downsample_metadata_path)

    sampling_rate_calibrator = float(calibrator_artifact["sampling_rate"])
    sampling_rate_downsample = float(downsample_metadata["negative_sampling_rate"])
    if abs(sampling_rate_calibrator - sampling_rate_downsample) > _TOL:
        raise AssertionError(
            "load_corrected_predictions(): calibrator artifact's "
            f"sampling_rate={sampling_rate_calibrator!r} does not match "
            f"downsampling metadata's negative_sampling_rate="
            f"{sampling_rate_downsample!r} -- these must be the exact same "
            "downsampling run's rate for the correction chain to be valid."
        )

    assert_correction_order(
        predictions, calibrator_artifact, sampling_rate=sampling_rate_calibrator
    )

    predictions_with_bidid = attach_bidid_to_predictions(
        predictions, seasons=seasons, date_range=date_range, processed_root=processed_root
    )

    return CorrectedPredictions(
        predictions_with_bidid=predictions_with_bidid,
        sampling_rate=sampling_rate_calibrator,
        calibrator_fit_split=calibrator_artifact["fit_on"]["split"],
        source_predictions_path=str(predictions_path),
        source_calibrator_path=str(calibrator_path),
    )
