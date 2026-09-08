"""Fit an isotonic-regression calibrator on top of the (already
downsampling-recalibrated) LightGBM predictions, using validation only,
and apply it to the already-persisted test predictions.

**LightGBM analogue of `calibrate_isotonic.py`.** Mirrors that script's
discipline exactly (see its module docstring for the full rationale on
scope, `increasing=True`, `out_of_bounds='clip'`, and the matched-edge
decile-table fix) but for the LightGBM artifact -- kept as a separate
module, rather than a `--model` flag on `calibrate_isotonic.py`, so the
already-committed LR calibration outputs are never at risk of being
touched by a future edit to this file.

**Scope: calibration only. Never retrains the underlying CTR model.**
This script loads:
  - the persisted val predictions (`ctr_lgbm_season2_val.parquet`,
    produced by `score_val_lgbm.py` -- a re-scoring of the already-fitted
    LightGBM artifact, never a refit), and
  - the persisted test predictions (`ctr_lgbm_season2_test.parquet`,
    produced by `score_test_lgbm.py`, the single test-set touch already
    made for this model),
and fits nothing but an `sklearn.isotonic.IsotonicRegression` on top of
their `p_calibrated` column (the downsampling-recalibrated probability).

**Fit on validation only, applied to test exactly once.** The isotonic
calibrator is `.fit(p_val_calibrated, y_val)` -- test never influences the
fit. Test predictions are transformed through the already-fitted
calibrator and scored exactly once, at the end, purely for reporting (no
threshold/knot selection happens against test).

**increasing=True, out_of_bounds='clip'.** Same reasoning as
`calibrate_isotonic.py`: the underlying model already ranks correctly
(test AUC ~0.929), so isotonic regression is used purely to recalibrate
magnitudes, never to re-rank; `out_of_bounds='clip'` clips any test-set
probability outside the val-fit range to the nearest fitted boundary.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.calibrate_isotonic_lgbm
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from .evaluate import expected_calibration_error

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_PREDICTIONS_ROOT = _REPO_ROOT / "backend" / "data" / "predictions"

_MODEL_PATH = _MODELS_ROOT / "ctr_lgbm_season2.joblib"
_VAL_PRED_PARQUET = _PREDICTIONS_ROOT / "ctr_lgbm_season2_val.parquet"
_VAL_PRED_SIDECAR = _PREDICTIONS_ROOT / "ctr_lgbm_season2_val.json"
_TEST_PRED_PARQUET = _PREDICTIONS_ROOT / "ctr_lgbm_season2_test.parquet"
_TEST_PRED_SIDECAR = _PREDICTIONS_ROOT / "ctr_lgbm_season2_test.json"

_CALIBRATOR_PATH = _MODELS_ROOT / "ctr_lgbm_season2_isotonic_calibrator.joblib"
_CALIBRATOR_METADATA_PATH = _METADATA_ROOT / "ctr_lgbm_isotonic_calibrator_season2.json"

_N_DECILES = 10
_EPS = 1e-12


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decile_table(p: np.ndarray, y: np.ndarray, *, n_bins: int = _N_DECILES) -> pd.DataFrame:
    """Quantile (equal-count) decile table -- identical convention to
    `calibrate_isotonic.py::_decile_table`.
    """
    df = pd.DataFrame({"y_true": y.astype(np.float64), "p": p.astype(np.float64)})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = 0

    grouped = df.groupby("bin", observed=True).agg(
        n=("y_true", "size"),
        prob_min=("p", "min"),
        prob_max=("p", "max"),
        mean_predicted=("p", "mean"),
        observed_click_rate=("y_true", "mean"),
        n_positive=("y_true", "sum"),
    ).reset_index(drop=True)

    ratios = []
    for _, row in grouped.iterrows():
        if row["n_positive"] == 0:
            ratios.append("undefined (0 observed positives)")
        else:
            ratios.append(float(row["mean_predicted"] / row["observed_click_rate"]))
    grouped["predicted_over_observed_ratio"] = ratios
    grouped.insert(0, "decile", range(1, len(grouped) + 1))
    return grouped


def _matched_edge_decile_tables(
    p_before: np.ndarray,
    p_after: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = _N_DECILES,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Before/after decile tables on matched bin edges -- identical
    convention to `calibrate_isotonic.py::_matched_edge_decile_tables`.
    Bin membership is derived once, from `p_before` (`p_calibrated`), and
    reused unchanged to aggregate `p_after` (`p_isotonic`), so bin *i*
    contains the exact same rows in both tables.
    """
    df = pd.DataFrame(
        {
            "y_true": y.astype(np.float64),
            "p_before": p_before.astype(np.float64),
            "p_after": p_after.astype(np.float64),
        }
    )
    try:
        bin_labels, edges = pd.qcut(df["p_before"], q=n_bins, duplicates="drop", retbins=True)
    except ValueError:
        bin_labels = pd.Series(0, index=df.index)
        edges = np.array([float(df["p_before"].min()), float(df["p_before"].max())])
    df["bin"] = bin_labels

    def _agg(value_col: str) -> pd.DataFrame:
        grouped = df.groupby("bin", observed=True).agg(
            n=("y_true", "size"),
            mean_predicted=(value_col, "mean"),
            observed_click_rate=("y_true", "mean"),
            n_positive=("y_true", "sum"),
        ).reset_index(drop=True)
        ratios = []
        for _, row in grouped.iterrows():
            if row["n_positive"] == 0:
                ratios.append("undefined (0 observed positives)")
            else:
                ratios.append(float(row["mean_predicted"] / row["observed_click_rate"]))
        grouped["predicted_over_observed_ratio"] = ratios
        grouped.insert(0, "bin", range(1, len(grouped) + 1))
        grouped["edge_low"] = edges[:-1]
        grouped["edge_high"] = edges[1:]
        return grouped

    before_table = _agg("p_before")
    after_table = _agg("p_after")
    return before_table, after_table, edges


def _print_matched_edge_table(title: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'bin':>4} {'edge_low':>12} {'edge_high':>12} {'n':>9} "
        f"{'mean_pred_before':>18} {'mean_pred_after':>17} {'observed':>12} "
        f"{'n_pos':>6} {'ratio_before':>14} {'ratio_after':>14}"
    )
    print(header)
    for (_, rb), (_, ra) in zip(before.iterrows(), after.iterrows()):
        rb_ratio = rb["predicted_over_observed_ratio"]
        ra_ratio = ra["predicted_over_observed_ratio"]
        rb_str = rb_ratio if isinstance(rb_ratio, str) else f"{rb_ratio:.4f}"
        ra_str = ra_ratio if isinstance(ra_ratio, str) else f"{ra_ratio:.4f}"
        print(
            f"{int(rb['bin']):>4} {rb['edge_low']:>12.6e} {rb['edge_high']:>12.6e} "
            f"{int(rb['n']):>9,} {rb['mean_predicted']:>18.6e} {ra['mean_predicted']:>17.6e} "
            f"{rb['observed_click_rate']:>12.6e} {int(rb['n_positive']):>6} "
            f"{rb_str:>14} {ra_str:>14}"
        )


def _print_decile_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'decile':>6} {'n':>10} {'prob_min':>12} {'prob_max':>12} "
        f"{'mean_pred':>12} {'observed':>12} {'n_pos':>7} {'pred/obs':>24}"
    )
    print(header)
    for _, row in table.iterrows():
        ratio = row["predicted_over_observed_ratio"]
        ratio_str = ratio if isinstance(ratio, str) else f"{ratio:.4f}"
        print(
            f"{int(row['decile']):>6} {int(row['n']):>10,} {row['prob_min']:>12.6e} "
            f"{row['prob_max']:>12.6e} {row['mean_predicted']:>12.6e} "
            f"{row['observed_click_rate']:>12.6e} {int(row['n_positive']):>7} {ratio_str:>24}"
        )


def _records(table: pd.DataFrame) -> list:
    return table.assign(
        predicted_over_observed_ratio=lambda d: d["predicted_over_observed_ratio"].apply(
            lambda v: v if isinstance(v, str) else float(v)
        )
    ).to_dict(orient="records")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p_clipped = np.clip(p, _EPS, 1 - _EPS)
    auc = float(roc_auc_score(y, p_clipped))
    ll = float(log_loss(y, p_clipped, labels=[0, 1]))
    ece, _ = expected_calibration_error(y, p_clipped, n_bins=_N_DECILES)
    return {"auc": auc, "log_loss": ll, "ece": ece}


def main() -> int:
    print(f"[calibrate_isotonic_lgbm] loading val predictions -> {_VAL_PRED_PARQUET}")
    if not _VAL_PRED_PARQUET.exists():
        print(
            f"[calibrate_isotonic_lgbm] REFUSING to proceed: {_VAL_PRED_PARQUET} does "
            "not exist. Run `python -m src.modeling.score_val_lgbm` first (scores the "
            "already-fitted model artifact; never refits).",
        )
        return 1
    val_preds = pd.read_parquet(_VAL_PRED_PARQUET)
    with open(_VAL_PRED_SIDECAR, "r", encoding="utf-8") as f:
        val_sidecar = json.load(f)
    print(
        f"[calibrate_isotonic_lgbm] val: {len(val_preds):,} rows, "
        f"{int(val_preds['y_true'].sum())} positives (split={val_sidecar['split_evaluated']})"
    )

    print(f"[calibrate_isotonic_lgbm] loading test predictions -> {_TEST_PRED_PARQUET}")
    if not _TEST_PRED_PARQUET.exists():
        print(
            f"[calibrate_isotonic_lgbm] REFUSING to proceed: {_TEST_PRED_PARQUET} does "
            "not exist. Run `python -m src.modeling.score_test_lgbm` first."
        )
        return 1
    test_preds = pd.read_parquet(_TEST_PRED_PARQUET)
    with open(_TEST_PRED_SIDECAR, "r", encoding="utf-8") as f:
        test_sidecar = json.load(f)
    print(
        f"[calibrate_isotonic_lgbm] test: {len(test_preds):,} rows, "
        f"{int(test_preds['y_true'].sum())} positives (split={test_sidecar['split_evaluated']})"
    )

    if val_sidecar["model_path"] != test_sidecar["model_path"]:
        print(
            "[calibrate_isotonic_lgbm] REFUSING to proceed: val and test predictions "
            f"come from different model artifacts (val={val_sidecar['model_path']!r}, "
            f"test={test_sidecar['model_path']!r})."
        )
        return 1
    if val_sidecar["sampling_rate"] != test_sidecar["sampling_rate"]:
        print(
            "[calibrate_isotonic_lgbm] REFUSING to proceed: val and test predictions "
            f"were recalibrated with different sampling rates (val={val_sidecar['sampling_rate']!r}, "
            f"test={test_sidecar['sampling_rate']!r})."
        )
        return 1
    print(
        f"[calibrate_isotonic_lgbm] OK: val and test predictions share model_path="
        f"{val_sidecar['model_path']!r}, sampling_rate={val_sidecar['sampling_rate']!r}."
    )
    print(
        "[calibrate_isotonic_lgbm] using column 'p_calibrated' (downsampling-recalibrated "
        "probability) as the calibrator's input; 'p_raw' is NOT used here."
    )

    y_val = val_preds["y_true"].to_numpy(dtype=np.int64)
    p_val_calibrated = val_preds["p_calibrated"].to_numpy(dtype=np.float64)
    y_test = test_preds["y_true"].to_numpy(dtype=np.int64)
    p_test_calibrated = test_preds["p_calibrated"].to_numpy(dtype=np.float64)

    print(
        f"[calibrate_isotonic_lgbm] fitting IsotonicRegression(out_of_bounds='clip', "
        f"increasing=True) on val ONLY (n={len(y_val):,}) ..."
    )
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(p_val_calibrated, y_val)
    print("[calibrate_isotonic_lgbm] fit complete. Test set has NOT been touched yet.")

    print("[calibrate_isotonic_lgbm] applying fitted calibrator to TEST predictions (single application) ...")
    p_test_isotonic = calibrator.predict(p_test_calibrated)

    before_table = _decile_table(p_test_calibrated, y_test)
    after_table = _decile_table(p_test_isotonic, y_test)

    _print_decile_table("TEST decile table -- BEFORE isotonic (p_calibrated)", before_table)
    _print_decile_table("TEST decile table -- AFTER isotonic (p_isotonic)", after_table)

    before_metrics = _metrics(y_test, p_test_calibrated)
    after_metrics = _metrics(y_test, p_test_isotonic)

    print("\n=== TEST pooled metrics ===")
    print(
        f"  BEFORE: auc={before_metrics['auc']:.10f} log_loss={before_metrics['log_loss']:.10f} "
        f"ece={before_metrics['ece']:.10f}"
    )
    print(
        f"  AFTER:  auc={after_metrics['auc']:.10f} log_loss={after_metrics['log_loss']:.10f} "
        f"ece={after_metrics['ece']:.10f}"
    )
    auc_delta = abs(after_metrics["auc"] - before_metrics["auc"])
    print(f"  AUC delta (should be ~0, isotonic is monotone): {auc_delta:.10f}")
    if auc_delta > 1e-4:
        print(
            "  [calibrate_isotonic_lgbm] WARNING: AUC changed by more than 1e-4 after a "
            "monotone (increasing=True) isotonic transform -- this can legitimately "
            "happen from tie-breaking when isotonic collapses previously-distinct "
            "probabilities into the same calibrated value, but investigate before "
            "trusting this number if the delta is large."
        )

    before_matched, after_matched, edges = _matched_edge_decile_tables(
        p_test_calibrated, p_test_isotonic, y_test
    )
    _print_matched_edge_table(
        "TEST matched-edge decile table -- bin membership from p_calibrated (BEFORE) quantiles, "
        "reused unchanged for p_isotonic (AFTER)",
        before_matched,
        after_matched,
    )
    n_bins_realized = len(before_matched)
    if n_bins_realized < _N_DECILES:
        print(
            f"[calibrate_isotonic_lgbm] NOTE: {n_bins_realized} bins realized (requested "
            f"{_N_DECILES}) -- duplicate quantile edges were dropped from p_calibrated "
            "itself (this can happen even before isotonic, from ties)."
        )

    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    calibrator_artifact = {
        "calibrator": calibrator,
        "calibrator_type": "sklearn.isotonic.IsotonicRegression",
        "isotonic_params": {"out_of_bounds": "clip", "increasing": True},
        "input_column": "p_calibrated",
        "source_model_path": str(_MODEL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "fit_on": {
            "predictions_path": str(_VAL_PRED_PARQUET.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "split": "season2_val",
            "n_rows": int(len(y_val)),
            "n_positive": int(y_val.sum()),
        },
        "sampling_rate": val_sidecar["sampling_rate"],
        "split_boundaries": val_sidecar["split_boundaries"],
    }
    joblib.dump(calibrator_artifact, _CALIBRATOR_PATH)
    print(f"\n[calibrate_isotonic_lgbm] saved fitted calibrator -> {_CALIBRATOR_PATH}")

    metadata = {
        "generated_at": _now_iso(),
        "source_model_artifact": str(_MODEL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "calibrator_artifact": str(_CALIBRATOR_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "prediction_artifacts_used": {
            "val": {
                "parquet": str(_VAL_PRED_PARQUET.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "sidecar": str(_VAL_PRED_SIDECAR.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "column_used": "p_calibrated",
                "note": (
                    "val predictions were regenerated by score_val_lgbm.py because no "
                    "persisted val-prediction parquet existed before this task -- "
                    "regeneration was a re-scoring of the already-fitted LightGBM "
                    "artifact (ctr_lgbm_season2.joblib) only; the model was NOT refit. "
                    "score_val_lgbm.py cross-checks its pooled AUC/log loss against the "
                    "val_selection_metrics already recorded in "
                    "ctr_lgbm_test_evaluation_season2.json to within 1e-9."
                ),
            },
            "test": {
                "parquet": str(_TEST_PRED_PARQUET.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "sidecar": str(_TEST_PRED_SIDECAR.relative_to(_REPO_ROOT)).replace("\\", "/"),
                "column_used": "p_calibrated",
                "note": "regenerated by score_test_lgbm.py, reproducing the recorded single test touch exactly.",
            },
        },
        "split_boundaries": val_sidecar["split_boundaries"],
        "n_val": int(len(y_val)),
        "n_val_positive": int(y_val.sum()),
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
        "isotonic_params": {"out_of_bounds": "clip", "increasing": True},
        "sampling_rate": val_sidecar["sampling_rate"],
        "test_metrics_before_isotonic": before_metrics,
        "test_metrics_after_isotonic": after_metrics,
        "auc_delta_after_isotonic": auc_delta,
        "decile_table_before": _records(before_table),
        "decile_table_after": _records(after_table),
        "decile_table_before_matched_edges": _records(before_matched),
        "decile_table_after_matched_edges": _records(after_matched),
        "matched_edge_bin_edges": [float(e) for e in edges],
        "matched_edge_note": (
            "decile_table_before_matched_edges / decile_table_after_matched_edges bin "
            "BOTH p_calibrated (before) and p_isotonic (after) on IDENTICAL bin edges, "
            "derived once from p_calibrated's own 10-quantile qcut on test "
            "(duplicates='drop'). Every row falls in the same bin index in both tables "
            "by construction, so n, observed_click_rate, and n_positive are IDENTICAL "
            "across the before/after table for a given bin -- only mean_predicted and "
            "the predicted_over_observed_ratio can differ between the two tables. The "
            "independent-qcut decile_table_before / decile_table_after above remain "
            "correct for a standalone before-only or after-only reading."
        ),
        "notes": (
            "Isotonic calibrator fit on validation split's p_calibrated (the "
            "downsampling-recalibrated LightGBM probability) -> y_true only; test set "
            "was never used for fitting, hyperparameter/knot selection, or choosing "
            "between calibration methods -- it was scored exactly once, after fitting, "
            "purely for this report."
        ),
    }
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_CALIBRATOR_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[calibrate_isotonic_lgbm] wrote calibrator metadata -> {_CALIBRATOR_METADATA_PATH}")

    print("[calibrate_isotonic_lgbm] persisting p_isotonic alongside the existing test predictions ...")
    test_preds_out = test_preds.copy()
    test_preds_out["p_isotonic"] = p_test_isotonic
    test_preds_out.to_parquet(_TEST_PRED_PARQUET, index=False)
    print(
        f"[calibrate_isotonic_lgbm] wrote {len(test_preds_out):,} rows (added p_isotonic "
        f"column, all pre-existing columns preserved unchanged) -> {_TEST_PRED_PARQUET}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
