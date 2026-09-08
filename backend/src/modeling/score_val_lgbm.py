"""Score the persisted LightGBM model on season-2 VALIDATION, once, and
save the predictions to disk -- the LightGBM analogue of `score_val.py`,
mirroring `score_test_lgbm.py`'s discipline for the val split.

**Not a new test-set decision, not a retrain.** `train_lgbm.py --full`
already scored every configuration in `CONFIG_GRID` against val as part of
model selection (see `backend/data/metadata/ctr_lgbm_runs.jsonl` and the
`val_selection_metrics` block recorded for the selected config in
`ctr_lgbm_test_evaluation_season2.json`) -- but it never persisted the
*per-row* val predictions, only pooled summary metrics. This script does
not select a model, does not tune anything, and does not fit anything: it
loads the already-fitted artifact (`ctr_lgbm_season2.joblib`, selected
config + `best_iteration` baked in) and reproduces the exact forward pass
over val that `train_lgbm.py` already made when it picked that config, so
the resulting per-row probabilities can be persisted and reused (e.g. for
fitting an isotonic calibrator on val only, never on test).

Because nothing here is stochastic (no fitting happens;
`lightgbm.Booster.predict()` is deterministic given a fixed
`num_iteration`), re-running it is idempotent, and its pooled AUC/log loss
are asserted, at runtime, to match the existing `val_selection_metrics`
recorded in `ctr_lgbm_test_evaluation_season2.json` to within `1e-9`. If
that assertion ever fails, report it loudly rather than silently accepting
a new number (same policy as `score_val.py` / `score_test_lgbm.py`).

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.score_val_lgbm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from . import features_lgbm, split
from .downsample import recalibrate_probability
from .evaluate import evaluate

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_PREDICTIONS_ROOT = _REPO_ROOT / "backend" / "data" / "predictions"

_MODEL_PATH = _MODELS_ROOT / "ctr_lgbm_season2.joblib"
_RECORDED_TEST_EVAL_PATH = _METADATA_ROOT / "ctr_lgbm_test_evaluation_season2.json"

_OUTPUT_PARQUET_PATH = _PREDICTIONS_ROOT / "ctr_lgbm_season2_val.parquet"
_OUTPUT_SIDECAR_PATH = _PREDICTIONS_ROOT / "ctr_lgbm_season2_val.json"

# Same carry-through columns as score_val.py / score_test_lgbm.py, for
# direct comparability across every persisted predictions parquet.
_RAW_CARRY_COLUMNS = ["advertiser", "timestamp"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_matches_recorded_val_selection(result, *, atol: float = 1e-9) -> None:
    """Cross-check this run's pooled val AUC/log loss against the
    `val_selection_metrics` block already recorded for the SELECTED
    configuration in `ctr_lgbm_test_evaluation_season2.json` (written once
    by `train_lgbm.py --full --touch-test`, from the same fit/config that
    produced the persisted `ctr_lgbm_season2.joblib` artifact this script
    loads). A mismatch means this script's forward pass over val diverged
    from the one that picked the selected configuration -- fail loudly.
    """
    if not _RECORDED_TEST_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"{_RECORDED_TEST_EVAL_PATH} does not exist -- this script "
            "expects the selected model's val_selection_metrics to already "
            "be recorded there; run `train_lgbm.py --full --touch-test` "
            "first if that record was never produced."
        )
    with open(_RECORDED_TEST_EVAL_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    recorded_metrics = recorded["val_selection_metrics"]

    mismatches = []
    for key in ("n_rows", "n_positive"):
        if int(recorded_metrics[key]) != int(getattr(result, key)):
            mismatches.append(
                f"{key}: recorded={recorded_metrics[key]!r} this_run={getattr(result, key)!r}"
            )
    for key in ("auc", "log_loss"):
        recorded_val = float(recorded_metrics[key])
        this_val = float(getattr(result, key))
        if abs(recorded_val - this_val) > atol:
            mismatches.append(
                f"{key}: recorded={recorded_val!r} this_run={this_val!r} "
                f"(diff={abs(recorded_val - this_val)!r} > atol={atol!r})"
            )
    if mismatches:
        raise AssertionError(
            "score_val_lgbm.py: this run's forward pass does NOT reproduce "
            f"the recorded val_selection_metrics at {_RECORDED_TEST_EVAL_PATH}. "
            "This is a reproducibility bug -- investigate before trusting "
            "any downstream analysis (e.g. isotonic calibration) built on "
            "this script's output. Mismatches:\n  " + "\n  ".join(mismatches)
        )


def _assert_persisted_column_reproduces_harness(
    y_true: np.ndarray,
    p_calibrated: np.ndarray,
    result,
    *,
    atol: float = 1e-9,
) -> None:
    """Same guard as `score_test_lgbm.py`'s equivalent function: recompute
    AUC/log loss directly from the array about to be persisted and check
    it matches what `evaluate()` reported for this run.
    """
    eps = 1e-12
    p_clipped = np.clip(p_calibrated, eps, 1 - eps)
    auc_check = float(roc_auc_score(y_true, p_clipped))
    ll_check = float(log_loss(y_true, p_clipped, labels=[0, 1]))

    problems = []
    if abs(auc_check - result.auc) > atol:
        problems.append(f"AUC: recomputed={auc_check!r} harness={result.auc!r}")
    if abs(ll_check - result.log_loss) > atol:
        problems.append(f"log_loss: recomputed={ll_check!r} harness={result.log_loss!r}")
    if problems:
        raise AssertionError(
            "score_val_lgbm.py: locally-recomputed metrics on the array "
            "about to be persisted do not match evaluate()'s reported "
            "metrics -- the persisted p_calibrated column would not be "
            "trustworthy. Mismatches:\n  " + "\n  ".join(problems)
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print(f"[score_val_lgbm] loading model artifact -> {_MODEL_PATH}")
    if not _MODEL_PATH.exists():
        print(
            f"[score_val_lgbm] REFUSING to proceed: {_MODEL_PATH} does not "
            "exist. This script scores an already-fitted artifact and "
            "never trains one.",
            file=sys.stderr,
        )
        return 1
    artifact = joblib.load(_MODEL_PATH)
    booster = artifact["model"]
    cat_state = artifact["cat_state"]
    best_config = artifact["best_config"]
    best_iteration = artifact["best_iteration"]
    sampling_rate = artifact["downsampling"]["negative_sampling_rate"]
    saved_boundaries = artifact["split_boundaries"]

    live_boundaries = split.get_split_boundaries(2)
    live_boundaries_dict = {
        "train": list(live_boundaries.train),
        "val": list(live_boundaries.val),
        "test": list(live_boundaries.test),
    }
    if saved_boundaries != live_boundaries_dict:
        print(
            "[score_val_lgbm] REFUSING to proceed: artifact's saved "
            f"split_boundaries={saved_boundaries!r} do not match live "
            f"split.get_split_boundaries(2)={live_boundaries_dict!r}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[score_val_lgbm] artifact: config={best_config['name']} "
        f"best_iteration={best_iteration} sampling_rate={sampling_rate} "
        f"split_boundaries={saved_boundaries}"
    )

    print("[score_val_lgbm] loading season-2 val split (true rate, never downsampled) ...")
    X_val_raw, y_val = split.load_split(2, "val")
    print(f"[score_val_lgbm] val: {X_val_raw.shape}, positives={int(y_val.sum())}")

    print("[score_val_lgbm] encoding val features with the SAVED category-encoder state (fit=False) ...")
    X_val_enc, _, _ = features_lgbm.encode_features(X_val_raw, state=cat_state, fit=False)

    print("[score_val_lgbm] scoring (single forward pass, no retraining) ...")
    p_raw = booster.predict(X_val_enc, num_iteration=best_iteration)

    result = evaluate(
        y_val, p_raw,
        model_name=f"lgbm_{best_config['name']}",
        split="season2_val",
        sampling_rate=sampling_rate,
    )
    print(
        f"[score_val_lgbm] pooled val metrics: auc={result.auc:.10f} "
        f"log_loss={result.log_loss:.10f} ece={result.ece:.10f}"
    )

    print("[score_val_lgbm] cross-checking against the recorded val_selection_metrics ...")
    _assert_matches_recorded_val_selection(result)
    print(f"[score_val_lgbm] OK: matches val_selection_metrics in {_RECORDED_TEST_EVAL_PATH} exactly (within 1e-9).")

    p_calibrated = recalibrate_probability(np.asarray(p_raw, dtype=np.float64), sampling_rate)
    _assert_persisted_column_reproduces_harness(np.asarray(y_val, dtype=np.int64), p_calibrated, result)
    print("[score_val_lgbm] OK: persisted recalibrated-probability column reproduces the harness's own metrics exactly.")

    print("[score_val_lgbm] assembling predictions frame ...")
    out_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_val, dtype=np.int64),
            "p_raw": np.asarray(p_raw, dtype=np.float64),
            "p_calibrated": p_calibrated,
        },
        index=X_val_raw.index,
    )
    for col in _RAW_CARRY_COLUMNS:
        out_df[col] = X_val_raw[col].to_numpy()
    out_df = out_df.reset_index(drop=True)

    _PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(_OUTPUT_PARQUET_PATH, index=False)
    print(f"[score_val_lgbm] wrote {len(out_df):,} rows -> {_OUTPUT_PARQUET_PATH}")

    sidecar = {
        "generated_at": _now_iso(),
        "model_path": str(_MODEL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "model_name": result.model_name,
        "selected_config": best_config,
        "best_iteration": best_iteration,
        "seed": artifact["seed"],
        "sampling_rate": sampling_rate,
        "downsampling": artifact["downsampling"],
        "split_boundaries": saved_boundaries,
        "split_evaluated": "season2_val",
        "n_rows": result.n_rows,
        "n_positive": result.n_positive,
        "positive_rate": result.positive_rate,
        "metrics": result.to_dict(),
        "recorded_val_selection_cross_check": {
            "path": str(_RECORDED_TEST_EVAL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "matched_within_atol": 1e-9,
        },
        "columns": list(out_df.columns),
        "notes": (
            "p_raw is booster.predict(..., num_iteration=best_iteration) on "
            "the downsampled-train-fitted model, BEFORE recalibration. "
            "p_calibrated = downsample.recalibrate_probability(p_raw, "
            "sampling_rate), verified to reproduce evaluate()'s reported "
            "AUC/log loss exactly (see "
            "score_val_lgbm.py::_assert_persisted_column_reproduces_harness). "
            "Use p_calibrated for any analysis that treats the column as a "
            "probability (e.g. fitting an isotonic calibrator on val); "
            "p_raw is kept only for traceability back to the raw model "
            "output. This file mirrors score_test_lgbm.py's output shape "
            "but for the VALIDATION split -- it is a re-scoring of the "
            "already-fitted artifact, not a new fit and not a second test "
            "touch (val was already used for model selection)."
        ),
    }
    with open(_OUTPUT_SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[score_val_lgbm] wrote sidecar metadata -> {_OUTPUT_SIDECAR_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
