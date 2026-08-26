"""Score the persisted LR baseline on season-2 test, once, and save the
predictions to disk so downstream analyses (per-advertiser AUC, decile
calibration ratios, anything else) never have to re-run the model or
re-touch the test set to get at its output.

**This is not a second test-set decision.** `train_lr.py --touch-test`
already evaluated `ctr_lr_baseline_season2.joblib` against season-2 test
exactly once and recorded the result in
`backend/data/metadata/ctr_lr_test_evaluation_season2.json` (per the
top-level protocol, "touch the test set once"). This script does not
select a model, does not tune anything, and does not decide test is now
fair game for a second look -- it deterministically reproduces the exact
same forward pass (same artifact, same scaler, same rows) so the
resulting per-row probabilities can be persisted and reused. Because nothing
here is stochastic (no fitting happens), re-running it is idempotent and
its pooled AUC/log loss are asserted, at import-safe runtime, to match the
existing `ctr_lr_test_evaluation_season2.json` record byte-for-byte on the
metrics that matter -- see `_assert_matches_recorded_evaluation()` below.
If that assertion ever fails, it means this script's forward pass diverged
from the one that produced the recorded evaluation (e.g. the artifact was
swapped, the split boundaries changed, or scikit-learn's prediction
became non-deterministic across versions) -- report that loudly, do not
silently accept a different number.

**Recalibration is applied by this script too, not just inside the
harness.** `modeling.evaluate.evaluate()` recalibrates internally (via
`modeling.downsample.recalibrate_probability()`) but only returns
summary statistics (AUC, log loss, ECE, the binned reliability table) --
it does not hand back the per-row recalibrated array. Since downstream
analyses (per-advertiser AUC, decile ratio tables) need the *per-row*
recalibrated probability, this script calls
`recalibrate_probability()` itself, directly, on the same raw
`predict_proba` output and the same `sampling_rate` passed to `evaluate()`
-- and then verifies (via `_assert_matches_recorded_evaluation`) that
scoring this locally-recalibrated array with `sklearn.metrics.roc_auc_score`
/ `log_loss` reproduces exactly what `evaluate()` reported, so there is no
risk of the persisted column silently drifting from what the harness
actually scored.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.score_test
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

from . import features_lr, split
from .downsample import recalibrate_probability
from .evaluate import evaluate

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_PREDICTIONS_ROOT = _REPO_ROOT / "backend" / "data" / "predictions"

_MODEL_PATH = _MODELS_ROOT / "ctr_lr_baseline_season2.joblib"
_RECORDED_TEST_EVAL_PATH = _METADATA_ROOT / "ctr_lr_test_evaluation_season2.json"

_OUTPUT_PARQUET_PATH = _PREDICTIONS_ROOT / "ctr_lr_baseline_season2_test.parquet"
_OUTPUT_SIDECAR_PATH = _PREDICTIONS_ROOT / "ctr_lr_baseline_season2_test.json"

# Columns carried through from the raw test-split X frame alongside the
# prediction columns -- cheap (already materialized, no extra query) join
# keys / analysis dimensions for downstream work (per-advertiser AUC,
# etc.). `bidid` is deliberately NOT in this list: it is excluded from
# the admitted feature set by `schema.NON_FEATURE_BID_COLUMNS` (row
# identifier, not predictive -- see `schema.py`), so
# `load_impression_features()` never hands it back at all; `timestamp` IS
# an admitted feature column and is carried through here both because
# it's useful for downstream time-based slicing and because it is
# already present in `X_raw` at zero extra cost.
_RAW_CARRY_COLUMNS = ["advertiser", "timestamp"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_matches_recorded_evaluation(result, *, atol: float = 1e-9) -> None:
    """Cross-check this run's pooled AUC/log loss against the existing,
    single-touch test-evaluation record. A mismatch here is a
    reproducibility bug (different artifact, different rows, or a
    non-deterministic model/environment), not a "score has drifted, that's
    fine" situation -- fail loudly rather than silently reporting a new
    number as if it were the recorded one.
    """
    if not _RECORDED_TEST_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"{_RECORDED_TEST_EVAL_PATH} does not exist -- this script "
            "expects to reproduce an already-recorded single test-touch "
            "evaluation (see module docstring); run "
            "`train_lr.py --full --touch-test` first if that record was "
            "never produced."
        )
    with open(_RECORDED_TEST_EVAL_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    recorded_metrics = recorded["metrics"]

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
            "score_test.py: this run's forward pass does NOT reproduce "
            f"the recorded test evaluation at {_RECORDED_TEST_EVAL_PATH}. "
            "This is a reproducibility bug -- investigate before trusting "
            "any downstream analysis built on this script's output. "
            "Mismatches:\n  " + "\n  ".join(mismatches)
        )


def _assert_persisted_column_reproduces_harness(
    y_true: np.ndarray,
    p_calibrated: np.ndarray,
    result,
    *,
    atol: float = 1e-9,
) -> None:
    """Recompute AUC/log loss directly from the array that is about to be
    persisted to parquet, and check it matches what `evaluate()` (the
    harness) reported for the same run -- guards against the persisted
    `p_calibrated` column silently being a different array than the one
    actually scored (e.g. a recalibration-formula or clipping mismatch
    between this script and `evaluate.py`).
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
            "score_test.py: locally-recomputed metrics on the array about "
            "to be persisted do not match evaluate()'s reported metrics -- "
            "the persisted p_calibrated column would not be trustworthy. "
            "Mismatches:\n  " + "\n  ".join(problems)
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print(f"[score_test] loading model artifact -> {_MODEL_PATH}")
    if not _MODEL_PATH.exists():
        print(
            f"[score_test] REFUSING to proceed: {_MODEL_PATH} does not exist. "
            "This script scores an already-fitted artifact and never trains "
            "one -- see module/task constraints.",
            file=sys.stderr,
        )
        return 1
    artifact = joblib.load(_MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    best_C = artifact["best_C"]
    sampling_rate = artifact["downsampling"]["negative_sampling_rate"]
    saved_boundaries = artifact["split_boundaries"]

    # Sanity: the artifact's recorded split boundaries must match
    # `split.get_split_boundaries(2)` exactly -- if they diverge, the
    # artifact was fit against a different split shape than this script is
    # about to score against, and nothing downstream would be trustworthy.
    live_boundaries = split.get_split_boundaries(2)
    live_boundaries_dict = {
        "train": list(live_boundaries.train),
        "val": list(live_boundaries.val),
        "test": list(live_boundaries.test),
    }
    if saved_boundaries != live_boundaries_dict:
        print(
            "[score_test] REFUSING to proceed: artifact's saved "
            f"split_boundaries={saved_boundaries!r} do not match live "
            f"split.get_split_boundaries(2)={live_boundaries_dict!r}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[score_test] artifact: best_C={best_C} sampling_rate={sampling_rate} "
        f"split_boundaries={saved_boundaries}"
    )

    print("[score_test] loading season-2 test split (true rate, never downsampled) ...")
    X_test_raw, y_test = split.load_split(2, "test")
    print(f"[score_test] test: {X_test_raw.shape}, positives={int(y_test.sum())}")

    print("[score_test] encoding test features with the SAVED scaler (fit_scaler=False) ...")
    X_test_enc, _ = features_lr.encode_features(X_test_raw, scaler=scaler, fit_scaler=False)

    print("[score_test] scoring (single forward pass, no retraining) ...")
    p_raw = model.predict_proba(X_test_enc)[:, 1]

    result = evaluate(
        y_test, p_raw,
        model_name=f"lr_hashed_C={best_C}",
        split="season2_test",
        sampling_rate=sampling_rate,
    )
    print(
        f"[score_test] pooled test metrics: auc={result.auc:.10f} "
        f"log_loss={result.log_loss:.10f} ece={result.ece:.10f}"
    )

    print("[score_test] cross-checking against the recorded single-touch test evaluation ...")
    _assert_matches_recorded_evaluation(result)
    print(f"[score_test] OK: matches {_RECORDED_TEST_EVAL_PATH} exactly (within 1e-9).")

    p_calibrated = recalibrate_probability(np.asarray(p_raw, dtype=np.float64), sampling_rate)
    _assert_persisted_column_reproduces_harness(np.asarray(y_test, dtype=np.int64), p_calibrated, result)
    print("[score_test] OK: persisted recalibrated-probability column reproduces the harness's own metrics exactly.")

    print("[score_test] assembling predictions frame ...")
    out_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "p_raw": np.asarray(p_raw, dtype=np.float64),
            "p_calibrated": p_calibrated,
        },
        index=X_test_raw.index,
    )
    for col in _RAW_CARRY_COLUMNS:
        out_df[col] = X_test_raw[col].to_numpy()
    out_df = out_df.reset_index(drop=True)

    _PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(_OUTPUT_PARQUET_PATH, index=False)
    print(f"[score_test] wrote {len(out_df):,} rows -> {_OUTPUT_PARQUET_PATH}")

    sidecar = {
        "generated_at": _now_iso(),
        "model_path": str(_MODEL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "model_name": result.model_name,
        "best_C": best_C,
        "seed": artifact["seed"],
        "sampling_rate": sampling_rate,
        "downsampling": artifact["downsampling"],
        "split_boundaries": saved_boundaries,
        "split_evaluated": "season2_test",
        "n_rows": result.n_rows,
        "n_positive": result.n_positive,
        "positive_rate": result.positive_rate,
        "metrics": result.to_dict(),
        "recorded_test_evaluation_cross_check": {
            "path": str(_RECORDED_TEST_EVAL_PATH.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "matched_within_atol": 1e-9,
        },
        "columns": list(out_df.columns),
        "notes": (
            "p_raw is model.predict_proba(...)[:, 1] on the downsampled-train-fitted "
            "model, BEFORE recalibration. p_calibrated = "
            "downsample.recalibrate_probability(p_raw, sampling_rate), verified to "
            "reproduce evaluate()'s reported AUC/log loss exactly (see "
            "score_test.py::_assert_persisted_column_reproduces_harness). Use "
            "p_calibrated for any analysis that treats the column as a probability "
            "(per-advertiser AUC, decile calibration ratios, etc.); p_raw is kept "
            "only for traceability back to the raw model output."
        ),
    }
    with open(_OUTPUT_SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[score_test] wrote sidecar metadata -> {_OUTPUT_SIDECAR_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
