"""Score the persisted LightGBM model on season-2 test, once, and save the
predictions to disk -- the LightGBM analogue of `score_test.py`, same
discipline, same guarantees.

**This is not a second test-set decision.** `train_lgbm.py --full
--touch-test` already evaluated `ctr_lgbm_season2.joblib` against
season-2 test exactly once and recorded the result in
`backend/data/metadata/ctr_lgbm_test_evaluation_season2.json`. This script
does not select a model, does not tune anything, and does not decide test
is now fair game for a second look -- it deterministically reproduces the
exact same forward pass (same artifact, same category-encoder state, same
rows) so the resulting per-row probabilities can be persisted and reused
for per-advertiser AUC etc. Because nothing here is stochastic (no
fitting happens; `lightgbm.Booster.predict()` is deterministic given a
fixed `num_iteration`), re-running it is idempotent and its pooled
AUC/log loss are asserted, at runtime, to match the existing
`ctr_lgbm_test_evaluation_season2.json` record to within `1e-9`.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.score_test_lgbm
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

_OUTPUT_PARQUET_PATH = _PREDICTIONS_ROOT / "ctr_lgbm_season2_test.parquet"
_OUTPUT_SIDECAR_PATH = _PREDICTIONS_ROOT / "ctr_lgbm_season2_test.json"

# Same carry-through columns as score_test.py, for direct comparability of
# the two predictions parquets (per-advertiser AUC grouping, etc.).
_RAW_CARRY_COLUMNS = ["advertiser", "timestamp"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_matches_recorded_evaluation(result, *, atol: float = 1e-9) -> None:
    if not _RECORDED_TEST_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"{_RECORDED_TEST_EVAL_PATH} does not exist -- this script "
            "expects to reproduce an already-recorded single test-touch "
            "evaluation; run `train_lgbm.py --full --touch-test` first."
        )
    with open(_RECORDED_TEST_EVAL_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    recorded_metrics = recorded["metrics"]

    mismatches = []
    for key in ("n_rows", "n_positive"):
        if int(recorded_metrics[key]) != int(getattr(result, key)):
            mismatches.append(f"{key}: recorded={recorded_metrics[key]!r} this_run={getattr(result, key)!r}")
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
            "score_test_lgbm.py: this run's forward pass does NOT "
            f"reproduce the recorded test evaluation at "
            f"{_RECORDED_TEST_EVAL_PATH}. Mismatches:\n  " + "\n  ".join(mismatches)
        )


def _assert_persisted_column_reproduces_harness(
    y_true: np.ndarray, p_calibrated: np.ndarray, result, *, atol: float = 1e-9,
) -> None:
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
            "score_test_lgbm.py: locally-recomputed metrics on the array "
            "about to be persisted do not match evaluate()'s reported "
            "metrics. Mismatches:\n  " + "\n  ".join(problems)
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print(f"[score_test_lgbm] loading model artifact -> {_MODEL_PATH}")
    if not _MODEL_PATH.exists():
        print(
            f"[score_test_lgbm] REFUSING to proceed: {_MODEL_PATH} does not "
            "exist. This script scores an already-fitted artifact and never "
            "trains one.",
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
            "[score_test_lgbm] REFUSING to proceed: artifact's saved "
            f"split_boundaries={saved_boundaries!r} do not match live "
            f"split.get_split_boundaries(2)={live_boundaries_dict!r}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[score_test_lgbm] artifact: config={best_config['name']} "
        f"best_iteration={best_iteration} sampling_rate={sampling_rate} "
        f"split_boundaries={saved_boundaries}"
    )

    print("[score_test_lgbm] loading season-2 test split (true rate, never downsampled) ...")
    X_test_raw, y_test = split.load_split(2, "test")
    print(f"[score_test_lgbm] test: {X_test_raw.shape}, positives={int(y_test.sum())}")

    print("[score_test_lgbm] encoding test features with the SAVED category-encoder state (fit=False) ...")
    X_test_enc, _, _ = features_lgbm.encode_features(X_test_raw, state=cat_state, fit=False)

    print("[score_test_lgbm] scoring (single forward pass, no retraining) ...")
    p_raw = booster.predict(X_test_enc, num_iteration=best_iteration)

    result = evaluate(
        y_test, p_raw,
        model_name=f"lgbm_{best_config['name']}",
        split="season2_test",
        sampling_rate=sampling_rate,
    )
    print(
        f"[score_test_lgbm] pooled test metrics: auc={result.auc:.10f} "
        f"log_loss={result.log_loss:.10f} ece={result.ece:.10f}"
    )

    print("[score_test_lgbm] cross-checking against the recorded single-touch test evaluation ...")
    _assert_matches_recorded_evaluation(result)
    print(f"[score_test_lgbm] OK: matches {_RECORDED_TEST_EVAL_PATH} exactly (within 1e-9).")

    p_calibrated = recalibrate_probability(np.asarray(p_raw, dtype=np.float64), sampling_rate)
    _assert_persisted_column_reproduces_harness(np.asarray(y_test, dtype=np.int64), p_calibrated, result)
    print("[score_test_lgbm] OK: persisted recalibrated-probability column reproduces the harness's own metrics exactly.")

    print("[score_test_lgbm] assembling predictions frame ...")
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
    print(f"[score_test_lgbm] wrote {len(out_df):,} rows -> {_OUTPUT_PARQUET_PATH}")

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
            "p_raw is booster.predict(..., num_iteration=best_iteration) on "
            "the downsampled-train-fitted model, BEFORE recalibration. "
            "p_calibrated = downsample.recalibrate_probability(p_raw, "
            "sampling_rate), verified to reproduce evaluate()'s reported "
            "AUC/log loss exactly. Use p_calibrated for any analysis that "
            "treats the column as a probability (per-advertiser AUC etc.); "
            "p_raw is kept only for traceability back to the raw model output."
        ),
    }
    with open(_OUTPUT_SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[score_test_lgbm] wrote sidecar metadata -> {_OUTPUT_SIDECAR_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
