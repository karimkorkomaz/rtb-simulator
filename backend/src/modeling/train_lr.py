"""Logistic-regression CTR baseline: hashed features
(`modeling.features_lr`), temporal split (`modeling.split`), negative
downsampling on train only (`modeling.downsample`), scored by the shared
evaluation harness (`modeling.evaluate`).

**Why this has to exist before LightGBM.** "Baselines before complexity":
a gradient-boosting result with nothing to compare against is not a
finding. This script is deliberately the floor -- a linear model over
hashed categorical features, no interactions, no tree structure -- so a
later LightGBM run has something concrete to beat, scored by the exact
same harness (`modeling.evaluate.evaluate()`), on the exact same val/test
rows, with the exact same recalibration correction applied.

**Model selection happens on val only; test is touched once.** Every
`LogisticRegression` fit in the `--c-grid` sweep is scored against val;
the best-by-log-loss `C` is what gets the (single, explicit) test
evaluation, gated behind `--touch-test`. Re-running with `--touch-test`
against an existing test-evaluation record raises unless `--overwrite`
is passed, so accidentally re-touching test requires a deliberate,
visible choice, not a rerun of the default command.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lr --dev
    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lr --full
    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lr --full --touch-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import features_lr, split
from .downsample import downsample_negatives, DownsampleInfo
from .evaluate import evaluate, save_reliability_plot

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_RUN_LOG_PATH = _METADATA_ROOT / "ctr_lr_runs.jsonl"
_DOWNSAMPLE_METADATA_PATH = _METADATA_ROOT / "ctr_train_downsampling_season2.json"
_TEST_EVAL_PATH = _METADATA_ROOT / "ctr_lr_test_evaluation_season2.json"
_MODEL_PATH = _MODELS_ROOT / "ctr_lr_baseline_season2.joblib"

# Fixed everywhere in this script -- see module docstring "Engineering
# standards" (top-level protocol): every number here must regenerate.
RANDOM_SEED = 42

# Fraction of TRUE negatives kept in the train split. Chosen to land the
# downsampled train's positive rate in a range (a few percent) that keeps
# logistic regression's optimizer well-behaved without discarding so much
# data that rare feature values lose all their negative examples; the
# EXACT resulting counts (not just this rate) are persisted to
# `_DOWNSAMPLE_METADATA_PATH` by every `--full` run -- see
# `modeling.downsample` for the recalibration this implies.
NEGATIVE_SAMPLING_RATE = 0.02

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_run_log(record: dict) -> None:
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _fit_and_score(X_train_enc, y_train, X_val_enc, y_val, *, C: float, sampling_rate: float):
    clf = LogisticRegression(
        # No explicit `penalty=`: scikit-learn 1.9 deprecated that param in
        # favour of `l1_ratio` (l1_ratio=0.0, the default, is L2 -- the same
        # penalty this used to request via penalty="l2").
        C=C,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    t0 = time.time()
    clf.fit(X_train_enc, y_train)
    fit_seconds = time.time() - t0

    p_val_raw = clf.predict_proba(X_val_enc)[:, 1]
    result = evaluate(
        y_val, p_val_raw,
        model_name=f"lr_hashed_C={C}",
        split="season2_val",
        sampling_rate=sampling_rate,
    )
    return clf, result, fit_seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="small row_limit, fast iteration")
    mode.add_argument("--full", action="store_true", help="full season-2 train/val (and test, with --touch-test)")
    parser.add_argument("--touch-test", action="store_true", help="also evaluate on season-2 test (once)")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting an existing test-evaluation record")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--downsample-rate", type=float, default=NEGATIVE_SAMPLING_RATE)
    parser.add_argument("--c-grid", type=str, default=",".join(str(c) for c in DEFAULT_C_GRID))
    parser.add_argument(
        "--dev-train-row-limit", type=int, default=300_000,
        help=(
            "row cap for --dev mode only. NOTE: DuckDB's LIMIT with no "
            "ORDER BY is not guaranteed stable across separate query "
            "executions (parallel scan order), so --dev row subsets can "
            "differ slightly run to run -- fine for fast iteration, but "
            "--full (no LIMIT, every row) is what's exactly reproducible "
            "and what gets reported."
        ),
    )
    parser.add_argument("--dev-val-row-limit", type=int, default=150_000)
    args = parser.parse_args(argv)

    if args.touch_test and not args.full:
        parser.error("--touch-test requires --full (never touch test from a --dev/sampled run)")

    seed = args.seed
    downsample_rate = args.downsample_rate
    c_grid = [float(c) for c in args.c_grid.split(",")]
    mode_name = "full" if args.full else "dev"

    train_row_limit = None if args.full else args.dev_train_row_limit
    val_row_limit = None if args.full else args.dev_val_row_limit

    print(f"[train_lr] mode={mode_name} seed={seed} downsample_rate={downsample_rate} c_grid={c_grid}")

    boundaries = split.get_split_boundaries(2)

    print("[train_lr] loading train split ...")
    X_train_raw, y_train_raw = split.load_split(2, "train", row_limit=train_row_limit)
    print(f"[train_lr] train raw: {X_train_raw.shape}, positives={int(y_train_raw.sum())}")

    X_train_ds, y_train_ds, ds_info = downsample_negatives(
        X_train_raw, y_train_raw, rate=downsample_rate, seed=seed
    )
    print(
        f"[train_lr] train downsampled: n={len(y_train_ds)} "
        f"positives={ds_info.n_positive} negatives={ds_info.n_negative_sampled} "
        f"(true negatives={ds_info.n_negative_true}) "
        f"pos_rate {ds_info.positive_rate_before:.6f} -> {ds_info.positive_rate_after:.6f}"
    )

    if args.full:
        downsample_record = {
            "season": 2,
            "split_boundaries": {
                "train": list(boundaries.train),
                "val": list(boundaries.val),
                "test": list(boundaries.test),
            },
            "seed": seed,
            **ds_info.to_dict(),
            "generated_at": _now_iso(),
        }
        _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
        with open(_DOWNSAMPLE_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(downsample_record, f, indent=2)
        print(f"[train_lr] wrote downsampling metadata -> {_DOWNSAMPLE_METADATA_PATH}")

    print("[train_lr] encoding train features ...")
    X_train_enc, scaler = features_lr.encode_features(X_train_ds, fit_scaler=True)
    print(f"[train_lr] train encoded: {X_train_enc.shape}, nnz={X_train_enc.nnz}")

    print("[train_lr] loading val split (true rate, never downsampled) ...")
    X_val_raw, y_val = split.load_split(2, "val", row_limit=val_row_limit)
    print(f"[train_lr] val: {X_val_raw.shape}, positives={int(y_val.sum())}")
    X_val_enc, _ = features_lr.encode_features(X_val_raw, scaler=scaler, fit_scaler=False)

    print(f"[train_lr] sweeping C over {c_grid} (selection metric: val log loss) ...")
    best = None
    for C in c_grid:
        clf, result, fit_seconds = _fit_and_score(
            X_train_enc, y_train_ds, X_val_enc, y_val,
            C=C, sampling_rate=ds_info.negative_sampling_rate,
        )
        print(
            f"[train_lr]   C={C:<8} auc={result.auc:.4f} log_loss={result.log_loss:.6f} "
            f"ece={result.ece:.6f} fit_seconds={fit_seconds:.1f}"
        )
        run_record = {
            "timestamp": _now_iso(),
            "mode": mode_name,
            "model_name": result.model_name,
            "split_evaluated": result.split,
            "hyperparameters": {"C": C, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000, "random_state": seed},
            "features": {
                "categorical_fields": features_lr.CATEGORICAL_FIELDS,
                "numeric_fields": features_lr.NUMERIC_FIELDS,
                "multi_valued_fields": ["usertag"],
                "derived_fields": ["hour"],
                "hash_dim": features_lr.HASH_DIM,
            },
            "split_boundaries": {
                "train": list(boundaries.train),
                "val": list(boundaries.val),
                "test": list(boundaries.test),
            },
            "downsampling": ds_info.to_dict(),
            "train_row_limit": train_row_limit,
            "val_row_limit": val_row_limit,
            "n_train_rows_used": int(X_train_enc.shape[0]),
            "n_val_rows_used": int(X_val_enc.shape[0]),
            "fit_seconds": fit_seconds,
            "metrics": result.to_dict(),
        }
        _append_run_log(run_record)

        if best is None or result.log_loss < best[1].log_loss:
            best = (clf, result, C)

    best_clf, best_result, best_C = best
    print(f"[train_lr] selected C={best_C} on val log_loss={best_result.log_loss:.6f} (auc={best_result.auc:.4f}, ece={best_result.ece:.6f})")

    if args.full:
        _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": best_clf,
                "scaler": scaler,
                "hash_dim": features_lr.HASH_DIM,
                "categorical_fields": features_lr.CATEGORICAL_FIELDS,
                "numeric_fields": features_lr.NUMERIC_FIELDS,
                "best_C": best_C,
                "seed": seed,
                "downsampling": ds_info.to_dict(),
                "split_boundaries": {
                    "train": list(boundaries.train),
                    "val": list(boundaries.val),
                    "test": list(boundaries.test),
                },
            },
            _MODEL_PATH,
        )
        print(f"[train_lr] saved fitted model + encoder -> {_MODEL_PATH}")

        save_reliability_plot(best_result, "part3_lr_calibration_val")
        print("[train_lr] saved val reliability plot -> reports/figures/part3_lr_calibration_val.{png,svg}")

    if args.touch_test:
        if _TEST_EVAL_PATH.exists() and not args.overwrite:
            print(
                f"[train_lr] REFUSING to touch test: {_TEST_EVAL_PATH} already exists. "
                "This protocol touches the test set once; pass --overwrite if this is "
                "a deliberate, reviewed re-run.",
                file=sys.stderr,
            )
            return 1

        print("[train_lr] loading test split (true rate, never downsampled) -- TOUCHING TEST SET ...")
        X_test_raw, y_test = split.load_split(2, "test")
        print(f"[train_lr] test: {X_test_raw.shape}, positives={int(y_test.sum())}")
        X_test_enc, _ = features_lr.encode_features(X_test_raw, scaler=scaler, fit_scaler=False)

        p_test_raw = best_clf.predict_proba(X_test_enc)[:, 1]
        test_result = evaluate(
            y_test, p_test_raw,
            model_name=f"lr_hashed_C={best_C}",
            split="season2_test",
            sampling_rate=ds_info.negative_sampling_rate,
        )
        print(
            f"[train_lr] TEST auc={test_result.auc:.4f} log_loss={test_result.log_loss:.6f} "
            f"ece={test_result.ece:.6f}"
        )

        test_record = {
            "timestamp": _now_iso(),
            "model_name": test_result.model_name,
            "selected_C": best_C,
            "hyperparameters": {"C": best_C, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000, "random_state": seed},
            "split_boundaries": {
                "train": list(boundaries.train),
                "val": list(boundaries.val),
                "test": list(boundaries.test),
            },
            "downsampling": ds_info.to_dict(),
            "val_selection_metrics": best_result.to_dict(),
            "metrics": test_result.to_dict(),
            "reliability_table": test_result.reliability_table.to_dict(orient="records"),
        }
        with open(_TEST_EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2)
        print(f"[train_lr] wrote test evaluation (single touch) -> {_TEST_EVAL_PATH}")

        save_reliability_plot(test_result, "part3_lr_calibration_test")
        print("[train_lr] saved test reliability plot -> reports/figures/part3_lr_calibration_test.{png,svg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
