"""LightGBM CTR model: native-categorical features (`modeling.features_lgbm`),
temporal split (`modeling.split`), negative downsampling on train only
(`modeling.downsample`), scored by the exact same shared evaluation harness
as the LR baseline (`modeling.evaluate`) -- mirrors `train_lr.py`'s
orchestration pattern and guardrails so the two models are comparable on
every axis except the model + encoding itself.

**Reuses, does not re-choose, the downsampling decision.** The negative
sampling rate (`0.02`) and seed (`42`) are the same fixed values
`train_lr.py` already used and persisted to
`backend/data/metadata/ctr_train_downsampling_season2.json`. This script
recomputes `downsample_negatives(X_train_raw, y_train_raw, rate=0.02,
seed=42)` (same call, same inputs) and **asserts** the resulting counts
match that file exactly -- it does NOT overwrite that file (it is shared
provenance for the LR baseline too, and re-deriving the identical numbers
is itself the check that nothing about the split/loader/downsampling
logic silently drifted since the LR baseline was trained).

**Encode once, reuse across every configuration.** `features_lgbm.
encode_features()` is called exactly once for train (`fit=True`) and once
for val (`fit=False`, reusing the train-fit `CategoryEncoderState`) before
the hyperparameter search loop begins. The resulting `lightgbm.Dataset`
objects (`train_set`, `val_set`) are built once and reused, unmodified,
across all 8 configurations in `CONFIG_GRID` -- only `lgb.train()`'s
`params` argument differs per configuration, never the underlying data.

**Hyperparameter grid: 8 hand-picked configurations, not a cross-product.**
See `CONFIG_GRID` below. Each configuration perturbs 1-2 axes away from a
baseline (`num_leaves=31, learning_rate=0.05, min_data_in_leaf=20,
feature_fraction=0.9, bagging_fraction=0.8, lambda_l2=0.0`) rather than
exhaustively combining every axis -- `num_leaves` (tree complexity: 15,
31, 63, 127), `learning_rate` (0.02, 0.05, 0.1), `min_data_in_leaf` (20,
50, 100 -- directly relevant here given the rare-categorical-value
overfitting risk documented in `features_lgbm.py`'s module docstring),
`feature_fraction`/`bagging_fraction` (row/column subsampling, 0.6-0.9),
and `lambda_l2` (0.0-5.0). `num_boost_round` (i.e. tree count) is NOT
swept explicitly -- it is selected per-configuration via LightGBM's own
early stopping against the val set (`EARLY_STOPPING_ROUNDS=50`,
`MAX_BOOST_ROUNDS` upper bound), which is model-selection-on-val, exactly
as sanctioned by the top-level protocol ("Use early stopping on the val
set for n_estimators -- that is selection on val, which is allowed").

**Two distinct notions of "log loss" appear in this script, deliberately
kept separate.** LightGBM's own early-stopping criterion
(`metric="binary_logloss"`, evaluated against `val_set`) uses the model's
RAW (uncalibrated, downsampled-scale) predicted probabilities -- standard
LightGBM practice, and a reasonable stopping signal since
`downsample.recalibrate_probability()` is a strictly monotone function of
the raw probability for a fixed sampling rate. It is used ONLY to pick
each configuration's tree count (`best_iteration`). The metric that
actually SELECTS among the 8 configurations is the properly recalibrated
val log loss reported by `modeling.evaluate.evaluate()` (same harness,
same recalibration correction, same selection criterion `train_lr.py`
uses) -- the two never get conflated in this script's config-selection
logic.

**Selection metric: validation log loss** (recalibrated, via
`evaluate()`), matching the LR baseline's selection criterion exactly, so
the two models are chosen by the same rule.

**Test touched once.** `--touch-test` (gated behind `--full`) evaluates
the single selected configuration against season-2 test exactly once and
refuses to overwrite an existing
`ctr_lgbm_test_evaluation_season2.json` without `--overwrite`.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lgbm --dev
    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lgbm --full
    backend\\.venv\\Scripts\\python.exe -m src.modeling.train_lgbm --full --touch-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from . import features_lgbm, split
from .downsample import downsample_negatives
from .evaluate import evaluate, save_reliability_plot

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_MODELS_ROOT = _REPO_ROOT / "backend" / "data" / "models"
_RUN_LOG_PATH = _METADATA_ROOT / "ctr_lgbm_runs.jsonl"
_DOWNSAMPLE_METADATA_PATH = _METADATA_ROOT / "ctr_train_downsampling_season2.json"
_TEST_EVAL_PATH = _METADATA_ROOT / "ctr_lgbm_test_evaluation_season2.json"
_MODEL_PATH = _MODELS_ROOT / "ctr_lgbm_season2.joblib"

# Fixed everywhere in this script -- same discipline as train_lr.py.
RANDOM_SEED = 42
NEGATIVE_SAMPLING_RATE = 0.02  # reused, not re-chosen -- see module docstring

MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50

# 8 hand-picked configurations -- see module docstring for the axes swept
# and why a full cross-product was not used.
CONFIG_GRID = [
    {
        "name": "c1_baseline",
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 0.0,
    },
    {
        "name": "c2_shallow_reg",
        "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0,
    },
    {
        "name": "c3_deep",
        "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 20,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 0.0,
    },
    {
        "name": "c4_fast_lr",
        "num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 20,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 0.0,
    },
    {
        "name": "c5_slow_lr",
        "num_leaves": 31, "learning_rate": 0.02, "min_data_in_leaf": 20,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 0.0,
    },
    {
        "name": "c6_strong_reg",
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 100,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
        "lambda_l2": 5.0,
    },
    {
        "name": "c7_very_deep_reg",
        "num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 50,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0,
    },
    {
        "name": "c8_low_feature_frac",
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
        "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 1,
        "lambda_l2": 0.1,
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_run_log(record: dict) -> None:
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _assert_downsample_matches_recorded(ds_info, *, seed: int, rate: float) -> None:
    """Reused, not re-chosen: cross-check this run's downsampling against
    the already-recorded `ctr_train_downsampling_season2.json` (produced by
    `train_lr.py --full`). Only performed when rate/seed match the fixed
    defaults this file documents reusing -- a deliberately different
    `--downsample-rate`/`--seed` (e.g. for a future ablation) skips the
    check rather than failing against a record that was never meant to
    apply to it.
    """
    if rate != NEGATIVE_SAMPLING_RATE or seed != RANDOM_SEED:
        print(
            f"[train_lgbm] downsample rate/seed ({rate}/{seed}) differ from "
            f"the fixed reused defaults ({NEGATIVE_SAMPLING_RATE}/{RANDOM_SEED}) "
            "-- skipping the cross-check against the recorded LR downsampling "
            "metadata (that record only applies to the default rate/seed)."
        )
        return
    if not _DOWNSAMPLE_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{_DOWNSAMPLE_METADATA_PATH} does not exist -- run "
            "`train_lr.py --full` first so there is a recorded downsampling "
            "decision to reuse and cross-check against (see module docstring: "
            "this script reuses, not re-chooses, that decision)."
        )
    with open(_DOWNSAMPLE_METADATA_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    mismatches = []
    for key in ("n_positive", "n_negative_true", "n_negative_sampled"):
        if int(recorded[key]) != int(getattr(ds_info, key)):
            mismatches.append(f"{key}: recorded={recorded[key]!r} this_run={getattr(ds_info, key)!r}")
    if float(recorded["negative_sampling_rate"]) != ds_info.negative_sampling_rate:
        mismatches.append(
            f"negative_sampling_rate: recorded={recorded['negative_sampling_rate']!r} "
            f"this_run={ds_info.negative_sampling_rate!r}"
        )
    if int(recorded["seed"]) != ds_info.seed:
        mismatches.append(f"seed: recorded={recorded['seed']!r} this_run={ds_info.seed!r}")
    if mismatches:
        raise AssertionError(
            "train_lgbm.py: this run's downsampled train does NOT match the "
            f"recorded {_DOWNSAMPLE_METADATA_PATH} exactly -- stopping rather "
            "than proceeding on a train set that silently diverged from the "
            "LR baseline's. Mismatches:\n  " + "\n  ".join(mismatches)
        )
    print(f"[train_lgbm] OK: downsampled train matches {_DOWNSAMPLE_METADATA_PATH} exactly.")


def _lgbm_params(config: dict, *, seed: int) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": config["num_leaves"],
        "learning_rate": config["learning_rate"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "bagging_freq": config["bagging_freq"],
        "lambda_l2": config["lambda_l2"],
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="small row_limit, fast iteration")
    mode.add_argument("--full", action="store_true", help="full season-2 train/val (and test, with --touch-test)")
    parser.add_argument("--touch-test", action="store_true", help="also evaluate on season-2 test (once)")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting an existing test-evaluation record")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--downsample-rate", type=float, default=NEGATIVE_SAMPLING_RATE)
    parser.add_argument("--min-frequency", type=int, default=features_lgbm.MIN_FREQUENCY)
    parser.add_argument(
        "--dev-train-row-limit", type=int, default=300_000,
        help="row cap for --dev mode only -- see train_lr.py's identical flag for the DuckDB LIMIT caveat.",
    )
    parser.add_argument("--dev-val-row-limit", type=int, default=150_000)
    args = parser.parse_args(argv)

    if args.touch_test and not args.full:
        parser.error("--touch-test requires --full (never touch test from a --dev/sampled run)")

    seed = args.seed
    downsample_rate = args.downsample_rate
    min_frequency = args.min_frequency
    mode_name = "full" if args.full else "dev"

    train_row_limit = None if args.full else args.dev_train_row_limit
    val_row_limit = None if args.full else args.dev_val_row_limit

    print(
        f"[train_lgbm] mode={mode_name} seed={seed} downsample_rate={downsample_rate} "
        f"min_frequency={min_frequency} n_configs={len(CONFIG_GRID)}"
    )

    boundaries = split.get_split_boundaries(2)

    print("[train_lgbm] loading train split ...")
    X_train_raw, y_train_raw = split.load_split(2, "train", row_limit=train_row_limit)
    print(f"[train_lgbm] train raw: {X_train_raw.shape}, positives={int(y_train_raw.sum())}")

    X_train_ds, y_train_ds, ds_info = downsample_negatives(
        X_train_raw, y_train_raw, rate=downsample_rate, seed=seed
    )
    print(
        f"[train_lgbm] train downsampled: n={len(y_train_ds)} "
        f"positives={ds_info.n_positive} negatives={ds_info.n_negative_sampled} "
        f"(true negatives={ds_info.n_negative_true}) "
        f"pos_rate {ds_info.positive_rate_before:.6f} -> {ds_info.positive_rate_after:.6f}"
    )

    if args.full:
        _assert_downsample_matches_recorded(ds_info, seed=seed, rate=downsample_rate)

    print("[train_lgbm] encoding train features (fit=True, native categorical) ...")
    X_train_enc, cat_state, categorical_feature_names = features_lgbm.encode_features(
        X_train_ds, fit=True, min_frequency=min_frequency
    )
    print(f"[train_lgbm] train encoded: {X_train_enc.shape}")
    print(f"[train_lgbm] per-field cardinality (post min_frequency={min_frequency} thresholding): {cat_state.cardinality}")
    print(f"[train_lgbm] usertag vocab size: {len(cat_state.usertag_vocab)}")

    print("[train_lgbm] loading val split (true rate, never downsampled) ...")
    X_val_raw, y_val = split.load_split(2, "val", row_limit=val_row_limit)
    print(f"[train_lgbm] val: {X_val_raw.shape}, positives={int(y_val.sum())}")
    X_val_enc, _, _ = features_lgbm.encode_features(X_val_raw, state=cat_state, fit=False)

    print("[train_lgbm] building lightgbm.Dataset objects ONCE (reused across every config) ...")
    train_set = lgb.Dataset(
        X_train_enc, label=y_train_ds.to_numpy(),
        categorical_feature=categorical_feature_names, free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val_enc, label=y_val.to_numpy(), reference=train_set,
        categorical_feature=categorical_feature_names, free_raw_data=False,
    )

    print(f"[train_lgbm] sweeping {len(CONFIG_GRID)} configs (selection metric: recalibrated val log loss) ...")
    best = None
    for config in CONFIG_GRID:
        params = _lgbm_params(config, seed=seed)
        t0 = time.time()
        booster = lgb.train(
            params, train_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        fit_seconds = time.time() - t0
        best_iteration = booster.best_iteration

        p_val_raw = booster.predict(X_val_enc, num_iteration=best_iteration)
        result = evaluate(
            y_val, p_val_raw,
            model_name=f"lgbm_{config['name']}",
            split="season2_val",
            sampling_rate=ds_info.negative_sampling_rate,
        )
        print(
            f"[train_lgbm]   {config['name']:<20} best_iter={best_iteration:<5} "
            f"auc={result.auc:.4f} log_loss={result.log_loss:.6f} ece={result.ece:.6f} "
            f"fit_seconds={fit_seconds:.1f}"
        )

        run_record = {
            "timestamp": _now_iso(),
            "mode": mode_name,
            "model_name": result.model_name,
            "split_evaluated": result.split,
            "config_name": config["name"],
            "hyperparameters": {**params, "num_boost_round_max": MAX_BOOST_ROUNDS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
            "best_iteration": int(best_iteration),
            "features": {
                "categorical_fields": categorical_feature_names,
                "numeric_fields": features_lgbm.NUMERIC_FIELDS,
                "multi_valued_fields": ["usertag"],
                "usertag_vocab_size": len(cat_state.usertag_vocab),
                "min_frequency": min_frequency,
                "per_field_cardinality": cat_state.cardinality,
                "encoding": "native_categorical",
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
            best = (booster, result, config, best_iteration)

    best_booster, best_result, best_config, best_iteration = best
    print(
        f"[train_lgbm] selected config={best_config['name']} (best_iteration={best_iteration}) "
        f"on val log_loss={best_result.log_loss:.6f} (auc={best_result.auc:.4f}, ece={best_result.ece:.6f})"
    )

    if args.full:
        _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": best_booster,
                "cat_state": cat_state,
                "categorical_feature_names": categorical_feature_names,
                "numeric_fields": features_lgbm.NUMERIC_FIELDS,
                "best_config": best_config,
                "best_iteration": best_iteration,
                "min_frequency": min_frequency,
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
        print(f"[train_lgbm] saved fitted model + encoder state -> {_MODEL_PATH}")

        save_reliability_plot(best_result, "part4_lgbm_calibration_val")
        print("[train_lgbm] saved val reliability plot -> reports/figures/part4_lgbm_calibration_val.{png,svg}")

    if args.touch_test:
        if _TEST_EVAL_PATH.exists() and not args.overwrite:
            print(
                f"[train_lgbm] REFUSING to touch test: {_TEST_EVAL_PATH} already exists. "
                "This protocol touches the test set once; pass --overwrite if this is "
                "a deliberate, reviewed re-run.",
                file=sys.stderr,
            )
            return 1

        print("[train_lgbm] loading test split (true rate, never downsampled) -- TOUCHING TEST SET ...")
        X_test_raw, y_test = split.load_split(2, "test")
        print(f"[train_lgbm] test: {X_test_raw.shape}, positives={int(y_test.sum())}")
        X_test_enc, _, _ = features_lgbm.encode_features(X_test_raw, state=cat_state, fit=False)

        p_test_raw = best_booster.predict(X_test_enc, num_iteration=best_iteration)
        test_result = evaluate(
            y_test, p_test_raw,
            model_name=f"lgbm_{best_config['name']}",
            split="season2_test",
            sampling_rate=ds_info.negative_sampling_rate,
        )
        print(
            f"[train_lgbm] TEST auc={test_result.auc:.4f} log_loss={test_result.log_loss:.6f} "
            f"ece={test_result.ece:.6f}"
        )

        test_record = {
            "timestamp": _now_iso(),
            "model_name": test_result.model_name,
            "selected_config": best_config,
            "best_iteration": int(best_iteration),
            "hyperparameters": {**_lgbm_params(best_config, seed=seed), "num_boost_round_max": MAX_BOOST_ROUNDS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
            "features": {
                "categorical_fields": categorical_feature_names,
                "numeric_fields": features_lgbm.NUMERIC_FIELDS,
                "multi_valued_fields": ["usertag"],
                "usertag_vocab_size": len(cat_state.usertag_vocab),
                "min_frequency": min_frequency,
                "per_field_cardinality": cat_state.cardinality,
                "encoding": "native_categorical",
            },
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
        print(f"[train_lgbm] wrote test evaluation (single touch) -> {_TEST_EVAL_PATH}")

        save_reliability_plot(test_result, "part4_lgbm_calibration_test")
        print("[train_lgbm] saved test reliability plot -> reports/figures/part4_lgbm_calibration_test.{png,svg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
