"""Full-scale re-run of the LR feature ablation from the "Leakage
investigation" section of `docs/analysis/ctr-results.md`.

**Why this exists.** The original ablation table in that section used a
2M-row train sample / 800K-row val sample at `C=1.0` ("full field set" AUC
0.810), which is NOT comparable to the reported baseline (full train, full
val, `C=0.1`, val AUC 0.9299) -- two confounds (sample size, `C`) stacked
in one number, with no way to attribute the ~0.12 gap to either. This
script re-runs the exact same 4 feature sets at the exact same scale and
`C` as the reported baseline, so the table is internally comparable AND
comparable to the headline number.

**Exact configuration, matching the reported baseline (`train_lr.py`):**
  - `split.load_split(2, "train"/"val")`, `row_limit=None` (never a
    row-limited call anywhere in this module -- see
    `backend/tests/test_dataset_row_order.py`: `row_limit` selects a
    non-reproducible row SUBSET, which is itself one reason the old 0.810
    figure cannot be exactly regenerated even if someone tried).
  - Negative downsampling on TRAIN only, rate 0.02, seed 42 -- reconstructed
    here (not reloaded from a cache) and cross-checked field-for-field
    against the recorded `ctr_train_downsampling_season2.json` before any
    variant is fit, so a silent drift in the downsampling logic since that
    file was written would be caught immediately rather than silently
    producing a table against a different train set.
  - `LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
    random_state=42)` -- same as `train_lr.py`'s selected `C`.
  - Evaluated on val ONLY, through `modeling.evaluate.evaluate()`
    (`sampling_rate=0.02`), same harness every other model in this repo is
    scored by. Test is never touched by this script.
  - A fresh `StandardScaler` is fit per-variant on that variant's own
    encoded downsampled train (`fit_scaler=True`) and reused, unrefit, on
    val (`fit_scaler=False`) -- the saved baseline scaler is never reused,
    since its dense numeric block is fine to reuse in principle (the
    NUMERIC_FIELDS never change across variants) but refitting per variant
    costs nothing and removes any risk of a stale-artifact assumption.

**Efficiency note.** The full train/val raw frames and the one
downsampled-train `(X, y)` pair are each loaded/computed ONCE and reused
across all 4 variants (only the categorical field LIST differs between
variants; `NUMERIC_FIELDS` and the raw rows are identical) -- avoids 4x
redundant Parquet scans of the ~8.8M-row train table.

**Validation gate.** Variant 1 (full field set) must reproduce the
reported baseline val AUC (0.9299) and val log loss (0.0045260) to ~4
decimal places. If it does not, this script prints a loud discrepancy
warning and still writes its output (marked accordingly) rather than
silently proceeding as if nothing were wrong -- see `main()`.

**What this does NOT touch.** `ctr_lr_baseline_season2.joblib`,
`ctr_lr_test_evaluation_season2.json`,
`ctr_train_downsampling_season2.json`, and the predictions parquets are
read (the downsampling JSON, for cross-checking only) but never
overwritten. Output goes to a new file,
`ctr_lr_ablation_fullscale_season2.json`, and run-log records are tagged
`mode="ablation_fullscale"` so they can never be confused with the
baseline C-sweep rows in `ctr_lr_runs.jsonl`.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.ablation_fullscale
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from . import features_lr, split
from .downsample import downsample_negatives
from .evaluate import evaluate

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_RUN_LOG_PATH = _METADATA_ROOT / "ctr_lr_runs.jsonl"
_RECORDED_DOWNSAMPLE_PATH = _METADATA_ROOT / "ctr_train_downsampling_season2.json"
_OUTPUT_PATH = _METADATA_ROOT / "ctr_lr_ablation_fullscale_season2.json"

RANDOM_SEED = 42
NEGATIVE_SAMPLING_RATE = 0.02
SELECTED_C = 0.1

# Reported baseline (docs/analysis/ctr-results.md "Results" section,
# `C=0.1` row) -- variant 1 below must reproduce these to ~4 dp.
BASELINE_VAL_AUC = 0.9299
BASELINE_VAL_LOG_LOSS = 0.0045260
_GATE_ATOL = 5e-4  # ~4 decimal places

_ALL_CATEGORICAL = features_lr.CATEGORICAL_FIELDS

VARIANTS = [
    {
        "name": "full_field_set",
        "description": "all CATEGORICAL_FIELDS + NUMERIC_FIELDS + usertag + derived hour",
        "dropped": [],
    },
    {
        "name": "drop_ipinyouid",
        "description": "drop ipinyouid",
        "dropped": ["ipinyouid"],
    },
    {
        "name": "drop_identity_fields",
        "description": "drop ipinyouid, useragent, IP",
        "dropped": ["ipinyouid", "useragent", "IP"],
    },
    {
        "name": "drop_all_high_cardinality",
        "description": "drop domain, url, slotid, creative, useragent, IP, ipinyouid",
        "dropped": ["domain", "url", "slotid", "creative", "useragent", "IP", "ipinyouid"],
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_run_log(record: dict) -> None:
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _verify_downsample_matches_recorded(ds_info) -> None:
    if not _RECORDED_DOWNSAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{_RECORDED_DOWNSAMPLE_PATH} does not exist -- run "
            "`train_lr.py --full` at least once first so there is a "
            "recorded downsampling decision to cross-check this "
            "reconstruction against."
        )
    with open(_RECORDED_DOWNSAMPLE_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)

    mismatches = []
    for key in ("n_positive", "n_negative_true", "n_negative_sampled"):
        if int(recorded[key]) != int(getattr(ds_info, key)):
            mismatches.append(f"{key}: recorded={recorded[key]!r} reconstructed={getattr(ds_info, key)!r}")
    if float(recorded["negative_sampling_rate"]) != ds_info.negative_sampling_rate:
        mismatches.append(
            f"negative_sampling_rate: recorded={recorded['negative_sampling_rate']!r} "
            f"reconstructed={ds_info.negative_sampling_rate!r}"
        )
    if int(recorded["seed"]) != ds_info.seed:
        mismatches.append(f"seed: recorded={recorded['seed']!r} reconstructed={ds_info.seed!r}")

    if mismatches:
        raise AssertionError(
            "ablation_fullscale.py: reconstructed downsampling does NOT "
            f"match the recorded {_RECORDED_DOWNSAMPLE_PATH}. This means "
            "the ablation train set is not the same train set the "
            "reported baseline was fit on. Mismatches:\n  " + "\n  ".join(mismatches)
        )


def main(argv=None) -> int:
    boundaries = split.get_split_boundaries(2)
    t_start = time.time()

    print("[ablation_fullscale] loading FULL season-2 train split (row_limit=None) ...")
    t0 = time.time()
    X_train_raw, y_train_raw = split.load_split(2, "train", row_limit=None)
    print(f"[ablation_fullscale] train raw: {X_train_raw.shape}, positives={int(y_train_raw.sum())} ({time.time() - t0:.1f}s)")

    print("[ablation_fullscale] downsampling negatives (rate=0.02, seed=42) ...")
    X_train_ds, y_train_ds, ds_info = downsample_negatives(
        X_train_raw, y_train_raw, rate=NEGATIVE_SAMPLING_RATE, seed=RANDOM_SEED
    )
    print(
        f"[ablation_fullscale] train downsampled: n={len(y_train_ds)} "
        f"positives={ds_info.n_positive} negatives={ds_info.n_negative_sampled} "
        f"(true negatives={ds_info.n_negative_true})"
    )
    _verify_downsample_matches_recorded(ds_info)
    print("[ablation_fullscale] OK: reconstructed downsampling matches ctr_train_downsampling_season2.json exactly.")

    print("[ablation_fullscale] loading FULL season-2 val split (row_limit=None, never downsampled) ...")
    t0 = time.time()
    X_val_raw, y_val = split.load_split(2, "val", row_limit=None)
    print(f"[ablation_fullscale] val: {X_val_raw.shape}, positives={int(y_val.sum())} ({time.time() - t0:.1f}s)")

    results = []
    for i, variant in enumerate(VARIANTS, start=1):
        cat_fields = [f for f in _ALL_CATEGORICAL if f not in variant["dropped"]]
        print(
            f"[ablation_fullscale] variant {i}/4 '{variant['name']}': "
            f"{len(cat_fields)} categorical fields (dropped={variant['dropped']}) ..."
        )

        t0 = time.time()
        X_train_enc, scaler = features_lr.encode_features(
            X_train_ds, fit_scaler=True, categorical_fields=cat_fields
        )
        X_val_enc, _ = features_lr.encode_features(
            X_val_raw, scaler=scaler, fit_scaler=False, categorical_fields=cat_fields
        )
        encode_seconds = time.time() - t0

        clf = LogisticRegression(
            C=SELECTED_C, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED,
        )
        t0 = time.time()
        clf.fit(X_train_enc, y_train_ds)
        fit_seconds = time.time() - t0

        p_val_raw = clf.predict_proba(X_val_enc)[:, 1]
        result = evaluate(
            y_val, p_val_raw,
            model_name=f"lr_hashed_ablation_{variant['name']}_C={SELECTED_C}",
            split="season2_val",
            sampling_rate=ds_info.negative_sampling_rate,
        )
        print(
            f"[ablation_fullscale]   variant '{variant['name']}': auc={result.auc:.4f} "
            f"log_loss={result.log_loss:.6f} ece={result.ece:.6f} "
            f"encode_seconds={encode_seconds:.1f} fit_seconds={fit_seconds:.1f}"
        )

        run_record = {
            "timestamp": _now_iso(),
            "mode": "ablation_fullscale",
            "note": (
                "Full-scale feature-ablation run (see "
                "modeling.ablation_fullscale docstring) -- NOT part of the "
                "baseline C-sweep; comparable in scale/C to the reported "
                "baseline (full train, full val, C=0.1), unlike the older "
                "sampled ablation table in ctr-results.md."
            ),
            "variant_name": variant["name"],
            "variant_description": variant["description"],
            "dropped_categorical_fields": variant["dropped"],
            "categorical_fields_used": cat_fields,
            "model_name": result.model_name,
            "split_evaluated": result.split,
            "hyperparameters": {
                "C": SELECTED_C, "l1_ratio": 0.0, "solver": "lbfgs",
                "max_iter": 1000, "random_state": RANDOM_SEED,
            },
            "features": {
                "categorical_fields": cat_fields,
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
            "train_row_limit": None,
            "val_row_limit": None,
            "n_train_rows_used": int(X_train_enc.shape[0]),
            "n_val_rows_used": int(X_val_enc.shape[0]),
            "encode_seconds": encode_seconds,
            "fit_seconds": fit_seconds,
            "metrics": result.to_dict(),
        }
        _append_run_log(run_record)
        results.append(run_record)

    total_seconds = time.time() - t_start

    variant1 = results[0]
    auc_gate_ok = abs(variant1["metrics"]["auc"] - BASELINE_VAL_AUC) <= _GATE_ATOL
    ll_gate_ok = abs(variant1["metrics"]["log_loss"] - BASELINE_VAL_LOG_LOSS) <= _GATE_ATOL
    gate_passed = auc_gate_ok and ll_gate_ok

    print("=" * 70)
    if gate_passed:
        print(
            f"[ablation_fullscale] VALIDATION GATE PASSED: variant 1 (full field "
            f"set) val AUC={variant1['metrics']['auc']:.4f} (baseline {BASELINE_VAL_AUC}), "
            f"val log_loss={variant1['metrics']['log_loss']:.7f} (baseline {BASELINE_VAL_LOG_LOSS})."
        )
    else:
        print(
            "[ablation_fullscale] VALIDATION GATE FAILED: variant 1 (full field "
            f"set) val AUC={variant1['metrics']['auc']:.4f} vs baseline "
            f"{BASELINE_VAL_AUC} (diff={abs(variant1['metrics']['auc'] - BASELINE_VAL_AUC):.4f}), "
            f"val log_loss={variant1['metrics']['log_loss']:.7f} vs baseline "
            f"{BASELINE_VAL_LOG_LOSS} (diff={abs(variant1['metrics']['log_loss'] - BASELINE_VAL_LOG_LOSS):.7f}). "
            "This means the reported baseline is NOT reproducible by this "
            "script's forward pass -- investigate before trusting the "
            "ablation table below."
        )
    print("=" * 70)

    output = {
        "generated_at": _now_iso(),
        "purpose": (
            "Full-scale (full train, full val, C=0.1) re-run of the 4-variant "
            "feature ablation originally reported at 2M/800K-row sample scale "
            "and C=1.0 in docs/analysis/ctr-results.md 'Leakage investigation' "
            "-- see that section, now superseded by this table for the "
            "scale-vs-C decomposition question."
        ),
        "config": {
            "seed": RANDOM_SEED,
            "negative_sampling_rate": NEGATIVE_SAMPLING_RATE,
            "C": SELECTED_C,
            "solver": "lbfgs",
            "max_iter": 1000,
            "train_row_limit": None,
            "val_row_limit": None,
        },
        "validation_gate": {
            "baseline_val_auc": BASELINE_VAL_AUC,
            "baseline_val_log_loss": BASELINE_VAL_LOG_LOSS,
            "variant1_val_auc": variant1["metrics"]["auc"],
            "variant1_val_log_loss": variant1["metrics"]["log_loss"],
            "atol": _GATE_ATOL,
            "passed": gate_passed,
        },
        "downsampling": ds_info.to_dict(),
        "split_boundaries": {
            "train": list(boundaries.train),
            "val": list(boundaries.val),
            "test": list(boundaries.test),
        },
        "variants": results,
        "total_runtime_seconds": total_seconds,
    }
    _METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[ablation_fullscale] wrote {_OUTPUT_PATH}")
    print(f"[ablation_fullscale] total runtime: {total_seconds:.1f}s")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
