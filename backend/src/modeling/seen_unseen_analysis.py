"""Placement-overlap analysis: how much of the LR baseline's test-set
discrimination depends on having seen the exact `(domain, slotid,
creative)` placement triple during training.

**Pure analysis over already-persisted artifacts -- no retraining, no
refitting, no re-scoring.** This script never fits a model or a scaler.
It reloads the already-fitted-and-scored season-2 test predictions
(`backend/data/predictions/ctr_lr_baseline_season2_test.parquet`, produced
once by `score_test.py`) and the raw season-2 train/test feature splits
(via `modeling.split.load_split()`, which itself never fits anything --
see that module), and computes ranking metrics on subsets of already-
computed `p_calibrated` values. Nothing here changes any prediction.

**Why the raw test split has to be reloaded and aligned positionally.**
The persisted predictions parquet carries only `y_true`, `p_raw`,
`p_calibrated`, `advertiser`, `timestamp` -- not the placement fields
(`domain`, `slotid`, `creative`), which were never part of that script's
`_RAW_CARRY_COLUMNS`. `features/dataset.py::load_impression_features()`
guarantees a deterministic row order across separate calls **only when
`row_limit=None`** (see that module's "DETERMINISTIC ROW ORDER" docstring
section and `backend/tests/test_dataset_row_order.py`) -- exactly the
case used here and in `score_test.py`'s own test-split load. That
guarantee is verified empirically in this run (not merely assumed from
the docstring) via `_assert_test_split_aligns_with_predictions()` below:
element-wise equality of `y_true`, `advertiser`, and `timestamp` between
the freshly-reloaded split and the persisted parquet, across all rows.
If that assertion fails, this script stops -- no fuzzy/key-based join is
attempted as a fallback (there is no natural join key available; `bidid`
is excluded from the admitted feature set by construction, see
`schema.py`).

**Null-handling policy for the triple key.** Verified directly against
season-2 train (2013-06-06..06-10): `domain` has real (pandas `NaN`)
missing values (confirmed via a direct query, no literal `"null"`-string
values in `domain`/`slotid`/`creative` anywhere in this dataset; `slotid`
and `creative` have zero nulls). A missing `domain` is mapped to the same
explicit sentinel token `features_lr.NULL_TOKEN` ("__null__") already
used by the LR baseline's own null-handling (`features_lr._field_tokens`)
before the triple key is built, rather than being silently coerced into a
single ambiguous pseudo-value by e.g. `str(NaN)` (`"nan"` would work by
accident, but relying on that is not a documented decision). This means:
two test rows with a missing `domain` and identical `slotid`/`creative`
are treated as the SAME triple (matching train rows with a missing domain
and the same `slotid`/`creative` count as "seen"), which is a deliberate,
stated choice, not an unremarked collapse -- consistent with
`features_lr.py`'s own treatment of missing categoricals as their own
explicit category rather than dropped rows.

**Two distinct "seen in train" definitions -- reported separately, never
conflated.** The natural reading of "seen in train" is the FULL train
split (`split.load_split(2, "train")`, all ~8.83M rows) -- this is the
headline number. But the LR baseline was actually FIT on the
NEGATIVE-DOWNSAMPLED train set (2% of true negatives kept, seed 42; see
`backend/data/metadata/ctr_train_downsampling_season2.json`), so many
triples present in the full train split were never actually seen by the
fitted model. This script reconstructs that downsampled set via
`downsample.downsample_negatives()` at the exact recorded rate/seed
(RESAMPLING an existing split deterministically -- not retraining
anything) and reports the seen-fraction against it too, labeled distinctly
from the full-train figure.

Usage (from `backend/`, using the project venv):

    backend\\.venv\\Scripts\\python.exe -m src.modeling.seen_unseen_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features_lr, split
from .downsample import downsample_negatives

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METADATA_ROOT = _REPO_ROOT / "backend" / "data" / "metadata"
_PREDICTIONS_ROOT = _REPO_ROOT / "backend" / "data" / "predictions"

_PREDICTIONS_PATH = _PREDICTIONS_ROOT / "ctr_lr_baseline_season2_test.parquet"
_DOWNSAMPLE_METADATA_PATH = _METADATA_ROOT / "ctr_train_downsampling_season2.json"
_OUTPUT_FLAGS_PATH = _PREDICTIONS_ROOT / "ctr_lr_baseline_season2_test_seen_flags.parquet"

_TRIPLE_FIELDS = ["domain", "slotid", "creative"]
_SEP = "\x1f"  # ASCII unit separator -- not expected to occur in any of these fields' raw values


def _build_triple_key(X: pd.DataFrame) -> pd.Series:
    """Vectorized `(domain, slotid, creative)` composite key, one string
    per row. `domain` nulls are mapped to `features_lr.NULL_TOKEN`
    (`"__null__"`) before concatenation -- see module docstring's
    "Null-handling policy" section. `slotid`/`creative` are cast to `str`
    defensively (verified zero-null in this dataset, but not assumed
    forever).
    """
    domain = X["domain"].astype(object).where(X["domain"].notna(), features_lr.NULL_TOKEN).astype(str)
    slotid = X["slotid"].astype(str)
    creative = X["creative"].astype(str)
    return domain + _SEP + slotid + _SEP + creative


def _assert_test_split_aligns_with_predictions(X_test_raw: pd.DataFrame, y_test: pd.Series, preds: pd.DataFrame) -> None:
    """Empirically verify (not merely assume) that the freshly-reloaded
    raw test split lines up row-for-row with the persisted predictions
    parquet, per the module docstring. Compares `y_true`, `advertiser`,
    `timestamp` element-wise across ALL rows. Raises `AssertionError` on
    any mismatch -- this script does not proceed past this check on a
    failure, and does not fall back to a fuzzy or key-based join.
    """
    if len(X_test_raw) != len(preds):
        raise AssertionError(
            "row count mismatch: raw test split has "
            f"{len(X_test_raw)} rows, predictions parquet has "
            f"{len(preds)} rows -- cannot align positionally."
        )

    y_test_arr = np.asarray(y_test, dtype=np.int64)
    y_pred_arr = np.asarray(preds["y_true"], dtype=np.int64)
    y_match = np.array_equal(y_test_arr, y_pred_arr)

    adv_raw = X_test_raw["advertiser"].to_numpy()
    adv_pred = preds["advertiser"].to_numpy()
    adv_match = np.array_equal(adv_raw, adv_pred)

    ts_raw = pd.to_datetime(X_test_raw["timestamp"]).to_numpy()
    ts_pred = pd.to_datetime(preds["timestamp"]).to_numpy()
    ts_match = np.array_equal(ts_raw, ts_pred)

    if not (y_match and adv_match and ts_match):
        raise AssertionError(
            "seen_unseen_analysis: raw test split does NOT align "
            f"element-wise with {_PREDICTIONS_PATH} -- "
            f"y_true match={y_match}, advertiser match={adv_match}, "
            f"timestamp match={ts_match}. Refusing to proceed with a "
            "positional join; no fuzzy/key-based fallback is used (see "
            "module docstring)."
        )


def _auc_or_na(y_true: np.ndarray, p: np.ndarray) -> tuple:
    """`roc_auc_score`, or `("n/a", reason)` if undefined (all-one-class
    subset) -- same convention as the per-advertiser table in
    `docs/analysis/ctr-results.md` (never silently 0.5, never dropped).
    """
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0:
        return "n/a", "zero positives"
    if n_neg == 0:
        return "n/a", "zero negatives"
    return float(roc_auc_score(y_true, p)), ""


def main() -> int:
    print(f"[seen_unseen] loading persisted predictions -> {_PREDICTIONS_PATH}")
    preds = pd.read_parquet(_PREDICTIONS_PATH)
    print(
        f"[seen_unseen] predictions: {len(preds):,} rows, "
        f"{int(preds['y_true'].sum())} positives"
    )

    print("[seen_unseen] reloading season-2 test split (row_limit=None, per module docstring) ...")
    X_test_raw, y_test = split.load_split(2, "test")
    print(f"[seen_unseen] raw test split: {X_test_raw.shape}, positives={int(y_test.sum())}")

    print("[seen_unseen] verifying positional alignment against the predictions parquet ...")
    _assert_test_split_aligns_with_predictions(X_test_raw, y_test, preds)
    print("[seen_unseen] OK: y_true, advertiser, timestamp match element-wise across all rows.")

    n_domain_null = int(X_test_raw["domain"].isna().sum())
    print(
        f"[seen_unseen] test split: domain nulls = {n_domain_null:,} "
        f"({n_domain_null / len(X_test_raw):.4%}); slotid/creative nulls verified zero at build time."
    )

    test_triple = _build_triple_key(X_test_raw)

    print("[seen_unseen] loading FULL season-2 train split (domain, slotid, creative only; row_limit=None) ...")
    X_train_full, y_train_full = split.load_split(2, "train", columns=_TRIPLE_FIELDS)
    print(f"[seen_unseen] full train: {X_train_full.shape}, positives={int(y_train_full.sum())}")
    full_train_triples = set(_build_triple_key(X_train_full).unique().tolist())
    print(f"[seen_unseen] distinct (domain, slotid, creative) triples in FULL train: {len(full_train_triples):,}")

    print(
        "[seen_unseen] reconstructing DOWNSAMPLED train via downsample_negatives() "
        f"(rate/seed from {_DOWNSAMPLE_METADATA_PATH}) ..."
    )
    with open(_DOWNSAMPLE_METADATA_PATH, "r", encoding="utf-8") as f:
        ds_meta = json.load(f)
    rate = ds_meta["negative_sampling_rate"]
    seed = ds_meta["seed"]
    X_train_ds, y_train_ds, ds_info = downsample_negatives(X_train_full, y_train_full, rate=rate, seed=seed)
    print(f"[seen_unseen] downsampled train: {X_train_ds.shape}, positives={int(y_train_ds.sum())}")

    ds_mismatches = []
    for key in ("n_positive", "n_negative_true", "n_negative_sampled"):
        recorded = ds_meta[key]
        reproduced = getattr(ds_info, key)
        if int(recorded) != int(reproduced):
            ds_mismatches.append(f"{key}: recorded={recorded!r} reproduced={reproduced!r}")
    if ds_mismatches:
        raise AssertionError(
            "seen_unseen_analysis: reconstructed downsampling does not match "
            f"{_DOWNSAMPLE_METADATA_PATH}:\n  " + "\n  ".join(ds_mismatches)
        )
    print(
        "[seen_unseen] OK: reconstructed downsampled train matches recorded metadata exactly "
        "(n_positive, n_negative_true, n_negative_sampled)."
    )

    ds_train_triples = set(_build_triple_key(X_train_ds).unique().tolist())
    print(f"[seen_unseen] distinct (domain, slotid, creative) triples in DOWNSAMPLED train: {len(ds_train_triples):,}")

    y_true_arr = np.asarray(y_test, dtype=np.int64)
    p_calibrated = preds["p_calibrated"].to_numpy(dtype=np.float64)
    n_total = len(y_true_arr)
    n_positive_total = int(y_true_arr.sum())

    seen_full = test_triple.isin(full_train_triples).to_numpy()
    seen_ds = test_triple.isin(ds_train_triples).to_numpy()

    # Sanity checks (per task): counts must partition exactly.
    assert int(seen_full.sum()) + int((~seen_full).sum()) == n_total
    assert int(y_true_arr[seen_full].sum()) + int(y_true_arr[~seen_full].sum()) == n_positive_total
    assert int(seen_ds.sum()) + int((~seen_ds).sum()) == n_total
    assert int(y_true_arr[seen_ds].sum()) + int(y_true_arr[~seen_ds].sum()) == n_positive_total

    def _bucket_stats(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        pos = int(y_true_arr[mask].sum())
        rate = pos / n if n else float("nan")
        auc, reason = _auc_or_na(y_true_arr[mask], p_calibrated[mask])
        return {"n": n, "positives": pos, "positive_rate": rate, "auc": auc, "auc_na_reason": reason}

    full_seen_stats = _bucket_stats(seen_full)
    full_unseen_stats = _bucket_stats(~seen_full)
    ds_seen_stats = _bucket_stats(seen_ds)
    ds_unseen_stats = _bucket_stats(~seen_ds)

    print("\n=== FULL-TRAIN definition of 'seen' ===")
    print(
        f"  seen:   n={full_seen_stats['n']:,} ({full_seen_stats['n']/n_total:.4%} of rows), "
        f"positives={full_seen_stats['positives']} "
        f"({full_seen_stats['positives']/n_positive_total:.4%} of clicks), "
        f"positive_rate={full_seen_stats['positive_rate']:.6%}, auc={full_seen_stats['auc']}"
    )
    print(
        f"  unseen: n={full_unseen_stats['n']:,} ({full_unseen_stats['n']/n_total:.4%} of rows), "
        f"positives={full_unseen_stats['positives']} "
        f"({full_unseen_stats['positives']/n_positive_total:.4%} of clicks), "
        f"positive_rate={full_unseen_stats['positive_rate']:.6%}, auc={full_unseen_stats['auc']}"
    )

    print("\n=== DOWNSAMPLED-TRAIN definition of 'seen' ===")
    print(
        f"  seen:   n={ds_seen_stats['n']:,} ({ds_seen_stats['n']/n_total:.4%} of rows), "
        f"positives={ds_seen_stats['positives']} "
        f"({ds_seen_stats['positives']/n_positive_total:.4%} of clicks), "
        f"positive_rate={ds_seen_stats['positive_rate']:.6%}, auc={ds_seen_stats['auc']}"
    )
    print(
        f"  unseen: n={ds_unseen_stats['n']:,} ({ds_unseen_stats['n']/n_total:.4%} of rows), "
        f"positives={ds_unseen_stats['positives']} "
        f"({ds_unseen_stats['positives']/n_positive_total:.4%} of clicks), "
        f"positive_rate={ds_unseen_stats['positive_rate']:.6%}, auc={ds_unseen_stats['auc']}"
    )

    print("\n[seen_unseen] writing companion seen/unseen flags parquet ...")
    flags_df = pd.DataFrame(
        {
            "seen_in_full_train": seen_full,
            "seen_in_downsampled_train": seen_ds,
        }
    )
    _PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    flags_df.to_parquet(_OUTPUT_FLAGS_PATH, index=False)
    print(f"[seen_unseen] wrote {len(flags_df):,} rows -> {_OUTPUT_FLAGS_PATH}")

    results = {
        "n_total": n_total,
        "n_positive_total": n_positive_total,
        "n_domain_null_test": n_domain_null,
        "full_train": {
            "n_distinct_triples": len(full_train_triples),
            "seen": full_seen_stats,
            "unseen": full_unseen_stats,
        },
        "downsampled_train": {
            "negative_sampling_rate": rate,
            "seed": seed,
            "n_distinct_triples": len(ds_train_triples),
            "seen": ds_seen_stats,
            "unseen": ds_unseen_stats,
        },
    }
    results_path = _METADATA_ROOT / "ctr_seen_unseen_placement_analysis_season2.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[seen_unseen] wrote results -> {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
