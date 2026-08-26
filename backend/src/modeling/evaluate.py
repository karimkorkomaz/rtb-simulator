"""Reusable evaluation harness for CTR models.

Operates on the TRUE-rate val/test distribution only (never a downsampled
one -- see `modeling.downsample` and the top-level protocol). Every model
scored by this harness -- the logistic-regression baseline here, LightGBM
later -- goes through the exact same function, so a results table entry is
always comparable to every other row in it.

**Why AUC and log loss, never accuracy.** The positive (click) rate is
roughly 0.075% (see `docs/analysis/eda-findings.md` §2, and the verified
per-split rates in `modeling.split`). A trivial "always predict no click"
classifier scores >99.9% accuracy while carrying zero ranking information
and being useless for bid shading or budget allocation -- accuracy cannot
distinguish that trivial model from a genuinely useful one at this
imbalance. AUC measures ranking/discrimination (can the model separate
clickers from non-clickers); log loss measures the quality of the
predicted *probabilities themselves* (calibration matters directly here,
since predictions feed a bidding function -- see module-level docstring of
the training scripts). Neither metric is fooled by the imbalance the way
accuracy is.

**Calibration.** AUC and log loss alone can still hide a systematically
mis-scaled probability (e.g. everything off by a constant multiplicative
factor can leave AUC unchanged and barely dent log loss at this
imbalance). This harness additionally reports Expected Calibration Error
(ECE) and the underlying reliability table. Binning scheme: **quantile
(equal-count) bins**, not fixed equal-width bins on `[0, 1]` -- at a
~0.08% true positive rate, almost every predicted probability lands below
0.01, so equal-width bins on `[0, 1]` would put >99% of rows in a single
bin and say nothing about calibration in the range that actually matters.
Equal-count bins spread the (mostly tiny) predicted probabilities out
across the bins that have data, at the cost of bin edges that aren't
round numbers -- an explicit, documented trade-off, not an oversight.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .downsample import recalibrate_probability

_FIGURES_DIR = Path(__file__).resolve().parents[3] / "reports" / "figures"


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    split: str  # e.g. "season2_val", "season2_test"
    n_rows: int
    n_positive: int
    positive_rate: float
    sampling_rate: float  # rate applied for recalibration; 1.0 if model needs none
    auc: float
    log_loss: float
    ece: float
    calibration_n_bins: int
    calibration_binning: str
    reliability_table: pd.DataFrame  # per-bin mean predicted vs. mean actual

    def to_dict(self) -> dict:
        d = {
            "model_name": self.model_name,
            "split": self.split,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "positive_rate": self.positive_rate,
            "sampling_rate": self.sampling_rate,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "calibration_n_bins": self.calibration_n_bins,
            "calibration_binning": self.calibration_binning,
        }
        return d


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Quantile-binned ECE -- see module docstring for why quantile bins,
    not fixed equal-width bins, are used here.

    ECE = sum over bins of (bin weight) * |mean(y_prob in bin) - mean(y_true in bin)|,
    i.e. the row-count-weighted average absolute gap between predicted and
    observed (empirical) click rate, per bin.

    Ties in `y_prob` (very common here -- many rows can share an identical
    predicted probability, e.g. all rows of one categorical combination)
    are handled via `pandas.qcut(..., duplicates="drop")`, which can yield
    fewer than `n_bins` bins when the predicted distribution is very
    concentrated; the returned table reflects however many bins actually
    resulted, and `calibration_n_bins` in `EvaluationResult` records the
    number requested, not necessarily the number produced.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        raise ValueError("expected_calibration_error(): empty input")

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    try:
        df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        # Fewer distinct values than n_bins even after dropping duplicate
        # edges (e.g. a degenerate all-identical-probability model) --
        # fall back to a single bin rather than raising, since "everything
        # in one bin" is still a meaningful (if coarse) answer.
        df["bin"] = 0

    grouped = df.groupby("bin", observed=True).agg(
        n=("y_true", "size"),
        mean_predicted=("y_prob", "mean"),
        mean_actual=("y_true", "mean"),
        prob_min=("y_prob", "min"),
        prob_max=("y_prob", "max"),
    ).reset_index(drop=True)
    grouped["abs_gap"] = (grouped["mean_predicted"] - grouped["mean_actual"]).abs()
    grouped["weight"] = grouped["n"] / n

    ece = float((grouped["abs_gap"] * grouped["weight"]).sum())
    return ece, grouped


def evaluate(
    y_true,
    y_prob_raw,
    *,
    model_name: str,
    split: str,
    sampling_rate: float = 1.0,
    n_bins: int = 10,
) -> EvaluationResult:
    """Score one set of raw predictions against true labels.

    `y_prob_raw` is whatever the model output on the (possibly
    downsampled-train-fitted) model, BEFORE recalibration -- this function
    applies `downsample.recalibrate_probability(y_prob_raw, sampling_rate)`
    itself, first, before computing any metric, per the top-level protocol
    ("apply the downsampling correction to predictions first"). Pass
    `sampling_rate=1.0` (the default) for a model that was not trained on
    downsampled data -- the correction is then the identity.

    `y_true`/`y_prob_raw` must come from a split that was NEVER
    downsampled (i.e. val or test) -- this harness has no way to verify
    that from the arrays alone, so it is the caller's responsibility (see
    `modeling.downsample` module docstring).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_calibrated = recalibrate_probability(np.asarray(y_prob_raw, dtype=np.float64), sampling_rate)
    # Clip away from the exact boundary for log_loss's sake; sklearn's
    # log_loss already clips internally, but doing it explicitly here
    # keeps the calibration table (which also uses y_prob_calibrated)
    # consistent with what log_loss actually scores.
    eps = 1e-12
    y_prob_calibrated = np.clip(y_prob_calibrated, eps, 1 - eps)

    n_rows = len(y_true_arr)
    n_positive = int(y_true_arr.sum())
    positive_rate = n_positive / n_rows if n_rows else float("nan")

    auc = float(roc_auc_score(y_true_arr, y_prob_calibrated))
    ll = float(log_loss(y_true_arr, y_prob_calibrated, labels=[0, 1]))
    ece, reliability_table = expected_calibration_error(y_true_arr, y_prob_calibrated, n_bins=n_bins)

    return EvaluationResult(
        model_name=model_name,
        split=split,
        n_rows=n_rows,
        n_positive=n_positive,
        positive_rate=positive_rate,
        sampling_rate=sampling_rate,
        auc=auc,
        log_loss=ll,
        ece=ece,
        calibration_n_bins=n_bins,
        calibration_binning="quantile",
        reliability_table=reliability_table,
    )


def save_reliability_plot(result: EvaluationResult, name: str) -> Path:
    """Log-log reliability curve (mean predicted vs. mean actual click
    rate, per quantile bin) for one `EvaluationResult`, saved as both PNG
    (300 DPI) and SVG under `reports/figures/`, matching the convention
    used by the Week 3 EDA figures. Log-log scale (not linear) because
    every quantity here -- predicted and actual click rates alike -- is a
    small positive fraction (~1e-4 to a few 1e-2 at most, given the
    ~0.075-0.09% base rate, see `docs/analysis/eda-findings.md` §2); a
    linear-scale plot would compress everything into an unreadable corner
    near the origin. A perfectly calibrated model's points sit exactly on
    the y=x diagonal, which is drawn for reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result.reliability_table
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        table["mean_predicted"], table["mean_actual"],
        s=(table["weight"] * 800).clip(lower=20),
        alpha=0.75, edgecolor="black", linewidth=0.5,
        label=f"{result.model_name} ({result.split})",
    )
    # `mean_actual` can be exactly 0.0 for a bin with zero observed clicks
    # (small bins, extreme imbalance) -- excluded from the axis-floor
    # computation rather than left in, since a literal 0 would force a
    # non-positive log-scale limit (matplotlib silently ignores that,
    # which is harmless, but noisy -- avoided here instead of tolerated).
    positive_values = pd.concat([table["mean_predicted"], table["mean_actual"]])
    positive_values = positive_values[positive_values > 0]
    lo = min(positive_values.min(), 1e-5) * 0.5
    hi = max(table["mean_predicted"].max(), table["mean_actual"].max()) * 2
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", label="perfect calibration (y=x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("mean predicted click probability (per quantile bin)")
    ax.set_ylabel("mean actual click rate (per quantile bin)")
    ax.set_title(f"Reliability curve: {result.model_name}\n{result.split}, ECE={result.ece:.6f}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_png = _FIGURES_DIR / f"{name}.png"
    out_svg = _FIGURES_DIR / f"{name}.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_png
