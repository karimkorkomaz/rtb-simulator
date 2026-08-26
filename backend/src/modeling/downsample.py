"""Negative downsampling for the TRAIN split, plus the recalibration
correction needed to undo the resulting probability inflation.

**Applies to train only.** Validation and test must always reflect the
true, undownsampled class distribution -- see
`backend/src/ingest/README.md`, "Where train/val/test splitting and
downsampling would slot in", and the top-level protocol this thesis
follows. Nothing in this module should ever be called on a val/test split.

**The sampling rate must be persisted, not implied by row counts.**
`downsample_negatives()` returns a `DownsampleInfo` record (rate, seed,
resulting pos/neg counts) alongside the downsampled data; callers are
expected to write it to `backend/data/metadata/` (see
`modeling.train_lr` for the concrete file) so a probability produced by a
model trained on the downsampled data can always be traced back to the
exact rate used to correct it -- "recalibrate using whatever rate seems
right" is exactly the failure mode this guards against.

**Recalibration formula.** Downsampling negatives at keep-rate `w` (i.e.
a fraction `w` of negatives is kept, positives are kept in full) inflates
the model's predicted probability. Given `p` = predicted probability on
the downsampled distribution, the corrected (true-population) probability
is:

    q = p / (p + (1 - p) / w)

Derivation: for a fixed feature value x with `n1` true positives and `n0`
true negatives, downsampling negatives at rate `w` leaves `n1` positives
and `w * n0` negatives, so a model fit on the downsampled data estimates
`p = n1 / (n1 + w*n0)`. Solving for `n0/n1 = (1-p)/(p*w)` and substituting
into the true rate `q = n1/(n1+n0) = 1/(1+n0/n1)` gives the formula above.
This is the same correction used in Facebook's "Practical Lessons from
Predicting Clicks on Ads" (He et al., 2014) and is standard practice for
downsampled CTR models generally. At `w = 1` (no downsampling), `q == p`
exactly (verified in `backend/tests/test_downsample.py`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DownsampleInfo:
    """Everything needed to trace a downsampled train set back to an exact,
    reproducible sampling decision, and to recalibrate predictions made on
    top of it. Persist via `to_dict()` -- see `modeling.train_lr` for the
    JSON file this feeds.
    """

    negative_sampling_rate: float  # `w`: fraction of true negatives kept
    seed: int
    n_positive: int  # positives kept (== all true positives, never downsampled)
    n_negative_true: int  # true negative count before downsampling
    n_negative_sampled: int  # negatives actually kept
    positive_rate_before: float
    positive_rate_after: float

    def to_dict(self) -> dict:
        return asdict(self)


def downsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, DownsampleInfo]:
    """Downsample the negative (`y == 0`) rows of a TRAIN split to `rate`
    (fraction kept, e.g. `0.02` keeps 2% of true negatives); all positive
    (`y == 1`) rows are always kept. Sampling is done with
    `numpy.random.default_rng(seed)` so it is exactly reproducible given
    the same `(X, y, rate, seed)`.

    Never call this on a validation or test split -- see module docstring.
    """
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"downsample_negatives(): rate must be in (0, 1], got {rate!r}")
    if len(X) != len(y):
        raise ValueError(
            f"downsample_negatives(): X has {len(X)} rows but y has {len(y)} -- must match"
        )

    y_bool = y.astype(bool)
    pos_mask = y_bool
    neg_mask = ~y_bool

    n_positive = int(pos_mask.sum())
    n_negative_true = int(neg_mask.sum())
    n_total_true = n_positive + n_negative_true

    rng = np.random.default_rng(seed)
    neg_index = X.index[neg_mask]
    n_negative_sampled = int(round(rate * n_negative_true))
    if n_negative_sampled > 0:
        sampled_neg_index = rng.choice(neg_index.to_numpy(), size=n_negative_sampled, replace=False)
    else:
        sampled_neg_index = np.array([], dtype=neg_index.dtype)

    keep_index = X.index[pos_mask].append(pd.Index(sampled_neg_index))
    # Sort by original position so row order stays deterministic and
    # doesn't (say) put every positive before every negative, which would
    # otherwise be an accidental artifact of how `keep_index` was built.
    keep_index = keep_index.sort_values()

    X_ds = X.loc[keep_index]
    y_ds = y.loc[keep_index]

    info = DownsampleInfo(
        negative_sampling_rate=rate,
        seed=seed,
        n_positive=n_positive,
        n_negative_true=n_negative_true,
        n_negative_sampled=n_negative_sampled,
        positive_rate_before=n_positive / n_total_true if n_total_true else float("nan"),
        positive_rate_after=(
            n_positive / (n_positive + n_negative_sampled)
            if (n_positive + n_negative_sampled)
            else float("nan")
        ),
    )
    return X_ds, y_ds, info


def recalibrate_probability(p, sampling_rate: float):
    """Undo the probability inflation caused by downsampling negatives at
    `sampling_rate` (== `w` in the module docstring's formula). Accepts a
    scalar, `numpy.ndarray`, or `pandas.Series`; returns the same shape.

    `sampling_rate == 1.0` (no downsampling) returns `p` unchanged, exactly
    (not just approximately) -- see `backend/tests/test_downsample.py`.
    """
    if not (0.0 < sampling_rate <= 1.0):
        raise ValueError(
            f"recalibrate_probability(): sampling_rate must be in (0, 1], "
            f"got {sampling_rate!r}"
        )
    if isinstance(p, pd.Series):
        p_arr = p.to_numpy(dtype=np.float64)
        q_arr = p_arr if sampling_rate == 1.0 else p_arr / (p_arr + (1.0 - p_arr) / sampling_rate)
        return pd.Series(q_arr, index=p.index, name=p.name)
    is_scalar = np.isscalar(p)
    p_arr = np.asarray(p, dtype=np.float64)
    if sampling_rate == 1.0:
        q_arr = p_arr
    else:
        q_arr = p_arr / (p_arr + (1.0 - p_arr) / sampling_rate)
    return float(q_arr) if is_scalar else q_arr
