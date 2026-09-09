"""
The three baseline bidding strategies for Phase 3 (constant / random /
linear-in-predicted-CTR), per the project's ordering requirement --
non-linear bidding or pacing is explicitly out of scope until these three
are measured and tabulated (see `docs/analysis/simulation-results.md`).

CONTRACT: every strategy function here is a pure function of a
`replay.BidTimeView` (plus its own parameters), returning a `numpy.ndarray`
of non-negative bids aligned to the view's (== the pool's) row order. None
of them accept, import, or reference `replay.AuctionPool`, `payprice`, or
`bidprice` -- the ONLY bid-time-observable input is `BidTimeView`, mirroring
`replay.py`'s "THE PAYPRICE GUARD" discipline exactly. A strategy that
needed anything settlement-only would be a methodology violation, not a
missing feature -- see the module docstring of `replay.py`.

`p_click_for_pool()` (this project's fully-corrected click probability,
see `simulation.predictions`) is deliberately NOT computed inside this
module: `LinearInCTR` takes `p_click` as an explicit, already-aligned
argument, so the caller (not this module) is responsible for having gone
through the two-stage correction chain and the bidid alignment -- keeping
this module blind to *how* a probability was produced, only that it is
already aligned to the pool it will be bid against.

REPRODUCIBILITY: `RandomBid` takes an explicit `numpy.random.Generator`
(never the module-global `numpy.random` state) so a caller can seed it
deterministically per `(advertiser, budget, parameter)` run -- see
`run_baselines.py` for the seeding convention actually used. Same
generator + same `n` always produces the identical bid vector.
"""
from __future__ import annotations

import numpy as np

from .replay import BidTimeView


def constant_bid(view: BidTimeView, *, amount: float) -> np.ndarray:
    """Bid `amount` on every auction in `view`. The floor baseline -- see
    `docs/analysis/simulation-results.md` baselines section: this
    establishes what a strategy with NO information at all (not even
    "some auctions are worth more than others") achieves, so any smarter
    strategy has something concrete to beat.

    Raises `ValueError` for a negative `amount` -- `settle_auctions()`
    would reject it anyway, but failing here gives a more specific error
    at the point the strategy was misconfigured.
    """
    if amount < 0:
        raise ValueError(f"constant_bid(): amount must be non-negative, got {amount!r}")
    n = len(view)
    return np.full(n, float(amount), dtype=np.float64)


def random_bid(
    view: BidTimeView,
    *,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bid i.i.d. `Uniform[low, high)` on every auction in `view`, drawn
    from the caller-supplied `rng` (never `numpy.random`'s global state --
    see module docstring "REPRODUCIBILITY"). Establishes whether pure
    variance without any structure (unlike `constant_bid`, at least
    *some* auctions get outbid where a constant would not, and vice
    versa) buys anything over the constant floor.

    Raises `ValueError` if `low < 0` or `high < low`.
    """
    if low < 0:
        raise ValueError(f"random_bid(): low must be non-negative, got {low!r}")
    if high < low:
        raise ValueError(f"random_bid(): high ({high!r}) must be >= low ({low!r})")
    n = len(view)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if high == low:
        return np.full(n, float(low), dtype=np.float64)
    return rng.uniform(low=low, high=high, size=n)


def linear_in_ctr_bid(view: BidTimeView, *, base: float, p_click: np.ndarray) -> np.ndarray:
    """`bid = base * p_click` -- the classic RTB linear-bidding baseline
    (see e.g. Zhang et al.'s iPinYou benchmark literature, cited in
    `replay.py`'s module docstring). `p_click` must already be the
    fully-corrected click probability (`simulation.predictions.p_isotonic`,
    aligned to `view`'s row order by the caller) -- this function performs
    NO correction and NO alignment of its own; it only multiplies.

    `len(p_click)` must equal `len(view)` -- checked here (not deferred to
    a downstream broadcasting surprise) because a caller passing an
    unaligned `p_click` array is exactly the "predictions parquet has no
    bidid" alignment bug this project explicitly guards against
    (`simulation.predictions` module docstring).

    Raises `ValueError` for a negative `base`, a length mismatch, or any
    negative/NaN entry in `p_click` (a probability cannot be negative or
    undefined; a NaN silently propagating into a bid would produce a NaN
    bid that `settle_auctions()`'s `bids < 0` check would not even catch).
    """
    if base < 0:
        raise ValueError(f"linear_in_ctr_bid(): base must be non-negative, got {base!r}")
    p_click_arr = np.asarray(p_click, dtype=np.float64)
    n = len(view)
    if p_click_arr.shape != (n,):
        raise ValueError(
            f"linear_in_ctr_bid(): p_click has shape {p_click_arr.shape}, "
            f"expected ({n},) to align with the {n}-row BidTimeView. "
            "Refusing to broadcast or reindex -- this is exactly the "
            "predictions-alignment failure mode this project guards "
            "against (see simulation.predictions module docstring)."
        )
    if n > 0 and np.any(np.isnan(p_click_arr)):
        raise ValueError("linear_in_ctr_bid(): p_click contains NaN.")
    if n > 0 and np.any(p_click_arr < 0):
        raise ValueError("linear_in_ctr_bid(): p_click contains negative value(s).")
    return base * p_click_arr
