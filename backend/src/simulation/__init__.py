"""Phase 3: the auction replay engine.

See `replay.py` for the loader (`load_auction_pool`), the second-price /
budget settlement function (`settle_auctions`), and the payprice-guarded
bid-time view (`BidTimeView`). This package is the engine only -- no
bidding strategies (constant/random/linear-in-CTR) and no CTR-probability
plumbing live here; both are separate, later stages.
"""
from __future__ import annotations
