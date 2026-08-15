"""Exposure risk score, derived from classifier output statistics.

The architectural rule from CLAUDE.md is enforced structurally here: `risk_score`
accepts a sequence of probabilities and nothing else. It has no parameter that could
carry a key identity or decoded text, so no future edit can accidentally feed the
transcript branch into the alert branch.

Two components, both describing the *shape* of the prediction distribution:

  margin      p1 - p2. A confident, well-separated prediction means the acoustic
              signal carried enough information to pick one key over its neighbours.
  1 - entropy Normalized Shannon entropy, inverted. A flat distribution means the
              audio was ambiguous; a peaked one means it was exploitable.

They disagree in useful ways. A distribution can have a large top-1 margin while the
remaining mass is smeared across many keys (high entropy), or a small margin between
two candidates while every other key is ruled out (low entropy). Averaging them is
more stable than either alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Relative weight of the margin term against the inverted-entropy term.
MARGIN_WEIGHT = 0.5


def risk_score(probs: Sequence[float], *, margin_weight: float = MARGIN_WEIGHT) -> float:
    """Return an exposure estimate in [0, 1] from a prediction distribution.

    `probs` is the probability column of `key_topk`, highest first. It is normally
    truncated to top-k rather than the full key vocabulary, so entropy is normalized
    by log(len(probs)) — the maximum entropy actually representable in what we were
    given. Comparing risk scores across different k values is not meaningful.

    Returns 0.0 for an empty or degenerate distribution: no signal, no exposure.
    """
    if not probs:
        return 0.0

    ranked = sorted((max(0.0, float(p)) for p in probs), reverse=True)
    total = sum(ranked)
    if total <= 0.0:
        return 0.0
    ranked = [p / total for p in ranked]

    # A single candidate carries no distributional information to score. Treat it as
    # fully ambiguous rather than fully exploitable — claiming certainty from one
    # number would overstate exposure, and this branch should stay conservative.
    if len(ranked) < 2:
        return 0.0

    margin = ranked[0] - ranked[1]

    entropy = -sum(p * math.log(p) for p in ranked if p > 0.0)
    normalized_entropy = entropy / math.log(len(ranked))
    peakedness = 1.0 - normalized_entropy

    score = margin_weight * margin + (1.0 - margin_weight) * peakedness
    return _clamp(score)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
