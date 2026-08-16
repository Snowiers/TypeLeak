"""Adapter: the model pipeline's event dict → our `Prediction`.

The pipeline (see `pipeline/`) emits its own event shape and speaks a different
dialect: 37 classes including digits and a `junk` class, a scalar confidence, and
its own 0-100 rolling `exposure_score`. This module translates one keystroke into
the `Prediction` the rest of this package consumes.

Deliberately a pure function on a plain dict. The pipeline only imports on its own
machine — it needs torch, timm, a config.yaml and a checkpoint that live outside
this repo — so translating dicts rather than pipeline objects keeps this fully
testable here, with no model dependencies at all.

What it decides, and why:

**Reportable vocabulary is a-z plus space.** The classifier also predicts digits and
`junk` (release clicks and non-keystroke sounds). Neither belongs in a transcript,
and the frontend keyboard only lights letters and space.

**An event whose best guess is outside that vocabulary is dropped entirely**, rather
than promoting the best surviving letter. If the model's argmax is `junk` or `7`,
that sound was not an a-z keystroke, and inventing a letter for it would put
characters in the transcript that the model never predicted.

**Risk is scored over the reportable keys only.** Once an event is accepted, the
non-vocabulary mass is dropped before scoring. The question the risk score answers is
"how identifiable is this keystroke among the possible keys" — `junk` is not a key it
could be confused with. Note this makes scores from this bridge systematically higher
than scores over the raw 37-class distribution, so the alert thresholds must be
calibrated against output that came *through here*, not against raw model output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from exposure.event import KeyTopK, Prediction

# The reportable class set: a-z plus space. Matches the frontend's VOCAB.
VOCAB: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyz ")

# The classifier's non-keystroke class (release clicks, other sounds).
JUNK_LABEL = "junk"

# How many candidates to keep. The risk score normalizes entropy by the number of
# candidates it is given, so this must stay fixed for scores to be comparable.
TOP_K = 5

# Labels the model may use for the space key, normalized to a literal space.
SPACE_ALIASES: frozenset[str] = frozenset({"space", "spacebar", "_", "<space>"})


def normalize_label(label: Any) -> str:
    """Map a raw model label to its reportable character.

    Handles the common spellings of the space key and lowercases letters, so a
    retrain that labels space differently does not silently stop producing spaces.
    """
    text = str(label)
    lowered = text.lower()
    if lowered in SPACE_ALIASES:
        return " "
    return lowered


def prediction_from_pipeline_event(event: dict[str, Any]) -> Prediction | None:
    """Translate one pipeline event, or return None if it should not be reported.

    Returns None for junk, for predictions below the pipeline's own confidence
    threshold, for events whose best guess is outside the reportable vocabulary,
    and for events carrying no usable probability vector.
    """
    if event.get("is_junk"):
        return None
    if event.get("below_confidence_threshold"):
        return None

    scored = _label_probability_pairs(event)
    if not scored:
        return None

    # Judge the winner on the full distribution, before any filtering — otherwise
    # a junk-dominated sound could be reported as whichever letter placed second.
    best_label, _ = max(scored, key=lambda pair: pair[1])
    if best_label == JUNK_LABEL:
        return None

    best_char = normalize_label(best_label)
    if best_char not in VOCAB:
        return None

    key_topk = _reportable_topk(scored)
    if len(key_topk) < 2:
        # A single candidate carries no distributional information; risk_score
        # would return 0.0 and the event could never alert. Drop it instead of
        # emitting a keystroke that is invisible to the alert path.
        return None

    return Prediction(
        key_topk=key_topk,
        confidence=key_topk[0][1],
        timestamp=_timestamp(event),
        # The pipeline has no VAD, so speech presence is genuinely unknown. The
        # frontend labels a false value "unknown" rather than claiming silence.
        speech_present=False,
    )


def _label_probability_pairs(event: dict[str, Any]) -> list[tuple[str, float]]:
    """Read the full class distribution, tolerating a few plausible shapes.

    `key_probs` as a list of pairs is what `pipeline.py` emits; a dict of
    label → probability is accepted too, in case a retrain changes the shape.
    """
    raw = event.get("key_probs")
    if raw is None:
        return []

    pairs: Iterable[tuple[Any, Any]]
    if isinstance(raw, dict):
        pairs = raw.items()
    elif isinstance(raw, Sequence):
        pairs = ((item[0], item[1]) for item in raw if len(item) >= 2)
    else:
        return []

    scored: list[tuple[str, float]] = []
    for label, probability in pairs:
        try:
            # float() also strips numpy scalars, which json.dumps cannot serialize.
            value = float(probability)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        scored.append((str(label), max(0.0, value)))
    return scored


def _reportable_topk(scored: list[tuple[str, float]]) -> KeyTopK:
    """Restrict to the reportable vocabulary, merge aliases, and take the top k."""
    merged: dict[str, float] = {}
    for label, probability in scored:
        char = normalize_label(label)
        if char not in VOCAB:
            continue
        # A retrain could emit two labels mapping to the same character; summing
        # is the only interpretation that keeps the distribution honest.
        merged[char] = merged.get(char, 0.0) + probability

    ranked = sorted(merged.items(), key=lambda pair: pair[1], reverse=True)[:TOP_K]
    total = sum(probability for _, probability in ranked)
    if total <= 0.0:
        return []
    return [(char, probability / total) for char, probability in ranked]


def _timestamp(event: dict[str, Any]) -> float | None:
    value = event.get("timestamp")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
