"""Event sources: where Predictions come from.

One interface, two implementations. `FakeSource` runs today; a `ClassifierSource`
wrapping the model side drops in later at the same point. Everything downstream —
risk, alert, assembly, transport — is written against `EventSource` and never learns
which one it got. Integration day is constructing a different object, not rewiring.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Protocol

from exposure.event import KeyTopK, Prediction

# One typist, one keyboard, per CLAUDE.md. Lowercase letters plus space is enough
# vocabulary to demo a transcript; the real set comes from the model side's labels.
VOCAB = "abcdefghijklmnopqrstuvwxyz "

TOP_K = 5


class EventSource(Protocol):
    """Yields Predictions until stopped. Both fake and real sources satisfy this."""

    def predictions(self) -> AsyncIterator[Prediction]: ...


class FakeSource:
    """Synthetic Predictions with realistic distribution shapes and typing rhythm.

    Deliberately produces a spread of exposure levels rather than uniformly confident
    output, so the alert panel has something to distinguish. Three regimes:

      clean      peaked distribution, high margin -> scores critical
      usable     one clear candidate with a real runner-up -> scores moderate
      ambiguous  near-flat distribution -> below threshold, log only

    Typing arrives in bursts with pauses between, so the frontend's latched exposure
    indicator can be seen sitting lit through a gap — which is the behaviour that
    makes the manual clear hotkey make sense to a viewer.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        speech_probability: float = 0.3,
        burst_length: tuple[int, int] = (5, 14),
        keystroke_delay: tuple[float, float] = (0.06, 0.22),
        pause_delay: tuple[float, float] = (1.2, 3.0),
    ) -> None:
        self._rng = random.Random(seed)
        self._speech_probability = speech_probability
        self._burst_length = burst_length
        self._keystroke_delay = keystroke_delay
        self._pause_delay = pause_delay

    async def predictions(self) -> AsyncIterator[Prediction]:
        while True:
            burst = self._rng.randint(*self._burst_length)
            speech = self._rng.random() < self._speech_probability
            for _ in range(burst):
                yield self._one_prediction(speech_present=speech)
                await asyncio.sleep(self._rng.uniform(*self._keystroke_delay))
            await asyncio.sleep(self._rng.uniform(*self._pause_delay))

    def sample(
        self,
        count: int,
        *,
        speech_present: bool | None = None,
        start_time: float | None = None,
    ) -> list[Prediction]:
        """Return `count` Predictions synchronously, for tests and fixture dumps.

        Timestamps follow the same burst-and-pause rhythm the async stream produces,
        advanced against a virtual clock rather than real sleeping. Without this a
        dumped fixture lands entirely within one millisecond, which makes it useless
        for replay and misleading for anyone building a UI against it.
        """
        clock = time.time() if start_time is None else start_time
        predictions: list[Prediction] = []
        remaining_in_burst = 0
        speech_for_burst = False

        for _ in range(count):
            if remaining_in_burst == 0:
                remaining_in_burst = self._rng.randint(*self._burst_length)
                speech_for_burst = self._rng.random() < self._speech_probability
                if predictions:
                    clock += self._rng.uniform(*self._pause_delay)
            else:
                clock += self._rng.uniform(*self._keystroke_delay)

            prediction = self._one_prediction(
                speech_present=(
                    speech_for_burst if speech_present is None else speech_present
                )
            )
            prediction.timestamp = clock
            predictions.append(prediction)
            remaining_in_burst -= 1

        return predictions

    def _one_prediction(self, *, speech_present: bool) -> Prediction:
        regime = self._rng.choices(
            ("clean", "usable", "ambiguous"), weights=(0.35, 0.4, 0.25)
        )[0]
        topk = self._distribution(regime)
        return Prediction(
            key_topk=topk,
            confidence=topk[0][1],
            speech_present=speech_present,
        )

    def _distribution(self, regime: str) -> KeyTopK:
        """Build a top-k distribution whose shape matches the requested regime.

        Weights are drawn then normalized, so the returned probabilities sum to 1 the
        way a real softmax output would. The regimes differ only in how much mass the
        leader takes and how flat the tail is.
        """
        # Leader ranges are chosen so the three regimes straddle the alert
        # thresholds in alert.py (0.80 / 0.95). Raising a threshold without
        # raising these would make the top severity band unreachable and the
        # panel would never demonstrate a critical alert.
        if regime == "clean":
            leader = self._rng.uniform(0.965, 0.998)
            tail_spread = 0.35
        elif regime == "usable":
            leader = self._rng.uniform(0.88, 0.96)
            tail_spread = 0.7
        else:
            leader = self._rng.uniform(0.22, 0.60)
            tail_spread = 1.0

        keys = self._rng.sample(VOCAB, TOP_K)
        tail_weights = [self._rng.uniform(1.0 - tail_spread, 1.0) for _ in range(TOP_K - 1)]
        tail_total = sum(tail_weights) or 1.0
        remaining = 1.0 - leader
        probs = [leader] + [remaining * w / tail_total for w in tail_weights]

        pairs = list(zip(keys, probs))
        pairs.sort(key=lambda kp: kp[1], reverse=True)
        return pairs
