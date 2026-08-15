"""The integration point for the model side.

Written now, deliberately, so that integration day is filling in a call rather than
designing an interface under time pressure. Both adapters here already work and are
tested; neither needs any audio or model code to exist.

Two shapes, because we do not yet know which one the model side will want:

`QueueSource` — they push. Their capture loop calls `submit()` whenever a keystroke is
classified, from whatever thread or task they like. Best fit if they own a running
audio loop, which they almost certainly will.

`CallableSource` — we pull. We repeatedly await a function they provide. Best fit if
their code is request-shaped rather than loop-shaped.

Either way, the object plugs into `EventServer` exactly where `FakeSource` does and
nothing downstream changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from exposure.event import Prediction

# Bounds the backlog if the model side outruns the socket. Dropping the oldest
# unsent prediction is the right failure mode for a live exposure monitor: a stale
# keystroke is worth less than a current one, and unbounded growth would eventually
# take the process down mid-demo.
DEFAULT_MAX_BACKLOG = 256


class QueueSource:
    """Model side pushes Predictions in; the server pulls them out.

    Usage from their code::

        source = QueueSource()
        server = EventServer(source)          # instead of FakeSource()
        ...
        source.submit(Prediction(key_topk=topk, speech_present=vad_flag))

    `submit` is safe to call from a synchronous thread. `submit_nowait` is the
    coroutine-side equivalent when already inside the event loop.
    """

    def __init__(self, *, max_backlog: int = DEFAULT_MAX_BACKLOG) -> None:
        self._queue: asyncio.Queue[Prediction | None] = asyncio.Queue()
        self._max_backlog = max_backlog
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Predictions discarded to backlog pressure. Worth logging during the demo."""
        return self._dropped

    def submit_nowait(self, prediction: Prediction) -> None:
        """Enqueue from inside the event loop."""
        while self._queue.qsize() >= self._max_backlog:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - racing an empty queue
                break
        self._queue.put_nowait(prediction)

    def submit(self, prediction: Prediction) -> None:
        """Enqueue from another thread, or from inside the loop if that is where we are."""
        loop = self._loop
        if loop is None or not loop.is_running():
            self.submit_nowait(prediction)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.submit_nowait(prediction)
        else:
            loop.call_soon_threadsafe(self.submit_nowait, prediction)

    def close(self) -> None:
        """End the stream. The server's pump returns once this is drained."""
        self.submit(None)  # type: ignore[arg-type]

    async def predictions(self) -> AsyncIterator[Prediction]:
        self._loop = asyncio.get_running_loop()
        while True:
            prediction = await self._queue.get()
            if prediction is None:
                return
            yield prediction


class CallableSource:
    """We pull: awaits a caller-supplied function for each next Prediction.

    The function may be sync or async. Returning None ends the stream.
    """

    def __init__(
        self, produce: Callable[[], Prediction | None | Awaitable[Prediction | None]]
    ) -> None:
        self._produce = produce

    async def predictions(self) -> AsyncIterator[Prediction]:
        while True:
            result = self._produce()
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                return
            yield result
