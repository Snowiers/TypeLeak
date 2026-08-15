"""Websocket transport. Pushes audio events to any connected frontend.

The `websockets` import is deliberately lazy so that risk, alert, assembly, and the
fake source can all be imported and unit-tested with no third-party dependency
installed. Only running the live server needs it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from exposure.event import assemble_event
from exposure.source import EventSource
from exposure.state import ExposureState

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765

log = logging.getLogger(__name__)


def _load_serve():
    """Import websockets' server entry point across library versions."""
    try:
        from websockets.asyncio.server import serve  # websockets >= 13
    except ImportError:  # pragma: no cover - depends on installed version
        from websockets import serve  # type: ignore[attr-defined]
    return serve


class EventServer:
    """Runs a source through the pipeline and broadcasts the resulting events."""

    def __init__(
        self,
        source: EventSource,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        state: ExposureState | None = None,
    ) -> None:
        self._source = source
        self._host = host
        self._port = port
        self._state = state or ExposureState()
        self._clients: set[Any] = set()

    async def run(self) -> None:
        serve = _load_serve()
        async with serve(self._handle_client, self._host, self._port):
            log.info("listening on ws://%s:%d", self._host, self._port)
            await self._pump()

    async def _handle_client(self, websocket: Any, *_: Any) -> None:
        """Accept a client, send it the current state, then hold the connection.

        The snapshot matters for demo reliability: a frontend that reloads mid-demo
        rejoins showing the alerts it missed rather than an empty log.
        """
        self._clients.add(websocket)
        log.info("client connected (%d total)", len(self._clients))
        try:
            await websocket.send(json.dumps(self._state.snapshot()))
            async for message in websocket:
                await self._handle_message(message)
        except Exception:  # noqa: BLE001 - a dropped client must not kill the server
            log.debug("client connection ended", exc_info=True)
        finally:
            self._clients.discard(websocket)
            log.info("client disconnected (%d remaining)", len(self._clients))

    async def _handle_message(self, raw: str | bytes) -> None:
        """Handle client -> server messages. Currently only the manual latch clear."""
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            log.debug("ignoring unparseable client message")
            return
        if message.get("type") == "clear_latch":
            self._state.clear_latch()
            await self._broadcast(self._state.snapshot())

    async def _pump(self) -> None:
        """Drive the pipeline: prediction in, event assembled, event broadcast."""
        async for prediction in self._source.predictions():
            event = assemble_event(prediction)
            self._state.record(event)
            await self._broadcast(event.to_dict())

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload)
        results = await asyncio.gather(
            *(client.send(message) for client in tuple(self._clients)),
            return_exceptions=True,
        )
        for client, result in zip(tuple(self._clients), results):
            if isinstance(result, Exception):
                self._clients.discard(client)
