"""Entry point.

  python -m exposure                 serve fake events on ws://localhost:8765
  python -m exposure --dump 40       print 40 events as JSONL, no websockets needed

The dump mode exists so the frontend can be built against a static fixture file
before anyone runs a server, and so the exact bytes the frontend will receive can be
eyeballed without a client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from exposure.event import assemble_event
from exposure.server import DEFAULT_HOST, DEFAULT_PORT, EventServer
from exposure.source import FakeSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exposure", description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--dump",
        type=int,
        metavar="N",
        help="print N events as JSONL and exit instead of serving",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed the fake source")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = FakeSource(seed=args.seed)

    if args.dump is not None:
        for prediction in source.sample(args.dump):
            print(json.dumps(assemble_event(prediction).to_dict()))
        return 0

    server = EventServer(source, host=args.host, port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
