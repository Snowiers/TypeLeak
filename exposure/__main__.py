"""Entry point.

  python -m exposure                      serve fake events on ws://localhost:8765
  python -m exposure --source replay      deterministic playback, for rehearsal
  python -m exposure --dump 40            print 40 events as JSONL, no server
  python -m exposure --list-sources       show what is available and why

The `--source` flag is the degradation switch. CLAUDE.md asks for degradation paths to
be assumed from the start rather than discovered under time pressure, so it exists
before there is anything to degrade from: when the real classifier lands as a third
option, falling back to `replay` mid-demo is already a flag rather than a code edit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from exposure.event import assemble_event
from exposure.recorder import EventRecorder
from exposure.replay import ReplaySource
from exposure.server import DEFAULT_HOST, DEFAULT_PORT, EventServer
from exposure.source import FakeSource

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_events.jsonl"

SOURCES = {
    "fake": "Randomised synthetic events. Different every run. Use for development.",
    "replay": "Deterministic playback of a recorded JSONL file. Use for rehearsal.",
    "classifier": "The real model side. Not wired up yet - see exposure/classifier.py.",
}

log = logging.getLogger(__name__)


def build_source(args: argparse.Namespace):
    if args.source == "fake":
        return FakeSource(seed=args.seed)
    if args.source == "replay":
        fixture = Path(args.fixture)
        if not fixture.exists():
            raise SystemExit(
                f"fixture not found: {fixture}\n"
                f"generate one with: python -m exposure --dump 200 > {fixture}"
            )
        return ReplaySource(fixture, loop=not args.no_loop, speed=args.speed)
    raise SystemExit(
        "the classifier source is not wired up yet.\n"
        "The model side implements it against exposure/classifier.py; until then use "
        "--source fake or --source replay."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exposure", description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="fake")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None, help="seed the fake source")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="replay source file")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("--no-loop", action="store_true", help="stop after one replay pass")
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="write this session to a replayable JSONL file (never overwrites)",
    )
    parser.add_argument(
        "--dump", type=int, metavar="N", help="print N events as JSONL and exit"
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="describe the available sources"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_sources:
        for name, description in sorted(SOURCES.items()):
            print(f"{name:12} {description}")
        return 0

    if args.dump is not None:
        for prediction in FakeSource(seed=args.seed).sample(args.dump):
            print(json.dumps(assemble_event(prediction).to_dict()))
        return 0

    source = build_source(args)
    if args.source == "replay":
        log.info("replaying %d events from %s", source.length, args.fixture)

    recorder = None
    if args.record:
        try:
            recorder = EventRecorder(args.record)
        except OSError as exc:
            # A mistyped path should not end a recording session in a traceback.
            raise SystemExit(f"cannot record to {args.record}: {exc}") from exc
        log.info("recording to %s", recorder.path)

    server = EventServer(source, host=args.host, port=args.port, recorder=recorder)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
            log.info("recorded %d events to %s", recorder.count, recorder.path)
            if recorder.count:
                log.info(
                    "replay it with: python -m exposure --source replay --fixture %s",
                    recorder.path,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
