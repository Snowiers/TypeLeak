"""Run the model pipeline and serve its output to the frontend.

This is the join between the two halves. The pipeline captures and classifies;
`exposure/` scores risk, decides alerts, and serves the websocket the dashboard
connects to. Run this on the machine that has the model, the checkpoint and the
config -- the pipeline only works there by design.

    python serve_frontend.py --checkpoint ~/Documents/keylogging/dataset/runs/.../best_model.pt

Then open frontend/index.html. Add --record take.jsonl to capture the session as a
replayable fixture.

Audio source defaults to the network receiver (mic on another machine, see
remote_mic_sender.py). Pass --mic to capture locally instead.

Threading: the websocket server owns an asyncio loop on a background thread, while
`pipeline.run()` blocks the main thread. `QueueSource.submit()` is thread-safe and
hands work across via call_soon_threadsafe, so the audio callback never touches the
loop directly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading

# exposure/ lives one level up from pipeline/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (pipeline-local, must resolve before exposure imports)
from pipeline import AcousticGuardPipeline  # noqa: E402

from exposure.bridge import prediction_from_pipeline_event  # noqa: E402
from exposure.classifier import QueueSource  # noqa: E402
from exposure.recorder import EventRecorder  # noqa: E402
from exposure.server import DEFAULT_HOST, DEFAULT_PORT, EventServer  # noqa: E402

log = logging.getLogger("serve_frontend")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=config.MODEL_CHECKPOINT_PATH,
        help="path to best_model.pt (defaults to config.MODEL_CHECKPOINT_PATH)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mic", action="store_true", help="capture locally instead of over the network")
    parser.add_argument("--audio-port", type=int, default=config.NETWORK_PORT)
    parser.add_argument("--record", metavar="PATH", help="also write a replayable JSONL fixture")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.checkpoint:
        log.warning(
            "no --checkpoint given: the model will run untrained and its predictions "
            "will be noise. Alerts shown will not mean anything."
        )

    source = QueueSource()
    recorder = EventRecorder(args.record) if args.record else None
    if recorder is not None:
        log.info("recording to %s", recorder.path)

    server = EventServer(source, host=args.host, port=args.port, recorder=recorder)

    def serve() -> None:
        asyncio.run(server.run())

    thread = threading.Thread(target=serve, name="exposure-server", daemon=True)
    thread.start()

    # Wait for the server's loop to bind before any audio can arrive, so the first
    # keystrokes are not submitted into a queue with no consumer thread attached.
    if not source.wait_until_ready(timeout=10.0):
        log.error("websocket server did not start within 10s")
        return 1
    log.info("serving on ws://%s:%d -- open frontend/index.html", args.host, args.port)

    counts = {"seen": 0, "forwarded": 0}

    def on_event(event: dict) -> None:
        counts["seen"] += 1
        prediction = prediction_from_pipeline_event(event)
        if prediction is None:
            return  # junk, low confidence, or a class we do not report
        counts["forwarded"] += 1
        source.submit(prediction)

    pipeline = AcousticGuardPipeline(on_event=on_event, checkpoint_path=args.checkpoint)

    if args.mic:
        from audio_io import MicAudioSource

        audio_source = MicAudioSource()
        log.info("capturing from local microphone")
    else:
        from network_audio import NetworkAudioSource

        audio_source = NetworkAudioSource(host="0.0.0.0", port=args.audio_port)
        log.info("waiting for network audio on port %d", args.audio_port)

    try:
        pipeline.run(audio_source)
    except KeyboardInterrupt:
        pass
    finally:
        log.info(
            "%d events from the pipeline, %d forwarded to the dashboard (%d filtered)",
            counts["seen"],
            counts["forwarded"],
            counts["seen"] - counts["forwarded"],
        )
        if source.dropped:
            log.warning("%d predictions dropped to backlog pressure", source.dropped)
        if recorder is not None:
            recorder.close()
            log.info("recorded %d events to %s", recorder.count, recorder.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
