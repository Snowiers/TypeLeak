"""
Runs on the CAPTURE DEVICE (e.g. your laptop, which has a mic) — NOT on the
GN100. Captures live audio with sounddevice and streams it over TCP to the
GN100, where NetworkAudioSource (network_audio.py) receives it and feeds it
into the real pipeline.

This script deliberately has minimal dependencies (just sounddevice + numpy)
so it can run on lightweight/non-GPU hardware.

Usage:
    python3 remote_mic_sender.py --host 192.168.1.42 --port 9999

    # list available input devices on this machine first, if unsure which mic to use:
    python3 remote_mic_sender.py --list-devices

    # pick a specific input device by index:
    python3 remote_mic_sender.py --host 192.168.1.42 --device 2

All parameters are configurable via CLI flags — see --help. Sample rate and
block size should match what the GN100-side pipeline expects (config.py's
SAMPLE_RATE / BLOCK_SIZE) — mismatches won't crash anything, but will distort
timing-sensitive analysis (onset detection, and especially TDoA if you add
multi-mic support later), so keep them in sync.
"""

from __future__ import annotations
import argparse
import socket
import struct
import sys

import numpy as np
import sounddevice as sd

HEADER_FORMAT = "<I"


def list_devices():
    print(sd.query_devices())


def send_chunk(sock: socket.socket, chunk: np.ndarray) -> None:
    chunk = chunk.astype("<f4")  # little-endian float32
    header = struct.pack(HEADER_FORMAT, len(chunk))
    sock.sendall(header + chunk.tobytes())


def run(host: str, port: int, sample_rate: int, block_size: int,
        device: int | None, channels: int = 1):
    print(f"Connecting to GN100 at {host}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print("Connected. Streaming mic audio — Ctrl+C to stop.")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[sender] status: {status}", file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        try:
            send_chunk(sock, mono)
        except (BrokenPipeError, ConnectionResetError):
            print("[sender] connection to GN100 lost.", file=sys.stderr)
            raise sd.CallbackStop

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels,
                             blocksize=block_size, device=device,
                             dtype="float32", callback=callback):
            sd.sleep(10**9)  # sleep "forever" (until Ctrl+C)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream this machine's mic audio to the GN100 pipeline.")
    parser.add_argument("--host", type=str, help="GN100's IP address on the local network")
    parser.add_argument("--port", type=int, default=9999, help="Must match config.NETWORK_PORT on the GN100 side")
    parser.add_argument("--sample-rate", type=int, default=44100,
                         help="Must match config.yaml's audio.sample_rate on the GN100 side (44100)")
    parser.add_argument("--block-size", type=int, default=1024,
                         help="Should match config.yaml's audio.blocksize on the GN100 side (1024)")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="List available input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if not args.host:
        parser.error("--host is required (unless using --list-devices)")

    run(args.host, args.port, args.sample_rate, args.block_size, args.device)
