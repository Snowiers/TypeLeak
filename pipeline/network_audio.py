"""
Network audio streaming: lets audio CAPTURE happen on one machine (e.g. your
laptop, which has a mic) while PROCESSING happens on another (the GN100,
headless/no mic).

Protocol (kept intentionally simple — this is a hackathon, not a hardened
network audio stack):
  - TCP, one connection at a time.
  - Each message = [4-byte little-endian uint32: N] + [N float32 samples,
    little-endian, mono].
  - Sender (capture device) is the CLIENT: connects out to the GN100.
  - Receiver (GN100) is the SERVER: listens and accepts.
    (Client-connects-to-server is usually easier on a hackathon LAN/hotspot
    than the reverse, since you don't need to know the laptop's IP/deal with
    its firewall — just point the sender at the GN100's known IP.)

Usage on the GN100 (processing machine):
    from network_audio import NetworkAudioSource
    pipeline.run(NetworkAudioSource(host="0.0.0.0", port=config.NETWORK_PORT))

Usage on the capture device (laptop with the mic): see remote_mic_sender.py,
run as a standalone script — NOT imported into the pipeline, since it has no
GPU/model dependencies and should be able to run on a totally different,
lightweight machine.
"""

from __future__ import annotations
import socket
import struct
import threading
from typing import Callable, Optional

import numpy as np

import config

HEADER_FORMAT = "<I"          # little-endian uint32
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a socket, or None if the connection closed early."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None  # connection closed
        buf.extend(chunk)
    return bytes(buf)


class NetworkAudioSource:
    """Server side — runs on the GN100. Listens for one incoming connection
    from a capture device and forwards received audio chunks into the
    pipeline, using the exact same `stream(on_chunk)` interface as
    MicAudioSource/ArrayAudioSource, so it's a drop-in replacement.
    """

    def __init__(self, host: str = "0.0.0.0",
                 port: int = config.NETWORK_PORT,
                 expected_sample_rate: int = config.SAMPLE_RATE):
        self.host = host
        self.port = port
        self.expected_sample_rate = expected_sample_rate
        self._server_sock: Optional[socket.socket] = None

    def stream(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        print(f"[NetworkAudioSource] listening on {self.host}:{self.port} "
              f"— waiting for capture device to connect...")

        conn, addr = self._server_sock.accept()
        print(f"[NetworkAudioSource] connected: {addr}")

        try:
            with conn:
                while True:
                    try:
                        header = _recv_exact(conn, HEADER_SIZE)
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        print(f"[NetworkAudioSource] sender disconnected abruptly ({e}). Stopping cleanly.")
                        break
                    if header is None:
                        print("[NetworkAudioSource] sender disconnected.")
                        break
                    (n_samples,) = struct.unpack(HEADER_FORMAT, header)
                    if n_samples == 0 or n_samples > 10_000_000:
                        print(f"[NetworkAudioSource] bad chunk size {n_samples}, dropping connection")
                        break

                    try:
                        payload = _recv_exact(conn, n_samples * 4)  # float32 = 4 bytes
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        print(f"[NetworkAudioSource] sender disconnected abruptly ({e}). Stopping cleanly.")
                        break
                    if payload is None:
                        print("[NetworkAudioSource] sender disconnected mid-chunk.")
                        break

                    chunk = np.frombuffer(payload, dtype="<f4").astype(np.float32)
                    on_chunk(chunk)
        except KeyboardInterrupt:
            print("\n[NetworkAudioSource] stopped by user (Ctrl+C).")
        finally:
            self._server_sock.close()

    def stop(self) -> None:
        if self._server_sock:
            self._server_sock.close()
