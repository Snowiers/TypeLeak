"""
Audio input handling.

Two source types are provided behind the same interface so the rest of the
pipeline never needs to know which one it's reading from:

  - MicAudioSource   : real-time capture from a microphone via `sounddevice`
                       (use this on the actual demo hardware).
  - ArrayAudioSource : replays an in-memory numpy array in fixed-size chunks,
                       simulating a live stream. Useful for offline testing
                       and for a "semi-live" demo fallback (pre-recorded
                       typing replayed through the real pipeline).

Both push raw float32 mono chunks into a RingBuffer that the rest of the
pipeline reads from.
"""

from __future__ import annotations
import numpy as np
import queue
import threading
import time
from typing import Callable, Optional

import config


class RingBuffer:
    """Fixed-length circular buffer of audio samples.

    Supports appending new chunks and reading the most recent N seconds,
    which is what onset detection and feature extraction need (a bit of
    lookback/lookahead around any candidate onset).
    """

    def __init__(self, seconds: float = config.RING_BUFFER_SECONDS,
                 sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.capacity = int(seconds * sample_rate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0          # next write index (mod capacity)
        self._total_written = 0      # monotonically increasing sample counter
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        """Append a 1D float32 chunk to the buffer."""
        chunk = chunk.astype(np.float32, copy=False)
        n = len(chunk)
        with self._lock:
            if n >= self.capacity:
                # chunk bigger than the whole buffer — just keep the tail
                self._buf[:] = chunk[-self.capacity:]
                self._write_pos = 0
            else:
                end = self._write_pos + n
                if end <= self.capacity:
                    self._buf[self._write_pos:end] = chunk
                else:
                    first_part = self.capacity - self._write_pos
                    self._buf[self._write_pos:] = chunk[:first_part]
                    self._buf[:end - self.capacity] = chunk[first_part:]
                self._write_pos = end % self.capacity
            self._total_written += n

    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def read_absolute_range(self, start_sample: int, end_sample: int) -> Optional[np.ndarray]:
        """Read samples in the global (monotonic) sample-index range [start, end).

        Returns None if the requested range has already been overwritten
        (fallen out of the buffer) or hasn't been written yet.
        """
        with self._lock:
            if start_sample < 0 or end_sample > self._total_written:
                return None
            oldest_available = self._total_written - self.capacity
            if start_sample < max(oldest_available, 0):
                return None  # requested data has been overwritten

            length = end_sample - start_sample
            # position of start_sample within the physical buffer
            offset_from_write = self._total_written - start_sample
            start_pos = (self._write_pos - offset_from_write) % self.capacity
            end_pos = start_pos + length
            if end_pos <= self.capacity:
                return self._buf[start_pos:end_pos].copy()
            else:
                first_part = self.capacity - start_pos
                out = np.empty(length, dtype=np.float32)
                out[:first_part] = self._buf[start_pos:]
                out[first_part:] = self._buf[:end_pos - self.capacity]
                return out


class ArrayAudioSource:
    """Replays a numpy array as a simulated live stream, chunk by chunk.

    Use for offline testing, and as a "semi-live" fallback demo path
    (pre-recorded typing session fed through the real streaming pipeline).
    """

    def __init__(self, samples: np.ndarray, sample_rate: int = config.SAMPLE_RATE,
                 block_size: int = config.BLOCK_SIZE, realtime: bool = False):
        assert samples.ndim == 1, "expected mono 1D audio array"
        self.samples = samples.astype(np.float32)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.realtime = realtime  # if True, sleeps to mimic real wall-clock timing

    def stream(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        n = len(self.samples)
        pos = 0
        chunk_duration = self.block_size / self.sample_rate
        while pos < n:
            chunk = self.samples[pos:pos + self.block_size]
            if len(chunk) < self.block_size:
                chunk = np.pad(chunk, (0, self.block_size - len(chunk)))
            on_chunk(chunk)
            pos += self.block_size
            if self.realtime:
                time.sleep(chunk_duration)


class MicAudioSource:
    """Live microphone capture via sounddevice. Use this on the demo machine.

    Requires the `sounddevice` package and a working input device.
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE,
                 channels: int = config.CHANNELS, block_size: int = config.BLOCK_SIZE,
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device

    def stream(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                # xruns / overflow warnings land here — log, don't crash
                print(f"[MicAudioSource] status: {status}")
            mono = indata[:, 0] if indata.ndim > 1 else indata
            on_chunk(mono.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                             blocksize=self.block_size, device=self.device,
                             dtype="float32", callback=callback):
            print("[MicAudioSource] streaming... press Ctrl+C to stop")
            threading.Event().wait()  # block forever until interrupted
