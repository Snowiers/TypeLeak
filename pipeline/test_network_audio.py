"""
End-to-end test of the network audio path — runs a NetworkAudioSource
"server" (as it would run on the GN100) and a minimal raw-socket sender (a
stand-in for remote_mic_sender.py, which needs a real mic and can't run in
this sandbox) on localhost, and confirms synthetic keystroke audio streamed
over the socket makes it through the full pipeline correctly.

This validates the network protocol and NetworkAudioSource's integration
with the pipeline — it does NOT test remote_mic_sender.py's actual mic
capture (that needs to be tested on real hardware with a real mic).

Run: python3 test_network_audio.py
"""

import socket
import struct
import threading
import time

import numpy as np

import config
from network_audio import NetworkAudioSource, HEADER_FORMAT
from pipeline import AcousticGuardPipeline
from test_offline import make_synthetic_audio


def fake_sender(host: str, port: int, audio: np.ndarray, block_size: int, connected_evt: threading.Event):
    # wait for the server to be ready to accept
    connected_evt.wait(timeout=5)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(20):
        try:
            sock.connect((host, port))
            break
        except ConnectionRefusedError:
            time.sleep(0.1)
    else:
        raise RuntimeError("fake_sender: could not connect to server")

    pos = 0
    n = len(audio)
    while pos < n:
        chunk = audio[pos:pos + block_size]
        if len(chunk) < block_size:
            chunk = np.pad(chunk, (0, block_size - len(chunk)))
        payload = chunk.astype("<f4").tobytes()
        header = struct.pack(HEADER_FORMAT, len(chunk))
        sock.sendall(header + payload)
        pos += block_size
    sock.close()


def main():
    audio, true_click_times = make_synthetic_audio()
    print(f"Streaming {len(audio)/config.SAMPLE_RATE:.2f}s of synthetic audio over "
          f"a loopback socket (simulating capture-device -> GN100)...")

    host, port = "127.0.0.1", 9998  # distinct test port so it won't collide with a real run
    server = NetworkAudioSource(host=host, port=port)

    received_events = []

    def on_event(event):
        received_events.append(event)
        print(f"  [event via network] t~{event['sample_index']/config.SAMPLE_RATE:.3f}s "
              f"key={event['predicted_key']} exposure={event['exposure_score']:.1f}")

    pipeline = AcousticGuardPipeline(on_event=on_event)

    connected_evt = threading.Event()
    sender_thread = threading.Thread(
        target=fake_sender, args=(host, port, audio, config.BLOCK_SIZE, connected_evt), daemon=True
    )
    connected_evt.set()  # server.stream() below starts accepting almost immediately;
                          # fake_sender's own retry loop handles the small race either way
    sender_thread.start()

    # Run the SERVER on the MAIN thread (matches real run_server.py usage) --
    # NOT a background thread. librosa's onset detection uses numba
    # (JIT-compiled), which has known issues when first invoked from a
    # non-main/daemon thread (silent failures, crash on interpreter exit).
    # This mirrors how the real pipeline is actually run, so it's not just
    # a workaround -- it's the correct structure for this test too.
    server.stream(pipeline._handle_chunk)  # returns once fake_sender disconnects
    sender_thread.join(timeout=5)

    print(f"\nReceived {len(received_events)} keystroke event(s) over the network "
          f"(vs. {len(true_click_times)} injected clicks).")
    assert len(received_events) > 0, "FAIL: no events received over the network path"
    print("✅ Network audio path works end-to-end (loopback test).")
    print("   Next: test remote_mic_sender.py on real hardware with a real mic,")
    print("   pointed at the GN100's actual LAN IP.")


if __name__ == "__main__":
    main()
