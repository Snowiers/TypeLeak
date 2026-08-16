# TypeLeak

TypeLeak reconstructs what's being typed from the **sound** of typing. The Mac streams
microphone audio to the DGX Spark, which detects + classifies each keystroke and
shows the live transcription (with optional LLM correction) in a web page. 

```
  MAC (client.py) ──mic audio──▶ SPARK (live_server.py) ──▶ browser (http://spark:8080)
```

---

## Architecture

```
+--------------------------------------------------------------------------------+
| MAC -- client.py                                                         [Mac] |
| sounddevice mic capture                                                        |
| pynput ground-truth logger (--eval only; never sent to the model)              |
+--------------------------------------------------------------------------------+
                                        |  raw PCM, TCP :9009 (length-prefixed float32 frames)
                                        v
+--------------------------------------------------------------------------------+
| SPARK -- audio path (live_server.py / server.py / detector.py)     [Spark GPU] |
| rolling audio buffer -> onset detection                                        |
|   (librosa spectral flux + scipy peak-picking)                                 |
| onset -> per-keystroke clip -> log-mel spectrogram                             |
| classifier: fine-tuned EfficientNetV2 (timm) -> predicted char + confidence    |
+--------------------------------------------------------------------------------+
                                        |  predicted char echoed to Mac over TCP + fed into the Engine
                                        v
+--------------------------------------------------------------------------------+
| SPARK -- Engine (live_server.py)                                   [Spark GPU] |
| owns the raw decoded text stream                                               |
| optional LLM correction on a typing pause                                      |
|   (llm_correct.py: NVIDIA Nemotron / Mistral-NeMo-Minitron, local GPU)         |
| auto-commits the line to history after a longer idle pause                     |
+--------------------------------------------------------------------------------+
                                        |  JSON events over Server-Sent-Events, HTTP :8080
                                        v
+--------------------------------------------------------------------------------+
| BROWSER -- frontend/ (vanilla JS, no build step)                     [browser] |
| EventSource /events: reset | key | transcript | commit | llm_state             |
| Mac keyboard replica (flat 2D DOM/CSS) flashes the predicted key               |
| live typed line + session history (localStorage)                               |
| control panel: LLM toggle -> POST /api/llm, force commit -> POST /api/commit   |
+--------------------------------------------------------------------------------+
```

**Offline training pipeline** — separate from the live path above, run ahead
of time to produce the checkpoint `live_server.py` loads at startup:

```
--- offline training pipeline (produces the checkpoint live_server.py loads) ---

+--------------------------------------------------------------------------------+
| record.py / record_ambient.py session(s)  (on machine, not in this repo)       |
| -> dataset/raw/<session>/{audio.wav, events.json}                [Mac,offline] |
+--------------------------------------------------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
| process.py (+ combine_dataset.py to merge multiple sessions)         [offline] |
| detector.py onset detection, matched against logged keypresses                 |
|   -> labeled per-keystroke clips: dataset/processed/*.npy + labels.csv         |
+--------------------------------------------------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
| train.py / train_sweep.py                                 [Spark GPU, offline] |
| fine-tunes EfficientNetV2 (timm) on the labeled clips                          |
| -> dataset/runs/<timestamp>/best_model.pt                                      |
+--------------------------------------------------------------------------------+
                                        |  auto-loaded at startup (newest checkpoint by mtime)
                                        v
+--------------------------------------------------------------------------------+
| live_server.py / server.py                                                     |
+--------------------------------------------------------------------------------+
```

`detector.py` is shared and unmodified between this offline path and the live
audio path above.

## Tech stack

| Layer | Tech |
|---|---|
| Mic capture (Mac) | Python 3, `sounddevice`; `pynput` for ground-truth logging in eval mode only |
| Audio transport | Raw TCP socket, length-prefixed float32 PCM frames — no HTTP/WebSocket overhead on the hot path |
| Onset detection & features | `librosa` (onset envelope, log-mel spectrogram), `scipy.signal.find_peaks`, `Pillow` (spectrogram PNG export for eyeballing) |
| Classifier | PyTorch + `timm` (fine-tuned EfficientNetV2), CUDA inference on the Spark |
| LLM correction (optional, `--llm`) | Hugging Face `transformers`, NVIDIA Nemotron / Mistral-NeMo-Minitron, local GPU inference — no cloud API call |
| Web/API server | Python stdlib `http.server` + Server-Sent Events — no Flask/FastAPI/Node |
| Frontend | Vanilla HTML/CSS/JS, no build step, no three.js — a flat 2D DOM/CSS Mac-keyboard replica instead |
| Config | YAML (`config.yaml`), the single source of truth shared by data collection, training, and live inference |
| Target hardware | NVIDIA DGX Spark — on-device GPU inference keeps the capture → classify → correct → display round trip local, with no cloud round-trip in the loop |

---

## 1. Server — on the DGX Spark

```bash
cd backend

# first time only: create the venv + install deps
python3 -m venv .env
.env/bin/pip install -r requirements.txt

# run it (serves the web UI + does all the ML)
.env/bin/python live_server.py --llm --http-port 8080
```

Then open the web UI:

- on the Spark itself: **http://localhost:8080**
- from another machine on the LAN: **http://<spark-ip>:8080**

Useful flags:

```bash
--http-port 8080        # web/browser port (avoid 8000 — taken by another service)
--port 9009             # TCP port the Mac client connects to
--llm                   # enable LLM correction (omit for raw decode only)
--correct-idle-sec 5    # correct the line after this pause (seconds)
--idle-commit-sec 12    # auto-log the line to history after this pause
```

Find the Spark's LAN IP with: `hostname -I`

---

## 2. Client — on the Mac

```bash
# first time only: install the light audio deps (no torch/model needed here)
pip install numpy sounddevice pynput pyyaml

# stream the mic to the Spark (use the Spark's IP)
cd backend
python client.py --host <spark-ip> --port 9009
```

Type in any app. Predictions appear on the web page. Press **ESC** (or Ctrl+C) to stop.
Add `--no-eval` for a mic-only run (skips the ground-truth keylogger, so only
Microphone permission is needed — no Input Monitoring prompt).

macOS: grant **Microphone** permission (and **Input Monitoring** if using eval mode).

---

## Reproducing the demo

The demo is a side-by-side of two windows: a terminal on the Mac running the
client-side recorder, and a browser showing the live monitor served off the
DGX Spark ("GN100"), with predictions appearing on the monitor as you type on
the Mac.

1. **Start the server on the Spark** (section 1 above), then open its web UI
   on the monitor connected to (or mirroring) the Spark: `http://<spark-ip>:8080`.
2. **Toggle LLM correction from the control panel** on that page before you
   start typing:
   - **LLM correction ON** — best for demoing on real words/sentences; raw
     per-keystroke predictions get cleaned up into coherent text on each
     typing pause.
   - **LLM correction OFF** — best for demoing on gibberish/random keys;
     shows the raw classifier output per keystroke with no language-model
     smoothing.
3. **Start the client on the Mac** (section 2 above) — this opens the mic
   stream to the Spark.
4. **Type on the Mac** in any app (or just into the terminal). Keystrokes are
   classified from audio on the Spark and streamed back over SSE: watch the
   monitor's on-screen keyboard flash each predicted key in real time, with
   the decoded (and optionally LLM-corrected) line building up beside it.
5. To switch modes mid-demo, just hit the LLM toggle again — it takes effect
   on the next typing pause, no restart needed.

---
## Datasets Used

All data used to train the detection model was self-made, including 6,965 self-recorded keystrokes turned into spectograms. This can be found under backend/dataset/processed.

## Known limitations

- The model was trained using 6,965 real, self-made keystrokes. All recorded keystrokes were recorded from a single Mac's scissor-switch keyboard, making the system function on one keyboard (but multiple typists); it does not scope up to multiple keyboards.
- This is a detection-and-disclosure system only. It does not mute, mask, or
  otherwise modify the digital audio stream or physical sound. However, this is a deliberate scope boundary, not a missing feature.
- Typing too fast (above ~50-55 wpm) makes it much more difficult to identify keystrokes; at this speed, sounds from the keystrokes overlap.