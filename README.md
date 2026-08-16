# Acoustic Keystroke Live Demo

Reconstructs what's being typed from the **sound** of typing. The Mac streams
microphone audio to the DGX Spark, which detects + classifies each keystroke and
shows the live transcription (with optional LLM correction) in a web page.

```
  MAC (client.py) ──mic audio──▶ SPARK (live_server.py) ──▶ browser (http://spark:8080)
```

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
