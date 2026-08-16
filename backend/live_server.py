#!/usr/bin/env python3
"""
live_server.py  --  runs on the LINUX / DGX SPARK box.

The web-facing live demo server. It ties three things together in ONE process:

  1. TCP audio server (:9009)  -- identical wire protocol to server.py, so the
     unchanged client.py on the Mac streams mic audio here and (as before) gets
     each predicted character echoed back. On every committed keystroke we ALSO
     feed the character into the live transcription Engine.

  2. HTTP + SSE server (:8000, stdlib only, no extra deps) -- serves the frontend
     static files AND a Server-Sent-Events stream at /events that pushes live
     transcription events to any number of browsers. A tiny POST /api/llm toggles
     LLM correction on/off; POST /api/commit forces an immediate commit.

  3. The Engine -- single source of truth for the decoded text. It owns the raw
     character stream, runs LLM correction every N detected spaces (in a worker
     thread, OFF the audio hot-path), and auto-commits the line to history after
     a few seconds of silence. Browsers just render whatever text it publishes.

Design notes
------------
* Space detection is NOT reliable, so the LLM is deliberately allowed to fix
  spacing (add/remove/merge '_'). We therefore never try to token-align the
  corrected text against the raw text. Instead the Engine keeps a "corrected
  prefix" (the whole decode up to the last N-space boundary, re-corrected as one
  string) and concatenates the still-raw live tail after it. No alignment, so
  nothing can desync -- corrections just refine the earlier part of the line
  while new keystrokes keep streaming in raw at the end.
* All heavy work (classifier + Nemotron) reuses the in-memory models loaded once
  at startup. The classifier + classify() are imported straight from server.py.

Run:  python live_server.py            # classifier only, LLM toggle disabled
      python live_server.py --llm      # also load Nemotron; correction ON by default
"""
import os
import sys
import json
import time
import errno
import socket
import struct
import select
import threading
import queue
import argparse
import http.server
import socketserver

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse the classifier plumbing from server.py verbatim (model load + classify +
# framed recv). Importing server.py is safe: its main() is guarded by __main__.
import detector  # noqa: E402
import server as tcp  # noqa: E402  (load_cfg, find_latest_model, load_model, classify, recv_chunk)

FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))


# ============================================================================
#  Broadcaster -- fan-out of JSON events to every connected SSE browser.
# ============================================================================
class Broadcaster:
    def __init__(self):
        self._subs = set()
        self._lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def publish(self, obj):
        data = json.dumps(obj)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(data)
            except queue.Full:
                # Slow/stuck browser: drop it rather than block the audio path.
                pass

    def count(self):
        with self._lock:
            return len(self._subs)


# ============================================================================
#  Engine -- owns the decoded text, correction cadence, and idle auto-commit.
# ============================================================================
class Engine:
    def __init__(self, broadcaster, corrector, correct_idle_sec, idle_commit_sec):
        self.b = broadcaster
        self.corrector = corrector                      # None if --llm not passed
        self.correct_idle = float(correct_idle_sec)     # correct the line after this pause (<=0 = only on commit/finish)
        self.idle_commit = float(idle_commit_sec)       # auto-log to history after this pause
        self.lock = threading.RLock()
        self.llm_available = corrector is not None
        self.llm_enabled = corrector is not None        # default ON when available
        self.correct_q = queue.Queue()
        self.session = 0
        self._new_state()

    # ---- state (must hold self.lock) --------------------------------------
    def _new_state(self):
        self.raw = []                # list[str] of emitted chars ('_' == space)
        self.corrected_prefix = ""   # corrected text (in '_' form) for raw[:prefix_len]
        self.prefix_len = 0          # how many raw chars corrected_prefix covers
        self.spaces = 0              # count of '_' seen so far
        self.pending_corr = False    # a correction is queued/in-flight for this line
        self.last_activity = time.time()
        self.session += 1            # invalidates in-flight corrections/commits

    def _display_locked(self):
        raw_str = "".join(self.raw)
        if self.llm_enabled and self.llm_available and self.prefix_len > 0:
            text = self.corrected_prefix + raw_str[self.prefix_len:]
        else:
            text = raw_str
        return text.replace("_", " ")

    def _publish_transcript_locked(self):
        self.b.publish({"type": "transcript", "text": self._display_locked()})

    # ---- called from the audio thread on a fresh client connection --------
    def start_session(self):
        with self.lock:
            self._new_state()
            self.b.publish({"type": "reset"})
            self._publish_transcript_locked()
            self._publish_llm_state_locked()

    def _publish_llm_state_locked(self):
        self.b.publish({
            "type": "llm_state",
            "enabled": bool(self.llm_enabled),
            "available": bool(self.llm_available),
            "correct_idle": self.correct_idle,
        })

    # ---- called from the audio thread once per committed keystroke --------
    def on_char(self, ch, conf):
        with self.lock:
            self.raw.append(ch)
            self.last_activity = time.time()
            if ch == "_":
                self.spaces += 1
            flash = " " if ch == "_" else ch
            self.b.publish({"type": "key", "char": flash, "confidence": float(conf)})
            self._publish_transcript_locked()
            # server-side live log (like the old server.py running "decoded:" line)
            sys.stdout.write("\r  decoded: " + "".join(self.raw) + " ")
            sys.stdout.flush()
            # NOTE: no mid-typing correction. Correction fires on a pause / finish
            # (see idle_monitor / force_commit) so the text never jumps mid-word.

    # ---- correction worker (its own thread) -------------------------------
    def correction_worker(self):
        while True:
            kind, idx, sess = self.correct_q.get()
            with self.lock:
                stale = (sess != self.session or not (self.llm_enabled and self.llm_available))
                to_correct = "".join(self.raw[:idx])
                if stale or not to_correct.strip("_ "):
                    self.pending_corr = False
                    continue
            try:
                corrected = self.corrector.correct(to_correct)
            except Exception as e:
                sys.stderr.write(f"\n[correction error] {e}\n")
                with self.lock:
                    self.pending_corr = False
                continue
            with self.lock:
                self.pending_corr = False
                # Apply only if still the same session and it covers >= what we
                # already have (single worker => in-order, but stay defensive).
                if corrected and sess == self.session and idx <= len(self.raw) and idx >= self.prefix_len:
                    self.corrected_prefix = corrected
                    self.prefix_len = idx
                    self._publish_transcript_locked()
                    sys.stdout.write("\n  corrected: " + corrected + "\n")
                    sys.stdout.flush()

    # ---- idle monitor (its own thread) ------------------------------------
    def idle_monitor(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.raw:
                    continue
                idle = time.time() - self.last_activity
                # 1) correct-on-pause: once typing stops for `correct_idle`, refine
                #    the WHOLE current line in place (nothing jumps while typing).
                if (self.llm_enabled and self.llm_available and not self.pending_corr
                        and self.correct_idle > 0 and self.prefix_len < len(self.raw)
                        and idle >= self.correct_idle):
                    self.pending_corr = True
                    self.correct_q.put(("idle", len(self.raw), self.session))
                # 2) commit-on-longer-idle: auto-log the (corrected) line to history.
                if idle < self.idle_commit:
                    continue
                sess = self.session
                raw_str = "".join(self.raw)
                enabled = self.llm_enabled and self.llm_available
            self._finalize_and_commit(sess, raw_str, enabled)

    def force_commit(self):
        with self.lock:
            if not self.raw:
                return
            sess = self.session
            raw_str = "".join(self.raw)
            enabled = self.llm_enabled and self.llm_available
        self._finalize_and_commit(sess, raw_str, enabled)

    def _finalize_and_commit(self, sess, raw_str, enabled):
        # Heavy final correction happens OUTSIDE the lock. Reuse the already-
        # corrected text if the idle-correction already ran on this exact line.
        final = raw_str
        if enabled:
            with self.lock:
                already = (sess == self.session and "".join(self.raw) == raw_str
                           and self.prefix_len == len(self.raw) and self.corrected_prefix)
                reuse = self.corrected_prefix if already else None
            if reuse is not None:
                final = reuse
            else:
                try:
                    out = self.corrector.correct(raw_str)
                    if out:
                        final = out
                except Exception as e:
                    sys.stderr.write(f"\n[final correction error] {e}\n")
        with self.lock:
            # Bail if the user typed more, or the session rolled, while we were
            # correcting -- we'll catch it on the next idle tick / commit.
            if sess != self.session or "".join(self.raw) != raw_str:
                return
            text = final.replace("_", " ").strip()
            if text:
                self.b.publish({"type": "commit", "text": text})
                sys.stdout.write("\n  logged → history: " + text + "\n")
                sys.stdout.flush()
            self._new_state()
            self.b.publish({"type": "reset"})
            self._publish_transcript_locked()

    # ---- LLM toggle from the browser --------------------------------------
    def set_llm(self, enabled):
        with self.lock:
            if not self.llm_available:
                enabled = False
            self.llm_enabled = bool(enabled)
            self._publish_llm_state_locked()
            if (self.llm_enabled and self.raw and not self.pending_corr
                    and self.prefix_len < len(self.raw)):
                # Turning it on: correct what we have so far right away.
                self.pending_corr = True
                self.correct_q.put(("toggle", len(self.raw), self.session))
            self._publish_transcript_locked()

    # ---- snapshot for a browser that just connected -----------------------
    def snapshot(self):
        with self.lock:
            return [
                {"type": "llm_state", "enabled": bool(self.llm_enabled),
                 "available": bool(self.llm_available), "correct_idle": self.correct_idle},
                {"type": "transcript", "text": self._display_locked()},
            ]


# ============================================================================
#  TCP audio server  (the Mac's client.py connects here; protocol unchanged)
# ============================================================================
def run_audio_server(engine, model, idx_to_label, input_size, device, cfg, args):
    sr = cfg["audio"]["sample_rate"]
    clip_len = int(cfg["clip"]["clip_ms"] / 1000.0 * sr)
    pre = int(cfg["clip"]["pre_onset_ms"] / 1000.0 * sr)
    _gap_ms = args.min_gap_ms if args.min_gap_ms is not None else cfg["detection"]["min_gap_ms"]
    min_gap = int(_gap_ms / 1000.0 * sr)
    min_conf = float(args.min_conf if args.min_conf is not None else cfg["detection"].get("min_confidence", 0.0))
    junk_label = cfg.get("junk", {}).get("label", "junk")
    buf_max = int(args.buffer_sec * sr)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"[audio] listening on {args.host}:{args.port}  "
          f"(min_amp={cfg['detection'].get('min_peak_amplitude')} "
          f"min_conf={min_conf} min_gap_ms={_gap_ms})")

    while True:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        print(f"\n[audio] client connected: {addr}\n" + "-" * 50)
        engine.start_session()
        _handle_conn(conn, addr, engine, model, idx_to_label, input_size, device,
                     cfg, args, sr, clip_len, pre, min_gap, min_conf, junk_label, buf_max)
        print(f"[audio] client disconnected: {addr}")


def _handle_conn(conn, addr, engine, model, idx_to_label, input_size, device,
                 cfg, args, sr, clip_len, pre, min_gap, min_conf, junk_label, buf_max):
    buffer = np.zeros(0, dtype=np.float32)
    total = 0
    state = {"last_committed": -10 ** 12, "classified": 0, "emitted": 0, "rejected": 0}

    def try_commit(local, buf, bs, tot, flush):
        gp = bs + local
        if gp <= state["last_committed"] + min_gap:
            return                                  # dedup / release click
        if (gp - pre) < bs:
            return                                  # clip start rolled off buffer
        if not flush and (gp - pre) + clip_len > tot:
            return                                  # full clip not arrived yet
        clip = detector.extract_clip(buf, local, sr, cfg)
        try:
            ch, conf = tcp.classify(model, clip, sr, cfg, idx_to_label, device, input_size)
        except Exception as e:
            if args.debug:
                print(f"  [classify error: {e}]")
            return
        state["last_committed"] = gp
        state["classified"] += 1
        if args.debug:
            print(f"  [det amp={float(np.max(np.abs(clip))):.3f} -> '{ch}' conf={conf:.2f}]")
        if ch == junk_label or conf < min_conf:
            state["rejected"] += 1
            return
        # Echo the char back to the Mac client (keeps client.py's live line +
        # eval/last_session.json working, exactly like server.py did).
        if not flush:
            try:
                b = ch.encode("utf-8")
                conn.sendall(struct.pack(">I", len(b)) + b)
            except OSError:
                pass
        # Drive the browser-facing engine.
        engine.on_char(ch, conf)
        state["emitted"] += 1

    try:
        while True:
            ready, _, _ = select.select([conn], [], [], 1.0)
            if not ready:
                continue                            # idle handled by Engine, keep conn
            closed = False
            while True:
                chunk = tcp.recv_chunk(conn)
                if chunk is None:
                    closed = True
                    break
                buffer = np.concatenate([buffer, chunk])
                total += len(chunk)
                more, _, _ = select.select([conn], [], [], 0)
                if not more:
                    break
            if len(buffer) > buf_max:
                buffer = buffer[-buf_max:]
            buf_start = total - len(buffer)
            for local in detector.detect_onsets(buffer, sr, cfg):
                try_commit(local, buffer, buf_start, total, flush=False)
            if closed:
                break
        # flush trailing keystrokes that never got post-onset audio
        if len(buffer):
            bs = total - len(buffer)
            for local in detector.detect_onsets(buffer, sr, cfg):
                try_commit(local, buffer, bs, total, flush=True)
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass
        print(f"\n  session totals: classified={state['classified']} "
              f"emitted={state['emitted']} rejected(junk/low-conf)={state['rejected']}")


# ============================================================================
#  HTTP + SSE server  (serves the frontend and pushes live events)
# ============================================================================
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".map": "application/json; charset=utf-8",
}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine = None          # set on the class before the server starts
    broadcaster = None

    def log_message(self, fmt, *a):
        pass               # quiet; the audio/engine logs are what matter

    # ---- helpers ----------------------------------------------------------
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_bytes(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, code, obj):
        self._send_bytes(code, "application/json; charset=utf-8",
                         json.dumps(obj).encode("utf-8"))

    # ---- verbs ------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}
        if path == "/api/llm":
            self.engine.set_llm(bool(payload.get("enabled", False)))
            self._send_json(200, {"ok": True})
        elif path == "/api/commit":
            self.engine.force_commit()
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/events":
            self._stream_events()
        elif path == "/api/status":
            self._send_json(200, {
                "clients": self.broadcaster.count(),
                "llm_available": self.engine.llm_available,
                "llm_enabled": self.engine.llm_enabled,
            })
        else:
            self._serve_static(path)

    # ---- SSE stream -------------------------------------------------------
    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        q = self.broadcaster.subscribe()
        try:
            # tell the browser how quickly to reconnect if the stream ever drops
            self.wfile.write(b"retry: 2000\n\n")
            # initial sync so a freshly opened tab shows current state
            for msg in self.engine.snapshot():
                self._sse_write(json.dumps(msg))
            while True:
                try:
                    data = q.get(timeout=3.0)
                    self._sse_write(data)
                except queue.Empty:
                    # frequent heartbeat: keeps proxies/tunnels from idle-killing
                    # the stream, lets us detect a dead browser within ~3s, and
                    # (as a real data event, not an SSE comment) feeds the client's
                    # watchdog so it doesn't misfire during idle.
                    self._sse_write('{"type": "ping"}')
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.broadcaster.unsubscribe(q)

    def _sse_write(self, data):
        self.wfile.write(b"data: " + data.encode("utf-8") + b"\n\n")
        self.wfile.flush()

    # ---- static files -----------------------------------------------------
    def _serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"
        rel = path.lstrip("/")
        full = os.path.abspath(os.path.join(FRONTEND_DIR, rel))
        # prevent path traversal outside the frontend dir
        if not full.startswith(FRONTEND_DIR + os.sep) and full != FRONTEND_DIR:
            self._send_json(403, {"error": "forbidden"})
            return
        if not os.path.isfile(full):
            self._send_bytes(404, "text/plain; charset=utf-8", b"404 Not Found")
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError:
            self._send_bytes(500, "text/plain; charset=utf-8", b"500")
            return
        self._send_bytes(200, ctype, body)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_http_server(engine, broadcaster, host, port, span=20):
    """Bind the HTTP server, auto-advancing past busy ports (e.g. :8000 already
    taken by another service). Returns (httpd, chosen_port). Raises if none free."""
    Handler.engine = engine
    Handler.broadcaster = broadcaster
    last = None
    for p in range(port, port + span + 1):
        try:
            return ThreadingHTTPServer((host, p), Handler), p
        except OSError as e:
            last = e
            if e.errno == errno.EADDRINUSE:
                continue
            raise
    raise last


# ============================================================================
#  main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Live acoustic-keystroke web demo server")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--host", default="0.0.0.0", help="bind host for the TCP audio server")
    ap.add_argument("--port", type=int, default=9009, help="TCP audio port (Mac client)")
    ap.add_argument("--http-host", default="0.0.0.0", help="bind host for the web server")
    ap.add_argument("--http-port", type=int, default=8000, help="HTTP/SSE port (browser)")
    ap.add_argument("--buffer-sec", type=float, default=2.0)
    ap.add_argument("--min-amp", type=float, default=None)
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--min-gap-ms", type=float, default=None)
    ap.add_argument("--sensitivity", type=float, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--llm", action="store_true", help="load Nemotron for live correction")
    ap.add_argument("--llm-model", default="nvidia/Nemotron-Mini-4B-Instruct")
    ap.add_argument("--correct-idle-sec", type=float, default=5.0,
                    help="run LLM correction after N seconds of no typing (0 = only on log/finish)")
    ap.add_argument("--idle-commit-sec", type=float, default=12.0,
                    help="auto-log the line to history after N idle seconds")
    args = ap.parse_args()

    import torch
    cfg = tcp.load_cfg()
    if args.min_amp is not None:
        cfg["detection"]["min_peak_amplitude"] = args.min_amp
    if args.sensitivity is not None:
        cfg["detection"]["sensitivity"] = args.sensitivity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_path = args.model_path or tcp.find_latest_model(cfg)
    model, idx_to_label, input_size = tcp.load_model(model_path, device)

    corrector = None
    if args.llm:
        from llm_correct import Corrector
        corrector = Corrector(args.llm_model)

    broadcaster = Broadcaster()
    engine = Engine(broadcaster, corrector, args.correct_idle_sec, args.idle_commit_sec)

    # Bind the web server up-front (main thread) so a port clash fails loudly and
    # clearly instead of dying silently in a background thread.
    try:
        httpd, http_port = make_http_server(engine, broadcaster, args.http_host, args.http_port)
    except OSError as e:
        sys.exit(f"\nCould not bind an HTTP port near {args.http_port} ({e}).\n"
                 f"Free the port or pass --http-port <N>.")
    if http_port != args.http_port:
        print(f"[http]  port {args.http_port} busy -> using {http_port} instead")
    print(f"[http]  serving frontend + SSE on http://{args.http_host}:{http_port}  "
          f"(open http://localhost:{http_port} on this box, or http://<spark-ip>:{http_port})")

    threading.Thread(target=engine.correction_worker, daemon=True).start()
    threading.Thread(target=engine.idle_monitor, daemon=True).start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    print(f"LLM correction: {'ON (' + args.llm_model + ')' if corrector else 'disabled (no --llm)'}"
          f"  | correct after {args.correct_idle_sec:.0f}s pause | log after {args.idle_commit_sec:.0f}s")
    try:
        run_audio_server(engine, model, idx_to_label, input_size, device, cfg, args)
    except KeyboardInterrupt:
        print("\nShutting down.")
    except OSError as e:
        sys.exit(f"\nCould not bind audio port {args.port} ({e}).\n"
                 f"Another live_server/server.py is probably still running — stop it "
                 f"(Ctrl+C in its terminal), or pass --port <N>.")


if __name__ == "__main__":
    main()
