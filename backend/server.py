#!/usr/bin/env python3
"""
server.py  --  runs on the LINUX / DGX SPARK box (does all the ML).

Receives streamed microphone audio from the Mac (client.py), maintains a rolling
audio buffer, detects keystroke onsets ACROSS chunk boundaries, classifies each
completed keystroke with the trained model, and prints the decoded text live.
Optionally sends each predicted character back to the client to display.

Why a rolling buffer (not per-chunk detection): a keystroke can land on the
boundary between two network chunks. We keep the last few seconds, detect on the
whole buffer, and only "commit" an onset once (a) enough audio AFTER it has
arrived to cut a full clip, and (b) it's far enough from the last committed key
(min_gap) -- which also swallows the release click and detector jitter.

Run:  python server.py                 # auto-loads newest run's best_model.pt
      python server.py --model-path dataset/runs/XX…/best_model.pt --port 9009
"""
import os, sys, glob, socket, struct, argparse, select
import numpy as np
import yaml
import torch
import torch.nn.functional as F

import detector

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def find_latest_model(cfg):
    runs = glob.glob(os.path.join(HERE, cfg["output"]["out_dir"], "runs", "*", "best_model.pt"))
    if not runs:
        sys.exit("No trained model found. Run train.py first, or pass --model-path.")
    # most-recently-MODIFIED (not lexicographic -- so a fresh train.py run wins
    # over an older sweep_best, and vice-versa)
    return max(runs, key=os.path.getmtime)


def load_model(model_path, device):
    try:
        import timm
    except ImportError:
        sys.exit("timm not installed: pip install timm")
    ckpt = torch.load(model_path, map_location=device)
    idx_to_label = ckpt["idx_to_label"]
    model = timm.create_model(ckpt["model_name"], pretrained=False,
                              num_classes=len(idx_to_label))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded {model_path}  (val_acc {ckpt.get('val_acc', '?')}, "
          f"{len(idx_to_label)} classes, image_size {ckpt['image_size']})")
    input_size = int(ckpt.get("input_size", ckpt["image_size"]))
    print(f"  model input size: {input_size}")
    return model, idx_to_label, input_size


def classify(model, clip, sr, cfg, idx_to_label, device, input_size):
    spec = detector.clip_to_melspec(clip, sr, cfg)              # [H,W] in [0,1]
    x = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,H,W]
    x = x.to(device)
    if x.shape[-1] != input_size:   # match the size the model was trained at
        x = F.interpolate(x, size=(input_size, input_size),
                          mode="bilinear", align_corners=False)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return idx_to_label[int(idx.item())], float(conf.item())


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def recv_chunk(conn):
    hdr = recv_exact(conn, 4)
    if hdr is None:
        return None
    (nbytes,) = struct.unpack(">I", hdr)
    data = recv_exact(conn, nbytes)
    if data is None:
        return None
    return np.frombuffer(data, dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--buffer-sec", type=float, default=2.0, help="rolling buffer length")
    ap.add_argument("--min-amp", type=float, default=None, help="override detection.min_peak_amplitude")
    ap.add_argument("--min-conf", type=float, default=None, help="override detection.min_confidence")
    ap.add_argument("--min-gap-ms", type=float, default=None, help="override detection.min_gap_ms")
    ap.add_argument("--sensitivity", type=float, default=None, help="override detection.sensitivity")
    ap.add_argument("--idle-timeout", type=float, default=30.0, help="drop client after N idle seconds")
    ap.add_argument("--debug", action="store_true", help="print each detection's amp/char/conf (for tuning)")
    ap.add_argument("--llm", action="store_true",
                    help="also print an LLM-corrected version of the decode (Nemotron, on this box)")
    ap.add_argument("--llm-model", default="nvidia/Nemotron-Mini-4B-Instruct")
    args = ap.parse_args()

    cfg = load_cfg()
    # live threshold overrides (tune without editing config / reprocessing)
    if args.min_amp is not None:
        cfg["detection"]["min_peak_amplitude"] = args.min_amp
    if args.sensitivity is not None:
        cfg["detection"]["sensitivity"] = args.sensitivity
    sr = cfg["audio"]["sample_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = args.model_path or find_latest_model(cfg)
    model, idx_to_label, input_size = load_model(model_path, device)

    corrector = None
    if args.llm:
        from llm_correct import Corrector
        corrector = Corrector(args.llm_model)

    clip_len = int(cfg["clip"]["clip_ms"] / 1000.0 * sr)
    pre = int(cfg["clip"]["pre_onset_ms"] / 1000.0 * sr)
    _gap_ms = args.min_gap_ms if args.min_gap_ms is not None else cfg["detection"]["min_gap_ms"]
    min_gap = int(_gap_ms / 1000.0 * sr)
    min_conf = float(args.min_conf if args.min_conf is not None else cfg["detection"].get("min_confidence", 0.0))
    print(f"  thresholds: min_amp={cfg['detection'].get('min_peak_amplitude')}  "
          f"min_conf={min_conf}  min_gap_ms={_gap_ms}  sensitivity={cfg['detection'].get('sensitivity')}")
    junk_label = cfg.get("junk", {}).get("label", "junk")
    buf_max = int(args.buffer_sec * sr)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    srv.settimeout(1.0)   # so accept() returns periodically -> Ctrl+C works
    print(f"Listening on {args.host}:{args.port}  (Ctrl+C to quit)")

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue   # no client yet; loop lets KeyboardInterrupt through
            print(f"\nClient connected: {addr}\n" + "-" * 50)
            buffer = np.zeros(0, dtype=np.float32)
            total = 0                 # total samples received (global sample clock)
            last_committed = -10 ** 12
            decoded = []
            n_classified = 0      # loud transients that reached the model
            n_rejected = 0        # classified as junk / below confidence
            idle = 0.0

            def try_commit(local, buf, bs, tot, flush):
                nonlocal last_committed, n_classified, n_rejected
                gp = bs + local
                if gp <= last_committed + min_gap:
                    return                                  # dedup / release click
                if (gp - pre) < bs:
                    return                                  # clip start rolled off buffer
                if not flush and (gp - pre) + clip_len > tot:
                    return                                  # full clip not arrived yet (live)
                clip = detector.extract_clip(buf, local, sr, cfg)
                try:
                    ch, conf = classify(model, clip, sr, cfg, idx_to_label, device, input_size)
                except Exception as e:
                    if args.debug:
                        print(f"\n  [classify error: {e}]")
                    return
                last_committed = gp
                n_classified += 1
                if args.debug:
                    print(f"\n  [det amp={float(np.max(np.abs(clip))):.3f} -> '{ch}' conf={conf:.2f}]")
                if ch == junk_label or conf < min_conf:
                    n_rejected += 1
                    return
                decoded.append(ch)
                sys.stdout.write("\r  decoded: " + "".join(decoded) + " ")
                sys.stdout.flush()
                if not flush:
                    try:
                        b = ch.encode("utf-8")
                        conn.sendall(struct.pack(">I", len(b)) + b)
                    except OSError:
                        pass

            try:
                while True:
                    ready, _, _ = select.select([conn], [], [], 1.0)
                    if not ready:
                        idle += 1.0
                        if idle >= args.idle_timeout:
                            print(f"\n  (client idle {args.idle_timeout:.0f}s — dropping)")
                            break
                        continue
                    idle = 0.0
                    # Drain EVERY currently-buffered chunk, then detect ONCE.
                    # Prevents latency build-up: if the client is ahead, we
                    # process the whole backlog per iteration instead of running
                    # detection on each 23ms block (which fell behind real-time
                    # and made it feel like the stream "cut out").
                    closed = False
                    while True:
                        chunk = recv_chunk(conn)
                        if chunk is None:
                            closed = True
                            break
                        buffer = np.concatenate([buffer, chunk])
                        total += len(chunk)
                        more, _, _ = select.select([conn], [], [], 0)
                        if not more:
                            break                  # no more chunks waiting -> detect now
                    if len(buffer) > buf_max:
                        buffer = buffer[-buf_max:]
                    buf_start = total - len(buffer)   # global index of buffer[0]
                    for local in detector.detect_onsets(buffer, sr, cfg):
                        try_commit(local, buffer, buf_start, total, flush=False)
                    if closed:
                        break                   # peer closed cleanly
                # ---- flush trailing keystrokes that never got post-onset audio ----
                # (fixes keystrokes at the very end being swallowed on stop)
                if len(buffer):
                    bs = total - len(buffer)
                    for local in detector.detect_onsets(buffer, sr, cfg):
                        try_commit(local, buffer, bs, total, flush=True)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                conn.close()
                print(f"\nClient disconnected.")
                print(f"  loud transients classified : {n_classified}")
                print(f"  emitted as keys            : {len(decoded)}")
                print(f"  rejected (junk/low-conf)   : {n_rejected}")
                raw = "".join(decoded)
                print(f"  decoded string             : {raw}")
                if corrector is not None and raw:
                    try:
                        print(f"  LLM-corrected              : {corrector.correct(raw)}")
                    except Exception as e:
                        print(f"  (LLM correction failed: {e})")
                print("Listening again (Ctrl+C to quit)...")
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
