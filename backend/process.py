#!/usr/bin/env python3
"""
process.py  --  Build a labelled mel-spectrogram dataset from raw sessions.

HYBRID pipeline:
  1. Detect keystroke onsets UNSUPERVISED from the audio (detector.detect_onsets)
     -- the SAME detector used at live inference, so training clips match test clips.
  2. LABEL each detected onset by matching to the nearest pynput timestamp.
  3. UNMATCHED detections (release clicks, noise, and -- for ambient sessions with
     no logged keys -- everything) become the "junk" class. This teaches the model
     to REJECT non-keystrokes, which fixes live false-positives AND release clicks
     without needing perfect detection.
  4. Clip -> mel-spectrogram -> save .npy (+ .png/.wav), append to labels.csv.

Outputs into <out_dir>/processed/:
    <session>__<idx>__<label>.npy / .png / .wav
    labels.csv  (filename, label, session, onset_time, dt_prev_ms, prev_label)
"""
import os, glob, json, csv, random
import numpy as np
import soundfile as sf
import yaml
from PIL import Image

import detector

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def match_onsets_to_events(onset_times, event_times, tol_s):
    """Greedy one-to-one nearest matching. returns {onset_index -> label}."""
    pairs = []
    for oi, ot in enumerate(onset_times):
        for ei, (et, _lbl) in enumerate(event_times):
            dt = abs(ot - et)
            if dt <= tol_s:
                pairs.append((dt, oi, ei))
    pairs.sort()
    used_o, used_e, matches = set(), set(), {}
    for dt, oi, ei in pairs:
        if oi in used_o or ei in used_e:
            continue
        used_o.add(oi); used_e.add(ei)
        matches[oi] = event_times[ei][1]
    return matches


def save_clip(out_proc, base, clip, spec, sr, cfg):
    if cfg["output"]["save_npy"]:
        np.save(os.path.join(out_proc, base + ".npy"), spec)
    if cfg["output"]["save_png"]:
        Image.fromarray((spec * 255).astype(np.uint8)).save(os.path.join(out_proc, base + ".png"))
    if cfg["output"].get("save_wav", False):
        sf.write(os.path.join(out_proc, base + ".wav"), clip, sr, subtype="FLOAT")


def process_session(session_dir, cfg, writer, out_proc):
    session = os.path.basename(session_dir)
    with open(os.path.join(session_dir, "events.json")) as f:
        meta = json.load(f)
    audio, sr = sf.read(os.path.join(session_dir, "audio.wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    stream_start = meta["stream_start"]
    event_times = [(ev["t"] - stream_start, ev["label"]) for ev in meta["events"]]

    onset_samples = detector.detect_onsets(audio, sr, cfg)
    onset_times = [s / sr for s in onset_samples]

    tol_s = cfg["detection"]["match_tolerance_ms"] / 1000.0
    matches = match_onsets_to_events(onset_times, event_times, tol_s)

    # --- matched keystrokes ---
    labelled = sorted(matches.keys(), key=lambda oi: onset_times[oi])
    prev_t = prev_label = None
    n_ok = 0
    for oi in labelled:
        label = matches[oi]
        clip = detector.extract_clip(audio, onset_samples[oi], sr, cfg)
        spec = detector.clip_to_melspec(clip, sr, cfg)
        base = f"{session}__{oi:04d}__{label}"
        save_clip(out_proc, base, clip, spec, sr, cfg)
        ot = onset_times[oi]
        dt = "" if prev_t is None else round((ot - prev_t) * 1000, 2)
        writer.writerow([base, label, session, round(ot, 4), dt,
                         prev_label if prev_label is not None else ""])
        prev_t, prev_label = ot, label
        n_ok += 1

    # --- junk class: unmatched detections (releases / noise / all-of-ambient) ---
    junk_cfg = cfg.get("junk", {})
    junk_label = junk_cfg.get("label", "junk")
    junk_cap = int(junk_cfg.get("max_per_session", 500))
    n_junk = 0
    if junk_cfg.get("enabled", True):
        unmatched = [oi for oi in range(len(onset_samples)) if oi not in matches]
        random.Random(0).shuffle(unmatched)
        for oi in unmatched[:junk_cap]:
            clip = detector.extract_clip(audio, onset_samples[oi], sr, cfg)
            spec = detector.clip_to_melspec(clip, sr, cfg)
            base = f"{session}__{oi:04d}__{junk_label}"
            save_clip(out_proc, base, clip, spec, sr, cfg)
            writer.writerow([base, junk_label, session, round(onset_times[oi], 4), "", ""])
            n_junk += 1

    n_det, n_ev = len(onset_samples), len(event_times)
    print(f"  {session}: detected={n_det}  logged_keys={n_ev}  labelled={n_ok}  "
          f"junk={n_junk}  missed_keys={n_ev - n_ok}")
    return n_ok, n_junk


def main():
    cfg = load_config()
    root = os.path.join(HERE, cfg["output"]["out_dir"])
    raw_root = os.path.join(root, "raw")
    out_proc = os.path.join(root, "processed")
    os.makedirs(out_proc, exist_ok=True)

    sessions = sorted(glob.glob(os.path.join(raw_root, "*")))
    if not sessions:
        print(f"No raw sessions in {raw_root}. Run record.py / record_ambient.py first.")
        return

    labels_path = os.path.join(out_proc, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "session", "onset_time",
                         "dt_prev_ms", "prev_label"])
        tot_k = tot_j = 0
        for s in sessions:
            if os.path.isfile(os.path.join(s, "events.json")):
                k, j = process_session(s, cfg, writer, out_proc)
                tot_k += k; tot_j += j

    print(f"\nDone. {tot_k} keystrokes + {tot_j} junk -> {out_proc}")
    print(f"Labels -> {labels_path}")


if __name__ == "__main__":
    main()
