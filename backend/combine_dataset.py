#!/usr/bin/env python3
"""
combine_dataset.py  --  Merge many per-folder datasets into one that train.py reads.

Your `dataset/` has multiple subfolders, each with its own .npy (+ .wav/.png) files
and one or more labels CSVs. train.py wants a single `dataset/processed/` folder with
one `labels.csv` and all the .npy alongside it. This script does that merge:

  - recursively finds every labels-style CSV under the source dir (must have
    'filename' and 'label' columns),
  - for each row, finds "<filename>.npy" in that CSV's own folder,
  - copies it into the output folder under a COLLISION-SAFE name
    (prefixed by the source folder), rewriting the 'filename' column to match,
  - writes one merged labels.csv with a unified column set,
  - fills a 'session' value (from the source folder) if a row lacks one,
  - optionally copies the matching .wav/.png too.

Usage:
    python combine_dataset.py
    python combine_dataset.py --source dataset --out dataset/processed --copy-extras
"""
import os, csv, glob, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
NEED_COLS = ("filename", "label")
KEEP_COLS = ["filename", "label", "session", "onset_time",
             "dt_prev_ms", "prev_label", "source_folder"]


def is_labels_csv(path):
    try:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        return all(c in header for c in NEED_COLS)
    except (StopIteration, OSError, UnicodeDecodeError):
        return False


def safe_prefix(source_root, csv_dir):
    """A filesystem-safe tag from the source folder, to guarantee unique names."""
    rel = os.path.relpath(csv_dir, source_root)
    rel = rel.strip(os.sep)
    if rel in (".", ""):
        rel = "root"
    return rel.replace(os.sep, "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(HERE, "dataset"),
                    help="root folder to scan for per-session subfolders")
    ap.add_argument("--out", default=os.path.join(HERE, "dataset", "processed"),
                    help="combined output folder (what train.py reads)")
    ap.add_argument("--copy-extras", action="store_true",
                    help="also copy matching .wav and .png files")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # find all labels CSVs, skipping anything already inside the output folder
    all_csvs = [p for p in glob.glob(os.path.join(source, "**", "*.csv"), recursive=True)
                if os.path.abspath(p).startswith(out + os.sep) is False
                and os.path.abspath(os.path.dirname(p)) != out]
    labels_csvs = [p for p in all_csvs if is_labels_csv(p)]

    if not labels_csvs:
        print(f"No labels-style CSVs (with 'filename'+'label' columns) found under {source}")
        return

    print(f"Found {len(labels_csvs)} labels CSV(s):")
    for p in labels_csvs:
        print("   " + os.path.relpath(p, source))

    merged_rows = []
    seen = set()
    n_missing = 0
    n_copied = 0

    for csv_path in labels_csvs:
        csv_dir = os.path.dirname(csv_path)
        prefix = safe_prefix(source, csv_dir)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("filename", "").strip()
                if not orig:
                    continue
                src_npy = os.path.join(csv_dir, orig + ".npy")
                if not os.path.isfile(src_npy):
                    n_missing += 1
                    continue
                new_name = f"{prefix}__{orig}"
                if new_name in seen:      # de-dupe identical rows across CSVs
                    continue
                seen.add(new_name)

                shutil.copy2(src_npy, os.path.join(out, new_name + ".npy"))
                n_copied += 1
                if args.copy_extras:
                    for ext in (".wav", ".png"):
                        src_extra = os.path.join(csv_dir, orig + ext)
                        if os.path.isfile(src_extra):
                            shutil.copy2(src_extra, os.path.join(out, new_name + ext))

                out_row = {c: "" for c in KEEP_COLS}
                for c in KEEP_COLS:
                    if c in row and row[c] != "":
                        out_row[c] = row[c]
                out_row["filename"] = new_name
                if not out_row["session"]:
                    out_row["session"] = prefix          # fallback session id
                out_row["source_folder"] = prefix
                merged_rows.append(out_row)

    labels_out = os.path.join(out, "labels.csv")
    with open(labels_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP_COLS)
        w.writeheader()
        w.writerows(merged_rows)

    # summary
    from collections import Counter
    by_label = Counter(r["label"] for r in merged_rows)
    by_session = Counter(r["session"] for r in merged_rows)
    print(f"\nMerged {len(merged_rows)} samples into {out}")
    print(f"  copied .npy: {n_copied}   missing .npy skipped: {n_missing}")
    print(f"  sessions: {len(by_session)}  -> {dict(by_session)}")
    print(f"  classes: {len(by_label)}")
    least = by_label.most_common()[-5:]
    print(f"  least-represented keys: {least}")
    print(f"\nlabels.csv -> {labels_out}")
    print("train.py will now read this combined folder.")


if __name__ == "__main__":
    main()
