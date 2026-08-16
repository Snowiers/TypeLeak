#!/usr/bin/env python3
"""
train.py  --  Image-by-image keystroke classifier (baseline, not CTC).

Trains a pretrained EfficientNetV2 (smallest variant by default) on the
mel-spectrograms produced by process.py. Grayscale specs are expanded to 3
channels so ImageNet-pretrained weights apply.

Reads dataset location + image size from config.yaml; all training
hyperparameters are argparse flags (sensible defaults below, roughly matching
the paper's setup where it specified anything, otherwise standard fine-tuning
values).

Outputs into <out_dir>/runs/<timestamp>/:
    best_model.pt      (weights + label_map + config snapshot)
    label_map.json
    history.csv

Usage:
    python train.py                          # random stratified split
    python train.py --holdout-session S      # train on all sessions except S (honest cross-session eval)
    python train.py --model tf_efficientnetv2_b0.in1k
"""
import os, sys, json, csv, glob, argparse, random, datetime
import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- config -----------------------------
def load_cfg():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


# ----------------------------- data -----------------------------
class KeystrokeDataset(Dataset):
    """Loads .npy mel-spectrograms + integer labels. Optional SpecAugment."""
    def __init__(self, rows, proc_dir, label_to_idx, image_size, augment=False):
        self.rows = rows
        self.proc_dir = proc_dir
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _specaugment(self, spec):
        # spec: [H, W] in [0,1]. Mask random freq bands + time bands (SpecAugment).
        H, W = spec.shape
        # up to 2 frequency masks
        for _ in range(2):
            if random.random() < 0.5:
                f = random.randint(1, max(1, H // 8))
                f0 = random.randint(0, H - f)
                spec[f0:f0 + f, :] = 0.0
        # up to 2 time masks
        for _ in range(2):
            if random.random() < 0.5:
                t = random.randint(1, max(1, W // 8))
                t0 = random.randint(0, W - t)
                spec[:, t0:t0 + t] = 0.0
        return spec

    def _time_shift(self, spec, p=0.5, lo=0.10, hi=0.20):
        # shift the spectrogram along TIME (columns) by 10-20% of width, random
        # direction, zero-filling the gap. Makes the model robust to the small
        # onset-centering differences between offline and live extraction.
        if random.random() > p:
            return spec
        W = spec.shape[1]
        mag = int(random.uniform(lo, hi) * W)
        if mag == 0:
            return spec
        shift = mag if random.random() < 0.5 else -mag
        out = np.zeros_like(spec)
        if shift > 0:
            out[:, shift:] = spec[:, :W - shift]
        else:
            s = -shift
            out[:, :W - s] = spec[:, s:]
        return out

    def __getitem__(self, i):
        fn, label = self.rows[i]
        spec = np.load(os.path.join(self.proc_dir, fn + ".npy")).astype(np.float32)
        if self.augment:
            spec = self._time_shift(spec.copy())
            spec = self._specaugment(spec)
        # -> 3-channel tensor for ImageNet-pretrained backbone
        t = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
        return t, self.label_to_idx[label]


def read_labels(proc_dir):
    rows = []
    with open(os.path.join(proc_dir, "labels.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def build_splits(label_rows, holdout_session, val_frac, seed):
    """Return (train_rows, val_rows) as lists of (filename, label)."""
    rng = random.Random(seed)
    if holdout_session:
        train = [(r["filename"], r["label"]) for r in label_rows
                 if r["session"] != holdout_session]
        val = [(r["filename"], r["label"]) for r in label_rows
               if r["session"] == holdout_session]
        if not val:
            sys.exit(f"No rows for holdout session '{holdout_session}'. "
                     f"Sessions present: {sorted({r['session'] for r in label_rows})}")
        return train, val
    # stratified random split by label
    by_label = {}
    for r in label_rows:
        by_label.setdefault(r["label"], []).append((r["filename"], r["label"]))
    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ----------------------------- model -----------------------------
def build_model(model_name, num_classes, device):
    try:
        import timm
    except ImportError:
        sys.exit("timm not installed. Run: pip install timm")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(device)


# ----------------------------- train -----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for yi, pi in zip(y.tolist(), pred.tolist()):
                per_class_total[yi] = per_class_total.get(yi, 0) + 1
                if yi == pi:
                    per_class_correct[yi] = per_class_correct.get(yi, 0) + 1
    acc = correct / max(1, total)
    return acc, per_class_correct, per_class_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tf_efficientnetv2_s.in1k",
                    help="timm model name (smallest EfficientNetV2 = tf_efficientnetv2_b0)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--holdout-session", default=None,
                    help="train on all sessions except this one (honest cross-session eval)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10, help="early-stop patience (epochs)")
    ap.add_argument("--no-digits", action="store_true",
                    help="drop 0-9 (train on letters + space + junk only)")
    ap.add_argument("--junk-ratio", type=float, default=0,
                    help="subsample junk to this multiple of the avg key-class size "
                         "(prevents the reject class from dominating). 0 = keep all junk.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg()
    proc_dir = os.path.join(HERE, cfg["output"]["out_dir"], "processed")
    image_size = cfg["output"]["image_size"]
    if not os.path.isfile(os.path.join(proc_dir, "labels.csv")):
        sys.exit(f"No labels.csv in {proc_dir}. Run process.py first.")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}  |  model: {args.model}  |  image_size: {image_size}")

    label_rows = read_labels(proc_dir)
    if args.no_digits:
        digits = set("0123456789")
        before = len(label_rows)
        label_rows = [r for r in label_rows if r["label"] not in digits]
        print(f"--no-digits: dropped {before - len(label_rows)} digit samples")

    # balance the junk (reject) class -- it's usually hugely over-represented,
    # which biases the model toward rejecting real keystrokes.
    junk_label = cfg.get("junk", {}).get("label", "junk")
    if args.junk_ratio > 0:
        non_junk = [r for r in label_rows if r["label"] != junk_label]
        junk = [r for r in label_rows if r["label"] == junk_label]
        n_key_classes = len({r["label"] for r in non_junk})
        avg = len(non_junk) / max(1, n_key_classes)
        target = int(avg * args.junk_ratio)
        if len(junk) > target:
            random.Random(args.seed).shuffle(junk)
            junk = junk[:target]
        label_rows = non_junk + junk
        print(f"junk balanced: {len(junk)} kept (~{args.junk_ratio}x avg class of {avg:.0f})")

    labels = sorted({r["label"] for r in label_rows})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    print(f"{len(label_rows)} samples, {len(labels)} classes")

    train_rows, val_rows = build_splits(label_rows, args.holdout_session,
                                        args.val_frac, args.seed)
    print(f"train: {len(train_rows)}  val: {len(val_rows)}"
          + (f"  (holdout session {args.holdout_session})" if args.holdout_session else ""))

    train_ds = KeystrokeDataset(train_rows, proc_dir, label_to_idx, image_size, augment=True)
    val_ds = KeystrokeDataset(val_rows, proc_dir, label_to_idx, image_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"),
                          drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = build_model(args.model, len(labels), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run_dir = os.path.join(HERE, cfg["output"]["out_dir"], "runs",
                           datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "label_map.json"), "w") as f:
        json.dump(idx_to_label, f, indent=2)

    hist_path = os.path.join(run_dir, "history.csv")
    hist_f = open(hist_path, "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["epoch", "train_loss", "val_acc", "lr"])

    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        train_loss = running / max(1, n)
        val_acc, pcc, pct = evaluate(model, val_dl, device)
        lr_now = opt.param_groups[0]["lr"]
        hist.writerow([epoch, round(train_loss, 4), round(val_acc, 4), round(lr_now, 6)])
        hist_f.flush()
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val_acc {val_acc:.4f}  "
              f"(best {best_acc:.4f})")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "idx_to_label": idx_to_label,
                "image_size": image_size,
                "val_acc": best_acc,
            }, os.path.join(run_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (no val improvement for {args.patience}).")
                break

    hist_f.close()
    print(f"\nBest val_acc: {best_acc:.4f}")
    print(f"Saved: {os.path.join(run_dir, 'best_model.pt')}")
    # quick per-class readout on final val
    _, pcc, pct = evaluate(model, val_dl, device)
    worst = sorted(((pcc.get(i, 0) / max(1, pct.get(i, 1)), idx_to_label[i])
                    for i in pct), )[:8]
    print("Weakest classes (acc, key):", [(round(a, 2), k) for a, k in worst])


if __name__ == "__main__":
    main()
