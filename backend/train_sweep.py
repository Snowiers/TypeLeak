#!/usr/bin/env python3
"""
train_sweep.py  --  sweep timm models + learning rates, print a leaderboard,
save the best model. Includes CoAtNet (the family the original paper used)
alongside EfficientNetV2 / ConvNeXt / ResNet / MobileNet.

Each model's input is resized to its native size (CoAtNet needs 224; CNNs accept
128 but we resize uniformly and STORE the size so server.py feeds the same size
at inference -- train/infer input size must match).

Usage:
  python train_sweep.py --epochs 20
  python train_sweep.py --holdout-session <SESSION>     # honest cross-session leaderboard
  python train_sweep.py --lr-list 1e-4 3e-4 1e-3        # also sweep LR
  python train_sweep.py --models tf_efficientnetv2_b0.in1k coatnet_0_rw_224.sw_in1k
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import load_cfg, KeystrokeDataset, read_labels, build_splits

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = [
    "tf_efficientnetv2_b0.in1k",        # smallest EfficientNetV2 (current baseline)
    "tf_efficientnetv2_s.in1k",         # larger EfficientNetV2
    "resnet18.a1_in1k",                 # classic small CNN
    "convnext_tiny.fb_in1k",            # modern pure-CNN
    "mobilenetv3_small_100.lamb_in1k",  # tiny / fast
    "coatnet_nano_rw_224.sw_in1k",      # CoAtNet (paper's family), small
    "coatnet_0_rw_224.sw_in1k",         # CoAtNet, larger
]


def build(model_name, num_classes, device):
    import timm
    m = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    try:
        tgt = int(m.pretrained_cfg["input_size"][-1])
    except Exception:
        tgt = 224
    return m.to(device), tgt


def _resize(x, tgt):
    if x.shape[-1] != tgt:
        x = F.interpolate(x, size=(tgt, tgt), mode="bilinear", align_corners=False)
    return x


def evaluate(model, loader, device, tgt):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = _resize(x, tgt)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item(); total += y.numel()
    return correct / max(1, total)


def train_one(model_name, lr, train_dl, val_dl, num_classes, device, epochs):
    model, tgt = build(model_name, num_classes, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            x = _resize(x, tgt)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        acc = evaluate(model, val_dl, device, tgt)
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return best, best_state, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--holdout-session", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-list", type=float, nargs="+", default=[3e-4])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-digits", action="store_true",
                    help="drop 0-9 (letters + space + junk only)")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_cfg()
    proc = os.path.join(HERE, cfg["output"]["out_dir"], "processed")
    image_size = cfg["output"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  image_size {image_size}")

    rows = read_labels(proc)
    if args.no_digits:
        rows = [r for r in rows if r["label"] not in set("0123456789")]
    labels = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in l2i.items()}
    print(f"{len(rows)} samples, {len(labels)} classes")

    tr, va = build_splits(rows, args.holdout_session, args.val_frac, args.seed)
    print(f"train {len(tr)}  val {len(va)}"
          + (f"  (holdout {args.holdout_session})" if args.holdout_session else ""))
    train_dl = DataLoader(KeystrokeDataset(tr, proc, l2i, image_size, augment=True),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(KeystrokeDataset(va, proc, l2i, image_size, augment=False),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    results = []
    best = (0.0, None, None, None, None)   # acc, name, lr, state, tgt
    for name in args.models:
        for lr in args.lr_list:
            t0 = time.time()
            try:
                acc, state, tgt = train_one(name, lr, train_dl, val_dl,
                                            len(labels), device, args.epochs)
            except Exception as e:
                print(f"  {name:34s} lr={lr:.0e}  FAILED: {e}")
                continue
            dt = time.time() - t0
            results.append((acc, name, lr, dt))
            print(f"  {name:34s} lr={lr:.0e}  best_val={acc:.4f}  ({dt:.0f}s)")
            if acc > best[0]:
                best = (acc, name, lr, state, tgt)

    results.sort(reverse=True)
    print("\n===== LEADERBOARD =====")
    for acc, name, lr, dt in results:
        print(f"  {acc:.4f}  {name}  lr={lr:.0e}  ({dt:.0f}s)")

    if best[3] is not None:
        run_dir = os.path.join(HERE, cfg["output"]["out_dir"], "runs", "sweep_best")
        os.makedirs(run_dir, exist_ok=True)
        torch.save({"model_state": best[3], "model_name": best[1],
                    "idx_to_label": idx_to_label, "image_size": image_size,
                    "input_size": best[4], "val_acc": best[0]},
                   os.path.join(run_dir, "best_model.pt"))
        with open(os.path.join(run_dir, "label_map.json"), "w") as f:
            json.dump(idx_to_label, f, indent=2)
        print(f"\nBest: {best[1]}  lr={best[2]:.0e}  val={best[0]:.4f}  "
              f"input_size={best[4]}\nSaved -> {run_dir}/best_model.pt")


if __name__ == "__main__":
    main()
