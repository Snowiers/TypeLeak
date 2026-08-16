"""
Quick utility to inspect a .pt checkpoint's structure — run this whenever a
checkpoint fails to load, before guessing at the fix.

Usage: python3 inspect_checkpoint.py /path/to/checkpoint.pt
"""
import sys
import torch

def describe(obj, prefix="", max_depth=3, depth=0):
    if depth > max_depth:
        print(f"{prefix}... (truncated)")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "shape"):  # tensor
                print(f"{prefix}{k}: Tensor{tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"{prefix}{k}: dict with {len(v)} keys")
                describe(v, prefix=prefix + "  ", max_depth=max_depth, depth=depth + 1)
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v!r}")
    else:
        print(f"{prefix}{type(obj).__name__} = {obj!r}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python3 inspect_checkpoint.py /path/to/checkpoint.pt")
        sys.exit(1)
    ckpt = torch.load(path, map_location="cpu")
    print(f"Top-level type: {type(ckpt).__name__}\n")
    describe(ckpt)
