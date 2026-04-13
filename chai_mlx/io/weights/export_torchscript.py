from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def export_torchscript_module(src: Path, dst: Path) -> None:
    import torch

    mod = torch.jit.load(str(src), map_location="cpu")
    arrays: dict[str, np.ndarray] = {}
    for name, param in mod.named_parameters():
        arrays[name] = param.detach().cpu().numpy()
    for name, buf in mod.named_buffers():
        if name not in arrays:
            arrays[name] = buf.detach().cpu().numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export Chai-1 TorchScript weights to NPZ")
    parser.add_argument("src", type=Path, help="TorchScript .pt file or directory")
    parser.add_argument("dst", type=Path, help="Output .npz file or directory")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.src.is_file():
        dst = args.dst if args.dst.suffix == ".npz" else args.dst / (args.src.stem + ".npz")
        export_torchscript_module(args.src, dst)
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    for pt in sorted(args.src.glob("*.pt")):
        export_torchscript_module(pt, args.dst / f"{pt.stem}.npz")


if __name__ == "__main__":  # pragma: no cover
    main()
