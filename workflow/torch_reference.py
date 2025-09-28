"""
Minimal, format-agnostic PyTorch Dataset for ESF (Encodec Sidecar Format),
plus a small CLI to inspect/verify a dataset.

Triplet per item (co-located):
  NAME.ecdc        # Encodec codes (torch.save checkpoint)
  NAME.cond.npy    # conditioning matrix [T, D] (float16/float32)
  NAME.cond.json   # metadata with names/norm/fps

CLI usage examples:
  python torch_reference.py --tokens-root /hf/my_ds/tokens --recursive --show 3 --verify
  python torch_reference.py --hf-ds /hf/my_ds --split train --show 5 --verify

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Union, Tuple
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _canonicalize_codes(codes: torch.Tensor) -> torch.Tensor:
    """
    Accepts shapes [1, Cb, T], [Cb, T], or [1, 1, Cb, T].
    Returns LongTensor of shape [T, Cb].
    """
    t = codes
    if t.ndim == 4:  # [1,1,Cb,T]
        t = t.squeeze(0).squeeze(0)  # [Cb, T]
    if t.ndim == 3:  # [1, Cb, T]
        t = t.squeeze(0)            # [Cb, T]
    if t.ndim != 2:
        raise ValueError(f"Unexpected code shape {tuple(codes.shape)}")
    t = t.transpose(0, 1).contiguous()  # [T, Cb]
    return t.long()


class EncodecSidecarDataset(Dataset):
    def __init__(
        self,
        tokens_root: Union[str, Path],
        recursive: bool = True,
        suffix: str = ".ecdc",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        paths: Optional[List[Path]] = None,
    ):
        self.tokens_root = Path(tokens_root)
        self.recursive = recursive
        self.suffix = suffix
        self.device = device
        self.dtype = dtype

        if paths is not None:
            self._paths: List[Path] = [Path(p) for p in paths]
        else:
            patt = f"*{suffix}"
            if recursive:
                paths = sorted(self.tokens_root.rglob(patt), key=lambda p: p.as_posix().lower())
            else:
                paths = sorted(self.tokens_root.glob(patt), key=lambda p: p.as_posix().lower())
            self._paths = [p for p in paths if p.is_file()]
        if not self._paths:
            raise FileNotFoundError(f"No {suffix} files found under {self.tokens_root}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ecdc = self._paths[idx]
        cpath = ecdc.with_suffix(".cond.npy")
        jpath = ecdc.with_suffix(".cond.json")

        # Load codes
        ckpt = torch.load(ecdc, map_location="cpu")
        codes = ckpt["audio_codes"]
        codes = _canonicalize_codes(torch.as_tensor(codes))
        T, Cb = codes.shape

        # Load sidecar
        cond = np.load(cpath, mmap_mode="r")
        if cond.ndim != 2 or cond.shape[0] != T:
            raise ValueError(f"Sidecar shape {cond.shape} does not match codes T={T}: {cpath}")
        names: List[str] = []
        meta = {}
        if jpath.exists():
            meta = json.loads(jpath.read_text(encoding="utf-8"))
            names = [str(x) for x in meta.get("names", [])]

        # To tensors
        cond_t = torch.from_numpy(np.asarray(cond)).to(self.dtype)
        if self.device is not None:
            codes = codes.to(self.device, non_blocking=True)
            cond_t = cond_t.to(self.device, non_blocking=True)

        return {
            "codes": codes,         # Long [T, Cb]
            "cond":  cond_t,        # Float [T, D]
            "names": names,         # List[str], may be empty
            "meta":  meta,          # dict (may be empty)
            "path":  str(ecdc),     # original path
        }

    
    @classmethod
    def from_hf_dataset(cls, ds_dir: Union[str, Path], split: str = "train",
                        device: Optional[torch.device] = None,
                        dtype: torch.dtype = torch.float32) -> "EncodecSidecarDataset":
        """
        Load a Dataset saved by build_dataset_from_folder.py that stores relative 'audio' paths.

        We accept either:
          - paths already prefixed with the tokens subdir (e.g., "tokens/ZOOM0001.ecdc"), or
          - paths relative to the tokens subdir (e.g., "ZOOM0001.ecdc").

        Resolution tries ds_dir/<path> first, else ds_dir/tokens/<path>.
        """
        from datasets import load_from_disk  # lazy import
        ds_dir = Path(ds_dir)
        dsd = load_from_disk(str(ds_dir))
        if split not in dsd:
            raise KeyError(f"Split '{split}' not found. Available: {list(dsd.keys())}")
        rel_paths = [Path(p) for p in dsd[split]["audio"]]
        abs_paths = []
        for rp in rel_paths:
            cand1 = ds_dir / rp
            cand2 = ds_dir / "tokens" / rp
            if cand1.exists():
                abs_paths.append(cand1)
            else:
                abs_paths.append(cand2)
        # tokens_root is the common parent of the files we'll access
        tokens_root = ds_dir
        return cls(tokens_root=tokens_root, recursive=False, device=device, dtype=dtype, paths=abs_paths)


def _parse_indices(s: Optional[str], n: int) -> List[int]:
    """
    Parse an indices string like "0,5,10" or "0:3" or "10:20:2".
    Returns a list of valid indices within [0, n).
    """
    if not s:
        return []
    s = s.strip()
    out: List[int] = []
    parts = s.split(",")
    for p in parts:
        p = p.strip()
        if ":" in p:
            # slice form a:b[:step]
            sl = [x for x in p.split(":") if len(x)]
            a = int(sl[0]) if len(sl) >= 1 else 0
            b = int(sl[1]) if len(sl) >= 2 else n
            step = int(sl[2]) if len(sl) >= 3 else 1
            out.extend(list(range(a, min(b, n), step)))
        else:
            out.append(int(p))
    # clamp and dedup preserve order
    seen = set()
    res = []
    for i in out:
        if 0 <= i < n and i not in seen:
            res.append(i); seen.add(i)
    return res



def _print_param_rows(cond_tensor, names, n_rows: int):
    """Pretty-print first n_rows of cond matrix with column names."""
    import numpy as _np
    n_rows = int(max(0, n_rows))
    if n_rows <= 0:
        return
    # move small slice to CPU numpy for pretty printing
    if hasattr(cond_tensor, "device"):
        arr = cond_tensor[:n_rows].detach().cpu().numpy()
    else:
        arr = _np.asarray(cond_tensor)[:n_rows]
    Tn, D = arr.shape
    cols = names if names and len(names) == D else [f"p{j}" for j in range(D)]
    # header
    header = "    t  | " + " | ".join(f"{c:>10s}" for c in cols)
    print(header)
    print("    " + "-" * (len(header)-4))
    for i in range(Tn):
        vals = " | ".join(f"{float(v):10.6f}" for v in arr[i])
        print(f"    {i:>3d} | {vals}")

def summarize_dataset(ds: EncodecSidecarDataset, show: int = 3, verify: bool = False, display_params: int = 0) -> None:
    N = len(ds)
    print(f"Items: {N}")
    if N == 0:
        return

    # Basic distribution stats (Cb, D)
    cb_set = set()
    d_set = set()
    problems: List[Tuple[int, str]] = []

    # Run a lightweight scan (shapes only)
    for i, ecdc in enumerate(ds._paths):
        try:
            ckpt = torch.load(ecdc, map_location="cpu")
            codes = _canonicalize_codes(torch.as_tensor(ckpt["audio_codes"]))
            T, Cb = codes.shape
            cpath = ecdc.with_suffix(".cond.npy")
            cond = np.load(cpath, mmap_mode="r")
            if cond.ndim != 2 or cond.shape[0] != T:
                problems.append((i, f"cond shape {cond.shape} vs T={T}"))
            else:
                cb_set.add(Cb)
                d_set.add(int(cond.shape[1]))
        except Exception as e:
            problems.append((i, f"{type(e).__name__}: {e}"))

    print(f"Distinct codebooks (Cb): {sorted(cb_set) or 'unknown'}")
    print(f"Distinct feature dims (D): {sorted(d_set) or 'unknown'}")
    if problems:
        print(f"Issues detected on {len(problems)} items (showing up to 10):")
        for i, msg in problems[:10]:
            print(f"  - idx {i}: {ds._paths[i].name}: {msg}")

    # Show a few full samples
    to_show = list(range(min(show, N)))
    print("\nSamples:")
    for idx in to_show:
        item = ds[idx]
        codes, cond, names, meta = item["codes"], item["cond"], item["names"], item["meta"]
        T, Cb = codes.shape
        D = cond.shape[1]
        print(f"  [{idx}] {Path(item['path']).name}: codes[T={T}, Cb={Cb}], cond[D={D}], names={names[:6]}")
        if meta.get("fps") is not None and int(meta["fps"]) != 75:
            print(f"      [warn] meta.fps={meta['fps']} (expected 75)")
        if display_params and D > 0:
            _print_param_rows(cond, names, min(display_params, T))

    # Optional verify pass (strict)
    if verify:
        print("\nVerify pass: checking existence and alignment for all items...")
        errors = 0
        for i, ecdc in enumerate(ds._paths):
            try:
                ckpt = torch.load(ecdc, map_location="cpu")
                codes = _canonicalize_codes(torch.as_tensor(ckpt["audio_codes"]))
                T, _ = codes.shape
                cpath = ecdc.with_suffix(".cond.npy")
                jpath = ecdc.with_suffix(".cond.json")
                if not Path(cpath).exists(): raise FileNotFoundError(f"missing {cpath.name}")
                if not Path(jpath).exists(): raise FileNotFoundError(f"missing {jpath.name}")
                cond = np.load(cpath, mmap_mode="r")
                if cond.ndim != 2 or cond.shape[0] != T:
                    raise ValueError(f"shape {cond.shape} vs T={T}")
                # names consistency
                meta = json.loads(Path(jpath).read_text(encoding="utf-8"))
                names = meta.get("names", [])
                if len(names) != int(cond.shape[1]):
                    raise ValueError(f"names length {len(names)} vs D={cond.shape[1]}")
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  [fail] idx {i}, {ecdc.name}: {e}")
        if errors == 0:
            print("Verify: all good.")
        else:
            print(f"Verify: {errors} item(s) failed. (Showing up to 10 above.)")


def main():
    ap = argparse.ArgumentParser(description="Inspect/verify an ESF (Encodec Sidecar Format) dataset.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens-root", type=Path, help="Root folder containing .ecdc/.cond.* files (typically .../tokens)")
    src.add_argument("--hf-ds", type=Path, help="Hugging Face dataset folder created by build_dataset_from_folder.py")
    ap.add_argument("--split", default="train", help="If using --hf-ds, which split to inspect (default: train)")
    ap.add_argument("--recursive", action="store_true", help="Recurse under --tokens-root (default: False)")
    ap.add_argument("--device", default=None, help="Move returned tensors to this device (e.g., cuda:0)")
    ap.add_argument("--dtype", choices=["float32","float16"], default="float32", help="Cond dtype for returned tensors")
    ap.add_argument("--show", type=int, default=3, help="Show this many sample items (default: 3)")
    ap.add_argument("--display-params", type=int, default=0,
                    help="For shown samples, print the first N rows (time steps) of the conditioning matrix")
    ap.add_argument("--indices", type=str, default=None,
                    help="Specific indices to print, e.g. '0,5,10' or '0:10:2'. Overrides --show.")
    ap.add_argument("--verify", action="store_true", help="Run a full integrity check across all items")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else None
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    if args.hf_ds:
        ds = EncodecSidecarDataset.from_hf_dataset(args.hf_ds, split=args.split, device=device, dtype=dtype)
    else:
        ds = EncodecSidecarDataset(tokens_root=args.tokens_root, recursive=args.recursive, device=device, dtype=dtype)

    print(f"Dataset source: {'HF:'+str(args.hf_ds) if args.hf_ds else str(args.tokens_root)}")
    print(f"Device: {device or 'cpu'} | dtype: {dtype}\n")

    # If indices provided, show those; else show first --show
    idxs = _parse_indices(args.indices, len(ds))
    if idxs:
        print(f"Showing indices: {idxs}")
        for i in idxs:
            item = ds[i]
            codes, cond, names, meta = item["codes"], item["cond"], item["names"], item["meta"]
            T, Cb = codes.shape
            D = cond.shape[1]
            print(f"  [{i}] {Path(item['path']).name}: codes[T={T}, Cb={Cb}], cond[D={D}], names={names[:6]}")
            if args.display_params and D > 0:
                _print_param_rows(cond, names, min(args.display_params, T))
    else:
        summarize_dataset(ds, show=args.show, verify=args.verify, display_params=args.display_params)
        # Note: summarize_dataset will handle --display-params via global args captured below.


if __name__ == "__main__":
    main()
