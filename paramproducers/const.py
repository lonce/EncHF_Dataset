# producers/const.py
from __future__ import annotations
import argparse
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

def add_cli_args(ap: argparse.ArgumentParser):
    ap.add_argument("--const", action="append", default=[],
                    help="Constant per-frame feature(s): name=value (repeatable)")

def _parse_const(items: List[str]) -> List[Tuple[str, float]]:
    out=[]
    for it in items:
        if "=" not in it:
            raise ValueError(f"--const expects name=value, got '{it}'")
        k, v = it.split("=", 1)
        out.append((k.strip(), float(v)))
    return out

def produce(ecdc_path: Path, T: int, *, args) -> Dict[str, np.ndarray]:
    consts = _parse_const(args.const or [])
    feats: Dict[str, np.ndarray] = {}
    for name, val in consts:
        feats[name] = np.full((T,), float(val), dtype=np.float32)
    return feats
