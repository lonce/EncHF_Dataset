# producers/lin.py
from __future__ import annotations
import argparse
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

def add_cli_args(ap: argparse.ArgumentParser):
    ap.add_argument("--lin", action="append", default=[],
                    help="Linear ramp per-frame feature(s): name=start:end (repeatable)")

def _parse_lin(items: List[str]) -> List[Tuple[str, float, float]]:
    out=[]
    for it in items:
        if "=" not in it or ":" not in it:
            raise ValueError(f"--lin expects name=start:end, got '{it}'")
        k, se = it.split("=", 1)
        s, e = se.split(":", 1)
        out.append((k.strip(), float(s), float(e)))
    return out

def produce(ecdc_path: Path, T: int, *, args) -> Dict[str, np.ndarray]:
    lins = _parse_lin(args.lin or [])
    feats: Dict[str, np.ndarray] = {}
    for name, s, e in lins:
        feats[name] = np.linspace(float(s), float(e), T, dtype=np.float32)
    return feats
