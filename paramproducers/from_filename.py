# producers/from_filename.py
from __future__ import annotations
import argparse, re
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

def add_cli_args(ap: argparse.ArgumentParser):
    ap.add_argument("--from-filename", action="store_true",
                    help="Enable filename parsing producer")
    ap.add_argument("--filename-regex",
                    default=r"_(?P<name>[A-Za-z]\w*)_(?P<value>-?\d+(?:\.\d+)?)",
                    help="Regex with (?P<name>...) and (?P<value>...) groups")

def produce(ecdc_path: Path, T: int, *, args) -> Dict[str, np.ndarray]:
    if not getattr(args, "from_filename", False):
        return {}
    pattern = re.compile(args.filename_regex)
    matches = list(pattern.finditer(ecdc_path.name))
    feats: Dict[str, np.ndarray] = {}
    for m in matches:
        name = m.group("name")
        val  = float(m.group("value"))
        feats[name] = np.full((T,), val, dtype=np.float32)
    return feats
