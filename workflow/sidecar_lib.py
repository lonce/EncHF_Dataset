#!/usr/bin/env python3
# sidecar_lib.py
# Shared utilities for EnCodec sidecar files (creation, reading, merging, atomic writes)

from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import torch

FPS = 75  # EnCodec frames per second
SCHEMA_VERSION = 1

SIDECAR_JSON_SUFFIX = ".cond.json"

# ---------------- I/O: atomic writes ----------------

def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def atomic_save_json(path: Path, obj: dict):
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False))

def atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure tmp also ends with .npy so numpy.save doesn't append another .npy
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)

# ---------------- Scanning & path helpers ----------------

def walk_ecdc(root: Path) -> Iterable[Path]:
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith(".ecdc"):
                p = Path(dp) / f
                if p.is_file():
                    yield p

def cond_path_for(audio_root: Path, cond_root: Optional[Path], ecdc_path: Path, suffix: str = ".cond.npy") -> Path:
    """Mirror relative path of ecdc under cond_root (or co-locate if cond_root is None)."""
    if cond_root is None:
        return ecdc_path.with_suffix(suffix)
    rel = ecdc_path.relative_to(audio_root)
    return (cond_root / rel).with_suffix(suffix)

def shard_ok(rel: str, shard_index: Optional[int], shard_count: Optional[int]) -> bool:
    if shard_index is None or shard_count is None:
        return True
    h = int(hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8], 16)
    return (h % shard_count) == shard_index

# ---------------- EnCodec metadata ----------------

def infer_frames_and_codebooks(ecdc_path: Path) -> Tuple[int, int]:
    """Return (T, Cb) from an .ecdc file."""
    ckpt = torch.load(ecdc_path, map_location="cpu")
    codes = ckpt["audio_codes"]
    if codes.ndim == 4:      # [1,1,Cb,T]
        codes = codes.squeeze(1)
    if codes.ndim == 2:      # [Cb,T]
        codes = codes[None, ...]
    T = int(codes.shape[-1])
    Cb = int(codes.shape[1])
    return T, Cb

# ---------------- Sidecar skeletons & metadata ----------------

def empty_meta() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "names": [],
        "source_rate": FPS,
        "norm": {"mean": [], "std": [], "min": [], "max": []},
    }

def ensure_sidecar_skeleton(
    ecdc_path: Path,
    audio_root: Path,
    cond_root: Optional[Path],
    suffix: str = ".cond.npy",
    dtype: str = "float16",
    overwrite: bool = False,
) -> Tuple[Path, int]:
    """
    Create an aligned, empty sidecar for an .ecdc file.
      - NPY shape: [T, 0], dtype
      - JSON metadata with empty name/stats arrays, fps/schema_version
    Returns: (cond_path, T)
    """
    np_dtype = np.float16 if dtype == "float16" else np.float32
    cond_path = cond_path_for(audio_root, cond_root, ecdc_path, suffix)
    json_path = cond_path.with_suffix(SIDECAR_JSON_SUFFIX)

    T, _ = infer_frames_and_codebooks(ecdc_path)
    if cond_path.exists() and not overwrite:
        # Keep existing; ensure it has correct first dimension
        arr = np.load(cond_path, mmap_mode="r")
        if arr.shape[0] != T:
            raise ValueError(f"Existing sidecar has T={arr.shape[0]} but codes have T={T}: {cond_path}")
        return cond_path, T

    # Fresh skeleton
    arr = np.empty((T, 0), dtype=np_dtype)
    meta = empty_meta()
    atomic_save_npy(cond_path, arr)
    atomic_save_json(json_path, meta)
    return cond_path, T

# ---------------- Load/merge/write ----------------

def load_sidecar(cond_path: Path) -> Tuple[np.ndarray, dict, List[str]]:
    arr = np.load(cond_path, mmap_mode="r")
    jpath = cond_path.with_suffix(SIDECAR_JSON_SUFFIX)
    meta = {}
    if jpath.exists():
        try:
            meta = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    names = [str(x) for x in meta.get("names", [])] if isinstance(meta.get("names", []), (list, tuple)) else []
    if arr.ndim != 2:
        raise ValueError(f"Sidecar must be 2D [T,D], got {arr.shape} at {cond_path}")
    if names and len(names) != arr.shape[1]:
        # Attempt to repair names length
        names = names[: arr.shape[1]]
    return arr, meta, names

def _ensure_norm_arrays(meta: dict, num_cols: int):
    meta.setdefault("norm", {"mean": [], "std": [], "min": [], "max": []})
    for k in ("mean", "std", "min", "max"):
        v = meta["norm"].get(k, [])
        if not isinstance(v, list):
            v = []
        if len(v) < num_cols:
            v = v + [0.0] * (num_cols - len(v))
        meta["norm"][k] = v

def _update_col_stats(meta: dict, col_idx: int, vec: np.ndarray):
    _ensure_norm_arrays(meta, col_idx + 1)
    meta["norm"]["min"][col_idx]  = float(vec.min()) if vec.size else 0.0
    meta["norm"]["max"][col_idx]  = float(vec.max()) if vec.size else 0.0
    meta["norm"]["mean"][col_idx] = float(vec.mean()) if vec.size else 0.0
    meta["norm"]["std"][col_idx]  = float(vec.std()) if vec.size else 0.0

def _append_col_stats(meta: dict, appended_stats: List[Tuple[float, float, float, float]]):
    """
    Append many (min,max,mean,std) entries to tail of norm arrays.
    """
    n_old = len(meta.get("norm", {}).get("min", []))
    for k in ("mean", "std", "min", "max"):
        meta["norm"].setdefault(k, [])
    for (mi, mx, me, sd) in appended_stats:
        meta["norm"]["min"].append(mi)
        meta["norm"]["max"].append(mx)
        meta["norm"]["mean"].append(me)
        meta["norm"]["std"].append(sd)
    # Ensure all arrays equal length
    n_new = len(meta["norm"]["min"])
    for k in ("mean", "std", "min", "max"):
        if len(meta["norm"][k]) != n_new:
            meta["norm"][k] = (meta["norm"][k] + [0.0] * (n_new - len(meta["norm"][k])))[:n_new]

def merge_features(
    base_arr: np.ndarray,
    base_names: List[str],
    meta: dict,
    new_features: Dict[str, np.ndarray],
    mode: str = "append",  # or "overwrite"
    out_dtype: str = "float16",
) -> Tuple[np.ndarray, List[str], dict]:
    """
    Merge new per-frame columns into existing sidecar.
      - base_arr: [T, D0], new_features: dict name -> [T]
      - 'append': skip names that already exist; append only new
      - 'overwrite': replace existing columns in-place; append new ones
      - returns updated (arr, names, meta) with stats updated only for changed columns
    """
    assert base_arr.ndim == 2, f"expected 2D array, got {base_arr.shape}"
    T, D0 = base_arr.shape
    names = list(base_names)
    arr = base_arr  # may be replaced with a copy
    appended_stats: List[Tuple[float, float, float, float]] = []

    np_dtype = np.float16 if out_dtype == "float16" else np.float32

    # keep core metadata current
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta["fps"] = meta.get("fps", FPS)
    meta.setdefault("source_rate", FPS)
    meta.setdefault("names", names)
    _ensure_norm_arrays(meta, D0)

    def as_col(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.ndim != 1 or v.shape[0] != T:
            raise ValueError(f"Feature column must be shape [T], got {v.shape}, expected T={T}")
        return v.reshape(T, 1).astype(np_dtype, copy=False)

    if mode not in ("append", "overwrite"):
        raise ValueError("mode must be 'append' or 'overwrite'")

    if mode == "append":
        for name, vec in new_features.items():
            if name in names:
                # skip existing
                continue
            col = as_col(vec)
            if arr.size:
                arr = np.concatenate([np.asarray(arr, dtype=np_dtype, copy=False), col], axis=1)
            else:
                arr = col
            names.append(name)
            # stats for appended column
            v32 = np.asarray(vec, dtype=np.float32)
            appended_stats.append((float(v32.min()), float(v32.max()), float(v32.mean()), float(v32.std())))
        # append stats to meta
        _append_col_stats(meta, appended_stats)

    else:  # overwrite
        # make sure we can write
        if arr.size:
            arr = np.array(arr, dtype=np_dtype, copy=True)
        for name, vec in new_features.items():
            col = as_col(vec)
            if name in names:
                idx = names.index(name)
                arr[:, idx] = col[:, 0]
                _update_col_stats(meta, idx, np.asarray(vec, dtype=np.float32))
            else:
                if arr.size:
                    arr = np.concatenate([arr, col], axis=1)
                else:
                    arr = col
                names.append(name)
                v32 = np.asarray(vec, dtype=np.float32)
                appended_stats.append((float(v32.min()), float(v32.max()), float(v32.mean()), float(v32.std())))
        # add per-appended stats
        if appended_stats:
            _append_col_stats(meta, appended_stats)

    meta["names"] = names
    return arr, names, meta
