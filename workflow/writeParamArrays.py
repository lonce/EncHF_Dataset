from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Dict, Optional, Union, Tuple

FeatureValue = Union[np.ndarray, Dict[str, object]]

def write_sidecar_features(
    *,
    root: Path,
    ecdc_rel: Path,                  # path to .ecdc RELATIVE to root
    features: Dict[str, FeatureValue],
    dtype: str = "float16",          # on-disk dtype for .cond.npy
    fps: Optional[float] = None,     # if provided, set fps/source_rate
    schema_version: int = 2,         # nested 'features' JSON
    mode: str = "overwrite",         # "overwrite" | "append"
    require_ecdc_exists: bool = False,
) -> None:
    """
    Create/merge sidecars for one .ecdc given as a path RELATIVE to 'root'.
    Sidecars are written next to the .ecdc with the same stem:
      <root>/<dir>/<name>.cond.npy and <root>/<dir>/<name>.json
    """
    # --- Resolve paths (and keep everything under root) ---
    ecdc_rel = Path(ecdc_rel)
    if ecdc_rel.is_absolute() or ".." in ecdc_rel.parts:
        raise ValueError("ecdc_rel must be a path under 'root' (no '..', not absolute).")
    ecdc_abs = (root / ecdc_rel).resolve()
    root_abs = root.resolve()
    try:
        ecdc_abs.relative_to(root_abs)
    except Exception:
        raise ValueError("ecdc_rel must resolve under 'root'.")

    if ecdc_abs.suffix != ".ecdc":
        raise ValueError(f"Expected a .ecdc file, got: {ecdc_abs.name}")

    if require_ecdc_exists and not ecdc_abs.exists():
        raise FileNotFoundError(f".ecdc not found: {ecdc_abs}")

    base = ecdc_abs.with_suffix("")            # drop .ecdc
    data_p = base.with_suffix(".cond.npy")
    meta_p = base.with_suffix(".cond.json")
    data_p.parent.mkdir(parents=True, exist_ok=True)

    # --- Normalize features and infer T ---
    def _norm_feature(f: FeatureValue) -> Tuple[np.ndarray, dict]:
        if isinstance(f, dict):
            if "values" not in f:
                raise ValueError("Feature dict must contain 'values'.")
            vals = np.asarray(f["values"], dtype=np.float32).reshape(-1)
            m = {
                "units": str(f.get("units") or ""),
                "doc_string": str(f.get("doc_string") or "")
            }
            for k in ("min", "max", "mean", "std"):
                if f.get(k) is not None:
                    m[k] = float(f[k])
            return vals, m
        vals = np.asarray(f, dtype=np.float32).reshape(-1)
        return vals, {"units": "", "doc_string": ""}

    norm: Dict[str, Tuple[np.ndarray, dict]] = {}
    lengths = set()
    for name, fv in features.items():
        vec, meta_frag = _norm_feature(fv)
        norm[name] = (vec, meta_frag)
        lengths.add(vec.shape[0])
    if not norm:
        return
    if len(lengths) != 1:
        raise ValueError(f"{ecdc_rel}: features have differing lengths: {sorted(lengths)}")
    T = lengths.pop()

    # --- Load existing sidecar or init empty ---
    def _load_or_init(T_required: int):
        if data_p.exists() and meta_p.exists():
            arr = np.load(data_p)
            if arr.ndim != 2:
                raise ValueError(f"{data_p} must be 2D (T,D); got {arr.shape}")
            if arr.shape[0] != T_required:
                raise ValueError(f"Frame mismatch: sidecar T={arr.shape[0]} vs provided T={T_required}")
            with open(meta_p) as f:
                meta = json.load(f)
            # Current order: keys() of features (insertion-ordered JSON)
            if "features" in meta and isinstance(meta["features"], dict):
                names = list(meta["features"].keys())
            else:
                # Legacy migration: names/stats/units/docs → nested features
                names = list(meta.get("names", []))
                feats = {}
                stats = meta.get("stats", {})
                units = meta.get("units", {})
                docs  = meta.get("docs", {})
                for n in names:
                    s = stats.get(n, {})
                    feats[n] = {
                        "min": float(s.get("min", 0.0)),
                        "max": float(s.get("max", 0.0)),
                        "mean": float(s.get("mean", 0.0)),
                        "std": float(s.get("std", 0.0)),
                        "units": str(units.get(n, "")),
                        "doc_string": str(docs.get(n, "")),
                    }
                meta = {
                    "features": feats,
                    "fps": meta.get("fps"),
                    "source_rate": meta.get("source_rate"),
                    "schema_version": schema_version,
                }
            name_to_idx = {n: i for i, n in enumerate(names)}
            return arr, meta, name_to_idx
        # init empty (T,0)
        return np.zeros((T_required, 0), dtype=np.float32), {"features": {}, "schema_version": schema_version}, {}

    arr, meta, name_to_idx = _load_or_init(T)

    # --- Merge columns ---
    cols = [] if arr.size == 0 else [arr[:, i] for i in range(arr.shape[1])]
    for name, (vec, _) in norm.items():
        if name in name_to_idx and mode == "overwrite":
            cols[name_to_idx[name]] = vec
        else:
            name_to_idx.setdefault(name, len(cols))
            cols.append(vec)

    names_ordered = [n for n, _ in sorted(name_to_idx.items(), key=lambda kv: kv[1])]
    new_arr = np.stack(cols, axis=1) if cols else np.zeros((T, 0), dtype=np.float32)
    out_arr = new_arr.astype(np.float16 if dtype == "float16" else np.float32, copy=False)

    # --- Update JSON (nested per-feature) ---
    def _fill_stats(vec: np.ndarray, meta_frag: dict) -> dict:
        v = vec.astype(np.float64, copy=False)
        meta_frag.setdefault("min",  float(np.min(v)) if v.size else 0.0)
        meta_frag.setdefault("max",  float(np.max(v)) if v.size else 0.0)
        meta_frag.setdefault("mean", float(np.mean(v)) if v.size else 0.0)
        meta_frag.setdefault("std",  float(np.std(v, ddof=0)) if v.size else 0.0)
        meta_frag.setdefault("units", "")
        meta_frag.setdefault("doc_string", "")
        return meta_frag

    feats_json = dict(meta.get("features", {}))
    for name in names_ordered:
        idx = name_to_idx[name]
        provided = dict(norm.get(name, (None, {}))[1]) if name in norm else {}
        feats_json[name] = _fill_stats(new_arr[:, idx], provided)

    meta["features"] = feats_json
    if fps is not None:
        meta["fps"] = meta.get("fps") or fps
        meta["source_rate"] = meta.get("source_rate") or fps
    meta["schema_version"] = schema_version

    # --- Write (atomic-ish) ---
    tmp_data = data_p.parent / (data_p.name + ".tmp")
    with open(tmp_data, "wb") as f:
        np.save(f, out_arr, allow_pickle=False)   # no extra ".npy"
    tmp_data.replace(data_p)
    
    tmp_meta = meta_p.parent / (meta_p.name + ".tmp")
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_p)




### USAGE   ###
# root = Path("/datasets/waterfill")
# ecdc_rel = Path("audio/ecdc/II_Double.ecdc")
# T = 12345

# feats = {
#   "gain_db": {
#       "values": np.linspace(-12, 0, T, dtype=np.float32),
#       "units": "dB",
#       "doc_string": "Per-frame gain in decibels"
#   },
#   "scene_id": {
#       "values": np.full(T, 7, np.float32),
#       "min": 7, "max": 7,
#       "doc_string": "Integer scene label"
#   },
#   "energy": np.random.rand(T).astype(np.float32)  # stats auto-computed
# }

# write_sidecar_features_for_rel(
#     root=root,
#     ecdc_rel=ecdc_rel,
#     features=feats,
#     fps=75.0,
#     mode="overwrite",
#     require_ecdc_exists=False,  # set True if you want a safety check
# )
