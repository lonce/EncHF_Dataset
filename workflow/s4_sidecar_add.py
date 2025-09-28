#!/usr/bin/env python3
# sidecar_add.py — plugin-driven producer runner for EnCodec sidecars.

from __future__ import annotations
import argparse, importlib, importlib.util, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from sidecar_lib import SIDECAR_JSON_SUFFIX

from sidecar_lib import (
    FPS, atomic_save_json, atomic_save_npy, cond_path_for, ensure_sidecar_skeleton,
    infer_frames_and_codebooks, load_sidecar, merge_features, shard_ok, walk_ecdc,
)

# --------------------- dynamic import helpers ---------------------

def _load_module_from_file(path: Path, name_hint: str):
    spec = importlib.util.spec_from_file_location(name_hint, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load producer from file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _load_producer(spec: str, producers_dir: Path | None):
    p = Path(spec)
    if p.suffix == ".py" and p.exists():
        return _load_module_from_file(p.resolve(), f"producer_{p.stem}")
    if producers_dir is not None:
        candidate = (producers_dir / f"{spec}.py").resolve()
        if candidate.exists():
            return _load_module_from_file(candidate, f"producer_{spec}")
    return importlib.import_module(spec)

def _list_producer_files(producers_dir: Path) -> list[Path]:
    if not producers_dir or not producers_dir.exists():
        return []
    return sorted([p for p in producers_dir.glob("*.py") if p.is_file()], key=lambda p: p.name.lower())

# --------------------- main ---------------------

def main():
    rawfmt = argparse.RawTextHelpFormatter

    DESCRIPTION = """\
Add per-frame features to EnCodec sidecars via plug-in producers.

HOW HELP WORKS
--------------
This CLI is two-stage:
  1) You tell it which producers to load (with --producer and --producers-dir).
  2) It loads those producers so their flags can appear in --help.

To see a producer’s options in --help, include it on the same command:
  sidecar_add.py --producers-dir ./producers --producer const --help

Discoverability:
  --list-producers                 list *.py producer files in --producers-dir
  --show-producer NAME             show detailed help for one producer (NAME = file stem)
"""

    EXAMPLES = """\
Examples
--------
# Append a constant 'scene_id=7' and a ramp 'gain=0->1' in one pass:
sidecar_add.py \\
  --audio-root /data_ecdc --cond-root /data_params \\
  --producers-dir ./producers \\
  --producer const --producer lin \\
  --const scene_id=7 --lin gain=0:1 \\
  --mode append --create-missing

# Parse constants from filenames like ..._tempo_120.0.ecdc:
sidecar_add.py \\
  --audio-root /data_ecdc --cond-root /data_params \\
  --producers-dir ./producers --producer from_filename \\
  --from-filename --filename-regex '_(?P<name>[A-Za-z]\\w*)_(?P<value>-?\\d+(?:\\.\\d+)?)'

# Decode EnCodec and compute per-frame RMS dB (dummy example):
sidecar_add.py \\
  --audio-root /data_ecdc --cond-root /data_params \\
  --producers-dir ./producers --producer from_audio \\
  --from-audio --hf-model facebook/encodec_24khz --device cuda:0 \\
  --audio-feats rms_db --mode append --create-missing
"""

    # Phase 1: lightweight parser to discover producers & help flags (NO required args here)
    base = argparse.ArgumentParser(add_help=False, formatter_class=rawfmt)
    base.add_argument("--audio-root", type=Path, help="Root containing .ecdc files")
    base.add_argument("--cond-root",  type=Path, help="Root where sidecars live (mirrors audio tree)")
    base.add_argument("--suffix",     default=".cond.npy", help="Sidecar data suffix (default: .cond.npy)")
    base.add_argument("--dtype",      choices=["float16","float32"], default="float16", help="Sidecar dtype on write")
    base.add_argument("--mode",       choices=["append","overwrite"], default="append",
                      help="If a feature already exists: append=keep existing column; overwrite=replace values")
    base.add_argument("--create-missing", action="store_true", help="Create empty .cond.npy/.json if missing")
    base.add_argument("--producer",   action="append", default=[],
                      help="Producer spec: file path, bare name (used with --producers-dir), or dotted module. Repeatable.")
    base.add_argument("--producers-dir", type=Path, default=None,
                      help="Directory containing producer .py files (used for bare --producer names).")
    base.add_argument("--list-producers", action="store_true",
                      help="List available producer files in --producers-dir and exit")
    base.add_argument("--show-producer", action="append", default=[],
                      help="Show detailed help for a specific producer name/file-stem (repeatable) and exit")
    base.add_argument("--shard-index", type=int, default=None, help="Shard index (0-based)")
    base.add_argument("--shard-count", type=int, default=None, help="Total number of shards")
    base.add_argument("--skip-errors", action="store_true", help="Log and continue on errors instead of aborting")

    prelim, _unknown = base.parse_known_args()

    # Producer discovery helpers (exit early)
    if prelim.list_producers:
        if prelim.producers_dir is None:
            print("--list-producers requires --producers-dir", file=sys.stderr)
            sys.exit(2)
        files = _list_producer_files(prelim.producers_dir)
        if not files:
            print(f"(no producers found in {prelim.producers_dir})")
        else:
            print("Producers in", prelim.producers_dir)
            for p in files:
                print("  -", p.stem, f"({p.name})")
        sys.exit(0)

    if prelim.show_producer:
        if prelim.producers_dir is None:
            print("--show-producer requires --producers-dir (for bare names)", file=sys.stderr)
            sys.exit(2)
        for name in prelim.show_producer:
            try:
                mod = _load_producer(name, prelim.producers_dir)
                tmp = argparse.ArgumentParser(
                    prog=f"producer:{name}",
                    formatter_class=rawfmt,
                    description=(mod.__doc__.strip() if getattr(mod, "__doc__", None) else "Producer"),
                )
                if callable(getattr(mod, "add_cli_args", None)):
                    mod.add_cli_args(tmp)
                tmp.print_help()
                print()
            except Exception as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)
                sys.exit(2)
        sys.exit(0)

    # Load the producers the user asked for (so their args appear in --help)
    mods = []
    for spec in prelim.producer:
        try:
            mods.append(_load_producer(spec, prelim.producers_dir))
        except Exception as e:
            print(f"[FATAL] Could not load producer '{spec}': {e}", file=sys.stderr)
            sys.exit(2)

    # Build the full parser with description + examples; keep args optional for --help to work
    ap = argparse.ArgumentParser(
        "Add per-frame features to EnCodec sidecars",
        parents=[base],
        formatter_class=rawfmt,
        description=DESCRIPTION,
        epilog=EXAMPLES,
    )
    for m in mods:
        add_cli = getattr(m, "add_cli_args", None)
        if callable(add_cli):
            add_cli(ap)

    args = ap.parse_args()

    # If user asked for help, argparse has already printed it and exited.
    # Now enforce required args ONLY if not asking for help.
    if "-h" not in sys.argv and "--help" not in sys.argv:
        if args.audio_root is None or args.cond_root is None:
            ap.error("--audio-root and --cond-root are required")

    if not mods:
        print("No producers selected (use --producer and optionally --producers-dir). Nothing to do.", file=sys.stderr)
        sys.exit(0)

    # Optional producer init
    for m in mods:
        init = getattr(m, "init", None)
        if callable(init):
            init(args)

    # ------------ main loop ------------
    ecdc_paths = sorted(walk_ecdc(args.audio_root), key=lambda p: p.as_posix().lower())
    changed = created = skipped = errors = 0

    for ecdc in tqdm(ecdc_paths, desc="Add features", unit="file"):
        rel = ecdc.relative_to(args.audio_root).as_posix()
        if not shard_ok(rel, args.shard_index, args.shard_count):
            continue
        try:
            T, _ = infer_frames_and_codebooks(ecdc)
            cpath = cond_path_for(args.audio_root, args.cond_root, ecdc, args.suffix)

            if not cpath.exists():
                if not args.create_missing:
                    skipped += 1
                    continue
                ensure_sidecar_skeleton(ecdc, args.audio_root, args.cond_root, args.suffix, args.dtype, overwrite=False)
                created += 1

            arr, meta, names = load_sidecar(cpath)
            if arr.shape[0] != T:
                raise ValueError(f"Frame mismatch: sidecar {arr.shape[0]} vs codes {T} at {cpath}")

            feats: Dict[str, np.ndarray] = {}
            for m in mods:
                produce = getattr(m, "produce", None)
                if not callable(produce):
                    raise TypeError(f"Producer {m.__name__} has no 'produce' function")
                new_feats = produce(ecdc, T, args=args) or {}
                for k, v in new_feats.items():
                    v = np.asarray(v, dtype=np.float32)
                    if v.ndim != 1 or v.shape[0] != T:
                        raise ValueError(
                            f"Producer '{m.__name__}' returned invalid shape for '{k}': {v.shape}, expected (T,)={T}"
                        )
                    feats[k] = v  # last-one-wins on collisions

            if not feats:
                continue

            arr2, names2, meta2 = merge_features(
                base_arr=arr,
                base_names=names,
                meta=meta,
                new_features=feats,
                mode=args.mode,
                out_dtype=args.dtype,
            )
            meta2["fps"] = FPS
            meta2.setdefault("schema_version", 1)
            meta2.setdefault("source_rate", FPS)
            meta2["names"] = names2

            atomic_save_npy(cpath, arr2)
            atomic_save_json(SIDECAR_JSON_SUFFIX, meta2)
            changed += 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors += 1
            if args.skip_errors:
                print(f"[WARN] {rel}: {e}", file=sys.stderr)
            else:
                raise

    for m in mods:
        shutdown = getattr(m, "shutdown", None)
        if callable(shutdown):
            shutdown(args)

    print(f"Done. changed={changed}, created={created}, skipped={skipped}, errors={errors}")

if __name__ == "__main__":
    main()
