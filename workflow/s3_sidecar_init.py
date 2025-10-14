#!/usr/bin/env python3
# sidecar_init.py
# Create empty, aligned sidecar skeletons for every .ecdc file (no features).


import argparse, csv, sys
from pathlib import Path
from tqdm import tqdm
from .sidecar_lib import (
    walk_ecdc, cond_path_for, ensure_sidecar_skeleton,
    infer_frames_and_codebooks, shard_ok
)

def main():
    ap = argparse.ArgumentParser("Initialize empty sidecar skeletons for an EnCodec .ecdc tree")
    ap.add_argument("--audio-root", type=Path, required=True, help="Root containing .ecdc files")
    ap.add_argument("--cond-root",  type=Path, default=None, help="Where to write sidecars. If omitted, co-locate next to .ecdc")
    ap.add_argument("--suffix",     default=".cond.npy", help="Sidecar data suffix (default: .cond.npy)")
    ap.add_argument("--dtype",      choices=["float16","float32"], default="float16")
    ap.add_argument("--overwrite",  action="store_true", help="Overwrite existing sidecars (re-create as empty)")
    ap.add_argument("--manifest-csv", type=Path, default=None, help="Optional CSV manifest to write (rel paths)")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument("--skip-errors", action="store_true")
    args = ap.parse_args()

    ecdc_paths = sorted(walk_ecdc(args.audio_root), key=lambda p: p.as_posix().lower())
    rows = []
    errors = 0

    for ecdc in tqdm(ecdc_paths, desc="Init skeletons", unit="file"):
        rel = ecdc.relative_to(args.audio_root).as_posix()
        if not shard_ok(rel, args.shard_index, args.shard_count):
            continue
        try:
            # ensure skeleton exists and aligned
            cond_path, T = ensure_sidecar_skeleton(
                ecdc_path=ecdc,
                audio_root=args.audio_root,
                cond_root=args.cond_root,
                suffix=args.suffix,
                dtype=args.dtype,
                overwrite=args.overwrite,
            )
            _, Cb = infer_frames_and_codebooks(ecdc)
            rel_cond = cond_path.relative_to(args.cond_root or args.audio_root).as_posix()
            rows.append([rel, rel_cond, T, 0, 75, Cb])

        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors += 1
            if args.skip_errors:
                print(f"[WARN] {rel}: {e}", file=sys.stderr)
            else:
                raise

    if args.manifest_csv:
        args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rel_ecdc","rel_cond","frames","D","fps","codebooks"])
            w.writerows(rows)

    print(f"Done. Sidecars initialized: {len(rows)}  Errors: {errors}")

if __name__ == "__main__":
    main()
