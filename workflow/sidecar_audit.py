#!/usr/bin/env python3
# sidecar_audit.py
# Audit EnCodec sidecars for presence, alignment, schema, and optional data-range checks.

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from sidecar_lib import (
    walk_ecdc,
    cond_path_for,
    load_sidecar,
    infer_frames_and_codebooks,
    shard_ok,
    FPS,
)

Issue = Tuple[str, str, str, str]  # (type, rel_ecdc, rel_cond, detail)


def audit_one(
    ecdc: Path,
    audio_root: Path,
    cond_root: Optional[Path],
    suffix: str,
    expect_fps: Optional[int],
    expect_dtype: Optional[str],
    require: List[str],
    verify_ranges: bool,
    range_tol: float,
) -> List[Issue]:
    """Return a list of issues for a single .ecdc."""
    issues: List[Issue] = []

    rel_ecdc = ecdc.relative_to(audio_root).as_posix()
    cpath = cond_path_for(audio_root, cond_root, ecdc, suffix)
    rel_cond = (
        cpath.relative_to(cond_root).as_posix()
        if cond_root
        else cpath.relative_to(ecdc.parent).as_posix()
    )

    # Read EnCodec meta
    try:
        T_codes, _ = infer_frames_and_codebooks(ecdc)
    except Exception as e:
        issues.append(("bad_ecdc", rel_ecdc, rel_cond, f"unreadable .ecdc: {e}"))
        return issues

    # Presence
    if not cpath.exists():
        issues.append(("missing_sidecar", rel_ecdc, rel_cond, "no sidecar"))
        return issues

    # Load sidecar
    try:
        arr, meta, names = load_sidecar(cpath)
    except Exception as e:
        issues.append(("bad_sidecar", rel_ecdc, rel_cond, f"unreadable sidecar: {e}"))
        return issues

    # Shape
    if arr.ndim != 2:
        issues.append(("bad_shape", rel_ecdc, rel_cond, f"expected 2D [T,D], got {arr.shape}"))
        return issues
    T_cond, D = int(arr.shape[0]), int(arr.shape[1])
    if T_cond != T_codes:
        issues.append(("frame_mismatch", rel_ecdc, rel_cond, f"codes={T_codes}, cond={T_cond}"))

    # Optional dtype check
    if expect_dtype:
        want = {"float16": np.float16, "float32": np.float32}[expect_dtype]
        if arr.dtype != want:
            issues.append(("dtype_mismatch", rel_ecdc, rel_cond, f"cond dtype={arr.dtype}, expected={want}"))

    # FPS check (metadata)
    if expect_fps is not None:
        got_fps = meta.get("fps")
        if got_fps is not None and int(got_fps) != int(expect_fps):
            issues.append(("fps_mismatch", rel_ecdc, rel_cond, f"meta.fps={got_fps}, expected={expect_fps}"))

    # Names / schema checks
    if names:
        if len(set(names)) != len(names):
            issues.append(("duplicate_names", rel_ecdc, rel_cond, "sidecar names contain duplicates"))
        if len(names) != D:
            issues.append(("names_length_mismatch", rel_ecdc, rel_cond, f"len(names)={len(names)} vs D={D}"))
        # required features
        missing = [n for n in require if n not in names]
        if missing:
            issues.append(("missing_features", rel_ecdc, rel_cond, f"missing: {missing}"))
    else:
        if require:
            issues.append(("missing_names", rel_ecdc, rel_cond, "sidecar has no 'names' in JSON"))

    # norm arrays length
    norm = meta.get("norm", {}) if isinstance(meta.get("norm", {}), dict) else {}
    for k in ("min", "max", "mean", "std"):
        v = norm.get(k, [])
        if not isinstance(v, list):
            issues.append(("norm_invalid", rel_ecdc, rel_cond, f"norm.{k} is not list"))
        elif D and len(v) not in (0, D):
            issues.append(("norm_length_mismatch", rel_ecdc, rel_cond, f"norm.{k} len={len(v)} vs D={D}"))

    # Optional value-in-range verification (memmap-friendly)
    if verify_ranges and D > 0:
        mins = norm.get("min", [])
        maxs = norm.get("max", [])
        if isinstance(mins, list) and isinstance(maxs, list) and len(mins) == len(maxs) == D:
            with np.errstate(all="ignore"):
                data_min = np.nanmin(arr, axis=0)
                data_max = np.nanmax(arr, axis=0)
            for j in range(D):
                vmin = float(mins[j])
                vmax = float(maxs[j])
                if vmax < vmin:
                    issues.append(("norm_inverted", rel_ecdc, rel_cond, f"col {j} max<min ({vmax}<{vmin})"))
                    continue
                if data_min[j] + range_tol < vmin:
                    issues.append((
                        "range_below_min",
                        rel_ecdc,
                        rel_cond,
                        f"col {j}: data_min={data_min[j]:.6g} < min={vmin:.6g}",
                    ))
                if data_max[j] - range_tol > vmax:
                    issues.append((
                        "range_above_max",
                        rel_ecdc,
                        rel_cond,
                        f"col {j}: data_max={data_max[j]:.6g} > max={vmax:.6g}",
                    ))
        else:
            issues.append(("range_unverifiable", rel_ecdc, rel_cond, "norm.min/max unavailable or wrong length"))

    return issues


def main():
    rawfmt = argparse.RawTextHelpFormatter

    ap = argparse.ArgumentParser(
        "Audit EnCodec sidecars (.cond.npy/.json) against .ecdc files",
        formatter_class=rawfmt,
    )
    ap.add_argument("--audio-root", type=Path, required=True, help="Root containing .ecdc files")
    ap.add_argument("--cond-root", type=Path, default=None, help="Sidecar root (mirrors audio tree); if omitted, co-located")
    ap.add_argument("--suffix", default=".cond.npy", help="Sidecar data suffix (default: .cond.npy)")
    ap.add_argument("--expect-fps", type=int, default=FPS, help=f"Expected sidecar fps in metadata (default: {FPS})")
    ap.add_argument("--expect-dtype", choices=["float16", "float32"], default=None, help="Check sidecar dtype")
    ap.add_argument("--require", default="", help="Comma-separated list of required feature names")
    ap.add_argument("--verify-ranges", action="store_true", help="Verify data lies within norm.min/max per column")
    ap.add_argument("--range-tol", type=float, default=1e-6, help="Tolerance for range checks")
    ap.add_argument("--list", default="50", help="Print up to N issues; use 'all' or -1 for unlimited, 0 for none")
    ap.add_argument("--export-csv", type=Path, default=None, help="Write full issue list as CSV (type,rel_ecdc,rel_cond,detail)")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument(
        "--fail-on",
        default="bad_ecdc,bad_sidecar,frame_mismatch,missing_sidecar",
        help="Comma-separated issue types that should cause a nonzero exit code",
    )

    args = ap.parse_args()

    require = [s.strip() for s in args.require.split(",") if s.strip()] if args.require else []
    fail_on = {s.strip() for s in args.fail_on.split(",") if s.strip()}

    ecdc_paths = sorted(walk_ecdc(args.audio_root), key=lambda p: p.as_posix().lower())

    issues: List[Issue] = []
    counts: Dict[str, int] = {}
    total = 0

    for ecdc in tqdm(ecdc_paths, desc="Auditing", unit="file"):
        rel = ecdc.relative_to(args.audio_root).as_posix()
        if not shard_ok(rel, args.shard_index, args.shard_count):
            continue
        total += 1
        try:
            its = audit_one(
                ecdc=ecdc,
                audio_root=args.audio_root,
                cond_root=args.cond_root,
                suffix=args.suffix,
                expect_fps=args.expect_fps,
                expect_dtype=args.expect_dtype,
                require=require,
                verify_ranges=bool(args.verify_ranges),
                range_tol=float(args.range_tol),
            )
            issues.extend(its)
            for t, *_ in its:
                counts[t] = counts.get(t, 0) + 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            issues.append(("auditor_error", rel, "", str(e)))
            counts["auditor_error"] = counts.get("auditor_error", 0) + 1

    # Summary
    ok = total - sum(counts.get(k, 0) for k in counts if k not in {"auditor_error"})
    print("=== SIDECAR AUDIT SUMMARY ===")
    print(f"Audio root: {args.audio_root}")
    print(f"Cond root : {args.cond_root or '(co-located)'}")
    print(f"Scanned   : {total} .ecdc files")
    if counts:
        print("Issues    :")
        for k in sorted(counts.keys()):
            print(f"  - {k}: {counts[k]}")
    print(f"OK files  : ~{ok}")

    # Limit for printing issues
    limit_arg = str(args.list).lower()
    if limit_arg in ("all", "-1"):
        limit = -1
    else:
        try:
            limit = int(args.list)
        except Exception:
            limit = 50

    if limit != 0 and issues:
        print("\n-- Sample issues --")
        to_show = issues if limit < 0 else issues[:limit]
        for t, re, rc, d in to_show:
            print(f"[{t}] {re} | {rc} :: {d}")

    # Export CSV
    if args.export_csv:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.export_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "rel_ecdc", "rel_cond", "detail"])
            w.writerows(issues)
        print(f"\nWrote CSV: {args.export_csv}")

    # Exit code
    should_fail = any(t in fail_on for (t, *_rest) in issues)
    sys.exit(1 if should_fail else 0)


if __name__ == "__main__":
    main()
