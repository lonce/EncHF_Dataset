#!/usr/bin/env python3
# build_dataset_from_folder.py
#
# Build a Hugging Face Dataset directly from a folder tree of .ecdc files.
# Provides the same ergonomics you liked:
#   --split, --recursive, --tokens-subdir, --materialize {link,copy,none}
#
# Notes:
# - With materialize=link (recommended), we symlink files into OUT_DIR/TOKENS_SUBDIR,
#   and store *relative* paths in the 'audio' column → portable dataset folder.
# - With materialize=copy, we copy files instead of linking.
# - With materialize=none, we still write relative paths into the dataset pointing
#   inside OUT_DIR/TOKENS_SUBDIR, but we do NOT create files there. Use --verify
#   to see missing paths (this mode is mostly for dry runs).
#
# Usage:
#   python build_dataset_from_folder.py /data_ecdc /hf/my_ds --split train \
#       --recursive --tokens-subdir tokens --materialize link --verify

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from datasets import Dataset, DatasetDict
import pandas as pd


def collect_tokens(tokens_root: Path, suffix: str, recursive: bool) -> List[Path]:
    """Return a sorted list of token file paths."""
    patt = f"*{suffix}"
    if recursive:
        paths = sorted(tokens_root.rglob(patt), key=lambda p: p.as_posix().lower())
    else:
        paths = sorted(tokens_root.glob(patt), key=lambda p: p.as_posix().lower())
    return [p for p in paths if p.is_file()]


def materialize_token(src: Path, dst: Path, how: str):
    """Create a link/copy at dst pointing to src, or do nothing if how == 'none'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if how == "none":
        return
    if dst.exists():
        return
    if how == "link":
        target = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        os.symlink(target, dst)
    elif how == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown materialize mode: {how}")


def verify_paths(df: pd.DataFrame, out_dir: Path) -> int:
    """Check that every 'audio' relative path exists under out_dir."""
    missing = []
    for _, row in df.iterrows():
        apath = out_dir / row["audio"]
        if not apath.exists():
            missing.append(str(apath))
    if missing:
        print(f"Verify: {len(missing)} missing audio files (showing up to 20):")
        for m in missing[:20]:
            print("  MISSING:", m)
    else:
        print("Verify: all audio paths exist.")
    return len(missing)

def materialize_siblings(src_ecdc: Path, dst_ecdc: Path, how: str):
    siblings = [
        src_ecdc.with_suffix(".cond.npy"),
        src_ecdc.with_suffix(".cond.json"),
    ]
    for s in siblings:
        if not s.exists():
            continue
        out = dst_ecdc.parent / s.name
        out.parent.mkdir(parents=True, exist_ok=True)
        if how == "none" or out.exists():
            continue
        if how == "link":
            target = os.path.relpath(s.resolve(), start=out.parent.resolve())
            os.symlink(target, out)
        elif how == "copy":
            shutil.copy2(s, out)
        else:
            raise ValueError(how)


def main():
    ap = argparse.ArgumentParser(
        description="Build a Hugging Face dataset from a folder of Encodec token files (.ecdc)."
    )
    ap.add_argument("tokens_dir", type=Path, help="Folder containing token files (.ecdc)")
    ap.add_argument("out_dir", type=Path, help="Output folder for datasets.save_to_disk")
    ap.add_argument("--split", default="train", help="Split name (default: train)")
    ap.add_argument("--suffix", default=".ecdc", help="Token file extension (default: .ecdc)")
    ap.add_argument("--recursive", action="store_true", help="Search tokens_dir recursively (recommended)")
    ap.add_argument(
        "--tokens-subdir",
        default="tokens",
        help="Subdirectory inside out_dir where links/copies live (default: tokens)",
    )
    ap.add_argument(
        "--materialize",
        choices=["link", "copy", "none"],
        default="link",
        help="How to place token files inside dataset dir (default: link)",
    )
    ap.add_argument("--verify", action="store_true", help="After saving, check that all audio paths exist")
    ap.add_argument("--max-files", type=int, default=None, help="(Optional) limit number of files (debug)")
    args = ap.parse_args()

    if not args.tokens_dir.exists() or not args.tokens_dir.is_dir():
        raise SystemExit(f"Tokens directory not found: {args.tokens_dir}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_tokens_root = out_dir / args.tokens_subdir

    # 1) Find all token files
    token_paths = collect_tokens(args.tokens_dir, args.suffix, args.recursive)
    if args.max_files:
        token_paths = token_paths[: args.max_files]
    if not token_paths:
        raise SystemExit("No token files found.")

    # 2) Materialize and build rows
    rows = []
    for src in token_paths:
        rel_inside_dataset = Path(args.tokens_subdir) / src.relative_to(args.tokens_dir)
        materialize_token(src, out_dir / rel_inside_dataset, args.materialize)
        rows.append({"audio": str(rel_inside_dataset)})

        materialize_token(src, out_dir / rel_inside_dataset, args.materialize)
        materialize_siblings(src, out_dir / rel_inside_dataset, args.materialize)

    # 3) Build dataset (single split)
    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df, preserve_index=False)
    dsd = DatasetDict({args.split: ds})
    dsd.save_to_disk(str(out_dir))

    print(f"Saved DatasetDict with split '{args.split}' to: {out_dir}")
    print(f"Rows: {len(ds)} | Columns: {ds.column_names}")
    print(
        f"Token materialization: {args.materialize} -> "
        f"{dst_tokens_root if args.materialize!='none' else '(external files; none created)'}"
    )

    if args.verify:
        verify_paths(df, out_dir)


if __name__ == "__main__":
    main()
