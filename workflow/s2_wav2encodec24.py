#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path
from typing import Iterable, Tuple, List

import torch
import librosa
from transformers import EncodecModel, AutoProcessor
from tqdm import tqdm


def iter_wavs(root: Path, recursive: bool) -> Iterable[Path]:
    patterns = ("*.wav", "*.WAV")
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def iter_token_files(root: Path, suffix: str, recursive: bool) -> Iterable[Path]:
    patterns = (f"*{suffix}",)
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def cpuify(obj):
    if hasattr(obj, "cpu"):
        return obj.cpu()
    if isinstance(obj, (list, tuple)):
        return [cpuify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: cpuify(v) for k, v in obj.items()}
    return obj


def encode_file(wav_path: Path, out_path: Path, model, processor, device, overwrite: bool):
    try:
        if out_path.exists() and not overwrite:
            return True, "exists"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio at 24kHz mono
        audio, _ = librosa.load(str(wav_path), sr=24000, mono=True)

        inputs = processor(
            raw_audio=audio,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            enc = model.encode(inputs["input_values"], inputs.get("padding_mask", None))

        save_data = {
            "audio_codes": cpuify(enc.audio_codes),
            "audio_scales": cpuify(enc.audio_scales),
            "audio_length": int(audio.shape[-1]),
        }

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(save_data, tmp_path)
        os.replace(tmp_path, out_path)

        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def expected_out_path(in_dir: Path, out_dir: Path, wav_path: Path, suffix: str) -> Path:
    rel = wav_path.relative_to(in_dir)  # keep relative structure
    return (out_dir / rel).with_suffix(suffix)


def list_missing_and_orphans(in_dir: Path, out_dir: Path, recursive: bool, suffix: str):
    wavs = set(iter_wavs(in_dir, recursive))
    toks = set(iter_token_files(out_dir, suffix, recursive))

    expected = {expected_out_path(in_dir, out_dir, w, suffix) for w in wavs}
    wavs_missing = sorted(expected - toks)
    orphans = sorted(toks - expected)
    return wavs_missing, orphans


def verify_tokens(out_dir: Path, recursive: bool, suffix: str, max_samples: int):
    toks = list(iter_token_files(out_dir, suffix, recursive))
    if not toks:
        return 0, [(out_dir, "No token files found")]

    sample = toks if len(toks) <= max_samples else random.sample(toks, max_samples)

    errors = []
    checked = 0
    for tp in sample:
        try:
            obj = torch.load(tp, map_location="cpu")
            if not isinstance(obj, dict):
                errors.append((tp, "Not a dict"))
                continue
            for key in ("audio_codes", "audio_scales", "audio_length"):
                if key not in obj:
                    errors.append((tp, f"Missing key {key}"))
            if isinstance(obj.get("audio_codes"), (list, tuple)):
                if len(obj["audio_codes"]) == 0:
                    errors.append((tp, "audio_codes empty"))
            else:
                errors.append((tp, "audio_codes not list/tuple"))
            if not isinstance(obj.get("audio_length"), int) or obj["audio_length"] <= 0:
                errors.append((tp, "audio_length invalid"))
            checked += 1
        except Exception as e:
            errors.append((tp, f"{type(e).__name__}: {e}"))
    return checked, errors


def main():
    ap = argparse.ArgumentParser(
        description="Mirror WAVs into Encodec .ecdc files (24 kHz mono) with verification and reporting."
    )
    ap.add_argument("in_dir", type=Path, help="Input folder with .wav files")
    ap.add_argument("out_dir", type=Path, help="Output folder for .ecdc files")
    ap.add_argument("--bandwidth", type=float, default=6.0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--suffix", default=".ecdc")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-missing", dest="list_missing", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-samples", dest="verify_samples", type=int, default=20)
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    if not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Missing/orphans report
    if args.list_missing:
        wavs_missing, orphans = list_missing_and_orphans(in_dir, out_dir, args.recursive, args.suffix)
        print(f"Missing tokens: {len(wavs_missing)}")
        for p in wavs_missing[:20]:
            print("  MISSING:", p)
        print(f"Orphans: {len(orphans)}")
        for p in orphans[:20]:
            print("  ORPHAN:", p)
        print()

    # Collect wavs
    wavs = sorted(iter_wavs(in_dir, args.recursive))
    if not wavs:
        raise SystemExit("No wav files found.")
    print(f"Found {len(wavs)} wav files under {in_dir}")

    if args.dry_run:
        to_write = 0
        for w in wavs:
            out_path = expected_out_path(in_dir, out_dir, w, args.suffix)
            status = "SKIP" if out_path.exists() and not args.overwrite else "WRITE"
            if status == "WRITE":
                to_write += 1
            print(status, "->", out_path)
        print(f"\nPlanned writes: {to_write}")
        if args.verify:
            checked, errs = verify_tokens(out_dir, args.recursive, args.suffix, args.verify_samples)
            print(f"Verify: checked {checked}, errors {len(errs)}")
        return

    # Device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Encodec model (facebook/encodec_24khz)...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model.config.target_bandwidths = [args.bandwidth]
    model.to(device).eval()

    ok, skipped, failed = 0, 0, 0
    errors = []

    for w in tqdm(wavs, desc="Encoding", unit="file"):
        out_path = expected_out_path(in_dir, out_dir, w, args.suffix)
        success, msg = encode_file(w, out_path, model, processor, device, args.overwrite)
        if success:
            if msg == "exists":
                skipped += 1
            else:
                ok += 1
        else:
            failed += 1
            errors.append((w, msg))

    print("\nSummary")
    print("-------")
    print(f"Wrote  : {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")
    for p, m in errors[:10]:
        print("ERR", p, ":", m)

    if args.verify:
        checked, errs = verify_tokens(out_dir, args.recursive, args.suffix, args.verify_samples)
        print(f"\nVerify: checked {checked}, errors {len(errs)}")
        for p, m in errs[:10]:
            print("ERR", p, ":", m)


if __name__ == "__main__":
    main()
