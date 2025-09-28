# Overview

This workflow is designed to create datasets of audio files coded with [Meta's Encodec](https://github.com/facebookresearch/encodec) along with aligned parameters or labels for training. 

# Installation

```bash
# Base tools
pip install "git+https://github.com/lonce/EncHF_Dataset@main"

# With extras (adds transformers/torch for audio producer & torch CLIs)
pip install "enc-hf-workflow[audio] @ git+https://github.com/<you>/EncHF_Dataset@main"
```
If you want to do an editable install (to add your own parameter producers for example):
```
git clone https://github.com/lonce/EncHF_Dataset
cd EncHF_Dataset
pip install -e .[audio]
```
If you do an editable install, you will also find a [paramproducers/README_paramproducers.md](paramproducers/README_paramproducers.md) to help write new parameter producers.
# Data Preparation Workflow

This repository provides a **5‑step pipeline** to convert a folder of `.wav` audio files into a Hugging Face `DatasetDict` that can be consumed by the network’s dataloaders. Each stage is implemented as a standalone script, so you can run them one by one, inspect the outputs, and rerun only the stages you need. ("sidecar" refers to the strategy of storing the conditional parameters and metatdata in filenames parallel to the audio data files).

The scripts are numbered in order of execution:

1. [`s1_audio_normalize.py`](./workflow/s1_audio_normalize.py)  
2. [`s2_wav2encodec24.py`](./workflow/s2_wav2encodec24.py)  
3. [`s3_sidecar_init.py`](./workflow/s3_sidecar_init.py)  
4. [`s4_sidecar_add.py`](./workflow/s4_sidecar_add.py)  
5. [`s5_build_dataset_from_folder.py`](./workflow/s5_build_dataset_from_folder.py)  

You may also use [`sidecar_audit.py`](./workflow/sidecar_audit.py) at any point to check sidecar integrity. 

--help gives you more detailed info on any of the scripts.

---

## Step 0

Collect your dataset of wave files (they can be in one, or in nested folders). Assuming you are going to train a model to associate parameters with the data for conditional training and interactive instrument-like performance, you'll need some record of the parameters. They might be is spread sheets, embedded in the file names, or just "in your head", but this workflow will get them into the "sidecar" files so that they can be read by a data loader. 

## Step 1 — Normalize Audio

**Script:** `s1_audio_normalize.py`

Normalizes `.wav` files to a consistent **peak 250ms windowed RMS level** and resamples to **24kHz mono**.

```bash
python s1_audio_normalize.py ./raw_wavs ./normalized_wavs   --target-rms 0.15   --window-ms 250
```

- Input: folder of arbitrary `.wav` files.  
- Output: 24kHz mono `.wav` files in the output folder, with consistent loudness.  
- Options:  
  - `--target-rms`: target RMS (default 0.1).  
  - `--window-ms`: sliding window size (default 250 ms).  
  - `--extensions`: which file extensions to scan.

**Example:** Normalize music clips and save to `./normed`:

```bash
python s1_audio_normalize.py ./music ./normed --target-rms 0.05
```

---

## Step 2 — Convert WAVs to EnCodec Tokens

**Script:** `s2_wav2encodec24.py`

Encodes `.wav` files into **EnCodec 24 kHz mono** compressed token files (`.ecdc`).

```bash
python s2_wav2encodec24.py ./normed ./ecdc_out   --bandwidth 6.0   --recursive   --verify
```

- Input: normalized `.wav` files.  
- Output: `.ecdc` token files with the same directory structure.  
- Options:  
  - `--bandwidth`: target bitrate (default 6.0 kbps).  
  - `--overwrite`: recompute if file already exists.  
  - `--verify`: sample and check `.ecdc` outputs.  
  - `--list-missing`: report which files still need encoding.

**Dry run example:**

```bash
python s2_wav2encodec24.py ./normed ./ecdc_out --dry-run
```

---

## Step 3 — Initialize Sidecar Skeletons

**Script:** `s3_sidecar_init.py`

Creates empty **sidecar files** (`.cond.npy` and `.cond.json`) aligned to each `.ecdc`. These are placeholders for conditioning parameters.

```bash
python s3_sidecar_init.py   --audio-root ./ecdc_out   --cond-root ./params   --manifest-csv ./manifest.csv
```

- Each `.ecdc` gets a matching `.cond.npy` with shape `[T, 0]` (frames, no features yet).  
- Manifest CSV (optional) records alignment.  
- Options for sharding (`--shard-index`, `--shard-count`) for parallel runs.

---

## Step 4 — Add Features with Param Producers

These are intended as examples for how to get parameters into the npy + json metadata file format that a dataloader can read (and less for doing the work of creating or extracting the parameters directly from the audio). They might be usable as is: If you have multiple constant parameters per file and the parameter names and values are embedded in the file names (e.g. foo\_pitch\_64.00_amp\_.75) then check out paramproducers/from_file.py. You can use paramproducers/lin.py to label frames with a normalized "position" parameter. You will likely have to write your own "sidecar param producer" to get your parameters from whatever format you have them in into this sidecar format by following these examples. 

**Script:** `s4_sidecar_add.py`

Populates sidecar `.cond.npy` with **per‑frame conditioning features** using plug‑in producers from `paramproducers/`.

```bash
python s4_sidecar_add.py   --audio-root ./ecdc_out   --cond-root ./params   --producers-dir ./paramproducers   --producer const --producer lin   --const scene_id=7   --lin gain=0:1   --mode append --create-missing
```

- Producers generate features such as constants, linear ramps, values parsed from filenames, or audio‑derived features.  
- Multiple producers can be chained.  
- `--mode append` keeps existing features; `--mode overwrite` replaces them.

Discover available producers:

```bash
python s4_sidecar_add.py --producers-dir ./paramproducers --list-producers
```

Get help for a specific producer:

```bash
python s4_sidecar_add.py --producers-dir ./paramproducers --producer const --help
```

---

## Step 5 — Build Hugging Face Dataset

**Script:** `s5_build_dataset_from_folder.py`

Packages the `.ecdc` + `.cond` pairs into a Hugging Face `DatasetDict` folder.

```bash
python s5_build_dataset_from_folder.py ./ecdc_out ./hf_dataset   --split train   --tokens-subdir tokens   --materialize link   --verify
```

- Creates a dataset with one column `audio` pointing to relative paths.  
- Sidecars are materialized alongside tokens (symlinked, copied, or skipped).  
- Saved with `datasets.save_to_disk` → reload with `datasets.load_from_disk`.

Options:  
- `--materialize link|copy|none` controls whether to symlink or copy files.  
- `--split` sets dataset split name (train/valid/test).  
- `--verify` checks that all referenced files exist.

---

## Optional — Audit

**Script:** `sidecar_audit.py`

Verifies sidecars for shape, dtype, FPS, and value ranges.

```bash
python sidecar_audit.py   --audio-root ./ecdc_out   --cond-root ./params   --expect-fps 75   --expect-dtype float16   --require tempo,bpm,key   --verify-ranges
```

Reports issues such as missing sidecars, frame mismatches, or invalid feature stats.

---

## Typical End‑to‑End Example

```bash
# 1. Normalize
python s1_audio_normalize.py ./raw ./norm

# 2. Encode
python s2_wav2encodec24.py ./norm ./ecdc --recursive --verify

# 3. Init sidecars
python s3_sidecar_init.py --audio-root ./ecdc --cond-root ./params

# 4. Add parameters
python s4_sidecar_add.py --audio-root ./ecdc --cond-root ./params   --producers-dir ./paramproducers --producer from_filename   --from-filename --filename-regex '_(?P<name>\w+)_(?P<value>-?\d+(\.\d+)?)'

# 5. Build dataset
python s5_build_dataset_from_folder.py ./ecdc ./hf_dataset   --split train --materialize link --verify

# (Optional) Audit
python sidecar_audit.py --audio-root ./ecdc --cond-root ./params
```

---

## Tips & Pitfalls

- Always normalize first (Step 1) before encoding, otherwise loudness differences affect downstream features.  
- Use `--dry-run` in Step 2 before heavy processing.  
- Sidecars are small; you can safely delete and regenerate them without touching `.ecdc`.  
- Keep your `paramproducers/` under version control — they define the conditioning schema.  
- After Step 5, you can copy or publish the Hugging Face dataset directory.

