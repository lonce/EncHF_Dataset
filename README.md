# Overview

This workflow is designed to create datasets of audio files coded with [Meta's Encodec](https://github.com/facebookresearch/encodec) along with aligned parameters or labels for training. 

# Installation

```
git clone https://github.com/lonce/EncHF_Dataset
cd EncHF_Dataset
```
You might want to create a basic conda environment before installing that packages. Then:
```bash
# Base tools
pip install "git+https://github.com/lonce/EncHF_Dataset@main"
```
or if you want an editable install:
```
pip install -e .
```

# Data Preparation Workflow

This repository provides a **4‑step pipeline** to convert a folder of `.wav` audio files into a Hugging Face `DatasetDict` that can be consumed by the network’s dataloaders. Each stage is implemented as a standalone script, so you can run them one by one, inspect the outputs, and rerun only the stages you need. ("sidecar" refers to the strategy of storing the conditional parameters and metatdata in filenames parallel to the audio data files).

The scripts are numbered in order of execution:

1. [`s1_audio_normalize.py`](./workflow/s1_audio_normalize.py)  - normalizes for rms and sr
2. [`s2_wav2encodec24.py`](./workflow/s2_wav2encodec24.py)   - codes wav files with Encodec
3. [`roll_your_own.py`] write param files using workflow.writeParamArrays.py (see MakeSidecars.ipynb as an example of how to write the params)
5. [`s5_build_dataset_from_folder.py`](./workflow/s5_build_dataset_from_folder.py)  - creates HF dataset ready for loading

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

## Step 3 — Add Parameter files

Parameter files have the same name as the data file. A .npy file has the frame-length lists of parameter values, and a .json file has the necessary meta-data.  You will create the features data structure as in this example:

feats = {
  "gain_db": {                                  # parameter name
      "values": [numpy array],        # only required attribute, same length as Encodec frames
      "units": "dB", 
      "doc_string": "Per-frame gain in decibels"
  },
  "scene_id": {
      "values": np.full(numFrames, 7, np.float32),
      "min": 7, "max": 7,             # used by dataloader to normalize for NN training
      "doc_string": "Integer scene label"
  },
  "energy": np.random.rand(numFrames).astype(np.float32)  # stats auto-computed
}

Then write the parameters (feat) like so:

write_sidecar_features_for_rel(
    root=Path("/datasets/waterfill"),
    ecdc_rel=Path("ecdc/II_Double.ecdc"),
    features=feats,
    fps=75.0,
    mode="overwrite",
    require_ecdc_exists=False,  # set True if you want a safety check
)



Here is an example of doing that for one data file in a jupyter notebook with checks and visualizations:

**Script:** `MakeSidecars.ipynb`



## Step 4 — Build Hugging Face Dataset

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

# 3. Extract/generate per-frame parameters and Write sidecars
python roll_your_own.py 

# 4. Build dataset
python s5_build_dataset_from_folder.py ./ecdc ./hf_dataset   --split train --materialize link --verify

# (Optional) Audit
python sidecar_audit.py --audio-root ./ecdc --cond-root ./params
```

---

## Tips & Pitfalls

- Probably best to  normalize first (Step 1) before encoding, otherwise loudness differences affect downstream features.  
- Use `--dry-run` in Step 2 before heavy processing.  
- Sidecars are small; you can safely delete and regenerate them without touching `.ecdc`.  
- After Step 4, you can copy or publish the Hugging Face dataset directory.



If you pip-installed this repository, you should also have these wf- shortcuts:
```bash
wf-audio-normalize --help
wf-wav2encodec24 --help
wf-build-dataset --help
wf-sidecar-audit --help
wf-inspect --help
```



