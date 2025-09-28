# Encodec Sidecar Format (ESF) — v1

A simple, model-agnostic on-disk format for pairing **Encodec token stacks** with
**aligned per-frame conditioning features**. ESF is designed so any training
pipeline can consume Encodec codes together with time-aligned parameters
(pitch, loudness, scene ids, controls, etc.) at **75 frames per second**.

---

## File triplet (co-located)

For each audio clip there are **three** files in the same directory, sharing a base name:

- `NAME.ecdc` — Encodec codes (PyTorch `torch.save` checkpoint).
- `NAME.cond.npy` — conditioning matrix, shape **[T, D]** (float16 by default).
- `NAME.cond.json` — metadata for the conditioning columns.

Where:
- `T` = number of Encodec frames.
- `D` = number of conditioning features.
- FPS is **fixed at 75** for Encodec code streams (24 kHz → hop 320, 48 kHz → hop 640).

> ESF **requires** that `NAME.cond.npy` has **exactly** the same number of frames `T` as `NAME.ecdc`.

---

## `NAME.ecdc` (Encodec codes)

A small `torch.save` checkpoint with (at minimum) the following keys:

- `audio_codes` — integer tensor of shape **[1, Cb, T]**, **[Cb, T]**, or **[1, 1, Cb, T]**.
  Readers must accept any of these and canonicalize to **[Cb, T]** or **[T, Cb]**.
- `audio_scales` — optional tensor/list of scalars (ignored by most consumers).
- `audio_length` — original sample count (optional but nice to have).

Notes:
- `Cb` = number of codebooks (a.k.a. quantizers).
- Store with `torch.save({...}, path)`; load with `torch.load(path, map_location="cpu")`.

---

## `NAME.cond.npy` (conditioning matrix)

- Numpy array, shape **[T, D]**.
- `dtype`: **float16** recommended (float32 also valid).
- Row **i** contains all `D` conditioning values aligned to **Encodec frame i**.

Empty sidecars are valid: **[T, 0]** with an empty names list in the JSON (see below).

---

## `NAME.cond.json` (metadata)

```jsonc
{
  "schema_version": 1,
  "fps": 75,
  "source_rate": 75,
  "names": ["pos", "gain"],      // length D, column order matches .cond.npy
  "norm": {                      // per-clip stats; arrays length D or 0
    "min":  [0.0, 0.0],
    "max":  [1.0, 1.0],
    "mean": [0.50, 0.65],
    "std":  [0.29, 0.20]
  }
}
```

Rules:
- `fps` **must** be 75 and match the token frame rate.
- `names` length must equal `D` in the `.cond.npy` (or be empty when `D=0`).
- `norm` arrays are either **all empty (length 0)** or **all length D**.
- Additional fields are allowed; readers should ignore unknown keys.

---

## Alignment & determinism

- Let the target Encodec model sampling rate be `sr`. One Encodec frame spans `hop = round(sr / 75)` samples.
- The **T** returned by the codes defines the authoritative number of frames, and the sidecar must match it.
- Consumers must treat `i` in `[0, T-1]` as a strict time index across codes and conditioning.

---

## Validation (recommended)

A sidecar pair **passes** ESF validation if:

1) `NAME.cond.npy` exists and is 2-D [T, D].  
2) `NAME.cond.json` exists and `names` matches `D`.  
3) JSON `fps == 75`.  
4) `T` equals the code tensor’s time dimension.  
5) If present, `norm.min/max/mean/std` either all have length D or all are empty.

Failures should be reported per-file (don’t halt the whole corpus).

---

## Typical workflow

1) **Initialize** empty sidecars aligned to codes (creates `[T, 0]` + JSON skeleton):
   ```bash
   encodec-sidecar-init --audio-root /path/to/ecdc
   ```

2) **Add features** with plugin producers (constants, ramps, filename parsing, audio-derived, etc.):
   ```bash
   encodec-sidecar-add --audio-root /path/to/ecdc \
     --producers-dir ./producers \
     --producer const --const scene_id=7 \
     --producer lin   --lin pos=0:1 \
     --mode append --create-missing
   ```

3) **Audit** alignment, schema, and (optionally) min/max range agreement:
   ```bash
   encodec-sidecar-audit --audio-root /path/to/ecdc --require pos,scene_id
   ```

4) **Package** as a Hugging Face dataset (symlink or copy tokens + sidecars):
   ```bash
   encodec-build-dataset /path/to/ecdc /hf/my_ds \
     --recursive --tokens-subdir tokens --materialize link
   ```

---

## Interoperability notes

- Co-location is required: `NAME.ecdc`, `NAME.cond.npy`, `NAME.cond.json` live in the same directory.
- Readers should cache JSON metadata per file for performance.
- Consumers **may** normalize features at load time using sidecar `norm.min/max` (common for [0, 1] scaling) but ESF does not prescribe a specific normalization policy.

---

## Versioning

- Current `schema_version` is **1**.  
- Future versions may add non-breaking fields. If a breaking change is necessary, bump the version and document it; readers should reject unknown major versions.

---

## Minimal example

Given `clip.ecdc` with `audio_codes` shape `[1, Cb, T]` and sidecar:

- `clip.cond.npy` → shape `[T, 1]` with column `"pos"` (float16)
- `clip.cond.json` → as in the snippet above

A generic loader should return:
- `codes`: LongTensor shape `[T, Cb]`
- `cond`:  FloatTensor shape `[T, 1]`
- `names`: `["pos"]`

---

## License

ESF text and reference tools: MIT or Apache-2.0 recommended.
