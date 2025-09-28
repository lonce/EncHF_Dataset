# Writing Param Producers

Param producers are **plug‑in modules** that generate per‑frame features for EnCodec sidecars. They live in the `paramproducers/` folder and are loaded dynamically by [`4_sidecar_add.py`](../4_sidecar_add.py).

Each producer is just a Python file that exposes a few optional functions. The loader takes care of aligning frames and writing the `.cond.npy/.json`.

---

## Producer API

A producer may implement any of these functions:

```python
def add_cli_args(parser):
    """Register custom CLI flags with argparse."""
    ...

def init(args):
    """Optional: initialize resources once (e.g., load a model)."""
    ...

def produce(ecdc_path: Path, T: int, args) -> dict[str, np.ndarray]:
    """Return a dict of {feature_name: np.ndarray(T,)}."""
    ...

def shutdown(args):
    """Optional: clean up resources."""
    ...
```

- **`produce` is required** — it must return 1D arrays of length `T` (frames).  
- Features are merged into the sidecar.  
- If multiple producers define the same key, **last one wins**.  
- Dtype is normalized to float32 internally.

---

## Example 1 — Constant Value

`paramproducers/const.py`:

```python
"""Constant value producer."""

import numpy as np

def add_cli_args(ap):
    ap.add_argument("--const", action="append", default=[],
        help="Constant features like name=value. Repeatable.")

def produce(ecdc_path, T, args):
    feats = {}
    for spec in args.const:
        name, val = spec.split("=")
        feats[name] = np.full(T, float(val), dtype=np.float32)
    return feats
```

Usage:

```bash
python 4_sidecar_add.py --audio-root ./ecdc --cond-root ./params   --producers-dir ./paramproducers --producer const   --const scene_id=7 --const instrument=2
```

---

## Example 2 — Linear Ramp

`paramproducers/lin.py`:

```python
"""Linear ramp producer."""

import numpy as np

def add_cli_args(ap):
    ap.add_argument("--lin", action="append", default=[],
        help="Linear ramp name=start:end. Repeatable.")

def produce(ecdc_path, T, args):
    feats = {}
    for spec in args.lin:
        name, rng = spec.split("=")
        start, end = map(float, rng.split(":"))
        feats[name] = np.linspace(start, end, T, dtype=np.float32)
    return feats
```

Usage:

```bash
python 4_sidecar_add.py --audio-root ./ecdc --cond-root ./params   --producers-dir ./paramproducers --producer lin   --lin gain=0:1
```

---

## Example 3 — Parse from Filenames

A producer that uses regex groups from filenames to extract parameters.

```python
"""Extract params from filenames using regex."""

import re, numpy as np

def add_cli_args(ap):
    ap.add_argument("--filename-regex", required=True,
        help="Regex with (?P<name>...) and (?P<value>...) groups")

def produce(ecdc_path, T, args):
    m = re.search(args.filename_regex, ecdc_path.name)
    if not m:
        return {}
    name = m.group("name")
    val = float(m.group("value"))
    return {name: np.full(T, val, dtype=np.float32)}
```

Usage:

```bash
python 4_sidecar_add.py --audio-root ./ecdc --cond-root ./params   --producers-dir ./paramproducers --producer from_filename   --filename-regex '_(?P<name>\w+)_(?P<value>-?\d+(\.\d+)?)'
```

---

## Example 4 — Audio‑Derived Features

You can also decode EnCodec tokens and compute per‑frame audio features.

```python
"""Compute RMS dB from decoded audio."""

import numpy as np, librosa, torch
from transformers import EncodecModel, AutoProcessor

_model = None
_proc = None

def add_cli_args(ap):
    ap.add_argument("--hf-model", default="facebook/encodec_24khz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--audio-feats", nargs="+", default=["rms_db"])

def init(args):
    global _model, _proc
    _model = EncodecModel.from_pretrained(args.hf_model).to(args.device)
    _proc = AutoProcessor.from_pretrained(args.hf_model)

def produce(ecdc_path, T, args):
    # TODO: decode ecdc_path (example assumes you already have wav)
    y, sr = librosa.load(str(ecdc_path.with_suffix(".wav")), sr=24000, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=320, hop_length=320)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8)
    return {"rms_db": np.resize(rms_db, T).astype(np.float32)}

def shutdown(args):
    pass
```

---

## Guidelines for Writing Producers

- Always return arrays of shape `(T,)`.  
- Use `np.float32`.  
- If parsing filenames, prefer regex with explicit `(?P<name>…)` and `(?P<value>…)` groups.  
- If you need state (e.g., ML model), initialize it in `init` and clean up in `shutdown`.  
- Validate output length — raise errors if mismatched.  
- Document your producer with a module docstring.

---

## Testing Producers

1. Add your new file to `paramproducers/`.  
2. Run with `--list-producers` to confirm it’s detected:  

   ```bash
   python 4_sidecar_add.py --producers-dir ./paramproducers --list-producers
   ```

3. Use `--show-producer NAME` to view help.  
4. Run with a small dataset and inspect the `.cond.npy` arrays.  
5. Audit with `sidecar_audit.py` to verify alignment and schema.

---

## Summary

- Producers are modular, self‑contained Python files.  
- They provide per‑frame features for conditioning.  
- Combine multiple producers for richer feature sets.  
- Keep them version‑controlled to ensure dataset reproducibility.

