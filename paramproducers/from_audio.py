# producers/from_audio.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from __future__ import annotations
import argparse
from typing import Dict, List
import numpy as np
from pathlib import Path

# lazy state
STATE = {"model": None, "sr": None, "device": "cpu"}

def add_cli_args(ap: argparse.ArgumentParser):
    ap.add_argument("--from-audio", action="store_true",
                    help="Enable audio-derived features by decoding .ecdc")
    ap.add_argument("--hf-model", default="facebook/encodec_24khz",
                    help="Hugging Face Encodec model id to use for decoding")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda:0")
    ap.add_argument("--audio-feats", default="rms_db",
                    help="Comma-separated features to compute (supported: rms_db)")

def init(args):
    if not getattr(args, "from_audio", False):
        return
    from transformers import EncodecModel
    device = args.device
    model = EncodecModel.from_pretrained(args.hf_model).to(device).eval()
    sr = int(getattr(getattr(model, "config", model), "sampling_rate",
                     getattr(model, "sample_rate", 24000)))
    STATE["model"] = model
    STATE["sr"] = sr
    STATE["device"] = device

def _decode_to_wave(ecdc_path: Path) -> np.ndarray:
    """
    Return mono waveform as float32 numpy array.
    """
    import torch
    ckpt = torch.load(ecdc_path, map_location="cpu")
    codes = ckpt["audio_codes"]  # [1,Cb,T] or [Cb,T] or [1,1,Cb,T]
    if codes.ndim == 4:
        codes = codes.squeeze(1)  # [1,Cb,T]
    if codes.ndim == 2:
        codes = codes[None, ...]  # [1,Cb,T]
    codes = torch.as_tensor(codes, dtype=torch.long)  # [1,Cb,T]

    model = STATE["model"]
    device = STATE["device"]

    # quantizer expects (n_q, B, T)
    z = model.quantizer.decode(codes.transpose(0, 1).to(device))  # (Cb, 1, T)
    # decoder usually accepts (B, T, 128) or (B, 128, T); handle both
    if z.dim() == 3:
        # try both permutations; prefer (B, T, 128)
        if z.shape[0] != 1:  # (Cb,1,T) -> (1,T,128) is model-internal; pass through decoder directly
            pass
    y = model.decoder(z)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y.detach().to("cpu").to(torch.float32)  # [B, C, N] or [B, N]
    if y.ndim == 1:
        wav = y.numpy()
    elif y.ndim == 2:
        # [B, N] (no channel dim)
        wav = y[0].numpy()
    else:
        # [B, C, N]
        arr = y[0].numpy()
        if arr.ndim == 2 and arr.shape[0] > 1:
            wav = arr.mean(axis=0)
        elif arr.ndim == 2:
            wav = arr[0]
        else:
            wav = arr.squeeze()
    return np.asarray(wav, dtype=np.float32)

def _per_frame_blocks(x: np.ndarray, T: int, hop: int) -> np.ndarray:
    need = T * hop
    if x.shape[0] < need:
        x = np.pad(x, (0, need - x.shape[0]))
    elif x.shape[0] > need:
        x = x[:need]
    return x.reshape(T, hop)

def produce(ecdc_path: Path, T: int, *, args) -> Dict[str, np.ndarray]:
    if not getattr(args, "from_audio", False):
        return {}
    feats_req: List[str] = [s.strip() for s in str(args.audio_feats).split(",") if s.strip()]
    if not feats_req:
        return {}

    sr = STATE["sr"]
    if sr is None or STATE["model"] is None:
        # not initialized; user likely forgot --from-audio before
        return {}

    hop = int(round(sr / 75))
    wav = _decode_to_wave(ecdc_path)     # mono waveform
    X = _per_frame_blocks(wav, T, hop)   # [T, hop]

    feats: Dict[str, np.ndarray] = {}
    for name in feats_req:
        if name == "rms_db":
            rms = np.sqrt((X.astype(np.float64)**2).mean(axis=1) + 1e-12)
            feats[name] = (20.0 * np.log10(rms + 1e-12)).astype(np.float32)
        else:
            # unknown: produce nothing; extend here as needed
            pass
    return feats
