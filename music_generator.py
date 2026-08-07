"""
music_generator.py
-------------------
Wraps Meta's Audiocraft MusicGen to turn a text prompt into a local .wav file.

Requires:
    pip install audiocraft

First run will download model weights from Hugging Face (one-time, needs
internet). After that, generation is fully offline.

Model sizes (VRAM roughly needed):
    small  ~2GB   - fast, decent quality, good for CPU too (slow on CPU)
    medium ~5GB   - better quality
    large  ~12GB  - best quality, needs a real GPU
"""

import os
from pathlib import Path

# Some ops used by MusicGen aren't yet implemented for the MPS (Apple GPU)
# backend; this makes torch fall back to CPU for just those ops instead of
# raising. Must be set before torch touches MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio
from audiocraft.models import MusicGen


_MODEL_CACHE = {}


def _resolve_device(device: str = "auto") -> str:
    """Pick the best available torch device. `MusicGen.get_pretrained` only
    auto-detects CUDA, so on Apple Silicon it would otherwise default to CPU."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_size: str, device: str = "auto") -> MusicGen:
    """Load (and cache) a MusicGen model by size and device."""
    resolved_device = _resolve_device(device)
    cache_key = (model_size, resolved_device)
    if cache_key not in _MODEL_CACHE:
        model_name = f"facebook/musicgen-{model_size}"
        _MODEL_CACHE[cache_key] = MusicGen.get_pretrained(model_name, device=resolved_device)
    return _MODEL_CACHE[cache_key]


def generate_music(
    prompt: str,
    duration: int = 10,
    model_size: str = "small",
    out_path: Path = Path("output.wav"),
    device: str = "auto",
) -> Path:
    """Generate audio from `prompt` and save it to `out_path`. Returns the path."""

    model = _load_model(model_size, device=device)
    model.set_generation_params(duration=duration)

    # MusicGen expects a list of prompts (batch of 1 here)
    wav = model.generate([prompt])  # shape: (batch, channels, samples)

    out_path = out_path.with_suffix(".wav")
    sample_rate = model.sample_rate

    # torchaudio expects (channels, samples) for a single clip
    torchaudio.save(str(out_path), wav[0].cpu(), sample_rate)

    return out_path
