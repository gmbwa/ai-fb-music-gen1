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

from pathlib import Path

import torch
import torchaudio
from audiocraft.models import MusicGen


_MODEL_CACHE = {}


def _load_model(model_size: str) -> MusicGen:
    """Load (and cache) a MusicGen model by size."""
    if model_size not in _MODEL_CACHE:
        model_name = f"facebook/musicgen-{model_size}"
        _MODEL_CACHE[model_size] = MusicGen.get_pretrained(model_name)
    return _MODEL_CACHE[model_size]


def generate_music(
    prompt: str,
    duration: int = 10,
    model_size: str = "small",
    out_path: Path = Path("output.wav"),
) -> Path:
    """Generate audio from `prompt` and save it to `out_path`. Returns the path."""

    model = _load_model(model_size)
    model.set_generation_params(duration=duration)

    # MusicGen expects a list of prompts (batch of 1 here)
    wav = model.generate([prompt])  # shape: (batch, channels, samples)

    out_path = out_path.with_suffix(".wav")
    sample_rate = model.sample_rate

    # torchaudio expects (channels, samples) for a single clip
    torchaudio.save(str(out_path), wav[0].cpu(), sample_rate)

    return out_path
