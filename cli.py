#!/usr/bin/env python3
"""
Local Creative Pipeline
------------------------
Turns a short theme/mood into a fully-produced music generation prompt
using a local LLM (via Ollama), then renders it to audio using MusicGen
(via Meta's Audiocraft) — entirely offline, no API keys required.

Usage:
    python cli.py "rainy night in Tokyo" --duration 15
    python cli.py "victory in an 8-bit video game" --model medium --out victory.wav
"""

import argparse
import sys
from pathlib import Path

from prompt_writer import generate_music_prompt
from music_generator import generate_music


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music from a theme using a local LLM + MusicGen."
    )
    parser.add_argument(
        "theme",
        type=str,
        help="A short theme or mood, e.g. 'rainy night in Tokyo'",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Length of generated audio in seconds (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="MusicGen model size (default: small; larger = better quality, needs more VRAM)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2",
        help="Ollama model to use for prompt writing (default: llama3.2)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM step and use the theme directly as the MusicGen prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_llm:
        music_prompt = args.theme
        print(f"[*] Using raw theme as prompt: {music_prompt}")
    else:
        print(f"[*] Asking local LLM ({args.llm_model}) to expand theme: '{args.theme}'")
        music_prompt = generate_music_prompt(args.theme, model=args.llm_model)
        print(f"[*] Generated music prompt:\n    {music_prompt}\n")

    print(f"[*] Generating {args.duration}s of audio with MusicGen ({args.model}) ...")
    out_path = generate_music(
        prompt=music_prompt,
        duration=args.duration,
        model_size=args.model,
        out_path=Path(args.out),
    )

    print(f"[✓] Done. Saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
