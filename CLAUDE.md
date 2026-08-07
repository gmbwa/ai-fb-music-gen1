# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline creative pipeline: a short text theme is expanded into a detailed
music-generation prompt by a local LLM (via Ollama), then rendered to a `.wav` file by
Meta's MusicGen (via Audiocraft). No API keys, no network calls after initial setup
(other than the one-time Hugging Face model download).

```
theme string → prompt_writer.py (Ollama LLM) → detailed prompt → music_generator.py (MusicGen) → output.wav
```

## Setup & running

```bash
# Ollama must be installed and running, with a model pulled
brew install ollama
brew services start ollama
ollama pull llama3.2

# Python deps (uv-based; pyproject.toml currently declares no deps — requirements.txt is authoritative)
uv add -r requirements.txt
# or: pip install -r requirements.txt

# Run the pipeline
python cli.py "rainy night in Tokyo" --duration 15
python cli.py "underwater ancient temple" --model medium --out temple.wav
python cli.py "slow ambient synth pads, 80 bpm, dreamy" --skip-llm
```

There is no build, lint, or test tooling configured in this repo (no test files,
no linter config). Don't assume `pytest`/`ruff`/etc. are set up — check before
invoking them.

## Architecture

The real pipeline lives in three root-level modules, run via `cli.py`:

- **`cli.py`** — argument parsing and orchestration only. Calls
  `prompt_writer.generate_music_prompt()` then `music_generator.generate_music()`.
  `--skip-llm` bypasses the LLM step and feeds the raw theme straight to MusicGen.
- **`prompt_writer.py`** — talks to Ollama's local HTTP API
  (`http://localhost:11434/api/generate`) via raw `urllib` (no SDK dependency). Wraps
  the theme in a fixed system instruction (`SYSTEM_INSTRUCTIONS`) that constrains the
  LLM to a single ≤40-word prompt describing genre/instrumentation/tempo/mood.
  Connection failures are re-raised as a `RuntimeError` with setup instructions.
- **`music_generator.py`** — loads `facebook/musicgen-{small,medium,large}` via
  Audiocraft's `MusicGen.get_pretrained`, caching loaded models per size in the
  module-level `_MODEL_CACHE` dict (relevant if generating multiple sizes in one
  process). Generates a batch of 1 and saves via `torchaudio.save`.

**`src/fb_music_gen1/`** is a separate, disconnected `uv init` package stub (just a
`main()` that prints "Hello from fb-music-gen1!"). `pyproject.toml`'s
`[project.scripts]` entry point (`fb-music-gen1 = "fb_music_gen1:main"`) refers to
this stub, not to the real pipeline in `cli.py` — the two are not wired together.
