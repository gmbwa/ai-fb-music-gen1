# Local Creative Pipeline 🎵

Turn a short theme or mood into a piece of generated music — entirely on your
own machine. No API keys, no cloud calls, no data leaving your computer.

```
"rainy night in Tokyo"
        │
        ▼
  Local LLM (Ollama)  →  "A slow, melancholic lo-fi jazz piece with soft
        │                  piano, muted trumpet, and rain ambience..."
        ▼
  MusicGen (Audiocraft)
        │
        ▼
   output.wav
```

## Why this exists

Most "AI music" demos hit an API. This one doesn't touch the network after
setup — the prompt-writing LLM and the music model both run locally. It's a
small example of chaining two local models together into a creative
pipeline.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- A GPU is strongly recommended for MusicGen (works on CPU, just slow)
- brew install python
- brew install uv


## Setup

1. **Install Ollama and pull a model**


   ```bash
   brew install ollama
   brew services start ollama
   # install Ollama: https://ollama.com/download
   ollama pull llama3.2
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt

   or

   uv init
   uv add -r requirements.txt
   ```

   > First run downloads MusicGen weights from Hugging Face (one-time).

## Usage

```bash
python cli.py "victory in an 8-bit video game" --duration 8
```

```bash
python cli.py "underwater ancient temple" --model medium --out temple.wav
```

Skip the LLM step and pass your own MusicGen prompt directly:

```bash
python cli.py "slow ambient synth pads, 80 bpm, dreamy" --skip-llm
```

### Options

| Flag          | Description                                      | Default     |
|---------------|---------------------------------------------------|-------------|
| `theme`       | Short theme or mood (required)                    | —           |
| `--duration`  | Length of audio in seconds                        | `10`        |
| `--model`     | MusicGen size: `small`, `medium`, `large`          | `small`     |
| `--llm-model` | Ollama model used for prompt writing               | `llama3.2`  |
| `--out`       | Output `.wav` file path                            | `output.wav`|
| `--skip-llm`  | Use the theme as the MusicGen prompt directly       | off         |
| `--device`    | Torch device: `auto`, `cpu`, `mps`, `cuda`          | `auto`      |

## Project structure

```
local-creative-pipeline/
├── cli.py              # entry point, argument parsing, orchestration
├── prompt_writer.py     # local LLM → detailed music prompt (via Ollama)
├── music_generator.py   # music prompt → .wav (via MusicGen/Audiocraft)
├── requirements.txt
└── README.md
```

## Ideas to extend

- Add local image generation (Stable Diffusion/ComfyUI) alongside the audio
  for a full "theme → album art + track" pipeline
- Swap in a Gradio/Streamlit UI instead of the CLI
- Cache generated prompts/audio by theme
- Add a `--genre` flag to steer the LLM's prompt style
- Batch mode: generate a whole soundtrack from a list of scene descriptions

## License

MIT
