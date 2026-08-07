"""
prompt_writer.py
-----------------
Uses a local LLM served by Ollama (http://localhost:11434) to turn a short
theme/mood into a detailed, MusicGen-friendly text prompt describing genre,
instrumentation, tempo, and mood.

Requires Ollama running locally: https://ollama.com
    ollama pull llama3.2
    ollama serve   (usually already running as a background service)
"""

import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_INSTRUCTIONS = """You are a music prompt writer for an AI music generation \
model (MusicGen). Given a short theme or mood, respond with ONE single-paragraph \
prompt (max 40 words) describing: genre, instrumentation, tempo, and emotional \
tone. Do not include song titles, lyrics, or explanations. Only output the \
prompt text itself, nothing else."""


def generate_music_prompt(theme: str, model: str = "llama3.2") -> str:
    """Ask the local LLM to expand `theme` into a rich music-generation prompt."""

    full_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nTheme: {theme}\nPrompt:"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.8},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434.\n"
            "Make sure Ollama is installed and running (`ollama serve`), "
            f"and that the model is pulled (`ollama pull {model}`).\n"
            f"Original error: {e}"
        )

    text = body.get("response", "").strip()

    if not text:
        raise RuntimeError("Ollama returned an empty response.")

    return text
