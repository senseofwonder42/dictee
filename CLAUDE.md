# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local, CPU-only voice dictation (French mostly, some English) served as a Gradio page on `127.0.0.1:7860`. Target deployment is a corporate Windows laptop with no GPU, possibly behind a proxy that blocks Hugging Face. No cloud services. The README is written in French; UI labels and code are in English.

## Commands

```bash
uv sync                                   # install deps (Python 3.12)
uv run app.py                             # launch the app, opens browser
uv run pytest                             # fast tests (slow ones excluded via addopts)
uv run pytest -m slow                     # real transcription of tests/fr_fixture.wav with both models (downloads models)
uv run pytest tests/test_app.py::test_stereo_to_mono   # single test
```

## Architecture

Core logic lives in `app.py` (Gradio UI) with `hotkey.py` as a push-to-talk entry point; tests in `tests/` import them as modules (`pythonpath = ["."]`).

Pipeline: Gradio `gr.Audio` (numpy) → `to_mono_16k` (int→float32, stereo→mono, resample with `soxr`) → `transcribe` → text + status line.

- **Two backends behind one `transcribe()`**: Parakeet TDT 0.6B v3 via `onnx-asr` (int8 ONNX, default, fast) and Whisper large-v3-turbo via `faster-whisper` (CTranslate2, float16 weights converted to int8 at load). Dispatch is by `isinstance(model, WhisperModel)`. Only Whisper uses the vocabulary field (as `initial_prompt`).
- **Model cache**: `load_model` lazily loads each model once into the module-level `_loaded` dict; switching models doesn't reload. Tests reset it with `monkeypatch.setattr(app, "_loaded", {})`.
- **Offline mode**: if `DICTEE_MODELS_DIR` is set, models are read only from `<dir>/<LOCAL_DIRS[name]>` (no network); a missing folder raises `FileNotFoundError`. Otherwise models download from Hugging Face.
- **UI handler `_on_stop`**: shared by `stop_recording`, `upload`, and the Transcribe button. It rejects audio shorter than `MIN_DURATION_S`, loads the model separately so load failures surface as a `⚠ Could not load …` status (with full traceback logged via loguru), and reports `"<Short> · X s audio · Y s"`.

**Push-to-talk (`hotkey.py`)**: a second entry point (`uv run hotkey.py`) that reuses `app.load_model` / `to_mono_16k` / `transcribe`. A `pynput` listener starts a `sounddevice` recording (device default rate) while `DICTEE_HOTKEY` (default `ctrl_r`) is held; on release, a worker thread transcribes with Parakeet and pastes via `pyperclip` + simulated Ctrl/Cmd+V. Keep the listener callbacks fast — slow ones lag system-wide typing. Tests in `tests/test_hotkey.py` patch `hotkey.paste` and `app.transcribe`.

When changing model loading, paths, or setup steps, keep the README's offline/proxy sections (expected folder tree, `hf download` commands) in sync.
