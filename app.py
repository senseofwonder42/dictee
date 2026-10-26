"""Dictée: local voice dictation in the browser, transcribed on CPU.

Run with ``uv run app.py``. Records from the microphone in a Gradio page on
127.0.0.1 and transcribes with Parakeet TDT 0.6B v3 (ONNX int8) or
Whisper large-v3-turbo (CTranslate2 int8).
"""

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import numpy.typing as npt
import onnx_asr
import soxr
from faster_whisper import WhisperModel
from loguru import logger

SAMPLE_RATE = 16_000
MIN_DURATION_S = 0.3

PARAKEET = "Parakeet TDT 0.6B v3 (default)"
WHISPER = "Whisper large-v3-turbo"
MODELS = [PARAKEET, WHISPER]
SHORT_NAMES = {PARAKEET: "Parakeet", WHISPER: "Whisper"}
# Folder names expected under DICTEE_MODELS_DIR for offline use.
LOCAL_DIRS = {PARAKEET: "parakeet-tdt-0.6b-v3-onnx", WHISPER: "faster-whisper-large-v3-turbo"}

_loaded: dict[str, onnx_asr.adapters.TextResultsAsrAdapter | WhisperModel] = {}


def _local_dir(model_name: str) -> Path | None:
    root = os.environ.get("DICTEE_MODELS_DIR")
    if not root:
        return None
    path = Path(root) / LOCAL_DIRS[model_name]
    if not path.is_dir():
        raise FileNotFoundError(f"DICTEE_MODELS_DIR is set but {path} does not exist")
    return path


def load_model(model_name: str) -> onnx_asr.adapters.TextResultsAsrAdapter | WhisperModel:
    """Load a model once and keep it cached in memory.

    Args:
        model_name: One of ``MODELS``.

    Returns:
        The loaded onnx-asr or faster-whisper model.

    Raises:
        ValueError: If model_name is unknown.
        FileNotFoundError: If DICTEE_MODELS_DIR is set but the model folder is missing.
    """
    if model_name in _loaded:
        return _loaded[model_name]
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    logger.info("Loading {}...", model_name)
    start = time.perf_counter()
    path = _local_dir(model_name)
    if model_name == PARAKEET:
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", path, quantization="int8", providers=["CPUExecutionProvider"]
        )
    else:
        model = WhisperModel(str(path or "large-v3-turbo"), device="cpu", compute_type="int8")
    logger.info("{} loaded in {:.1f} s", model_name, time.perf_counter() - start)
    _loaded[model_name] = model
    return model


def to_mono_16k(sample_rate: int, audio: npt.NDArray) -> npt.NDArray[np.float32]:
    """Convert raw audio to 16 kHz mono float32 in [-1, 1].

    Args:
        sample_rate: Sample rate of ``audio`` in Hz.
        audio: Samples as int or float, shape ``(n,)`` or ``(n, channels)``.

    Returns:
        A 1-D float32 array sampled at 16 kHz.
    """
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max + 1)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE)
    return audio.astype(np.float32)


def transcribe(model_name: str, audio: npt.NDArray[np.float32], vocabulary: str | None = None) -> str:
    """Transcribe 16 kHz mono float32 audio with the chosen model.

    Args:
        model_name: One of ``MODELS``.
        audio: 16 kHz mono float32 samples.
        vocabulary: Optional jargon passed to Whisper as ``initial_prompt``; ignored by Parakeet.

    Returns:
        The transcribed text, or an empty string if the audio is shorter than ``MIN_DURATION_S``.
    """
    if len(audio) < MIN_DURATION_S * SAMPLE_RATE:
        return ""
    model = load_model(model_name)
    if isinstance(model, WhisperModel):
        segments, _ = model.transcribe(audio, initial_prompt=vocabulary or None)
        return " ".join(s.text.strip() for s in segments).strip()
    return model.recognize(audio, sample_rate=SAMPLE_RATE).strip()


def _on_stop(recording: tuple[int, npt.NDArray] | None, model_name: str, vocabulary: str) -> tuple[str, str]:
    if recording is None:
        return "", "No audio received."
    audio = to_mono_16k(*recording)
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_DURATION_S:
        return "", f"Recording too short ({duration:.1f} s)."
    short = SHORT_NAMES[model_name]
    try:
        load_model(model_name)
    except Exception as exc:
        logger.exception("Could not load {}", model_name)
        return "", f"⚠ Could not load {short}: {exc}"
    start = time.perf_counter()
    text = transcribe(model_name, audio, vocabulary.strip() or None)
    elapsed = time.perf_counter() - start
    logger.info("{} · {:.1f} s audio · {:.1f} s", short, duration, elapsed)
    return text, f"{short} · {duration:.1f} s audio · {elapsed:.1f} s"


def build_ui() -> gr.Blocks:
    """Build the Gradio interface.

    Returns:
        The Blocks app, not yet launched.
    """
    with gr.Blocks(title="Dictée", analytics_enabled=False) as demo:
        mic = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")
        model = gr.Dropdown(MODELS, value=PARAKEET, label="Model")
        vocab = gr.Textbox(label="Vocabulary / jargon (Whisper only)", max_lines=1)
        output = gr.Textbox(label="Transcription", lines=8, interactive=True, buttons=["copy"])
        status = gr.Markdown()
        mic.stop_recording(_on_stop, inputs=[mic, model, vocab], outputs=[output, status])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
