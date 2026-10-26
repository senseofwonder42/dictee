"""Tests for audio conversion and transcription in app.py."""

import time
import wave
from pathlib import Path

import numpy as np
import pytest

import app

FIXTURE = Path(__file__).parent / "fr_fixture.wav"


def test_int16_to_float32() -> None:
    audio = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    out = app.to_mono_16k(16_000, audio)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0, 32767 / 32768])


def test_stereo_to_mono() -> None:
    audio = np.stack([np.full(1600, 0.2), np.full(1600, 0.6)], axis=1).astype(np.float32)
    out = app.to_mono_16k(16_000, audio)
    assert out.shape == (1600,)
    np.testing.assert_allclose(out, 0.4, rtol=1e-6)


@pytest.mark.parametrize("sample_rate", [8_000, 44_100, 48_000])
def test_resample_to_16k(sample_rate: int) -> None:
    t = np.arange(sample_rate) / sample_rate  # 1 s
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = app.to_mono_16k(sample_rate, audio)
    assert out.dtype == np.float32
    assert abs(len(out) - 16_000) <= 1


@pytest.mark.parametrize("n_samples", [0, 1000])
def test_short_audio_returns_empty(n_samples: int) -> None:
    audio = np.zeros(n_samples, dtype=np.float32)
    assert app.transcribe(app.PARAKEET, audio) == ""


def test_short_recording_status() -> None:
    text, status = app._on_stop((48_000, np.zeros(4800, dtype=np.int16)), app.PARAKEET, "")
    assert text == ""
    assert "too short" in status


def test_load_failure_shows_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DICTEE_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_loaded", {})
    text, status = app._on_stop((16_000, np.zeros(16_000, dtype=np.int16)), app.WHISPER, "")
    assert text == ""
    assert "Could not load Whisper" in status


@pytest.mark.slow
@pytest.mark.parametrize("model_name", app.MODELS)
def test_transcribe_french_fixture(model_name: str) -> None:
    with wave.open(str(FIXTURE)) as f:
        rate = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    audio = app.to_mono_16k(rate, audio)
    app.load_model(model_name)
    start = time.perf_counter()
    text = app.transcribe(model_name, audio).lower()
    print(f"\n{model_name}: {time.perf_counter() - start:.2f} s -> {text!r}")
    for word in ["bonjour", "réserver", "table", "personnes", "demain"]:
        assert word in text
