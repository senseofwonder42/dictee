"""Tests for the push-to-talk flow in hotkey.py (no microphone or keyboard needed)."""

import numpy as np
import pytest

import app
import hotkey


@pytest.fixture
def pasted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(hotkey, "paste", calls.append)
    return calls


def test_transcription_is_pasted(pasted: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    received = []

    def fake_transcribe(model_name, audio):
        received.append(audio)
        return "Bonjour."

    monkeypatch.setattr(app, "transcribe", fake_transcribe)
    hotkey.handle_recording(48_000, np.zeros(48_000, dtype=np.float32))
    assert pasted == ["Bonjour."]
    assert abs(len(received[0]) - 16_000) <= 1  # resampled to 16 kHz


def test_short_recording_not_pasted(pasted: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "transcribe", lambda *_: pytest.fail("should not transcribe"))
    hotkey.handle_recording(48_000, np.zeros(4800, dtype=np.float32))
    assert pasted == []


def test_empty_transcription_not_pasted(pasted: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "transcribe", lambda *_: "")
    hotkey.handle_recording(16_000, np.zeros(16_000, dtype=np.float32))
    assert pasted == []
