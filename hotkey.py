"""Push-to-talk dictation: hold a key, speak, release to paste the text at the cursor.

Run with ``uv run hotkey.py``. Works in any application: the transcription is put
on the clipboard and pasted with a simulated Ctrl+V (Cmd+V on macOS). The key is
right Ctrl by default; set ``DICTEE_HOTKEY`` to another pynput key name (e.g. ``alt_r``).
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import numpy.typing as npt
import pyperclip
import sounddevice as sd
from loguru import logger
from pynput import keyboard

import app

HOTKEY = keyboard.Key[os.environ.get("DICTEE_HOTKEY", "ctrl_r")]
PASTE_MODIFIER = keyboard.Key.cmd if sys.platform == "darwin" else keyboard.Key.ctrl
MODEL = app.PARAKEET


class Recorder:
    """Records the default microphone between ``start()`` and ``stop()``."""

    def __init__(self) -> None:
        self.sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])
        self._chunks: list[npt.NDArray[np.float32]] = []
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        """Start recording; ignored if already recording (key auto-repeat)."""
        if self._stream is not None:
            return
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._callback
        )
        self._stream.start()

    def _callback(self, indata: npt.NDArray[np.float32], frames: int, time_info: object, status: object) -> None:
        self._chunks.append(indata[:, 0].copy())

    def stop(self) -> npt.NDArray[np.float32]:
        """Stop recording.

        Returns:
            The recorded samples at ``sample_rate``, empty if nothing was recorded.
        """
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)


def paste(text: str) -> None:
    """Paste text into the focused application via the clipboard.

    Args:
        text: Text to paste.
    """
    pyperclip.copy(text)
    kb = keyboard.Controller()
    with kb.pressed(PASTE_MODIFIER):
        kb.tap("v")


def handle_recording(sample_rate: int, audio: npt.NDArray[np.float32]) -> None:
    """Transcribe a recording and paste the result.

    Args:
        sample_rate: Sample rate of ``audio`` in Hz.
        audio: Recorded samples.
    """
    audio = app.to_mono_16k(sample_rate, audio)
    duration = len(audio) / app.SAMPLE_RATE
    if duration < app.MIN_DURATION_S:
        logger.info("Recording too short ({:.1f} s)", duration)
        return
    start = time.perf_counter()
    text = app.transcribe(MODEL, audio)
    logger.info("{} · {:.1f} s audio · {:.1f} s", app.SHORT_NAMES[MODEL], duration, time.perf_counter() - start)
    if text:
        paste(text)


def _worker(jobs: queue.Queue[tuple[int, npt.NDArray[np.float32]]]) -> None:
    # Transcription runs off the listener thread: a slow keyboard callback would lag all typing.
    while True:
        sample_rate, audio = jobs.get()
        try:
            handle_recording(sample_rate, audio)
        except Exception:
            logger.exception("Dictation failed")


def main() -> None:
    """Load the model and listen for the hotkey until Ctrl+C."""
    app.load_model(MODEL)
    recorder = Recorder()
    jobs: queue.Queue[tuple[int, npt.NDArray[np.float32]]] = queue.Queue()
    threading.Thread(target=_worker, args=(jobs,), daemon=True).start()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == HOTKEY:
            recorder.start()

    def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == HOTKEY:
            jobs.put((recorder.sample_rate, recorder.stop()))

    logger.info("Ready. Hold {} to dictate, Ctrl+C to quit.", HOTKEY.name)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
