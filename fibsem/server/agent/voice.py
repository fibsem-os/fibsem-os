"""Voice input for the operator's message box: a recorded clip in, its words out.

The clip is transcribed on this PC by faster-whisper (``pip install fibsem[voice]``),
so no audio leaves it. The words go back to the page's message box; nothing is
sent to the agent until the operator presses Send.
"""

import base64
import io
import logging
import os
import threading
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fibsem.server.auth import refusal_error, require_operator

logger = logging.getLogger(__name__)

# Words Whisper would otherwise mishear, given as the start of the conversation.
VOCABULARY = (
    "FIB, SEM, FIB-SEM, lamella, lamellae, cryo, milling, polishing, trench, "
    "fiducial, coincidence, autofocus, AutoLamella, grid, stage, eucentric, "
    "fluorescence, z-stack, platinum, GIS, sputter."
)
# At most a minute or so of compressed speech.
CLIP_LIMIT = 4 * 1024 * 1024

_lock = threading.Lock()
_model: Any = None


class TranscribeBody(BaseModel):
    """POST /app/agent/transcribe: a recorded clip, base64, as the browser recorded it."""

    audio: str


class Transcribed(BaseModel):
    text: str
    seconds: float
    model: str


def _model_name() -> str:
    return os.environ.get("FIBSEM_WHISPER_MODEL", "base.en")


def _load() -> Any:
    global _model
    with _lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise refusal_error(
                    501,
                    "voice_not_installed",
                    "Voice input needs faster-whisper: pip install fibsem[voice].",
                )
            try:
                # A model name is downloaded once, then read from the cache;
                # an offline PC names a folder holding the model instead.
                _model = WhisperModel(_model_name(), device="cpu", compute_type="int8")
            except Exception as error:
                logger.warning("Whisper model %s not loaded: %s", _model_name(), error)
                raise refusal_error(
                    503,
                    "voice_model_missing",
                    f"The speech model {_model_name()!r} couldn't be loaded. Set "
                    "FIBSEM_WHISPER_MODEL to a downloaded model's folder on an offline PC.",
                )
        return _model


def transcribe(audio: bytes, prompt: Optional[str] = VOCABULARY) -> str:
    segments, _ = _load().transcribe(
        io.BytesIO(audio), beam_size=1, initial_prompt=prompt, vad_filter=True
    )
    return " ".join(s.text.strip() for s in segments).strip()


def build_voice_router() -> APIRouter:
    router = APIRouter(prefix="/app/agent", dependencies=[Depends(require_operator())])

    @router.post("/transcribe", response_model=Transcribed)
    def transcribe_clip(body: TranscribeBody):
        """The words in a recorded clip, for the message box."""
        try:
            audio = base64.b64decode(body.audio, validate=True)
        except ValueError:
            raise refusal_error(422, "invalid", "Pass 'audio' as base64.")
        if not audio:
            raise refusal_error(422, "missing_field", "The clip is empty.")
        if len(audio) > CLIP_LIMIT:
            raise refusal_error(422, "too_long", "The clip is too long.")
        started = time.monotonic()
        text = transcribe(audio)
        seconds = round(time.monotonic() - started, 2)
        logger.info("Transcribed %d bytes in %.2f s", len(audio), seconds)
        return Transcribed(text=text, seconds=seconds, model=_model_name())

    return router
