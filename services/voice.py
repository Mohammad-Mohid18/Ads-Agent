"""Text-to-speech via ElevenLabs with Supabase audio storage."""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

import httpx

from models import Script, VoiceResult, VoiceSegment
from services.asset_storage import upload_bytes

logger = logging.getLogger("ai_ad_engine.voice")

TTS_PROVIDER = (os.getenv("TTS_PROVIDER") or "elevenlabs").strip().lower()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
EDGE_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
ALLOW_TTS_FALLBACK = (os.getenv("ALLOW_TTS_FALLBACK") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _tts_api_key() -> str | None:
    return os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")


async def synthesize_voice_for_script(script: Script, project_id: str) -> VoiceResult:
    """Synthesize a full-script voiceover (and per-scene mirrors) for Creatomate."""
    provider = (os.getenv("TTS_PROVIDER") or TTS_PROVIDER).strip().lower()
    full_text = " ".join(scene.text.strip() for scene in script.scenes if scene.text.strip())
    if not full_text:
        raise ValueError("Script has no text to synthesize")

    if provider == "elevenlabs":
        try:
            audio_bytes = await _elevenlabs_tts(full_text)
        except Exception as exc:
            if not ALLOW_TTS_FALLBACK:
                raise
            logger.error(
                "ElevenLabs TTS failed (%s). Falling back to edge-tts so the pipeline can finish. "
                "Replace TTS_API_KEY with a valid ElevenLabs key.",
                exc,
            )
            audio_bytes = await _edge_tts(full_text)
    elif provider in {"edge", "edge-tts"}:
        audio_bytes = await _edge_tts(full_text)
    else:
        raise RuntimeError(
            f"Unsupported TTS_PROVIDER={provider!r}. Use elevenlabs or edge."
        )

    object_path = f"media/audio/{project_id}/voiceover_{uuid.uuid4().hex[:10]}.mp3"
    audio_url = await upload_bytes(object_path, audio_bytes, "audio/mpeg")
    logger.info("Uploaded voiceover -> %s (%d bytes)", object_path, len(audio_bytes))

    segments: List[VoiceSegment] = []
    total = 0.0
    for scene in script.scenes:
        duration = round(max(scene.end - scene.start, 0.5), 3)
        words = _approx_word_timestamps(scene.text, duration)
        segments.append(
            VoiceSegment(
                scene_id=scene.id,
                audio_url=audio_url,
                duration=duration,
                words=words,
            )
        )
        total += duration

    return VoiceResult(segments=segments, total_duration=total, full_audio_url=audio_url)


async def _elevenlabs_tts(text: str) -> bytes:
    api_key = _tts_api_key()
    if not api_key or api_key in {"local_no_key_required", "changeme"}:
        raise RuntimeError("TTS_API_KEY is missing. Add a valid ElevenLabs API key to .env.")

    url = ELEVENLABS_TTS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, headers=headers, json=body)
        if response.status_code == 401:
            raise RuntimeError(
                "ElevenLabs rejected TTS_API_KEY (401). "
                "Replace TTS_API_KEY in .env with a valid ElevenLabs key (usually starts with sk_)."
            )
        if response.is_error:
            raise RuntimeError(
                f"ElevenLabs TTS failed ({response.status_code}): {response.text[:400]}"
            )
        if not response.content:
            raise RuntimeError("ElevenLabs returned an empty audio payload")
        return response.content


async def _edge_tts(text: str) -> bytes:
    """Neural TTS fallback used only when ElevenLabs is unavailable."""
    import edge_tts

    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        await communicate.save(str(tmp_path))
        data = tmp_path.read_bytes()
        if not data:
            raise RuntimeError("edge-tts produced an empty audio file")
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def _approx_word_timestamps(text: str, duration: float) -> list[dict]:
    tokens = text.split()
    if not tokens:
        return []
    per = duration / len(tokens)
    words = []
    t = 0.0
    for token in tokens:
        words.append({"word": token, "start": round(t, 3), "end": round(t + per, 3)})
        t += per
    return words
