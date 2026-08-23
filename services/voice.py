"""Text-to-speech with per-scene duration measurement and script timing sync."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

import httpx
from mutagen.mp3 import MP3

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
# Small pad so text overlays never cut off before audio ends.
SCENE_TAIL_PAD_SEC = float(os.getenv("SCENE_AUDIO_TAIL_PAD", "0.15"))
MIN_SCENE_DURATION = 1.2


def _tts_api_key() -> str | None:
    return os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")


def measure_mp3_duration(audio_bytes: bytes) -> float:
    """Return audio duration in seconds from in-memory bytes using mutagen."""
    try:
        audio = MP3(BytesIO(audio_bytes))
        length = float(audio.info.length or 0.0)
        if length > 0:
            return length
    except Exception as exc:
        logger.debug("mutagen.mp3.MP3 duration read failed: %s, trying mutagen.File", exc)

    try:
        import mutagen
        audio_file = mutagen.File(BytesIO(audio_bytes))
        if audio_file and audio_file.info and getattr(audio_file.info, "length", 0) > 0:
            return float(audio_file.info.length)
    except Exception as exc:
        logger.warning("mutagen.File duration read failed: %s", exc)

    # Fallback estimate based on typical 128kbps MP3 bit rate if header parse fails
    estimated = max(1.5, len(audio_bytes) / (128 * 1024 / 8))
    logger.warning("Using bitrate duration estimation: %.2fs", estimated)
    return estimated


async def synthesize_voice_for_script(script: Script, project_id: str) -> VoiceResult:
    """
    Synthesize per-scene voiceovers, measure exact audio lengths, and rewrite
    script.start/end/duration so Creatomate text overlays match the VO timeline.
    Total video length expands dynamically with the spoken audio.
    """
    provider = (os.getenv("TTS_PROVIDER") or TTS_PROVIDER).strip().lower()
    if not script.scenes:
        raise ValueError("Script has no scenes to synthesize")

    scene_audio: list[bytes] = []
    measured: list[float] = []

    for scene in script.scenes:
        text = (scene.text or "").strip()
        if not text:
            text = "Learn more today."
        audio_bytes = await _synthesize_text(provider, text)
        duration = measure_mp3_duration(audio_bytes)
        duration = max(duration + SCENE_TAIL_PAD_SEC, MIN_SCENE_DURATION)
        scene_audio.append(audio_bytes)
        measured.append(round(duration, 3))
        logger.info(
            "Scene %s (%s) audio duration=%.3fs (text_len=%d)",
            scene.id[:8],
            scene.role,
            duration,
            len(text),
        )

    # Rewrite scene timeline to match real spoken lengths (may exceed 15s).
    cursor = 0.0
    for scene, duration in zip(script.scenes, measured):
        scene.start = round(cursor, 3)
        scene.end = round(cursor + duration, 3)
        cursor += duration
    script.duration = round(cursor, 3)
    logger.info("Synced script timeline to voiceover total_duration=%.3fs", script.duration)

    # Stitch per-scene MP3s into one continuous voiceover for Creatomate.
    full_audio = await _stitch_mp3s(scene_audio)
    object_path = f"media/audio/{project_id}/voiceover_{uuid.uuid4().hex[:10]}.mp3"
    audio_url = await upload_bytes(object_path, full_audio, "audio/mpeg")
    logger.info("Uploaded synced voiceover -> %s (%d bytes)", object_path, len(full_audio))

    segments: List[VoiceSegment] = []
    for scene, duration in zip(script.scenes, measured):
        segments.append(
            VoiceSegment(
                scene_id=scene.id,
                audio_url=audio_url,
                duration=duration,
                words=_approx_word_timestamps(scene.text, duration),
            )
        )

    return VoiceResult(
        segments=segments,
        total_duration=script.duration,
        full_audio_url=audio_url,
    )


async def _synthesize_text(provider: str, text: str) -> bytes:
    if provider == "elevenlabs":
        try:
            return await _elevenlabs_tts(text)
        except Exception as exc:
            if not ALLOW_TTS_FALLBACK:
                raise
            logger.error(
                "ElevenLabs TTS failed (%s). Falling back to edge-tts. "
                "Replace TTS_API_KEY with a valid ElevenLabs key.",
                exc,
            )
            return await _edge_tts(text)
    if provider in {"edge", "edge-tts"}:
        return await _edge_tts(text)
    raise RuntimeError(f"Unsupported TTS_PROVIDER={provider!r}. Use elevenlabs or edge.")


async def _stitch_mp3s(parts: list[bytes]) -> bytes:
    """Concatenate MP3 segments with ffmpeg into a single stream."""
    if len(parts) == 1:
        return parts[0]

    with tempfile.TemporaryDirectory(prefix="ad_voice_") as tmp:
        tmp_path = Path(tmp)
        list_file = tmp_path / "concat.txt"
        inputs: list[Path] = []
        for idx, blob in enumerate(parts):
            path = tmp_path / f"part_{idx:02d}.mp3"
            path.write_bytes(blob)
            inputs.append(path)
        list_file.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in inputs),
            encoding="utf-8",
        )
        output = tmp_path / "full.mp3"
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0 or not output.is_file():
            # Re-encode fallback when stream copy fails across encoder variants.
            command = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(output),
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0 or not output.is_file():
                raise RuntimeError(
                    f"Failed to stitch scene voiceovers: {stderr.decode(errors='replace')[-400:]}"
                )
        return output.read_bytes()


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
