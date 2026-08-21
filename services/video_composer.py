"""Build Creatomate render payloads and persist finished MP4s to Supabase."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from models import BrandAssets, EditRequest, Script, Storage, VideoProject, VoiceResult
from services.asset_storage import download_url_bytes, upload_bytes

logger = logging.getLogger("ai_ad_engine.video")

VIDEO_API_KEY = os.getenv("CREATOMATE_API_KEY")
VIDEO_API_URL = os.getenv("CREATOMATE_URL", "https://api.creatomate.com/v1/renders").rstrip("/")
CREATOMATE_TEMPLATE_ID = (os.getenv("CREATOMATE_TEMPLATE_ID") or "").strip()
# template | renderscript  — RenderScript is used for multi-scene image+VO ads.
CREATOMATE_RENDER_MODE = (os.getenv("CREATOMATE_RENDER_MODE") or "renderscript").strip().lower()
RENDER_DIR = Path(__file__).resolve().parent.parent / "data" / "session_renders"
POLL_INTERVAL_SEC = float(os.getenv("CREATOMATE_POLL_INTERVAL", "2.5"))
POLL_TIMEOUT_SEC = float(os.getenv("CREATOMATE_POLL_TIMEOUT", "300"))


def _normalize_renders_url(url: str) -> str:
    """Force the official Creatomate v1 renders endpoint."""
    if not url:
        return "https://api.creatomate.com/v1/renders"
    # Common misconfigs: /v2/renders, /execute, trailing paths
    if "creatomate.com" in url:
        return "https://api.creatomate.com/v1/renders"
    return url


VIDEO_API_URL = _normalize_renders_url(VIDEO_API_URL)


def _normalize_template_id(value: str) -> str:
    """Strip accidental trailing characters so the ID is a valid UUID."""
    value = (value or "").strip()
    uuid_re = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    match = uuid_re.match(value)
    return match.group(0) if match else value


CREATOMATE_TEMPLATE_ID = _normalize_template_id(CREATOMATE_TEMPLATE_ID)


async def build_and_render_video(
    project: VideoProject,
    assets: BrandAssets,
    script: Script,
    voice: VoiceResult,
) -> str:
    """Submit a Creatomate render, wait for completion, upload MP4 to Supabase."""
    image_urls = _collect_image_urls(script, assets)
    voice_url = voice.full_audio_url or (voice.segments[0].audio_url if voice.segments else None)
    if not voice_url or not voice_url.startswith("https://"):
        raise RuntimeError("Voiceover URL must be a public HTTPS URL for Creatomate")

    for idx, url in enumerate(image_urls):
        if not url or not str(url).startswith("https://"):
            raise RuntimeError(f"Scene image {idx + 1} must be a public HTTPS URL for Creatomate")

    if CREATOMATE_RENDER_MODE == "template" and CREATOMATE_TEMPLATE_ID:
        body = _build_template_payload(script, image_urls, voice_url)
    else:
        body = _build_renderscript_payload(project, script, image_urls, voice_url)

    project.layers = {
        "mode": body.get("template_id") and "template" or "renderscript",
        "modifications": body.get("modifications"),
        "image_urls": image_urls,
        "voiceover_url": voice_url,
        "source": body.get("source"),
    }

    if not VIDEO_API_KEY:
        preview = await _render_local_video(project, script)
        logger.info("Local video render for project %s -> %s", project.id, preview)
        return preview

    headers = {
        "Authorization": f"Bearer {VIDEO_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VIDEO_API_URL, headers=headers, json=body)
        if response.is_error:
            raise RuntimeError(
                f"Creatomate render request failed ({response.status_code}): {response.text[:500]}"
            )
        payload = response.json()
        render = payload[0] if isinstance(payload, list) else payload
        render_id = render.get("id")
        if not render_id:
            raise RuntimeError(f"Creatomate response missing render id: {payload}")
        logger.info("Creatomate render queued id=%s status=%s", render_id, render.get("status"))

        final = await _poll_render(client, headers, render_id)
        remote_url = final.get("url")
        if not remote_url:
            raise RuntimeError(f"Creatomate render succeeded without url: {final}")

    # Download finished MP4 and store in Supabase (streamed via GET /media/renders/...).
    video_bytes = await download_url_bytes(remote_url)
    object_path = f"renders/{project.id}_v{project.version}.mp4"
    stored_url = await upload_bytes(object_path, video_bytes, "video/mp4")
    preview = f"/media/{object_path}"
    project.layers = {
        **(project.layers or {}),
        "creatomate_render_id": render_id,
        "creatomate_url": remote_url,
        "supabase_url": stored_url,
        "storage_path": object_path,
    }
    logger.info("Creatomate render saved to Supabase %s", object_path)
    return preview


def _collect_image_urls(script: Script, assets: BrandAssets) -> list[str]:
    urls: list[str] = []
    for idx, scene in enumerate(script.scenes):
        url = scene.image_url
        if not url:
            if assets.product_images:
                url = assets.product_images[idx % len(assets.product_images)]
            else:
                url = assets.logo_url
        if not url:
            raise RuntimeError(f"No image URL available for scene {idx + 1}")
        urls.append(url)
    return urls


def _build_template_payload(script: Script, image_urls: list[str], voice_url: str) -> dict[str, Any]:
    """Creatomate template_id + modifications schema."""
    modifications: dict[str, str] = {
        "Voiceover": voice_url,
    }
    # Map scene text/images onto Text-N / Image-N layers when present in the template.
    for idx, scene in enumerate(script.scenes):
        n = idx + 1
        modifications[f"Text-{n}"] = scene.text
        if idx < len(image_urls):
            modifications[f"Image-{n}"] = image_urls[idx]
    # Always fill Text-1 with the hook / concatenated headline for templates that only have Text-1/2.
    if script.scenes:
        modifications["Text-1"] = script.scenes[0].text
        if len(script.scenes) > 1:
            modifications["Text-2"] = script.scenes[-1].text
    return {
        "template_id": CREATOMATE_TEMPLATE_ID,
        "modifications": modifications,
    }


def _build_renderscript_payload(
    project: VideoProject,
    script: Script,
    image_urls: list[str],
    voice_url: str,
) -> dict[str, Any]:
    """
    Custom RenderScript with multi-scene durations, images, text, and voiceover.
    This avoids single-frame static output from empty/invalid template payloads.
    """
    width, height = (1080, 1920) if project.aspect_ratio == "9:16" else (1920, 1080)
    elements: list[dict[str, Any]] = [
        {
            "name": "Voiceover",
            "type": "audio",
            "track": 1,
            "time": 0,
            "source": voice_url,
            "duration": None,
        }
    ]

    cursor = 0.0
    for idx, scene in enumerate(script.scenes):
        duration = max(float(scene.end - scene.start), 1.5)
        image_url = image_urls[idx]
        elements.append(
            {
                "name": f"Scene-{idx + 1}",
                "type": "composition",
                "track": 2,
                "time": cursor,
                "duration": duration,
                "width": "100%",
                "height": "100%",
                "elements": [
                    {
                        "name": f"Image-{idx + 1}",
                        "type": "image",
                        "track": 1,
                        "source": image_url,
                        "width": "100%",
                        "height": "100%",
                        "fit": "cover",
                        "animations": [
                            {
                                "time": 0,
                                "duration": duration,
                                "easing": "linear",
                                "type": "scale",
                                "scope": "element",
                                "start_scale": "100%",
                                "end_scale": "112%",
                                "fade": False,
                            }
                        ],
                    },
                    {
                        "name": f"Text-{idx + 1}",
                        "type": "text",
                        "track": 2,
                        "text": scene.text,
                        "width": "86%",
                        "height": "40%",
                        "x": "50%",
                        "y": "78%",
                        "x_alignment": "50%",
                        "y_alignment": "50%",
                        "fill_color": "#FFFFFF",
                        "font_family": "Montserrat",
                        "font_weight": "700",
                        "font_size": "6.2 vmin",
                        "background_color": "rgba(0,0,0,0.45)",
                        "background_x_padding": "6%",
                        "background_y_padding": "4%",
                        "animations": [
                            {
                                "time": 0,
                                "duration": 0.45,
                                "easing": "quadratic-out",
                                "type": "text-slide",
                                "direction": "up",
                                "split": "line",
                            }
                        ],
                    },
                ],
                "animations": [
                    {
                        "time": 0,
                        "duration": 0.35,
                        "transition": True,
                        "type": "fade",
                    }
                ],
            }
        )
        cursor += duration

    source = {
        "output_format": "mp4",
        "width": width,
        "height": height,
        "frame_rate": 30,
        "duration": cursor,
        "elements": elements,
    }
    return {"source": source}


async def _poll_render(
    client: httpx.AsyncClient, headers: dict[str, str], render_id: str
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT_SEC
    status_url = f"https://api.creatomate.com/v1/renders/{render_id}"
    while True:
        response = await client.get(status_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        render = payload[0] if isinstance(payload, list) else payload
        status = (render.get("status") or "").lower()
        logger.info("Creatomate poll %s -> %s", render_id, status)
        if status in {"succeeded", "success", "completed"}:
            return render
        if status in {"failed", "error"}:
            raise RuntimeError(
                f"Creatomate render failed: {render.get('error_message') or render}"
            )
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"Creatomate render timed out after {POLL_TIMEOUT_SEC}s")
        await asyncio.sleep(POLL_INTERVAL_SEC)


async def edit_ad_component(project: VideoProject, req: EditRequest, storage: Storage) -> str:
    """Perform targeted edits and re-render."""
    target = req.target_layer
    new_value = req.new_value

    if target.startswith("script_line_"):
        idx = int(target.split("_")[-1]) - 1
        project.script.scenes[idx].text = new_value
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    if target.startswith("scene_") and target.endswith("_image"):
        parts = target.split("_")
        idx = int(parts[1]) - 1
        key = f"scene_{idx + 1}_image"
        if not project.layers:
            project.layers = {}
        project.layers[key] = {"type": "image", "source": new_value}
        if project.script and 0 <= idx < len(project.script.scenes):
            project.script.scenes[idx].image_url = new_value
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    if target == "brand_color":
        if project.brand_assets:
            project.brand_assets.primary_color = new_value
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    if target.startswith("voiceover_"):
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    raise ValueError("Unknown target_layer")


async def _trigger_render_sim(project: VideoProject) -> str:
    if not project.script or not project.voice or not project.brand_assets:
        raise ValueError("Cannot re-render an incomplete project")
    preview = await build_and_render_video(
        project, project.brand_assets, project.script, project.voice
    )
    project.preview_url = preview
    project.status = "ready"
    return preview


async def _render_local_video(project: VideoProject, script: Script) -> str:
    """Create a real MP4 in the temporary session cache (demo fallback)."""
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    width, height = (1080, 1920) if project.aspect_ratio == "9:16" else (1920, 1080)
    color = _safe_color(project.brand_assets.primary_color if project.brand_assets else None)
    filename = f"{project.id}_v{project.version}.mp4"
    output_path = RENDER_DIR / filename
    text_path = RENDER_DIR / f"{project.id}_v{project.version}.txt"
    text_path.write_text("\n\n".join(scene.text for scene in script.scenes), encoding="utf-8")
    font_size = 60 if width == 1080 else 64
    filter_graph = (
        f"drawtext=textfile='{text_path}':fontcolor=white:fontsize={font_size}:"
        "x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=20:"
        "box=1:boxcolor=black@0.45:boxborderw=36"
    )
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:d={max(float(script.duration), 1.0)}",
        "-vf",
        filter_graph,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Local video render failed: {stderr.decode(errors='replace')[-500:]}")
    return f"/media/{filename}"


def _safe_color(value: str | None) -> str:
    return value if value and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else "#1E3A8A"
