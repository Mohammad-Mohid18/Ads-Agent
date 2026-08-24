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
from services.visual_sanitizer import (
    clean_asset_url,
    sanitize_audio_url,
    sanitize_image_list,
)

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
    raw_images = _collect_image_urls(script, assets)
    prompts = [scene.visual_prompt for scene in script.scenes]
    image_urls = await sanitize_image_list(
        raw_images,
        prompts,
        aspect_ratio=project.aspect_ratio or "16:9",
        category=script.business_category,
    )
    # Persist sanitized public URLs back onto the script for status/debug.
    for scene, url in zip(script.scenes, image_urls):
        scene.image_url = url

    raw_voice = voice.full_audio_url or (voice.segments[0].audio_url if voice.segments else None)
    voice_url = await sanitize_audio_url(raw_voice)
    voice.full_audio_url = voice_url
    for segment in voice.segments:
        segment.audio_url = voice_url

    if CREATOMATE_RENDER_MODE == "template" and CREATOMATE_TEMPLATE_ID:
        body = _build_template_payload(script, image_urls, voice_url)
    else:
        body = _build_renderscript_payload(project, script, image_urls, voice_url)

    project.layers = {
        "mode": "template" if body.get("template_id") else "renderscript",
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
        "supabase_url": clean_asset_url(stored_url) or stored_url,
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
    """Creatomate template_id + modifications schema for multi-scene rendering."""
    modifications: dict[str, str] = {
        "Voiceover": voice_url,
    }
    # Map scene text/images onto Text-N / Image-N layers when present in the template.
    for idx, scene in enumerate(script.scenes):
        n = idx + 1
        modifications[f"Text-{n}"] = scene.text
        modifications[f"Text-{n}.font_size"] = "44 px"
        modifications[f"Text-{n}.font_weight"] = "700"
        modifications[f"Text-{n}.shadow_color"] = "rgba(0,0,0,0.85)"
        modifications[f"Text-{n}.shadow_blur"] = "14 px"
        if idx < len(image_urls):
            modifications[f"Image-{n}"] = image_urls[idx]

    if script.scenes:
        modifications["Text-1"] = script.scenes[0].text
        modifications["Text-1.font_size"] = "44 px"
        if len(script.scenes) > 1:
            modifications["Text-2"] = script.scenes[-1].text
            modifications["Text-2.font_size"] = "40 px"
    return {
        "template_id": CREATOMATE_TEMPLATE_ID,
        "modifications": modifications,
    }


def _get_ken_burns_animation(idx: int, duration: float) -> list[dict[str, Any]]:
    """
    Generate alternating Ken Burns zoom and pan motion effects per scene
    to turn static images into dynamic cinematic footage.
    """
    motion_presets = [
        # Scene 1 (Hook): Smooth Zoom In from center
        {"start_scale": "100%", "end_scale": "116%", "x_anchor": "50%", "y_anchor": "50%"},
        # Scene 2 (Problem): Slow Zoom Out with slight pan right
        {"start_scale": "118%", "end_scale": "103%", "x_anchor": "40%", "y_anchor": "50%"},
        # Scene 3 (Solution): Dynamic Zoom In with focal point
        {"start_scale": "104%", "end_scale": "118%", "x_anchor": "60%", "y_anchor": "45%"},
        # Scene 4 (Benefit): Smooth Zoom Out
        {"start_scale": "116%", "end_scale": "102%", "x_anchor": "50%", "y_anchor": "55%"},
        # Scene 5 (CTA): Cinematic Slow Push In
        {"start_scale": "100%", "end_scale": "115%", "x_anchor": "50%", "y_anchor": "50%"},
    ]
    preset = motion_presets[idx % len(motion_presets)]
    return [
        {
            "time": 0,
            "duration": duration,
            "easing": "linear",
            "type": "scale",
            "scope": "element",
            "start_scale": preset["start_scale"],
            "end_scale": preset["end_scale"],
            "x_anchor": preset["x_anchor"],
            "y_anchor": preset["y_anchor"],
            "fade": False,
        }
    ]


def _get_scene_transition(idx: int, duration: float) -> list[dict[str, Any]]:
    """
    Generate dynamic transitions between scene compositions (fade, slide left, slide right, wipe).
    """
    if idx == 0:
        return [
            {
                "time": 0,
                "duration": min(0.4, duration * 0.15),
                "transition": True,
                "type": "fade",
            }
        ]
    
    transition_patterns = [
        {"type": "slide", "direction": "left", "easing": "quadratic-out"},
        {"type": "fade", "easing": "linear"},
        {"type": "slide", "direction": "right", "easing": "quadratic-out"},
        {"type": "slide", "direction": "left", "easing": "quadratic-out"},
    ]
    pattern = transition_patterns[(idx - 1) % len(transition_patterns)]
    trans_obj: dict[str, Any] = {
        "time": 0,
        "duration": min(0.45, duration * 0.18),
        "transition": True,
        "type": pattern["type"],
    }
    if "direction" in pattern:
        trans_obj["direction"] = pattern["direction"]
    if "easing" in pattern:
        trans_obj["easing"] = pattern["easing"]
    return [trans_obj]


def _build_renderscript_payload(
    project: VideoProject,
    script: Script,
    image_urls: list[str],
    voice_url: str,
) -> dict[str, Any]:
    """
    Build high-converting Creatomate RenderScript with Ken Burns motion,
    dynamic scene transitions, animated text entrance, and high-contrast styling.
    """
    is_portrait = project.aspect_ratio in {"9:16", "portrait"}
    width, height = (1080, 1920) if is_portrait else (1920, 1080)
    
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
        image_url = image_urls[idx] if idx < len(image_urls) else image_urls[-1]
        
        # Ken Burns Motion on Image
        image_animations = _get_ken_burns_animation(idx, duration)
        
        # Scene transition on composition
        scene_transitions = _get_scene_transition(idx, duration)

        # Subtitle layout & styling
        text_font_size = "4.8 vmin" if is_portrait else "4.2 vmin"
        text_y = "82%" if is_portrait else "80%"
        text_width = "86%" if is_portrait else "82%"
        
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
                        "animations": image_animations,
                    },
                    {
                        "name": f"Text-{idx + 1}",
                        "type": "text",
                        "track": 2,
                        "text": scene.text,
                        "width": text_width,
                        "height": "22%",
                        "x": "50%",
                        "y": text_y,
                        "x_alignment": "50%",
                        "y_alignment": "50%",
                        "fill_color": "#FFFFFF",
                        "font_family": "Montserrat",
                        "font_weight": "800",
                        "font_size": text_font_size,
                        "line_height": "125%",
                        "background_color": "rgba(10, 15, 30, 0.68)",
                        "background_border_radius": "15%",
                        "background_x_padding": "5%",
                        "background_y_padding": "3.5%",
                        "shadow_color": "rgba(0, 0, 0, 0.85)",
                        "shadow_blur": "2vmin",
                        "shadow_y": "1vmin",
                        "animations": [
                            {
                                "time": 0.1,
                                "duration": min(0.45, duration * 0.22),
                                "easing": "quadratic-out",
                                "type": "text-slide",
                                "direction": "up",
                                "split": "line",
                                "scope": "split-clip",
                            }
                        ],
                    },
                ],
                "animations": scene_transitions,
            }
        )
        cursor += duration

    # Expand composition length to voiceover total
    total_duration = max(cursor, float(script.duration or 0.0), 1.0)
    source = {
        "output_format": "mp4",
        "width": width,
        "height": height,
        "frame_rate": 30,
        "duration": total_duration,
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
    """Perform targeted edits and re-render with flexible layer name mapping."""
    target = req.target_layer.strip()
    new_value = req.new_value.strip()

    # Normalize target strings (e.g. "Image-4", "image_4", "scene_4_image", "scene_4_prompt")
    image_layer_match = re.search(r"(?:image[_-]?|scene[_-]?(\d+)[_-]?(?:image|prompt)?|(\d+))", target, re.IGNORECASE)
    
    # Extract scene index if target matches image pattern
    scene_idx = None
    if image_layer_match:
        digits = [g for g in image_layer_match.groups() if g and g.isdigit()]
        if digits:
            scene_idx = int(digits[0]) - 1

    # 1. Handle Image Edits (e.g., Image-4, scene_4_image, image_4, scene_4_prompt)
    if scene_idx is not None and project.script and 0 <= scene_idx < len(project.script.scenes):
        scene = project.script.scenes[scene_idx]

        # If new_value is an HTTP/HTTPS URL, assign it directly
        if new_value.startswith("http://") or new_value.startswith("https://"):
            scene.image_url = new_value
            logger.info("Updated Scene %d image URL directly -> %s", scene_idx + 1, new_value)
        # Otherwise, treat new_value as a prompt and generate a new visual via fal.ai
        else:
            logger.info("Triggering fal.ai re-generation for Scene %d prompt: %s", scene_idx + 1, new_value)
            scene.visual_prompt = new_value
            from services.visual_sanitizer import generate_fal_fallback_image
            new_image_url = await generate_fal_fallback_image(
                prompt=new_value,
                aspect_ratio=project.aspect_ratio or "16:9"
            )
            scene.image_url = new_image_url

        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    # 2. Handle Script Line Edits (e.g., script_line_1, Text-1, text_1)
    text_match = re.search(r"(?:script_line_|text[_-]?)(\d+)", target, re.IGNORECASE)
    if text_match and project.script:
        idx = int(text_match.group(1)) - 1
        if 0 <= idx < len(project.script.scenes):
            project.script.scenes[idx].text = new_value
            project.version += 1
            storage.save_project(project)
            return await _trigger_render_sim(project)

    # 3. Handle Brand Color Edits
    if target.lower() in {"brand_color", "primary_color", "color"}:
        if project.brand_assets:
            project.brand_assets.primary_color = new_value
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    # 4. Handle Voiceover Edits
    if target.lower().startswith("voiceover"):
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    # If layer doesn't match any supported target format, return clean exception
    raise ValueError(
        f"Unknown target_layer: '{target}'. Supported layer formats include: "
        "'Image-4', 'scene_4_image', 'scene_4_prompt', 'Text-1', 'script_line_1', 'brand_color', or 'voiceover'."
    )


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
