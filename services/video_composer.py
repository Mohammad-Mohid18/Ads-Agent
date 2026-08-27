"""Build Creatomate render payloads and persist finished MP4s to Supabase."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from models import BrandAssets, EditRequest, Script, Storage, VideoProject, VoiceResult
from services.asset_storage import download_url_bytes, upload_bytes
from services.llm_script import generate_script
from services.visuals import generate_scene_visuals
from services.voice import synthesize_voice_for_script
from services.visual_sanitizer import (
    clean_asset_url,
    sanitize_audio_url,
)

logger = logging.getLogger("ai_ad_engine.video")

VIDEO_API_KEY = os.getenv("CREATOMATE_API_KEY")
VIDEO_API_URL = os.getenv("CREATOMATE_URL", "https://api.creatomate.com/v1/renders").rstrip("/")
CREATOMATE_TEMPLATE_ID = (os.getenv("CREATOMATE_TEMPLATE_ID") or "").strip()
# template | renderscript — RenderScript is used for multi-scene image+VO ads.
CREATOMATE_RENDER_MODE = (os.getenv("CREATOMATE_RENDER_MODE") or "renderscript").strip().lower()
RENDER_DIR = Path(__file__).resolve().parent.parent / "data" / "session_renders"
POLL_INTERVAL_SEC = float(os.getenv("CREATOMATE_POLL_INTERVAL", "2.5"))
POLL_TIMEOUT_SEC = float(os.getenv("CREATOMATE_POLL_TIMEOUT", "300"))


# ==============================================================================
# CREATOMATE TEMPLATE REGISTRY
# ==============================================================================

TEMPLATES = {
    "service_local": {
        "id": "d8110d66-798e-432e-b744-3818a54ff3da",
        "name": "Service / Local Business",
        "required_scenes": 2,
        "required_images": 1,
    },
    "news_showcase": {
        "id": "0c177ed9-ee0b-46fb-b26b-36737d8cc738",
        "name": "News / Multi-Scene Highlight",
        "required_scenes": 3,
        "required_images": 3,
    },
    "product": {
        "id": "a248f122-6ecc-4869-9782-75f90890517e",
        "name": "Product / E-commerce",
        "required_scenes": 2,
        "required_images": 1,
    },
}

DEFAULT_TEMPLATE_ID = TEMPLATES["service_local"]["id"]


def _normalize_renders_url(url: str) -> str:
    """Force the official Creatomate v1 renders endpoint."""
    if not url:
        return "https://api.creatomate.com/v1/renders"
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


# CREATOMATE_TEMPLATE_ID = _normalize_template_id(CREATOMATE_TEMPLATE_ID)

# ==============================================================================
# TEMPLATE-SPECIFIC PAYLOAD BUILDERS
# ==============================================================================

def _build_service_local_payload(
    script: Script, 
    image_urls: list[str], 
    voice_url: str
) -> dict[str, Any]:
    """
    Template 1: Service / Local Business (Quick Promo)
    - Layers: Text-1, Text-2, Video, Audio
    """
    text_1_content = script.scenes[0].text if script.scenes else "Discover Our Services"
    text_2_content = " ".join([s.text for s in script.scenes[1:]]) if len(script.scenes) > 1 else text_1_content

    return {
        "Audio": voice_url,
        "Audio.source": voice_url,
        "Voiceover": voice_url,
        "Voiceover.source": voice_url,
        "Video": image_urls[0] if image_urls else "https://creatomate.com/files/assets/7347c3b7-e1a8-4439-96f1-f3dfc95c3d28",
        "Text-1": text_1_content,
        "Text-2": text_2_content,
    }


def _build_news_showcase_payload(
    script: Script, 
    image_urls: list[str], 
    voice_url: str
) -> dict[str, Any]:
    """
    Template 2: News / Multi-Scene Highlight (0c177ed9-ee0b-46fb-b26b-36737d8cc738)
    - Layers: Image-1.source, Image-2.source, Image-3.source,
              Text-1.text, Text-2.text, Text-3.text, Audio
    """
    modifications: dict[str, Any] = {
        "Audio": voice_url,
        "Audio.source": voice_url,
        "Voiceover": voice_url,
        "Voiceover.source": voice_url,
    }

    # Populate 3 image and headline pairs
    for i in range(3):
        n = i + 1
        scene_text = script.scenes[i].text if i < len(script.scenes) else (script.scenes[-1].text if script.scenes else "")
        img_src = image_urls[i] if i < len(image_urls) else (image_urls[-1] if image_urls else "")

        modifications[f"Image-{n}.source"] = img_src
        modifications[f"Text-{n}.text"] = scene_text

    return modifications



def _build_ecommerce_payload(
    script: Script, 
    image_urls: list[str], 
    voice_url: str,
    assets: Optional[BrandAssets] = None
) -> dict[str, Any]:
    """
    Template 3: Product / E-commerce Highlight
    - Layers: Product-Image.source, Product-Name.text, Product-Description.text, 
              Normal-Price.text, Discounted-Price.text, CTA.text, Website.text, Audio
    """
    hook_text = script.scenes[0].text if script.scenes else "Exclusive Offer Available Now"
    cta_text = script.scenes[-1].text if len(script.scenes) > 1 else "Shop Today!"
    site_name = assets.site_title if assets and assets.site_title else "Featured Product"

    return {
        "Audio": voice_url,
        "Audio.source": voice_url,
        "Voiceover": voice_url,
        "Voiceover.source": voice_url,
        "Product-Image.source": image_urls[0] if image_urls else "https://creatomate.com/files/assets/fe61553c-4274-4586-affe-54cffe99ccdc",
        "Product-Name.text": site_name[:30],
        "Product-Description.text": hook_text,
        "Normal-Price.text": "",
        "Discounted-Price.text": "Best Deal",
        "CTA.text": cta_text,
        "Website.text": site_name.lower().replace(" ", "") + ".com"
    }


# ==============================================================================
# MAIN ROUTER PAYLOAD BUILDER
# ==============================================================================

def _build_template_payload(
    script: Script, 
    image_urls: list[str], 
    voice_url: str,
    template_id: Optional[str] = None,
    assets: Optional[BrandAssets] = None
) -> dict[str, Any]:
    """
    Routes render payloads to the exact schema required by each Creatomate template.
    """
    # 1. Resolve active template ID
    active_template = (template_id or "").strip()

    if active_template in TEMPLATES:
        active_template = TEMPLATES[active_template]["id"]
    if not active_template:
        active_template = DEFAULT_TEMPLATE_ID

    # 2. Route to specialized payload generator
    if active_template == TEMPLATES["news_showcase"]["id"]:
        modifications = _build_news_showcase_payload(script, image_urls, voice_url)

    elif active_template == TEMPLATES["product"]["id"]:
        modifications = _build_ecommerce_payload(script, image_urls, voice_url, assets)

    else:
        # Default: Service / Local Business
        modifications = _build_service_local_payload(script, image_urls, voice_url)

    return {
        "template_id": active_template,
        "modifications": modifications,
        # Force-inject voiceover track into templates lacking an Audio layer
        "elements": [
            {
                "type": "audio",
                "source": voice_url,
                "volume": "100%",
            }
        ],
    }


async def render_single_template(
    project: VideoProject,
    assets: BrandAssets,
    script: Script,
    voice_url: str,
    image_urls: list[str],
    template_key: str,
    template_id: str,
) -> tuple[str, str]:
    """Renders a single template and uploads its finished video to Supabase."""
    body = _build_template_payload(
        script=script, 
        image_urls=image_urls, 
        voice_url=voice_url, 
        template_id=template_id, 
        assets=assets
    )

    if not VIDEO_API_KEY:
        preview = await _render_local_video(project, script, template_key=template_key)
        return template_key, preview

    headers = {
        "Authorization": f"Bearer {VIDEO_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(VIDEO_API_URL, headers=headers, json=body)
            if response.is_error:
                logger.error("Template '%s' failed (%d): %s", template_key, response.status_code, response.text[:300])
                return template_key, ""

            payload = response.json()
            render = payload[0] if isinstance(payload, list) else payload
            render_id = render.get("id")

            final = await _poll_render(client, headers, render_id)
            remote_url = final.get("url")

        # Download rendered MP4 and persist to Supabase
        video_bytes = await download_url_bytes(remote_url)
        object_path = f"renders/{project.id}_{template_key}_v{project.version}.mp4"
        stored_url = await upload_bytes(object_path, video_bytes, "video/mp4")

        # Return public direct Supabase URL for immediate playback
        public_url = clean_asset_url(stored_url) or stored_url
        return template_key, public_url

    except Exception as exc:
        logger.exception("Failed rendering template %s: %s", template_key, exc)
        return template_key, ""


async def render_all_templates(
    project: VideoProject,
    assets: BrandAssets,
    script: Script,
    voice: VoiceResult,
) -> dict[str, str]:
    """Generates visual and voice assets once, then renders all 3 templates simultaneously."""
    
    # 1. Prepare visual assets & audio once
    image_urls = await generate_scene_visuals(
        script=script,
        assets=assets,
        project_id=project.id,
        aspect_ratio=project.aspect_ratio or "16:9",
    )

    raw_voice = voice.full_audio_url or (voice.segments[0].audio_url if voice.segments else None)
    voice_url = await sanitize_audio_url(raw_voice)

    # 2. Run all 3 renders in parallel
    tasks = [
        render_single_template(
            project=project,
            assets=assets,
            script=script,
            voice_url=voice_url,
            image_urls=image_urls,
            template_key=t_key,
            template_id=t_info["id"],
        )
        for t_key, t_info in TEMPLATES.items()
    ]

    results = await asyncio.gather(*tasks)
    return {key: url for key, url in results if url}


async def render_templates_sequentially(
    project: VideoProject,
    assets: BrandAssets,
) -> dict[str, str]:
    """Render each template with script, voice, and visuals sized for its slots.

    ``assets`` is intentionally supplied by the caller: site scraping happens once
    before this function and is shared across the sequential template renders.
    """
    storage = Storage()
    project.preview_urls = project.preview_urls or {}

    for template_key, template in TEMPLATES.items():
        try:
            script = await generate_script(
                assets,
                project.aspect_ratio or "16:9",
                required_scenes=template["required_scenes"],
            )
            voice = await synthesize_voice_for_script(script, project_id=project.id)
            image_urls = await generate_scene_visuals(
                script=script,
                assets=assets,
                project_id=project.id,
                aspect_ratio=project.aspect_ratio or "16:9",
                required_images=template["required_images"],
            )
            raw_voice = voice.full_audio_url or (
                voice.segments[0].audio_url if voice.segments else None
            )
            voice_url = await sanitize_audio_url(raw_voice)
            if not voice_url:
                raise RuntimeError("Voiceover synthesis returned no usable audio URL")

            key, preview_url = await render_single_template(
                project=project,
                assets=assets,
                script=script,
                voice_url=voice_url,
                image_urls=image_urls,
                template_key=template_key,
                template_id=template["id"],
            )
            if not preview_url:
                raise RuntimeError("Creatomate did not return a render URL")

            project.script = script
            project.voice = voice
            project.layers = {
                "template_key": template_key,
                "template_id": template["id"],
                "required_scenes": template["required_scenes"],
                "required_images": template["required_images"],
                "image_urls": image_urls,
                "voiceover_url": voice_url,
            }
            project.preview_urls[key] = preview_url
            project.preview_url = project.preview_url or preview_url
            project.error = None
            storage.save_project(project)
            logger.info("Completed template '%s' for project %s", template_key, project.id)
        except Exception as exc:
            # A bad template payload must not prevent the remaining variants.
            logger.exception("Template '%s' failed for project %s: %s", template_key, project.id, exc)

    return project.preview_urls

def _get_ken_burns_animation(idx: int, duration: float) -> list[dict[str, Any]]:
    """Generate alternating Ken Burns motion effects per scene."""
    motion_presets = [
        {"start_scale": "100%", "end_scale": "116%", "x_anchor": "50%", "y_anchor": "50%"},
        {"start_scale": "118%", "end_scale": "103%", "x_anchor": "40%", "y_anchor": "50%"},
        {"start_scale": "104%", "end_scale": "118%", "x_anchor": "60%", "y_anchor": "45%"},
        {"start_scale": "116%", "end_scale": "102%", "x_anchor": "50%", "y_anchor": "55%"},
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
    """Generate dynamic transitions between scene compositions."""
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
    """Build high-converting Creatomate RenderScript."""
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
        
        image_animations = _get_ken_burns_animation(idx, duration)
        scene_transitions = _get_scene_transition(idx, duration)

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

    # Flexible matching for image layers (e.g. Image-4, scene_4_image, image_4, scene_4_prompt)
    image_layer_match = re.search(r"(?:image[_-]?|scene[_-]?(\d+)[_-]?(?:image|prompt)?|(\d+))", target, re.IGNORECASE)
    
    scene_idx = None
    if image_layer_match:
        digits = [g for g in image_layer_match.groups() if g and g.isdigit()]
        if digits:
            scene_idx = int(digits[0]) - 1

    # 1. Image Edits
    if scene_idx is not None and project.script and 0 <= scene_idx < len(project.script.scenes):
        scene = project.script.scenes[scene_idx]

        if new_value.startswith("http://") or new_value.startswith("https://"):
            scene.image_url = new_value
            logger.info("Updated Scene %d image URL directly -> %s", scene_idx + 1, new_value)
        else:
            logger.info("Triggering fal.ai re-generation for Scene %d prompt: %s", scene_idx + 1, new_value)
            scene.visual_prompt = new_value
            from services.visual_sanitizer import generate_fal_fallback_image
            new_image_url = await generate_fal_fallback_image(
                prompt=new_value,
                aspect_ratio=project.aspect_ratio or "16:9",
                scene_idx=scene_idx,
                category=project.script.business_category
            )
            scene.image_url = new_image_url

        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    # 2. Script Line Edits
    text_match = re.search(r"(?:script_line_|text[_-]?)(\d+)", target, re.IGNORECASE)
    if text_match and project.script:
        idx = int(text_match.group(1)) - 1
        if 0 <= idx < len(project.script.scenes):
            project.script.scenes[idx].text = new_value
            project.version += 1
            storage.save_project(project)
            return await _trigger_render_sim(project)

    # 3. Brand Color Edits
    if target.lower() in {"brand_color", "primary_color", "color"}:
        if project.brand_assets:
            project.brand_assets.primary_color = new_value
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    # 4. Voiceover Edits
    if target.lower().startswith("voiceover"):
        project.version += 1
        storage.save_project(project)
        return await _trigger_render_sim(project)

    raise ValueError(
        f"Unknown target_layer: '{target}'. Supported layer formats include: "
        "'Image-4', 'scene_4_image', 'scene_4_prompt', 'Text-1', 'script_line_1', 'brand_color', or 'voiceover'."
    )


async def _trigger_render_sim(project: VideoProject) -> str:
    if not project.script or not project.voice or not project.brand_assets:
        raise ValueError("Cannot re-render an incomplete project")
        
    # Re-render all template variants in parallel
    preview_urls = await render_all_templates(
        project, project.brand_assets, project.script, project.voice
    )
    
    project.preview_urls = preview_urls
    preview = next(iter(preview_urls.values()), "")
    project.preview_url = preview
    project.status = "ready"
    return preview


async def _render_local_video(
    project: VideoProject,
    script: Script,
    *,
    template_key: str = "preview",
) -> str:
    """Create a real MP4 in the temporary session cache (demo fallback)."""
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    width, height = (1080, 1920) if project.aspect_ratio == "9:16" else (1920, 1080)
    color = _safe_color(project.brand_assets.primary_color if project.brand_assets else None)
    filename = f"{project.id}_{template_key}_v{project.version}.mp4"
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
