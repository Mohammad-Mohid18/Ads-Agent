import logging
import os
import re
import uuid
from typing import List, Optional

from models import BrandAssets, Script
from services.asset_storage import download_url_bytes, upload_bytes
from services.llm_script import format_natural_color_palette, infer_business_category
from services.visual_sanitizer import (
    clean_asset_url,
    generate_image_with_fallbacks,
    get_pollinations_fallback_image,
)

logger = logging.getLogger("ai_ad_engine.visuals")

HF_API_TOKEN = (os.getenv("HF_API_TOKEN") or "").strip().strip("'\"")

# Clean the environment variable explicitly
raw_url = (os.getenv("HF_MODEL_URL") or "").strip().strip('"').strip("'")

# Primary active endpoint via Together provider on HF Router
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL", 
    "https://router.huggingface.co/together/v1/models/black-forest-labs/FLUX.1-schnell"
).strip()

# Explicitly define HF_FALLBACK_URL to fix the import error
HF_FALLBACK_URL = os.getenv(
    "HF_FALLBACK_URL",
    "https://router.huggingface.co/hf-inference/v1/models/stabilityai/stable-diffusion-xl-base-1.0"
).strip()

QUALITY_ANCHORS = ", highly detailed, photorealistic, professional brand photography, 8k resolution"
NEGATIVE_ANCHORS = ", zero text overlay, no typography, no watermark, no logo text"


def build_image_prompt(
    prompt: str,
    assets: Optional[BrandAssets] = None,
    business_category: Optional[str] = None,
) -> str:
    """
    Format and anchor the LLM-generated scene image prompt with natural language colors,
    brand aesthetics, photorealism quality, and zero-text constraints.
    """
    base_prompt = (prompt or "").strip()
    brand = (assets.site_title if assets else None) or "the brand"
    niche = business_category or (assets.business_niche if assets else None) or "technology & modern services"

    # Eliminate any generic fallback terms
    base_prompt = base_prompt.replace("Commercial Business & Services", niche).replace("Commercial Business", niche)

    if not base_prompt or len(base_prompt) < 15:
        color_phrase = format_natural_color_palette(assets.brand_colors if assets else [], assets.primary_color if assets else None)
        base_prompt = f"A sleek modern {niche} workspace for {brand} featuring software and tech solutions, illuminated by {color_phrase}, cinematic wide shot, clean aesthetic"

    # Translate any raw hex color codes in prompt to natural language descriptions
    hex_codes = re.findall(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b", base_prompt)
    if hex_codes:
        natural_color_phrase = format_natural_color_palette(hex_codes)
        base_prompt = re.sub(r"(?:with\s+)?brand\s+color\s+palette\s*\([^\)]*\)", natural_color_phrase, base_prompt, flags=re.IGNORECASE)
        base_prompt = re.sub(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})", "", base_prompt)

    # Append quality & aspect ratio anchors
    if "photorealistic" not in base_prompt.lower() or "8k" not in base_prompt.lower():
        base_prompt = f"{base_prompt}{QUALITY_ANCHORS}"

    # Append negative text constraints
    if "no text" not in base_prompt.lower() and "zero text" not in base_prompt.lower():
        base_prompt = f"{base_prompt}{NEGATIVE_ANCHORS}"

    # Clean punctuation and double spaces
    base_prompt = re.sub(r"\s*,\s*", ", ", base_prompt)
    base_prompt = re.sub(r"\s+", " ", base_prompt).strip(" ,")
    return base_prompt


async def generate_single_visual(
    prompt: str,
    *,
    scene_idx: int = 0,
    aspect_ratio: str = "16:9",
    category: Optional[str] = None,
    project_id: Optional[str] = None,
    cache_to_supabase: bool = True,
    assets: Optional[BrandAssets] = None,
) -> str:
    """
    Generate a single visual using the LLM-generated image_prompt.
    Directs to Hugging Face Serverless (FLUX.1-schnell / SDXL) with quality anchors,
    and falls back gracefully to Pexels and Pollinations.ai with the same tailored brand prompt.
    """
    full_prompt = build_image_prompt(prompt, assets=assets, business_category=category)
    logger.info(
        "Generating visual for scene %s via Hugging Face/fallbacks (aspect=%s): %s",
        scene_idx + 1,
        aspect_ratio,
        full_prompt[:130],
    )

    candidate: Optional[str] = None
    try:
        candidate = await generate_image_with_fallbacks(
            full_prompt,
            aspect_ratio=aspect_ratio,
            scene_idx=scene_idx,
            category=category,
            project_id=project_id,
        )
    except Exception as exc:
        logger.warning("Image generation raised for scene %s: %s", scene_idx + 1, exc)

    safe_url = clean_asset_url(candidate) if candidate else None
    if not safe_url:
        safe_url = get_pollinations_fallback_image(
            full_prompt, scene_idx=scene_idx, aspect_ratio=aspect_ratio
        )

    # Cache freshly generated image to Supabase if requested
    if cache_to_supabase and safe_url.startswith("http") and "supabase.co" not in safe_url:
        try:
            data = await download_url_bytes(safe_url)
            content_type = "image/jpeg"
            lower = safe_url.lower()
            if ".png" in lower:
                content_type = "image/png"
            elif ".webp" in lower:
                content_type = "image/webp"
            object_path = (
                f"media/images/{project_id or 'generated'}/scene_{scene_idx + 1}_{uuid.uuid4().hex[:8]}.jpg"
            )
            safe_url = await upload_bytes(object_path, data, content_type)
        except Exception as exc:
            logger.warning(
                "Supabase image cache failed for scene %s; using source URL: %s",
                scene_idx + 1,
                exc,
            )

    return clean_asset_url(safe_url) or get_pollinations_fallback_image(
        full_prompt, scene_idx=scene_idx, aspect_ratio=aspect_ratio
    )


async def generate_scene_visuals(
    script: Script,
    assets: BrandAssets,
    project_id: str,
    *,
    cache_to_supabase: bool = True,
    aspect_ratio: str = "16:9",
    required_images: int | None = None,
) -> List[str]:
    """
    Generate the exact number of requested scene visuals using LLM-tailored brand prompts.
    """
    image_count = len(script.scenes) if required_images is None else required_images
    if image_count < 1:
        raise ValueError("required_images must be at least 1")
    if len(script.scenes) < image_count:
        raise ValueError(
            f"Script has {len(script.scenes)} scenes but {image_count} images were requested"
        )

    category = (script.business_category or "").strip() or infer_business_category(assets)
    script.business_category = category
    urls: List[str] = []
    seen_urls: set[str] = set()

    for idx, scene in enumerate(script.scenes[:image_count]):
        prompt = (scene.image_prompt or scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {idx + 1} ({scene.id}) is missing a required image_prompt/visual_prompt")

        safe_url = await generate_single_visual(
            prompt,
            scene_idx=idx,
            aspect_ratio=aspect_ratio,
            category=category,
            project_id=project_id,
            cache_to_supabase=cache_to_supabase,
            assets=assets,
        )

        # Enforce anti-duplication: never reuse the exact same URL for multiple scenes
        if safe_url in seen_urls:
            full_prompt = build_image_prompt(prompt, assets=assets, business_category=category)
            safe_url = get_pollinations_fallback_image(
                full_prompt, scene_idx=idx + 10, aspect_ratio=aspect_ratio
            )

        seen_urls.add(safe_url)
        urls.append(safe_url)

        # Save fresh asset state onto scene
        scene.visual_prompt = prompt
        scene.image_prompt = prompt
        scene.image_url = safe_url
        logger.info("Scene %s image ready (slot Image-%s): %s", idx + 1, idx + 1, safe_url[:120])

    return urls

