"""Generate per-scene visual assets via fal.ai using business-category context."""
from __future__ import annotations

import logging
import os
import uuid
from typing import List, Optional

from models import BrandAssets, Script
from services.asset_storage import download_url_bytes, upload_bytes
from services.llm_script import infer_business_category
from services.visual_sanitizer import (
    STATIC_FALLBACK_IMAGE,
    clean_asset_url,
    generate_fal_fallback_image,
    sanitize_scene_visual,
)

logger = logging.getLogger("ai_ad_engine.visuals")

FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def build_fal_prompt(business_category: str, visual_prompt: str, assets: BrandAssets) -> str:
    """Combine niche + scene visual prompt into a commercial fal.ai prompt."""
    brand = assets.site_title or "the brand"
    color = f", brand accent color {assets.primary_color}" if assets.primary_color else ""
    return (
        f"Professional high-end cinematic advertisement photo for a {business_category}: "
        f"{visual_prompt}. Brand: {brand}{color}, 8k resolution, commercial lighting, "
        f"photorealistic, no text overlay, no watermark, no logo typography"
    )


async def generate_scene_visuals(
    script: Script,
    assets: BrandAssets,
    project_id: str,
    *,
    cache_to_supabase: bool = True,
    aspect_ratio: str = "16:9",
) -> List[str]:
    """
    Generate all scene visuals directly via fal.ai (business-category aware).
    Scraped site images are NOT used as primary assets.
    """
    category = (script.business_category or "").strip() or infer_business_category(assets)
    script.business_category = category
    urls: List[str] = []
    fal_enabled = bool(os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or FAL_KEY)
    if not fal_enabled:
        logger.warning(
            "FAL_KEY not set — fal prompts will fall back through sanitizer to static public imagery"
        )

    for idx, scene in enumerate(script.scenes):
        prompt = (scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {scene.id} is missing a required visual_prompt")

        fal_prompt = build_fal_prompt(category, prompt, assets)
        logger.info(
            "Generating fal.ai visual for scene %s (%s) category=%s",
            idx + 1,
            scene.role,
            category,
        )

        candidate: Optional[str] = None
        try:
            candidate = await generate_fal_fallback_image(fal_prompt, aspect_ratio=aspect_ratio)
            # generate_fal_fallback_image returns Unsplash when FAL_KEY missing — still sanitize.
        except Exception as exc:
            logger.warning("fal.ai generation raised for scene %s: %s", idx + 1, exc)

        # Validate / heal (never ship broken URLs to Creatomate).
        safe_url = await sanitize_scene_visual(
            candidate, fal_prompt, aspect_ratio=aspect_ratio
        )

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
                    f"media/images/{project_id}/scene_{idx + 1}_{uuid.uuid4().hex[:8]}.jpg"
                )
                safe_url = await upload_bytes(object_path, data, content_type)
            except Exception as exc:
                logger.warning("Supabase image cache failed; using source URL: %s", exc)

        safe_url = clean_asset_url(safe_url) or STATIC_FALLBACK_IMAGE
        urls.append(safe_url)
        scene.visual_prompt = prompt
        scene.image_url = safe_url
        logger.info("Scene %s fal image ready: %s", idx + 1, safe_url[:140])

    return urls
