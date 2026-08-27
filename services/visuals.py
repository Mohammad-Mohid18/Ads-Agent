"""Generate per-scene visuals through Hugging Face with resilient fallbacks."""
from __future__ import annotations

import logging
import os
import uuid
from typing import List, Optional

from models import BrandAssets, Script
from services.asset_storage import download_url_bytes, upload_bytes
from services.llm_script import infer_business_category
from services.visual_sanitizer import (
    clean_asset_url,
    generate_image_with_fallbacks,
    get_pollinations_fallback_image,
)

logger = logging.getLogger("ai_ad_engine.visuals")

HF_API_TOKEN = (os.getenv("HF_API_TOKEN") or "").strip()


def build_image_prompt(business_category: str, visual_prompt: str, assets: BrandAssets) -> str:
    """
    Combine niche, brand identity, and scene visual prompt with quality-boosting commercial photography keywords.
    """
    brand = assets.site_title or "the brand"
    color = f", brand accent {assets.primary_color}" if assets.primary_color else ""
    return (
        f"Professional high-end commercial advertising photography for {brand} ({business_category}): "
        f"{visual_prompt}{color}. "
        f"Cinematic lighting, high-end commercial aesthetic, hyperrealistic, 8k resolution, "
        f"vibrant color grading, professional cinematography, masterwork advertising still, "
        f"no text overlay, no typography, no watermark, no logo text"
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
    Generate the exact number of requested scene visuals. When omitted, one visual
    is generated for every script scene for backwards compatibility.
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

    if not HF_API_TOKEN:
        logger.warning(
            "HF_API_TOKEN not set — using Pexels and Pollinations fallbacks for all %d scenes",
            image_count,
        )

    for idx, scene in enumerate(script.scenes[:image_count]):
        prompt = (scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {idx + 1} ({scene.id}) is missing a required visual_prompt")

        image_prompt = build_image_prompt(category, prompt, assets)
        logger.info(
            "Generating visual via Hugging Face for scene %s/%s (%s) category=%s",
            idx + 1,
            len(script.scenes),
            scene.role,
            category,
        )

        # The generator logs its Hugging Face/Pexels/Pollinations provider choice.
        candidate: Optional[str] = None
        try:
            candidate = await generate_image_with_fallbacks(
                image_prompt,
                aspect_ratio=aspect_ratio,
                scene_idx=idx,
                category=category,
                project_id=project_id,
            )
        except Exception as exc:
            logger.warning("Image generation raised for scene %s: %s", idx + 1, exc)

        safe_url = clean_asset_url(candidate) if candidate else None
        if not safe_url:
            safe_url = get_pollinations_fallback_image(image_prompt, scene_idx=idx)

        # Enforce anti-duplication: never reuse the same URL for multiple scenes
        if safe_url in seen_urls:
            safe_url = get_pollinations_fallback_image(image_prompt, scene_idx=idx)

        # Cache freshly generated image to Supabase with unique scene-specific path
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
                logger.warning("Supabase image cache failed for scene %s; using source URL: %s", idx + 1, exc)

        safe_url = clean_asset_url(safe_url) or get_pollinations_fallback_image(image_prompt, scene_idx=idx)
        seen_urls.add(safe_url)
        urls.append(safe_url)
        
        # Save fresh asset state onto scene
        scene.visual_prompt = prompt
        scene.image_url = safe_url
        logger.info("Scene %s image ready (slot Image-%s): %s", idx + 1, idx + 1, safe_url[:120])

    return urls
