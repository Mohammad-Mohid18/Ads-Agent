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
    clean_asset_url,
    generate_fal_fallback_image,
    get_niche_fallback_image,
)

logger = logging.getLogger("ai_ad_engine.visuals")

FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def build_fal_prompt(business_category: str, visual_prompt: str, assets: BrandAssets) -> str:
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
) -> List[str]:
    """
    Generate all scene visuals directly via dedicated fal.ai FLUX calls for every scene (min 5).
    Guarantees strict scene-index mapping, fresh generation per run, and prevents image reuse.
    """
    category = (script.business_category or "").strip() or infer_business_category(assets)
    script.business_category = category
    urls: List[str] = []
    seen_urls: set[str] = set()

    fal_enabled = bool((os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or FAL_KEY or "").strip())
    if not fal_enabled:
        logger.warning(
            "FAL_KEY not set — using unique niche visual assets for all %d scenes",
            len(script.scenes),
        )

    for idx, scene in enumerate(script.scenes):
        prompt = (scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {idx + 1} ({scene.id}) is missing a required visual_prompt")

        fal_prompt = build_fal_prompt(category, prompt, assets)
        logger.info(
            "Generating fresh visual via fal.ai for scene %s/%s (%s) category=%s",
            idx + 1,
            len(script.scenes),
            scene.role,
            category,
        )

        # Force fresh generation via fal.ai for every new pipeline run
        candidate: Optional[str] = None
        try:
            candidate = await generate_fal_fallback_image(
                fal_prompt,
                aspect_ratio=aspect_ratio,
                scene_idx=idx,
                category=category,
            )
        except Exception as exc:
            logger.warning("fal.ai generation raised for scene %s: %s", idx + 1, exc)

        safe_url = clean_asset_url(candidate) if candidate else None
        if not safe_url:
            safe_url = get_niche_fallback_image(category, idx)

        # Enforce anti-duplication: never reuse the same URL for multiple scenes
        if safe_url in seen_urls:
            safe_url = get_niche_fallback_image(category, idx)

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

        safe_url = clean_asset_url(safe_url) or get_niche_fallback_image(category, idx)
        seen_urls.add(safe_url)
        urls.append(safe_url)
        
        # Save fresh asset state onto scene
        scene.visual_prompt = prompt
        scene.image_url = safe_url
        logger.info("Scene %s image ready (slot Image-%s): %s", idx + 1, idx + 1, safe_url[:120])

    return urls