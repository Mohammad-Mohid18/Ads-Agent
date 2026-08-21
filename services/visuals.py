"""Generate per-scene visual assets with fal.ai, then sanitize for Creatomate."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import List, Optional

from models import BrandAssets, Script
from services.asset_storage import download_url_bytes, upload_bytes
from services.visual_sanitizer import (
    STATIC_FALLBACK_IMAGE,
    clean_asset_url,
    generate_fal_fallback_image,
    sanitize_scene_visual,
)

logger = logging.getLogger("ai_ad_engine.visuals")

FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def _ensure_fal_key() -> None:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or FAL_KEY
    if not key:
        raise RuntimeError(
            "FAL_KEY is not set. Add your fal.ai API key to .env to generate scene visuals."
        )
    os.environ["FAL_KEY"] = key


def _brand_suffix(assets: BrandAssets) -> str:
    parts = [
        f"Brand: {assets.site_title}" if assets.site_title else "",
        f"Style color accent {assets.primary_color}" if assets.primary_color else "",
        "cinematic advertising still, high quality, no text overlay, no watermark",
    ]
    return ", ".join(p for p in parts if p)


async def _generate_one(prompt: str, aspect_ratio: str = "16:9") -> str:
    """Call fal.ai and return the remote image URL."""
    return await generate_fal_fallback_image(prompt, aspect_ratio=aspect_ratio)


async def generate_scene_visuals(
    script: Script,
    assets: BrandAssets,
    project_id: str,
    *,
    cache_to_supabase: bool = True,
    aspect_ratio: str = "16:9",
) -> List[str]:
    """
    Resolve one Creatomate-safe image URL per scene.
    Prefers fal.ai when FAL_KEY is set; otherwise validates scraped images and
    auto-heals broken ones via sanitize_scene_visual (fal / static fallback).
    """
    suffix = _brand_suffix(assets)
    urls: List[str] = []
    fal_enabled = bool(os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or FAL_KEY)
    if not fal_enabled:
        logger.warning(
            "FAL_KEY not set — validating scraped brand images and auto-healing failures"
        )

    for idx, scene in enumerate(script.scenes):
        prompt = (scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {scene.id} is missing a required visual_prompt")

        candidate: Optional[str] = None
        if fal_enabled:
            full_prompt = f"{prompt}. {suffix}"
            logger.info("Generating fal.ai visual for scene %s (%s)", idx + 1, scene.role)
            try:
                candidate = await _generate_one(full_prompt, aspect_ratio=aspect_ratio)
            except Exception as exc:
                logger.warning("fal.ai failed for scene %s (%s)", idx + 1, exc)

        if not candidate:
            candidate = _fallback_image(assets, idx)

        # Always sanitize: reject 404s / hotlink blocks / signed tokens / bad MIME types.
        safe_url = await sanitize_scene_visual(
            candidate, prompt, aspect_ratio=aspect_ratio
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
        logger.info("Scene %s image ready: %s", idx + 1, safe_url[:140])

    return urls


def _fallback_image(assets: BrandAssets, idx: int) -> Optional[str]:
    if assets.product_images:
        return assets.product_images[idx % len(assets.product_images)]
    return assets.logo_url
