"""Generate per-scene visual assets with fal.ai (Flux)."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import List, Optional

import httpx

from models import BrandAssets, Script
from services.asset_storage import download_url_bytes, upload_bytes

logger = logging.getLogger("ai_ad_engine.visuals")

FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def _ensure_fal_key() -> None:
    if not FAL_KEY:
        raise RuntimeError(
            "FAL_KEY is not set. Add your fal.ai API key to .env to generate scene visuals."
        )
    os.environ["FAL_KEY"] = FAL_KEY


def _brand_suffix(assets: BrandAssets) -> str:
    parts = [
        f"Brand: {assets.site_title}" if assets.site_title else "",
        f"Style color accent {assets.primary_color}" if assets.primary_color else "",
        "cinematic advertising still, high quality, no text overlay, no watermark",
    ]
    return ", ".join(p for p in parts if p)


async def _generate_one(prompt: str) -> str:
    """Call fal.ai and return the remote image URL."""
    _ensure_fal_key()
    import fal_client  # lazy import so missing package only fails when used

    def _run() -> str:
        result = fal_client.subscribe(
            FAL_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "num_images": 1,
            },
            with_logs=False,
        )
        images = result.get("images") or []
        if not images:
            raise RuntimeError(f"fal.ai returned no images for prompt: {prompt[:80]}")
        url = images[0].get("url")
        if not url:
            raise RuntimeError("fal.ai image payload missing url")
        return url

    return await asyncio.to_thread(_run)


async def generate_scene_visuals(
    script: Script,
    assets: BrandAssets,
    project_id: str,
    *,
    cache_to_supabase: bool = True,
) -> List[str]:
    """
    Generate one image URL per scene from visual_prompt.
    Returns a list aligned with script.scenes order.
    """
    suffix = _brand_suffix(assets)
    urls: List[str] = []
    fal_enabled = bool(FAL_KEY)
    if not fal_enabled:
        logger.warning("FAL_KEY not set — using scraped brand images as scene visuals")

    for idx, scene in enumerate(script.scenes):
        prompt = (scene.visual_prompt or "").strip()
        if not prompt:
            raise ValueError(f"Scene {scene.id} is missing a required visual_prompt")

        fal_url: Optional[str] = None
        if fal_enabled:
            full_prompt = f"{prompt}. {suffix}"
            logger.info("Generating fal.ai visual for scene %s (%s)", idx + 1, scene.role)
            try:
                fal_url = await _generate_one(full_prompt)
            except Exception as exc:
                logger.warning(
                    "fal.ai failed for scene %s (%s); using scraped fallback image",
                    idx + 1,
                    exc,
                )

        if not fal_url:
            fal_url = _fallback_image(assets, idx)
            if not fal_url:
                raise RuntimeError(
                    "No fal.ai image and no scraped product/logo image available for visuals"
                )

        if cache_to_supabase and fal_url.startswith("http"):
            try:
                data = await download_url_bytes(fal_url)
                content_type = "image/jpeg"
                if ".png" in fal_url.lower():
                    content_type = "image/png"
                elif ".webp" in fal_url.lower():
                    content_type = "image/webp"
                object_path = f"media/images/{project_id}/scene_{idx + 1}_{uuid.uuid4().hex[:8]}.jpg"
                fal_url = await upload_bytes(object_path, data, content_type)
            except Exception as exc:
                logger.warning("Supabase image cache failed; using source URL directly: %s", exc)

        urls.append(fal_url)
        scene.visual_prompt = prompt
        scene.image_url = fal_url

    return urls


def _fallback_image(assets: BrandAssets, idx: int) -> Optional[str]:
    if assets.product_images:
        return assets.product_images[idx % len(assets.product_images)]
    return assets.logo_url


async def validate_public_https(url: str) -> bool:
    if not url or not url.startswith("https://"):
        return False
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.head(url)
            if response.status_code >= 400:
                response = await client.get(url)
            return response.status_code < 400
    except Exception:
        return False
