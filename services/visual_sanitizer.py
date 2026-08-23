"""Validate media URLs for Creatomate and fall back to fal.ai when assets are broken."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import httpx

logger = logging.getLogger("ai_ad_engine.sanitizer")

VALID_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/jpg")
VALID_AUDIO_TYPES = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac")
STATIC_FALLBACK_IMAGE = (
    "https://images.unsplash.com/photo-1460925895917-afdab827c52f"
    "?q=80&w=1200&auto=format&fit=crop"
)
FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def clean_asset_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize URLs for Creatomate:
    - Convert Supabase /object/sign/... paths to /object/public/...
    - Strip ?token= / signature query strings from Supabase URLs only
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    # Supabase signed -> public (Creatomate cannot use long ?token= signed links)
    if "supabase.co" in url and "/storage/v1/object/sign/" in url:
        url = url.replace("/storage/v1/object/sign/", "/storage/v1/object/public/", 1)

    parsed = urlparse(url)
    query = parsed.query
    if "supabase.co" in url:
        query = ""

    safe_path = quote(parsed.path, safe="/:@-&+$,;=()")
    return urlunparse((parsed.scheme, parsed.netloc, safe_path, "", query, ""))


def public_supabase_url(supabase_url: str, bucket: str, object_path: str) -> str:
    base = supabase_url.rstrip("/")
    object_path = object_path.lstrip("/")
    return f"{base}/storage/v1/object/public/{bucket}/{quote(object_path, safe='/')}"


async def is_valid_creatomate_image(image_url: Optional[str]) -> bool:
    """True when the URL is publicly reachable and returns a standard image Content-Type."""
    url = clean_asset_url(image_url) if image_url else None
    if not url or not url.startswith("http"):
        return False
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            response = await client.head(url)
            if response.status_code >= 400 or not _content_type_ok(
                response.headers.get("Content-Type"), VALID_IMAGE_TYPES
            ):
                # Some CDNs reject HEAD — fall back to a ranged GET.
                response = await client.get(url, headers={"Range": "bytes=0-1023"})
            if response.status_code >= 400:
                return False
            content_type = (response.headers.get("Content-Type") or "").lower()
            if _content_type_ok(content_type, VALID_IMAGE_TYPES):
                return True
            # Accept extension-based images when CDNs omit Content-Type.
            path = urlparse(url).path.lower()
            return path.endswith((".jpg", ".jpeg", ".png", ".webp"))
    except Exception as exc:
        logger.warning("Image validation failed for %s: %s", image_url, exc)
        return False


async def is_valid_creatomate_audio(audio_url: Optional[str]) -> bool:
    url = clean_asset_url(audio_url) if audio_url else None
    if not url or not url.startswith("http"):
        return False
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            response = await client.head(url)
            if response.status_code >= 400:
                response = await client.get(url, headers={"Range": "bytes=0-1023"})
            if response.status_code >= 400:
                return False
            content_type = (response.headers.get("Content-Type") or "").lower()
            if _content_type_ok(content_type, VALID_AUDIO_TYPES):
                return True
            return urlparse(url).path.lower().endswith((".mp3", ".wav", ".m4a", ".aac"))
    except Exception as exc:
        logger.warning("Audio validation failed for %s: %s", audio_url, exc)
        return False


def _content_type_ok(content_type: Optional[str], allowed: tuple[str, ...]) -> bool:
    ct = (content_type or "").lower()
    return any(vt in ct for vt in allowed)


async def generate_fal_fallback_image(prompt: str, aspect_ratio: str = "16:9") -> str:
    """Generate a reliable public visual via fal.ai when site images are invalid/damaged."""
    logger.info("Generating fallback image with fal.ai for prompt: %s", (prompt or "")[:120])
    fal_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not fal_key:
        logger.error("FAL_KEY missing — using static Unsplash fallback image")
        return STATIC_FALLBACK_IMAGE

    os.environ["FAL_KEY"] = fal_key
    # fal-ai/flux/schnell supports landscape_16_9 and portrait_16_9 (9:16 vertical video)
    image_size = "portrait_16_9" if aspect_ratio in {"9:16", "portrait"} else "landscape_16_9"
    safe_prompt = (
        prompt
        or "Professional clean modern business showcase, high resolution, cinematic lighting"
    )

    def _run() -> str:
        import fal_client

        result = fal_client.subscribe(
            FAL_MODEL,
            arguments={
                "prompt": safe_prompt,
                "image_size": image_size,
                "num_images": 1,
            },
            with_logs=False,
        )
        images = result.get("images") or []
        if not images or not images[0].get("url"):
            raise RuntimeError("fal.ai returned no image URL")
        return images[0]["url"]

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        logger.error("fal.ai generation failed: %s", exc)
        return STATIC_FALLBACK_IMAGE


async def sanitize_scene_visual(
    scraped_url: Optional[str],
    visual_prompt: str,
    *,
    aspect_ratio: str = "16:9",
) -> str:
    """Validate scraped/generated image URL; replace with fal.ai (or static) when broken."""
    candidate = clean_asset_url(scraped_url)
    if candidate and await is_valid_creatomate_image(candidate):
        return candidate

    logger.warning(
        "Scraped/generated image invalid or damaged (%s). Falling back to fal.ai generation.",
        (scraped_url or "")[:120],
    )
    fal_url = await generate_fal_fallback_image(visual_prompt, aspect_ratio=aspect_ratio)
    return clean_asset_url(fal_url) or STATIC_FALLBACK_IMAGE


async def sanitize_audio_url(audio_url: Optional[str]) -> str:
    """Ensure voiceover URL is a clean public HTTPS link Creatomate can download."""
    candidate = clean_asset_url(audio_url)
    if candidate and await is_valid_creatomate_audio(candidate):
        return candidate
    raise RuntimeError(
        f"Voiceover URL is not publicly downloadable for Creatomate: {(audio_url or '')[:160]}"
    )


async def sanitize_image_list(
    urls: list[Optional[str]],
    prompts: list[str],
    *,
    aspect_ratio: str = "16:9",
) -> list[str]:
    """Sanitize every image layer (Image-1, Image-2, ...) before Creatomate payload build."""
    out: list[str] = []
    for idx, url in enumerate(urls):
        prompt = prompts[idx] if idx < len(prompts) else "Professional modern business visual"
        cleaned = await sanitize_scene_visual(url, prompt, aspect_ratio=aspect_ratio)
        out.append(cleaned)
        logger.info("Sanitized Image-%s -> %s", idx + 1, cleaned[:140])
    return out
