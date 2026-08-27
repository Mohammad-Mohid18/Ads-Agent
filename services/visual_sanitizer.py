"""Validate Creatomate media and generate image fallbacks when assets are broken."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from services.asset_storage import upload_bytes

logger = logging.getLogger("ai_ad_engine.sanitizer")

VALID_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/jpg")
VALID_AUDIO_TYPES = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac")

HF_API_TOKEN = (os.getenv("HF_API_TOKEN") or "").strip()
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
).strip()


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

async def fetch_pexels_fallback_image(
    prompt: str,
    scene_idx: int = 0,
    category: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Fetch a high-res stock photo from Pexels using scene-specific script keywords."""
    pexels_key = (os.getenv("PEXELS_API_KEY") or "").strip()
    if not pexels_key:
        logger.warning("PEXELS_API_KEY not set in .env")
        return None

    # Construct search query directly from scene visual prompt and category
    search_query = (prompt or category or "commercial product").strip()
    
    orientation = "portrait" if aspect_ratio in {"9:16", "portrait"} else "landscape"
    url = f"https://api.pexels.com/v1/search?query={quote(search_query)}&orientation={orientation}&per_page=15"
    
    headers = {"Authorization": pexels_key}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                photos = data.get("photos", [])
                if photos:
                    # Pick unique photo matching scene index
                    photo = photos[scene_idx % len(photos)]
                    selected_url = photo["src"]["large2x"]
                    logger.info("Pexels fetched image for query '%s' (Scene %d): %s", search_query, scene_idx + 1, selected_url[:80])
                    return selected_url
            else:
                logger.warning("Pexels API returned status %d for query '%s'", res.status_code, search_query)
    except Exception as exc:
        logger.warning("Pexels search failed for query '%s': %s", search_query, exc)

    return None


# 2. HUGGING FACE SERVERLESS GENERATION WITH PEXELS + POLLINATIONS FALLBACKS
def get_pollinations_fallback_image(prompt: str, *, scene_idx: int = 0) -> str:
    """Return a distinct Pollinations image URL when generated or stock media fails."""
    safe_prompt = quote((prompt or "professional commercial product photography").strip())
    return (
        f"https://image.pollinations.ai/prompt/{safe_prompt}"
        f"?width=1280&height=720&nologo=true&seed={scene_idx + 1}"
    )


def _image_content_type(content: bytes, reported_type: str) -> Optional[str]:
    """Accept Hugging Face image bytes even when its response omits a MIME type."""
    if reported_type.startswith("image/"):
        return reported_type
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


async def generate_image_with_fallbacks(
    prompt: str,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """Generate with Hugging Face, then fall back to Pexels and Pollinations."""
    safe_prompt = prompt or "Professional clean modern business showcase, cinematic lighting"
    if HF_API_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    HF_MODEL_URL,
                    headers={
                        "Authorization": f"Bearer {HF_API_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": safe_prompt,
                        "parameters": {"width": 1280, "height": 720},
                    },
                )
            reported_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            content_type = _image_content_type(response.content, reported_type)
            if response.status_code == 200 and content_type and response.content:
                extension = {"image/png": "png", "image/webp": "webp"}.get(content_type, "jpg")
                object_path = (
                    f"media/images/{project_id or 'generated'}/"
                    f"hf_scene_{scene_idx + 1}_{uuid.uuid4().hex[:10]}.{extension}"
                )
                public_url = await upload_bytes(object_path, response.content, content_type)
                logger.info("Hugging Face generated scene %s and uploaded %s", scene_idx + 1, object_path)
                return public_url
            logger.warning(
                "Hugging Face returned status=%s content_type=%s for scene %s; using Pexels fallback",
                response.status_code,
                reported_type or "unknown",
                scene_idx + 1,
            )
        except Exception as exc:
            logger.warning("Hugging Face generation failed for scene %s: %s; using Pexels fallback", scene_idx + 1, exc)
    else:
        logger.warning("HF_API_TOKEN is not configured; using Pexels fallback for scene %s", scene_idx + 1)

    pexels_url = await fetch_pexels_fallback_image(
        prompt=safe_prompt,
        scene_idx=scene_idx,
        category=category,
        aspect_ratio=aspect_ratio,
    )
    if pexels_url:
        return pexels_url

    pollinations_url = get_pollinations_fallback_image(safe_prompt, scene_idx=scene_idx)
    logger.warning("Pexels returned no image; using Pollinations fallback for scene %s", scene_idx + 1)
    return pollinations_url


async def sanitize_scene_visual(
    scraped_url: Optional[str],
    visual_prompt: str,
    *,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
) -> str:
    """Validate an image URL; regenerate via the configured fallback chain if needed."""
    candidate = clean_asset_url(scraped_url)
    if candidate and await is_valid_creatomate_image(candidate):
        return candidate

    logger.warning(
        "Scene %s image invalid or damaged (%s). Generating fresh visual.",
        scene_idx + 1,
        (scraped_url or "")[:100],
    )
    fresh_url = await generate_image_with_fallbacks(
        visual_prompt, aspect_ratio=aspect_ratio, scene_idx=scene_idx, category=category
    )
    return clean_asset_url(fresh_url) or get_pollinations_fallback_image(visual_prompt, scene_idx=scene_idx)


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
    category: Optional[str] = None,
) -> list[str]:
    """Sanitize every image layer (Image-1, Image-2, ...) before Creatomate payload build."""
    out: list[str] = []
    seen_urls: set[str] = set()
    for idx, url in enumerate(urls):
        prompt = prompts[idx] if idx < len(prompts) else "Professional modern commercial visual"
        cleaned = await sanitize_scene_visual(
            url, prompt, aspect_ratio=aspect_ratio, scene_idx=idx, category=category
        )
        # Prevent duplicate image reuse across scenes
        if cleaned in seen_urls:
            cleaned = get_pollinations_fallback_image(prompt, scene_idx=idx)
        seen_urls.add(cleaned)
        out.append(cleaned)
        logger.info("Sanitized Image-%s -> %s", idx + 1, cleaned[:140])
    return out
