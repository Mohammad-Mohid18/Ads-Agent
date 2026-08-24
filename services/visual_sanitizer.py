"""Validate media URLs for Creatomate and fall back to fal.ai (or niche-tailored visuals) when assets are broken."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import httpx

logger = logging.getLogger("ai_ad_engine.sanitizer")

VALID_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/jpg")
VALID_AUDIO_TYPES = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac")

# Curated, distinct 5-scene high-res commercial imagery per niche to eliminate single-image repetition
NICHE_FALLBACK_IMAGES: dict[str, list[str]] = {
    "Social Media & Camera Platform": [
        "https://images.unsplash.com/photo-1516251193007-45ef944ab0c6?q=80&w=1600&auto=format&fit=crop",  # Neon Gen-Z smartphone
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=1600&auto=format&fit=crop",  # Expressive portrait
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?q=80&w=1600&auto=format&fit=crop",  # Friends laughing outdoors
        "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1600&auto=format&fit=crop",  # Vibrant dynamic abstract/AR
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?q=80&w=1600&auto=format&fit=crop",  # Glowing phone lifestyle
    ],
    "Artisanal Coffee Brand": [
        "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?q=80&w=1600&auto=format&fit=crop",  # Espresso pouring
        "https://images.unsplash.com/photo-1509785307050-d4066910ec1e?q=80&w=1600&auto=format&fit=crop",  # Moody cafe setting
        "https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?q=80&w=1600&auto=format&fit=crop",  # Barista latte art
        "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?q=80&w=1600&auto=format&fit=crop",  # Joyful cafe customer
        "https://images.unsplash.com/photo-1447933601403-0c6688de566e?q=80&w=1600&auto=format&fit=crop",  # Roasted coffee beans
    ],
    "Fitness & Wellness Brand": [
        "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?q=80&w=1600&auto=format&fit=crop",  # Intense athletic workout
        "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?q=80&w=1600&auto=format&fit=crop",  # Focused training
        "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1600&auto=format&fit=crop",  # Dynamic gym performance
        "https://images.unsplash.com/photo-1506126613408-eca07ce68773?q=80&w=1600&auto=format&fit=crop",  # Sunrise wellness/yoga
        "https://images.unsplash.com/photo-1434596922112-19c563067271?q=80&w=1600&auto=format&fit=crop",  # Athletic gear/finish
    ],
    "Fintech Platform": [
        "https://images.unsplash.com/photo-1563986768609-322da13575f3?q=80&w=1600&auto=format&fit=crop",  # Modern digital finance
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?q=80&w=1600&auto=format&fit=crop",  # Analytical fintech
        "https://images.unsplash.com/photo-1556742049-0a67e557224f?q=80&w=1600&auto=format&fit=crop",  # Contactless mobile payment
        "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?q=80&w=1600&auto=format&fit=crop",  # Confident business leader
        "https://images.unsplash.com/photo-1559526324-593bc073d938?q=80&w=1600&auto=format&fit=crop",  # Premium modern investment
    ],
    "SaaS & AI Platform": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1600&auto=format&fit=crop",  # AI & Tech innovation
        "https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=1600&auto=format&fit=crop",  # Creative team collaboration
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=1600&auto=format&fit=crop",  # Modern clean visual insights
        "https://images.unsplash.com/photo-1531482615713-2afd69097998?q=80&w=1600&auto=format&fit=crop",  # High-energy presentation
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1600&auto=format&fit=crop",  # Futuristic global network
    ],
    "Default": [
        "https://images.unsplash.com/photo-1557804506-669a67965ba0?q=80&w=1600&auto=format&fit=crop",  # Modern creative showcase
        "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?q=80&w=1600&auto=format&fit=crop",  # Problem-solving workflow
        "https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1600&auto=format&fit=crop",  # Modern digital tools
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=1600&auto=format&fit=crop",  # Joyful collaborative achievement
        "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1600&auto=format&fit=crop",  # Inspiring leadership call-to-action
    ],
}

FAL_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")


def get_niche_fallback_image(category: Optional[str] = None, scene_idx: int = 0) -> str:
    """Retrieve a unique, niche-appropriate image for each scene index to avoid repeated imagery."""
    matched_key = "Default"
    if category:
        cat_lower = category.lower()
        if any(w in cat_lower for w in ("social", "camera", "chat", "lens", "snap", "media", "tiktok", "instagram", "creator")):
            matched_key = "Social Media & Camera Platform"
        elif any(w in cat_lower for w in ("coffee", "cafe", "espresso", "bakery", "roast")):
            matched_key = "Artisanal Coffee Brand"
        elif any(w in cat_lower for w in ("fit", "gym", "workout", "yoga", "train", "run", "wellness", "athlet")):
            matched_key = "Fitness & Wellness Brand"
        elif any(w in cat_lower for w in ("fin", "bank", "pay", "invest", "crypto", "trade", "money")):
            matched_key = "Fintech Platform"
        elif any(w in cat_lower for w in ("saas", "software", "ai", "tech", "cloud", "dev", "app", "platform")):
            matched_key = "SaaS & AI Platform"

    pool = NICHE_FALLBACK_IMAGES.get(matched_key) or NICHE_FALLBACK_IMAGES["Default"]
    base_url = pool[scene_idx % len(pool)]
    # Append unique signature / cache buster
    return f"{base_url}&sig={uuid.uuid4().hex[:8]}"



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

from urllib.parse import quote

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


# 2. UPDATED FAL.AI GENERATION WITH PEXELS FALLBACK
async def generate_fal_fallback_image(
    prompt: str,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
) -> str:
    """Generate visual via fal.ai; fall back to Pexels API, then Unsplash niche pools."""
    logger.info("Generating visual (scene %s) for prompt: %s", scene_idx + 1, (prompt or "")[:100])
    
    fal_key = (os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or "").strip()
    
    # If FAL_KEY is missing or balance exhausted, try Pexels first
    if fal_enabled := bool(fal_key):
        os.environ["FAL_KEY"] = fal_key
        image_size = "portrait_16_9" if aspect_ratio in {"9:16", "portrait"} else "landscape_16_9"
        safe_prompt = prompt or "Professional clean modern business showcase, cinematic lighting, 8k"

        def _run() -> str:
            import fal_client
            result = fal_client.subscribe(
                FAL_MODEL,
                arguments={
                    "prompt": safe_prompt,
                    "image_size": image_size,
                    "num_images": 1,
                    "enable_safety_checker": True,
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
            logger.warning("fal.ai generation failed for scene %s (%s). Attempting Pexels fallback...", scene_idx + 1, exc)

    # Fallback 1: Query Pexels API
    pexels_url = await fetch_pexels_fallback_image(
        prompt=prompt,
        scene_idx=scene_idx,
        category=category,
        aspect_ratio=aspect_ratio,
    )
    if pexels_url:
        return pexels_url

    # Fallback 2: Local Unsplash niche fallback pools
    logger.warning("Pexels fallback unavailable. Falling back to local niche pool for scene %s", scene_idx + 1)
    return get_niche_fallback_image(category, scene_idx)


async def sanitize_scene_visual(
    scraped_url: Optional[str],
    visual_prompt: str,
    *,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
) -> str:
    """Validate image URL; replace with fal.ai or unique niche visual when invalid."""
    candidate = clean_asset_url(scraped_url)
    if candidate and await is_valid_creatomate_image(candidate):
        return candidate

    logger.warning(
        "Scene %s image invalid or damaged (%s). Generating fresh visual.",
        scene_idx + 1,
        (scraped_url or "")[:100],
    )
    fresh_url = await generate_fal_fallback_image(
        visual_prompt, aspect_ratio=aspect_ratio, scene_idx=scene_idx, category=category
    )
    return clean_asset_url(fresh_url) or get_niche_fallback_image(category, scene_idx)


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
            cleaned = get_niche_fallback_image(category, idx)
        seen_urls.add(cleaned)
        out.append(cleaned)
        logger.info("Sanitized Image-%s -> %s", idx + 1, cleaned[:140])
    return out

