"""Validate Creatomate media and generate image fallbacks when assets are broken."""
from __future__ import annotations

import base64
import logging
import os
import re
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
                response = await client.get(url, headers={"Range": "bytes=0-1023"})
            if response.status_code >= 400:
                return False
            content_type = (response.headers.get("Content-Type") or "").lower()
            if _content_type_ok(content_type, VALID_IMAGE_TYPES):
                return True
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


NICHE_PEXELS_KEYWORDS = {
    "Software & Technology": ["coding software technology", "software developer programming", "modern tech workspace", "computer software engineer"],
    "SaaS & Cloud Technology": ["cloud software technology", "modern tech office developers", "data analytics software", "tech startup coding"],
    "HVAC & Climate Control": ["hvac technician repair", "air conditioning maintenance", "heating cooling technician", "modern home air condition"],
    "Luxury Real Estate": ["luxury modern villa", "contemporary home architecture", "luxury house interior", "modern real estate villa"],
    "Artisan Coffee Shop & Roastery": ["coffee shop barista", "specialty coffee espresso", "artisan cafe interior", "coffee roasting cafe"],
    "Dental Care & Orthodontics": ["modern dental clinic", "dentist patient smile", "dental care professional", "healthy teeth dentistry"],
    "Plumbing & Rooter Services": ["plumber repairing pipes", "modern bathroom plumbing", "plumbing technician tools", "residential plumber"],
    "Fitness & Wellness Studio": ["fitness workout athlete", "modern gym training", "personal trainer fitness", "wellness yoga studio"],
    "Legal Services & Law Practice": ["law firm office", "attorney consultation", "corporate law library", "executive boardroom law"],
    "Automotive Service & Repair": ["car mechanic repair", "auto detailing vehicle", "automotive garage mechanic", "modern car service bay"],
    "Restaurant & Culinary Dining": ["gourmet culinary dining", "chef kitchen plating", "restaurant dining ambiance", "fine dining dish"],
    "Professional Cleaning Services": ["professional cleaning service", "clean modern home housekeeping", "commercial janitorial cleaning", "impeccable clean room"],
    "Landscaping & Outdoor Design": ["landscaping garden design", "modern lawn care outdoor", "beautiful landscape architecture", "residential garden lawn"],
    "Construction & Home Remodeling": ["home remodeling construction", "modern kitchen renovation", "architect contractor blueprints", "renovated home interior"],
    "Veterinary & Pet Care Services": ["veterinarian dog clinic", "animal hospital care", "pet clinic veterinary", "happy dog vet exam"],
    "Fashion & Apparel Retail": ["fashion apparel boutique", "stylish clothing lifestyle", "modern fashion model", "boutique fashion retail"],
    "E-Commerce & Digital Retail": ["modern product photography", "premium packaging unboxing", "ecommerce shopping lifestyle", "clean product showcase"],
}


def extract_pexels_search_terms(
    prompt: str,
    category: Optional[str] = None,
    scene_idx: int = 0,
) -> str:
    niche = (category or "").strip()
    
    if niche in NICHE_PEXELS_KEYWORDS:
        options = NICHE_PEXELS_KEYWORDS[niche]
        return options[scene_idx % len(options)]

    for n_key, options in NICHE_PEXELS_KEYWORDS.items():
        if any(w.lower() in niche.lower() or w.lower() in (prompt or "").lower() for w in n_key.split()):
            return options[scene_idx % len(options)]

    combined = f"{niche} {prompt}".lower()
    if any(k in combined for k in ("code", "coding", "software", "developer", "programming", "tech", "web dev", "app")):
        tech_terms = ["coding software technology", "software developer programming", "modern tech workspace", "computer programmer"]
        return tech_terms[scene_idx % len(tech_terms)]

    clean_q = re.sub(
        r"(?:photorealistic|8k|4k|resolution|cinematic|lighting|zero text overlay|no text|no typography|no watermark|professional brand photography|highly detailed|establishing shot|close-up shot|wide shot|eye-level shot|commercial business|services|workspace|featuring|illuminated by|accents|clean aesthetic|setting|brand|color|palette)",
        "",
        prompt or "",
        flags=re.IGNORECASE,
    )
    clean_q = re.sub(r"[,\.\:\;\(\)\{\}\[\]\"'#\-_/]", " ", clean_q)
    words = [w.strip() for w in clean_q.split() if len(w.strip()) > 2 and w.lower() not in {"and", "the", "for", "with", "our", "team", "modern", "sleek", "subtle", "rich", "deep", "soft", "warm", "shot"}]
    
    if len(words) >= 2:
        return " ".join(words[:3])

    return niche or "modern technology workspace"


async def fetch_pexels_fallback_image(
    prompt: str,
    scene_idx: int = 0,
    category: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    pexels_key = (os.getenv("PEXELS_API_KEY") or "").strip()
    if not pexels_key:
        logger.warning("PEXELS_API_KEY not set in .env")
        return None

    search_query = extract_pexels_search_terms(prompt, category=category, scene_idx=scene_idx)
    orientation = "portrait" if aspect_ratio in {"9:16", "portrait"} else "landscape"
    url = f"https://api.pexels.com/v1/search?query={quote(search_query)}&orientation={orientation}&per_page=15"
    
    headers = {"Authorization": pexels_key}
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                photos = data.get("photos", [])
                if photos:
                    photo = photos[scene_idx % len(photos)]
                    selected_url = photo["src"]["large2x"]
                    logger.info("Pexels fetched image for query '%s' (Scene %d): %s", search_query, scene_idx + 1, selected_url[:80])
                    return selected_url
            else:
                logger.warning("Pexels API returned status %d for query '%s'", res.status_code, search_query)
    except Exception as exc:
        logger.warning("Pexels search failed for query '%s': %s", search_query, exc)

    return None


def get_aspect_ratio_dimensions(aspect_ratio: str) -> tuple[int, int]:
    ar = (aspect_ratio or "16:9").lower().strip()
    if ar in {"9:16", "portrait", "vertical"}:
        return 576, 1024
    elif ar in {"1:1", "square"}:
        return 1024, 1024
    return 1024, 576


def get_pollinations_fallback_url(
    prompt: str,
    *,
    scene_idx: int = 0,
    aspect_ratio: str = "16:9",
) -> str:
    """Constructs a direct Pollinations URL with commercial ad prompt modifiers."""
    enhanced = (
        f"Professional advertisement photography, {prompt.strip()}, "
        f"hyperrealistic, 8k resolution, studio commercial lighting, no text, no watermark"
    )
    safe_prompt = quote(enhanced)
    width, height = get_aspect_ratio_dimensions(aspect_ratio)

    return (
        f"https://image.pollinations.ai/prompt/{safe_prompt}"
        f"?width={width}&height={height}&model=flux&nologo=true&enhance=true&seed={scene_idx + 1}"
    )

# Alias for backwards compatibility with services/visuals.py
get_pollinations_fallback_image = get_pollinations_fallback_url

__all__ = [
    "sanitize_scene_visual",
    "sanitize_image_list",
    "sanitize_audio_url",
    "generate_image_with_fallbacks",
    "get_pollinations_fallback_url",
    "get_pollinations_fallback_image",
]

async def generate_image_pollinations(
    prompt: str,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    project_id: Optional[str] = None,
) -> Optional[str]:
    """
    Primary Generator: Fetches FLUX-generated visual bytes from Pollinations AI
    and uploads them directly to Supabase storage to give Creatomate permanent CDN access.
    """
    enhanced_prompt = (
        f"High quality advertisement photography of {prompt.strip()}, "
        f"hyperrealistic 8k, cinematic lighting, sharp focus, commercial studio quality, no text or logo"
    )
    encoded_prompt = quote(enhanced_prompt)
    width, height = get_aspect_ratio_dimensions(aspect_ratio)
    
    pollinations_api_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&model=flux&nologo=true&enhance=true&seed={scene_idx + 10}"
    )

    try:
        logger.info("Generating Pollinations AI (FLUX) image for scene %d...", scene_idx + 1)
        async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
            res = await client.get(pollinations_api_url)
            
            if res.status_code == 200 and len(res.content) > 2000:
                object_path = (
                    f"media/images/{project_id or 'generated'}/"
                    f"pollination_scene_{scene_idx + 1}_{uuid.uuid4().hex[:10]}.png"
                )
                # Upload raw PNG bytes to Supabase Storage
                public_url = await upload_bytes(object_path, res.content, "image/png")
                logger.info("Pollinations AI generated scene %d -> Uploaded: %s", scene_idx + 1, public_url)
                return public_url
            else:
                logger.warning("Pollinations HTTP status %d or empty payload for scene %d", res.status_code, scene_idx + 1)
    except Exception as exc:
        logger.warning("Pollinations AI request failed for scene %d: %s", scene_idx + 1, exc)

    return None


async def generate_image_with_fallbacks(
    prompt: str,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """
    Fallback Chain:
    1. Pollinations.ai (FLUX) -> Saves to Supabase Storage
    2. Pexels API Stock Images
    3. Direct Pollinations URL fallback
    """
    safe_prompt = prompt or "Professional clean modern business showcase, cinematic lighting"

    # 1. Primary: Pollinations FLUX Engine
    pollinations_url = await generate_image_pollinations(
        safe_prompt, aspect_ratio=aspect_ratio, scene_idx=scene_idx, project_id=project_id
    )
    if pollinations_url:
        return pollinations_url

    # 2. Secondary: Pexels Stock Photos
    logger.warning("Pollinations storage generation failed; trying Pexels stock photo fallback for scene %d", scene_idx + 1)
    pexels_url = await fetch_pexels_fallback_image(
        prompt=safe_prompt, scene_idx=scene_idx, category=category, aspect_ratio=aspect_ratio
    )
    if pexels_url:
        return pexels_url

    # 3. Tertiary: Direct Pollinations URL
    logger.warning("Pexels returned no image; using direct Pollinations URL for scene %d", scene_idx + 1)
    return get_pollinations_fallback_url(safe_prompt, scene_idx=scene_idx, aspect_ratio=aspect_ratio)


async def sanitize_scene_visual(
    scraped_url: Optional[str],
    visual_prompt: str,
    *,
    aspect_ratio: str = "16:9",
    scene_idx: int = 0,
    category: Optional[str] = None,
) -> str:
    """Validate an image URL; regenerate via Pollinations if scraping fails or image is broken."""
    candidate = clean_asset_url(scraped_url)
    if candidate and await is_valid_creatomate_image(candidate):
        return candidate

    logger.warning(
        "Scene %s image invalid or broken (%s). Generating fresh visual.",
        scene_idx + 1,
        (scraped_url or "")[:100],
    )
    fresh_url = await generate_image_with_fallbacks(
        visual_prompt, aspect_ratio=aspect_ratio, scene_idx=scene_idx, category=category
    )
    return clean_asset_url(fresh_url) or get_pollinations_fallback_url(visual_prompt, scene_idx=scene_idx, aspect_ratio=aspect_ratio)


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
        if cleaned in seen_urls:
            cleaned = get_pollinations_fallback_url(prompt, scene_idx=idx, aspect_ratio=aspect_ratio)
        seen_urls.add(cleaned)
        out.append(cleaned)
        logger.info("Sanitized Image-%s -> %s", idx + 1, cleaned[:140])
    return out