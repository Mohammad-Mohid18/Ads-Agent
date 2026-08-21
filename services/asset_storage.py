"""Upload media bytes to Supabase Storage and return clean public HTTPS URLs."""
from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger("ai_ad_engine.storage")

SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
).rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or ""
)
BUCKET = os.getenv("SUPABASE_RENDER_BUCKET") or os.getenv("S3_BUCKET", "ad-assets")


def _headers(*, content_type: Optional[str] = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    if content_type:
        headers["Content-Type"] = content_type
        headers["x-upsert"] = "true"
    return headers


def public_object_url(object_path: str) -> str:
    """Stable Creatomate-safe public URL — no /sign/ path and no ?token= query."""
    object_path = object_path.lstrip("/")
    return (
        f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/"
        f"{quote(object_path, safe='/')}"
    )


async def upload_bytes(
    object_path: str,
    data: bytes,
    content_type: str,
) -> str:
    """Upload to the public ad-assets bucket and return a clean public HTTPS URL."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials are not configured for asset upload")

    object_path = object_path.lstrip("/")
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{object_path}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            upload_url, headers=_headers(content_type=content_type), content=data
        )
        if response.is_error:
            raise RuntimeError(
                f"Supabase upload failed ({response.status_code}): {response.text[:300]}"
            )

    public_url = public_object_url(object_path)
    logger.info("Uploaded public asset %s", public_url)
    return public_url


async def download_url_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
