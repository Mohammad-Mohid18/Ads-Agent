"""Upload media bytes to Supabase Storage and return HTTPS URLs Creatomate can fetch."""
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
SIGNED_URL_TTL = int(os.getenv("SUPABASE_SIGNED_URL_TTL", "86400"))


def _headers(*, content_type: Optional[str] = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    if content_type:
        headers["Content-Type"] = content_type
        headers["x-upsert"] = "true"
    return headers


async def upload_bytes(
    object_path: str,
    data: bytes,
    content_type: str,
) -> str:
    """Upload to the private ad-assets bucket and return a long-lived signed HTTPS URL."""
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

        # Prefer a signed URL so Creatomate can fetch from a private bucket.
        sign_url = f"{SUPABASE_URL}/storage/v1/object/sign/{BUCKET}/{object_path}"
        signed = await client.post(
            sign_url,
            headers=_headers(content_type="application/json"),
            json={"expiresIn": SIGNED_URL_TTL},
        )
        if not signed.is_error:
            payload = signed.json()
            token_path = payload.get("signedURL") or payload.get("signedUrl") or ""
            if token_path.startswith("http"):
                return token_path
            if token_path:
                return f"{SUPABASE_URL}/storage/v1{token_path}"

        # Fallback: public object URL (works only if the bucket/object is public).
        public = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{quote(object_path)}"
        logger.warning("Signed URL unavailable; falling back to public URL for %s", object_path)
        return public


async def download_url_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
