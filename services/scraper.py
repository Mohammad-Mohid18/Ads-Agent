import httpx
from bs4 import BeautifulSoup
from typing import Optional
import logging
from urllib.parse import urljoin
from models import BrandAssets

logger = logging.getLogger("ai_ad_engine.scraper")


async def scrape_site(url: str) -> BrandAssets:
    """Rich brand scraping using httpx + BeautifulSoup.
    Extracts site title, description, headings, core features, logo, product images, and theme color.
    """
    html = ""
    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.warning("Scraping request failed for %s: %s", url, exc)

    if not html:
        # Fallback to URL-based brand extraction
        domain = url.split("//")[-1].split("/")[0].replace("www.", "")
        brand_name = domain.split(".")[0].capitalize()
        return BrandAssets(
            site_title=brand_name,
            description=f"{brand_name} digital platform and services.",
            logo_url=None,
            product_images=[],
            primary_color="#1E3A8A",
            raw_text_snippet=f"Welcome to {brand_name}. Modern solutions built for you.",
        )

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer tags for clean text extraction
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()

    # 1. Site Title / Brand Name
    title = None
    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    if og_title and og_title.get("content"):
        title = og_title.get("content").strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    # 2. Description
    desc = None
    desc_tag = (
        soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", property="og:description")
        or soup.find("meta", attrs={"name": "twitter:description"})
    )
    if desc_tag and desc_tag.get("content"):
        desc = desc_tag.get("content").strip()

    # 3. Headings & Hero Text
    headings = []
    for h in soup.find_all(["h1", "h2"]):
        h_text = h.get_text(separator=" ", strip=True)
        if h_text and len(h_text) > 4 and len(h_text) < 120:
            headings.append(h_text)
    headings = list(dict.fromkeys(headings))[:5]

    # 4. Logo extraction
    logo = None
    icon = soup.find("link", rel=lambda x: x and any(i in x for i in ("icon", "apple-touch-icon", "shortcut")))
    if icon and icon.get("href"):
        logo = _abspath(url, icon.get("href"))
    og_image = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
    if og_image and og_image.get("content"):
        if not logo:
            logo = og_image.get("content")

    # 5. Product / Content Images
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("srcset")
        if not src:
            continue
        if "," in src:
            src = src.split(",")[0].strip().split(" ")[0]
        src = _abspath(url, src)
        if any(ext in src.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
            imgs.append(src)
    product_images = list(dict.fromkeys(imgs))[:10]

    # 6. Primary Color
    primary_color = None
    theme = soup.find("meta", attrs={"name": "theme-color"})
    if theme and theme.get("content"):
        primary_color = theme.get("content").strip()

    # 7. Raw Text Snippet
    paragraphs = []
    for p in soup.find_all(["p", "li"]):
        p_text = p.get_text(separator=" ", strip=True)
        if len(p_text) > 20 and not any(k in p_text.lower() for k in ("cookie", "privacy policy", "terms of service", "all rights reserved")):
            paragraphs.append(p_text)
    snippet_parts = []
    if headings:
        snippet_parts.append("Key Highlights: " + " | ".join(headings[:3]))
    if paragraphs:
        snippet_parts.append("Summary: " + " ".join(paragraphs[:3]))
    raw_text_snippet = "\n".join(snippet_parts)[:800] if snippet_parts else (desc or title or "")

    assets = BrandAssets(
        site_title=title,
        description=desc,
        logo_url=logo,
        product_images=product_images,
        primary_color=primary_color,
        raw_text_snippet=raw_text_snippet,
    )

    logger.info("Scraped assets: title=%s, desc_len=%d, headings=%d, images=%d",
                title, len(desc or ""), len(headings), len(product_images))
    return assets


def _abspath(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href

