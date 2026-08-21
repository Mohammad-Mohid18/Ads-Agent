import httpx
from bs4 import BeautifulSoup
from typing import Optional
import logging
from models import BrandAssets

logger = logging.getLogger("ai_ad_engine.scraper")


async def scrape_site(url: str) -> BrandAssets:
    """Lightweight scraping using httpx + BeautifulSoup.
    Attempts to extract site title, description, logo, product images and theme color.
    """
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # title/description
    title = None
    desc = None
    if soup.title:
        title = soup.title.string
    desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if desc_tag:
        desc = desc_tag.get("content")

    # logo: look for rel=icon or og:image
    logo = None
    icon = soup.find("link", rel=lambda x: x and "icon" in x)
    if icon and icon.get("href"):
        logo = _abspath(url, icon.get("href"))
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        # prefer og:image as product image, but could be logo
        if not logo:
            logo = og_image.get("content")

    # images - collect large images from img tags
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        src = _abspath(url, src)
        imgs.append(src)
    # dedupe and limit
    product_images = list(dict.fromkeys(imgs))[:10]

    # primary color: try meta theme-color
    primary_color = None
    theme = soup.find("meta", attrs={"name": "theme-color"})
    if theme and theme.get("content"):
        primary_color = theme.get("content")

    # raw text snippet - first paragraph
    snippet = None
    p = soup.find("p")
    if p:
        snippet = p.get_text().strip()

    assets = BrandAssets(site_title=title, description=desc, logo_url=logo,
                         product_images=product_images, primary_color=primary_color,
                         raw_text_snippet=snippet)

    logger.info("Scraped assets: title=%s, logo=%s, images=%d", title, logo, len(product_images))
    return assets


from urllib.parse import urljoin


def _abspath(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href
