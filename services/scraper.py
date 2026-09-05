import json
import logging
import re
from typing import Any, List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from models import BrandAssets

logger = logging.getLogger("ai_ad_engine.scraper")

# Keyword maps for business niche classification
NICHE_MAPPINGS = [
    (r"\b(ecommerce|shop|store|cart|buy online|retail)\b", "E-Commerce & Digital Retail"),
    (r"\b(code|coding|software|developer|programming|bootcamp|club|web development|dev|tech|tech workspace|python|javascript|fullstack|engineering)\b", "Software & Technology"),
    (r"\b(tactical|firearms|outdoor|hunting|shooting|defense|military|gear)\b","Outdoors & Tactical Gear"),
    (r"\b(hvac|air condition|heating|furnace|duct|cooling|ventilation|ac repair)\b", "HVAC & Climate Control"),
    (r"\b(real estate|realtor|realty|property|villas|condos|apartments|homes for sale|mortgage)\b", "Luxury Real Estate"),
    (r"\b(coffee|roast|espresso|cafe|barista|latte|brew)\b", "Artisan Coffee Shop & Roastery"),
    (r"\b(dentist|dental|teeth|orthodont|oral care|smile)\b", "Dental Care & Orthodontics"),
    (r"\b(plumb|drain|pipe|water heater|clog)\b", "Plumbing & Rooter Services"),
    (r"\b(roof|roofing|shingle|gutters)\b", "Roofing & Exterior Contractors"),
    (r"\b(gym|fitness|workout|personal train|crossfit|pilates|yoga)\b", "Fitness & Wellness Studio"),
    (r"\b(lawyer|attorney|law firm|legal|litigation|counsel)\b", "Legal Services & Law Practice"),
    (r"\b(auto repair|mechanic|car service|tires|brakes|oil change|detailing)\b", "Automotive Service & Repair"),
    (r"\b(restaurant|dining|bistro|cuisine|menu|chef|eatery|bakery)\b", "Restaurant & Culinary Dining"),
    (r"\b(saas|cloud|platform|ai|analytics|crm|api)\b", "SaaS & Cloud Technology"),
    (r"\b(clean|maid|janitorial|carpet cleaning|housekeeping)\b", "Professional Cleaning Services"),
    (r"\b(landscape|lawn care|tree service|hardscape|irrigation|gardening)\b", "Landscaping & Outdoor Design"),
    (r"\b(construct|remodel|renovat|contractor|builder|carpentry)\b", "Construction & Home Remodeling"),
    (r"\b(vet|veterinar|animal hospital|pet care|dog grooming)\b", "Veterinary & Pet Care Services"),
    (r"\b(cloth|apparel|fashion|boutique|jewelry|shoes|wear|accessories)\b", "Fashion & Apparel Retail"),
]

# Niche-specific default visual elements for rich image synthesis
NICHE_VISUAL_DEFAULTS = {
    "Software & Technology": [
        "Sleek modern tech workspace with software developers collaborating on dual-monitor code workstations",
        "Close-up of modern IDE code editor and glowing syntax on ultra-wide monitor with ambient studio backlighting",
        "Dynamic tech team gathered around a modern conference table reviewing software architecture",
        "Modern laptop displaying clean code and dashboard in a bright sunlit tech office",
    ],
        # Add to NICHE_VISUAL_DEFAULTS in your sanitizer / visual service:
    "Outdoors & Tactical Gear": [
        "High-end tactical outdoor apparel and rugged gear displayed in a clean modern showroom",
        "Close-up of durable outdoor equipment and precision accessories on a dark slate background",
        "Cinematic outdoor photography of adventure gear in rugged wilderness lighting",
        "Professional product display of tactical boots, backpacks, and outdoor utilities",
    ],
    "HVAC & Climate Control": [
        "Modern service van parked outside a clean suburban home",
        "Certified technician inspecting high-efficiency air conditioning condenser",
        "Sleek digital smart thermostat glowing in a bright modern living room",
        "Clean, comfortable family home interior with cool refreshing airflow",
    ],
    "Luxury Real Estate": [
        "Sun-drenched architectural villa with floor-to-ceiling glass walls",
        "Spacious open-concept living room with designer furniture and marble finishes",
        "Infinity pool overlooking scenic mountain or ocean horizon at sunset",
        "Elegant kitchen with waterfall marble island and pendant lighting",
    ],
    "Artisan Coffee Shop & Roastery": [
        "Warm, inviting cafe interior with rustic wooden accents and emerald plants",
        "Barista skillfully crafting latte art on a polished espresso machine",
        "Single-origin roasted coffee beans cascading into ceramic bowls",
        "Happy patrons enjoying artisanal brews by large sunlit cafe windows",
    ],
    "Dental Care & Orthodontics": [
        "State-of-the-art modern dental clinic with soothing ambient lighting",
        "Friendly dentist discussing treatment in a high-tech exam room",
        "Bright, radiant healthy smile with perfect white teeth",
        "Ultra-clean clinical environment with advanced digital scanning equipment",
    ],
    "Plumbing & Rooter Services": [
        "Professional plumber with modern diagnostic tools and inspection camera",
        "Immaculate modern bathroom with pristine polished chrome fixtures",
        "Clean, organized service vehicle equipped with specialized repair equipment",
        "Relieved homeowner shaking hands with technician after quick fix",
    ],
    "Fitness & Wellness Studio": [
        "Dynamic modern fitness studio with sleek weights and ambient LED lighting",
        "Athletic individuals performing energized workouts with focus",
        "Clean wellness area with yoga mats, natural wood, and green plants",
        "High-energy coach motivating clients with positive enthusiasm",
    ],
    "Legal Services & Law Practice": [
        "Prestigious contemporary law office with mahogany conference table and city views",
        "Professional attorney reviewing documents with confident authority",
        "Warm, reassuring consultation with client in an executive boardroom",
        "Architectural marble courthouse columns under clear blue sky",
    ],
    "Automotive Service & Repair": [
        "Spotless high-tech auto repair bay with car raised on hydraulic lift",
        "Master mechanic examining engine diagnostics on a tablet computer",
        "Gleaming freshly detailed vehicle reflecting showroom spotlighting",
        "Precision tools organized neatly against modern shop wall",
    ],
    "Restaurant & Culinary Dining": [
        "Gourmet culinary dish beautifully plated with vibrant microgreens",
        "Cozy, atmospheric restaurant dining room with warm candlelight",
        "Executive chef finishing artful presentation in an open kitchen",
        "Happy dinner guests toasting glasses in a vibrant dining ambiance",
    ],
    "SaaS & Cloud Technology": [
        "Sleek glass tech office with glowing data visualization dashboards",
        "Collaborative engineering team working on modern ultra-wide monitors",
        "Modern laptop on minimalist wooden desk displaying clean UI graphs",
        "Vibrant futuristic tech workspace with natural daylight and indoor greenery",
    ],
    "E-Commerce & Digital Retail": [
        "High-end product showcase on minimalist studio pedestal with soft shadows",
        "Crisp, beautifully packaged items arranged in a clean aesthetic layout",
        "Satisfied customer unboxing a premium order in a stylish sunlit room",
        "Modern shopping bag and lifestyle accessories on textured marble",
    ],
    "Construction & Home Remodeling": [
        "Beautifully renovated open-concept kitchen and living area with recessed lighting",
        "Architect and contractor reviewing blueprints at an active job site",
        "High-end craftsmanship detail on custom hardwood and stonework",
        "Stunning before-and-after home exterior transformation in golden sunlight",
    ],
}

NICHE_COLOR_PALETTES = {
    "Software & Technology": ["#1E3A8A", "#3B82F6"],         # Deep Navy Blue & Vibrant Royal Blue
    "HVAC & Climate Control": ["#0284C7", "#F97316"],       # Cool Sky Blue & Warm Orange
    "Luxury Real Estate": ["#1E293B", "#D4AF37"],           # Deep Slate & Champagne Gold
    "Artisan Coffee Shop & Roastery": ["#451A03", "#0F5338"], # Dark Roast Brown & Forest Green
    "Dental Care & Orthodontics": ["#0EA5E9", "#10B981"],   # Fresh Cyan & Mint Green
    "Plumbing & Rooter Services": ["#1D4ED8", "#F59E0B"],   # Cobalt Blue & Amber
    "Fitness & Wellness Studio": ["#EF4444", "#111827"],    # High-Energy Crimson & Charcoal
    "Legal Services & Law Practice": ["#1E3A8A", "#94A3B8"], # Navy Blue & Classic Silver
    "Automotive Service & Repair": ["#DC2626", "#374151"],  # Racing Red & Industrial Steel
    "Restaurant & Culinary Dining": ["#B91C1C", "#F59E0B"], # Warm Wine Red & Saffron Gold
    "SaaS & Cloud Technology": ["#6366F1", "#06B6D4"],      # Indigo & Cyber Cyan
    "E-Commerce & Digital Retail": ["#0F172A", "#EC4899"],  # Midnight Obsidian & Vibrant Magenta
    "Construction & Home Remodeling": ["#B45309", "#334155"], # Warm Amber & Slate
}


def _extract_colors_from_css_and_html(soup: BeautifulSoup, html: str) -> List[str]:
    """Extract distinct brand hex colors from meta tags, inline styles, and CSS properties."""
    colors: List[str] = []

    # 1. Meta theme colors
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in {"theme-color", "msapplication-tilecolor", "msapplication-navbutton-color"}:
            val = (meta.get("content") or "").strip()
            if re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", val):
                colors.append(val.upper())

    # 2. CSS custom properties and styles in raw HTML (looking for brand / primary color vars)
    css_var_matches = re.findall(
        r"(?:--primary|--brand|--accent|--theme|--main-color|--color-primary)[^:]*:\s*(#[0-9a-fA-F]{3,6})\b",
        html,
        re.IGNORECASE,
    )
    for c in css_var_matches:
        colors.append(c.upper())

    # 3. Scan common color declarations in inline styles
    hex_matches = re.findall(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b", html[:25000])
    generic_colors = {"#FFFFFF", "#FFF", "#000000", "#000", "#EEE", "#EEEEEE", "#CCC", "#CCCCCC", "#F8F8F8"}
    for h in hex_matches:
        normalized = h.upper()
        if normalized not in generic_colors and normalized not in colors:
            colors.append(normalized)
            if len(colors) >= 4:
                break

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(colors))
    return deduped[:3]


def _infer_niche(text_corpus: str, site_title: str, json_ld_types: List[str]) -> str:
    """Infer the specific business niche from JSON-LD types, keywords, title, and body text."""
    combined = f"{site_title} {' '.join(json_ld_types)} {text_corpus}".lower()

    for pattern, niche_name in NICHE_MAPPINGS:
        if re.search(pattern, combined, re.IGNORECASE):
            return niche_name

    # Check title specifically for tech / code / digital
    if any(k in site_title.lower() for k in ("code", "club", "tech", "soft", "dev", "app", "program")):
        return "Software & Technology"

    # Fallback to specific descriptive categories
    if any(k in combined for k in ("product", "shop", "cart", "buy", "store")):
        return "E-Commerce & Digital Retail"
    if any(k in combined for k in ("service", "solution", "consult", "agency", "firm")):
        return f"{site_title} Professional Solutions" if site_title else "Modern Commercial Solutions"
    return f"{site_title} Digital Solutions" if site_title else "Modern Technology & Solutions"


def _extract_json_ld(soup: BeautifulSoup) -> tuple[List[str], List[str], List[str]]:
    """Extract structured data from JSON-LD scripts before decomposition."""
    types: List[str] = []
    services: List[str] = []
    products: List[str] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text()
            if not raw:
                continue
            data = json.loads(raw.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                t = item.get("@type")
                if t:
                    if isinstance(t, list):
                        types.extend(t)
                    else:
                        types.append(str(t))

                # Extract services / offers
                offers = item.get("hasOfferCatalog", {}).get("itemListElement", []) or item.get("offers", [])
                if isinstance(offers, list):
                    for offer in offers:
                        if isinstance(offer, dict):
                            name = offer.get("name") or offer.get("itemOffered", {}).get("name")
                            if name and isinstance(name, str):
                                services.append(name.strip())

                # Check direct name/description of services/products
                if t in {"Product", "IndividualProduct"}:
                    p_name = item.get("name")
                    if p_name and isinstance(p_name, str):
                        products.append(p_name.strip())
                elif t in {"Service", "ProfessionalService"}:
                    s_name = item.get("name")
                    if s_name and isinstance(s_name, str):
                        services.append(s_name.strip())
        except Exception:
            continue

    return types, services, products


def _extract_services_and_products(soup: BeautifulSoup, json_services: List[str], json_products: List[str]) -> tuple[List[str], List[str]]:
    """Extract clean core services and products from headings, list elements, and feature cards."""
    services: List[str] = list(json_services)
    products: List[str] = list(json_products)

    # Search for headings and cards in service/product sections
    for el in soup.find_all(["h2", "h3", "h4", "li", "p"]):
        # Check parents or class names for hints
        classes = " ".join(el.get("class", [])) if isinstance(el.get("class"), list) else str(el.get("class") or "")
        parent_classes = " ".join(el.parent.get("class", [])) if el.parent and isinstance(el.parent.get("class"), list) else ""
        combined_attrs = f"{classes} {parent_classes}".lower()

        text = el.get_text(separator=" ", strip=True)
        if not text or len(text) < 4 or len(text) > 80:
            continue
        # Skip generic web navigation phrases
        if any(ign in text.lower() for ign in ("privacy", "cookie", "terms", "copyright", "home", "contact", "about us", "menu", "sign in")):
            continue

        if any(k in combined_attrs for k in ("service", "feature", "offer", "solution", "benefit")):
            services.append(text)
        elif any(k in combined_attrs for k in ("product", "shop", "item", "catalog", "goods")):
            products.append(text)

    # Deduplicate and clean up
    clean_services = list(dict.fromkeys([s for s in services if len(s) > 3]))[:5]
    clean_products = list(dict.fromkeys([p for p in products if len(p) > 3]))[:5]
    return clean_services, clean_products


def _extract_key_visual_elements(soup: BeautifulSoup, niche: str, services: List[str]) -> List[str]:
    """Extract visual cues from image alt text, niche templates, and service offerings."""
    visuals: List[str] = []

    # 1. Alt attributes of relevant images
    for img in soup.find_all("img"):
        alt = (img.get("alt") or img.get("title") or "").strip()
        if alt and len(alt) > 5 and len(alt) < 100:
            if not any(ign in alt.lower() for ign in ("logo", "icon", "arrow", "badge", "button", "avatar", "tracking")):
                visuals.append(alt)

    # 2. Add niche-aligned visual defaults
    defaults = NICHE_VISUAL_DEFAULTS.get(niche, [
        f"Professional commercial showcase for {niche}",
        "Modern bright interior with clean architectural lighting",
        "High quality product and service highlight with crisp detail",
        "Happy satisfied customers enjoying premium experience",
    ])
    for d in defaults:
        if d not in visuals:
            visuals.append(d)

    # Deduplicate while preserving variety
    clean_visuals = list(dict.fromkeys(visuals))[:6]
    return clean_visuals


async def scrape_site(url: str) -> BrandAssets:
    """Rich brand scraping using httpx + BeautifulSoup.
    Extracts brand identity metadata: site_title, business_niche, brand_colors,
    core_services/products, key_visual_elements, logos, and product images.
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
        fallback_niche = _infer_niche("", brand_name, [])
        fallback_colors = NICHE_COLOR_PALETTES.get(fallback_niche, ["#1E3A8A", "#3B82F6"])
        fallback_visuals = NICHE_VISUAL_DEFAULTS.get(fallback_niche, [
            f"Cinematic showcase of {brand_name} modern operations",
            "Professional high-end service highlight with pristine lighting",
            "Happy customers interacting with premium solutions",
        ])

        return BrandAssets(
            site_title=brand_name,
            description=f"{brand_name} provides premier {fallback_niche} solutions.",
            business_niche=fallback_niche,
            brand_colors=fallback_colors,
            core_services=[f"{brand_name} Core Solutions", "Premium On-Demand Services", "Quality Customer Care"],
            products=[f"{brand_name} Signature Offering"],
            key_visual_elements=fallback_visuals,
            logo_url=None,
            product_images=[],
            primary_color=fallback_colors[0],
            raw_text_snippet=f"Welcome to {brand_name}. Premier {fallback_niche} solutions built for excellence.",
        )

    soup = BeautifulSoup(html, "html.parser")

    # Step 1: Extract structured JSON-LD before tag decomposition
    json_ld_types, json_services, json_products = _extract_json_ld(soup)

    # Step 2: Extract Brand Colors from CSS and meta tags before styles are decomposed
    brand_colors = _extract_colors_from_css_and_html(soup, html)

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

    if not title:
        domain = url.split("//")[-1].split("/")[0].replace("www.", "")
        title = domain.split(".")[0].capitalize()

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
        if h_text and 4 < len(h_text) < 120:
            headings.append(h_text)
    headings = list(dict.fromkeys(headings))[:5]

    # 4. Paragraphs / Corpus text
    paragraphs = []
    for p in soup.find_all(["p", "li"]):
        p_text = p.get_text(separator=" ", strip=True)
        if len(p_text) > 20 and not any(k in p_text.lower() for k in ("cookie", "privacy policy", "terms of service", "all rights reserved")):
            paragraphs.append(p_text)

    text_corpus = f"{' '.join(headings)} {' '.join(paragraphs[:6])}"

    # 5. Infer Business Niche
    business_niche = _infer_niche(text_corpus, title, json_ld_types)

    # 6. Resolve Brand Colors & ensure at least 2 brand tones exist
    if not brand_colors:
        brand_colors = NICHE_COLOR_PALETTES.get(business_niche, ["#1E3A8A", "#3B82F6"])
    elif len(brand_colors) == 1:
        palette = NICHE_COLOR_PALETTES.get(business_niche, ["#1E3A8A", "#3B82F6"])
        if palette[1] not in brand_colors:
            brand_colors.append(palette[1])
        else:
            brand_colors.append("#F8FAFC")

    primary_color = brand_colors[0] if brand_colors else "#1E3A8A"

    # 7. Core Services & Products
    core_services, products = _extract_services_and_products(soup, json_services, json_products)
    if not core_services:
        if headings:
            core_services = [h for h in headings if len(h) < 60][:3]
        else:
            core_services = [f"Professional {business_niche} Services", "Custom Tailored Solutions"]

    # 8. Key Visual Elements
    key_visual_elements = _extract_key_visual_elements(soup, business_niche, core_services)

    # 9. Logo extraction
    logo = None
    icon = soup.find("link", rel=lambda x: x and any(i in x for i in ("icon", "apple-touch-icon", "shortcut")))
    if icon and icon.get("href"):
        logo = _abspath(url, icon.get("href"))
    og_image = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
    if og_image and og_image.get("content"):
        if not logo:
            logo = og_image.get("content")

    # 10. Product / Content Images
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

    # 11. Raw Text Snippet
    snippet_parts = []
    if headings:
        snippet_parts.append("Key Highlights: " + " | ".join(headings[:3]))
    if core_services:
        snippet_parts.append("Services: " + ", ".join(core_services[:4]))
    if paragraphs:
        snippet_parts.append("Summary: " + " ".join(paragraphs[:3]))
    raw_text_snippet = "\n".join(snippet_parts)[:800] if snippet_parts else (desc or title or "")

    assets = BrandAssets(
        site_title=title,
        description=desc,
        business_niche=business_niche,
        brand_colors=brand_colors,
        core_services=core_services,
        products=products,
        key_visual_elements=key_visual_elements,
        logo_url=logo,
        product_images=product_images,
        primary_color=primary_color,
        raw_text_snippet=raw_text_snippet,
    )

    logger.info(
        "Scraped assets: title=%s, niche=%s, colors=%s, services=%d, visuals=%d",
        title,
        business_niche,
        brand_colors,
        len(core_services),
        len(key_visual_elements),
    )
    return assets


def _abspath(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href

