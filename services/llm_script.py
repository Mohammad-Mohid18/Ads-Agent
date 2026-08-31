import json
import logging
import os
import re
import uuid
from typing import List, Optional

import httpx
from models import BrandAssets, Scene, Script

logger = logging.getLogger("ai_ad_engine.llm")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = (os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3.3-70b-instruct:free").strip().strip("'\"")


def _clean_text(text: str) -> str:
    """Clean stage directions, markdown, or raw JSON leaks from voiceover lines."""
    if not text:
        return ""
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
    text = re.sub(r"[\*\_\#\~]", "", text)
    return text.strip(' "\'\n\r\t')


NAMED_COLORS = [
    # Blues & Navies
    ((30, 58, 138), "deep navy blue"),
    ((59, 130, 246), "vibrant royal blue"),
    ((37, 99, 235), "electric blue"),
    ((2, 132, 199), "crisp sky blue"),
    ((14, 165, 233), "bright cyan blue"),
    ((6, 182, 212), "cyber cyan"),
    ((99, 102, 241), "modern tech indigo"),
    ((79, 70, 229), "deep indigo"),
    ((30, 41, 59), "deep slate"),
    ((15, 23, 42), "midnight dark slate"),
    # Greens
    ((15, 83, 56), "emerald forest green"),
    ((16, 185, 129), "fresh mint green"),
    ((34, 197, 94), "vibrant leafy green"),
    ((22, 101, 52), "deep forest green"),
    # Warm, Earthy & Browns
    ((69, 26, 3), "warm dark roast coffee brown"),
    ((120, 53, 15), "rich amber brown"),
    ((139, 90, 43), "warm natural wood brown"),
    ((180, 83, 9), "warm bronze amber"),
    ((212, 175, 55), "champagne gold"),
    ((245, 158, 11), "warm golden amber"),
    ((249, 115, 22), "vibrant sunset orange"),
    # Reds & Pinks
    ((239, 68, 68), "energetic crimson red"),
    ((185, 28, 28), "deep wine red"),
    ((220, 38, 38), "bold ruby red"),
    ((236, 72, 153), "vibrant magenta"),
    ((168, 85, 247), "glowing violet"),
    # Neutrals
    ((17, 24, 39), "sleek charcoal dark"),
    ((0, 0, 0), "obsidian black"),
    ((255, 255, 255), "pure crisp white"),
    ((148, 163, 184), "cool silver steel"),
]


def hex_to_natural_color(hex_code: str) -> str:
    """Convert a hex color code like #1E3A8A into a vivid natural language color name."""
    if not hex_code or not isinstance(hex_code, str):
        return "modern brand palette"

    clean = hex_code.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return "brand color tone"

    try:
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
    except ValueError:
        return "brand color tone"

    best_color = "modern brand accent"
    best_dist = float("inf")
    for (cr, cg, cb), name in NAMED_COLORS:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_color = name

    return best_color


def format_natural_color_palette(brand_colors: Optional[List[str]], primary_color: Optional[str] = None) -> str:
    """
    Convert a list of brand hex colors into a natural language lighting and ambiance phrase.
    Example: ['#1E3A8A', '#3B82F6'] -> 'deep navy blue and vibrant royal blue studio lighting accents'
    """
    colors = [c for c in (brand_colors or []) if c]
    if not colors and primary_color:
        colors = [primary_color]

    if not colors:
        return "warm studio lighting accents"

    natural_names: List[str] = []
    for c in colors[:2]:
        name = hex_to_natural_color(c)
        if name not in natural_names:
            natural_names.append(name)

    if len(natural_names) == 1:
        return f"{natural_names[0]} studio lighting accents"
    elif len(natural_names) >= 2:
        return f"{natural_names[0]} and {natural_names[1]} studio lighting accents"
    return "balanced cinematic lighting"


def infer_business_category(assets: BrandAssets) -> str:
    """Dynamic universal fallback using the extracted business niche or brand title."""
    if assets.business_niche and assets.business_niche != "Commercial Business & Services":
        return assets.business_niche.strip()

    title = (assets.site_title or "").lower()
    if any(k in title for k in ("code", "club", "software", "tech", "dev", "app", "program")):
        return "Software & Technology"
    if any(k in title for k in ("coffee", "cafe", "roast", "brew", "espresso")):
        return "Artisan Coffee Shop & Roastery"
    if any(k in title for k in ("hvac", "air", "heat", "cool", "climate")):
        return "HVAC & Climate Control"
    if any(k in title for k in ("real estate", "realtor", "home", "villa", "property")):
        return "Luxury Real Estate"
    if any(k in title for k in ("dent", "smile", "ortho")):
        return "Dental Care & Orthodontics"
    if any(k in title for k in ("plumb", "pipe", "drain")):
        return "Plumbing & Rooter Services"
    if any(k in title for k in ("gym", "fit", "workout", "train")):
        return "Fitness & Wellness Studio"
    if any(k in title for k in ("clean", "maid", "wash")):
        return "Professional Cleaning Services"

    return f"{assets.site_title} Solutions" if assets.site_title else "Modern Technology & Solutions"


def _generate_fallback_scene_prompt(
    assets: BrandAssets,
    role: str,
    scene_idx: int,
    spoken_text: str = "",
) -> str:
    """Generate a hyper-specific brand-consistent image prompt for a given scene role."""
    brand = assets.site_title or "Our Team"
    niche = assets.business_niche or infer_business_category(assets)
    color_phrase = format_natural_color_palette(assets.brand_colors, assets.primary_color)
    visuals = assets.key_visual_elements or []
    services = assets.core_services or []

    v_elem = visuals[scene_idx % len(visuals)] if visuals else f"modern professional setting for {brand}"
    s_elem = services[scene_idx % len(services)] if services else f"expert {niche} solutions"

    if role == "hook" or scene_idx == 0:
        return (
            f"A sleek modern {niche} workspace for {brand} featuring {v_elem}, "
            f"illuminated by {color_phrase}, cinematic wide shot, clean aesthetic, "
            f"photorealistic, 8k resolution, zero text overlay, no text, no typography, no watermark"
        )
    elif role == "problem" or scene_idx == 1:
        return (
            f"Shallow depth-of-field close-up shot for {brand} focusing on {s_elem} and {v_elem}, "
            f"subtly accentuated by {color_phrase}, dramatic studio lighting, razor-sharp details, "
            f"photorealistic, 8k resolution, zero text overlay, no text, no typography, no watermark"
        )
    elif role == "solution" or scene_idx == 2:
        return (
            f"Professional eye-level shot of specialists delivering {s_elem} at {brand}, "
            f"in a high-end modern atmosphere with {color_phrase}, authentic commercial aesthetic, "
            f"photorealistic, 8k resolution, zero text overlay, no text, no typography, no watermark"
        )
    else:  # cta or closing
        return (
            f"Hero showcase for {brand} ({niche}) displaying {v_elem}, "
            f"with dramatic lighting, clean minimalist composition, {color_phrase}, "
            f"professional advertising cinematography, 8k resolution, zero text overlay, no text, no typography, no watermark"
        )


async def generate_script(
    assets: BrandAssets,
    aspect_ratio: str = "16:9",
    required_scenes: int | None = None,
) -> Script:
    """Generate an ad script with exact scene count, voiceover lines, and tailored image prompts."""
    if required_scenes is not None and required_scenes < 1:
        raise ValueError("required_scenes must be at least 1")
    if not LLM_API_KEY:
        logger.warning("LLM_API_KEY not set; generating heuristic script")
        return _heuristic_script(assets, required_scenes=required_scenes)

    prompt = _build_prompt(assets, aspect_ratio, required_scenes=required_scenes)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert commercial advertising director and AI visual prompt designer. "
                    "Always output strict, valid JSON with voiceover text and hyper-detailed photorealistic image prompts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.25,
        "max_tokens": 1000,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(LLM_API_URL, headers=headers, json=body)
            r.raise_for_status()
            j = r.json()

        content = j.get("choices", [])[0].get("message", {}).get("content", "")

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        parsed = json.loads(content)
        business_category = parsed.get("business_category", "").strip() or infer_business_category(assets)

        scenes = []
        t = 0.0
        for idx, s in enumerate(parsed.get("scenes", [])):
            dur = float(s.get("duration", 3.5))
            clean_text = _clean_text(s.get("text", ""))

            # Extract detailed image_prompt with fallback
            v_prompt = (s.get("image_prompt") or s.get("visual_prompt") or "").strip()
            if not v_prompt or len(v_prompt) < 15:
                v_prompt = _generate_fallback_scene_prompt(
                    assets, s.get("role", "scene"), idx, clean_text
                )

            scene = Scene(
                id=str(uuid.uuid4()),
                start=t,
                end=t + dur,
                role=s.get("role", "scene"),
                text=clean_text,
                visual_prompt=v_prompt,
                image_prompt=v_prompt,
            )
            scenes.append(scene)
            t += dur

        if required_scenes is not None:
            scenes = _fit_scene_count(scenes, assets, required_scenes)
            t = 0.0
            for scene in scenes:
                duration = max(scene.end - scene.start, 1.5)
                scene.start = t
                scene.end = t + duration
                t += duration

        logger.info(
            "Generated %d-scene ad script for niche '%s' with rich visual prompts",
            len(scenes),
            business_category,
        )
        return Script(duration=t, scenes=scenes, business_category=business_category)

    except Exception as e:
        logger.warning("LLM generation failed; using heuristic fallback: %s", e)
        return _heuristic_script(assets, required_scenes=required_scenes)


def _fit_scene_count(scenes: list[Scene], assets: BrandAssets, required_scenes: int) -> list[Scene]:
    """Trim or supplement an LLM response so template slots are always satisfied."""
    scenes = scenes[:required_scenes]
    fallback = _heuristic_script(assets, required_scenes=required_scenes).scenes
    while len(scenes) < required_scenes:
        fallback_scene = fallback[len(scenes)]
        scenes.append(fallback_scene)
    return scenes


def _heuristic_script(assets: BrandAssets, required_scenes: int | None = None) -> Script:
    """Universal high-quality fallback script with rich brand-specific visual prompts."""
    category = infer_business_category(assets)
    scene_count = required_scenes or 4
    durations = [3.5, 4.0, 4.5, 3.5]
    roles = ["hook", "problem", "solution", "cta"]

    brand_name = assets.site_title or "Our Team"
    services = assets.core_services or [f"Top-Rated {category} Solutions", "Trusted Local Experts"]
    svc_1 = services[0] if len(services) > 0 else f"{category} Services"
    svc_2 = services[1] if len(services) > 1 else "Fast, reliable results"

    texts = [
        f"Discover premium {category} with {brand_name}.",
        f"Tired of unreliable service? Get {svc_1} you can trust.",
        f"Experience {svc_2} tailored to your exact needs.",
        f"Contact {brand_name} today and experience the difference.",
    ]

    scenes = []
    t = 0.0
    for i in range(scene_count):
        duration = durations[i % len(durations)]
        role = roles[i % len(roles)]
        prompt = _generate_fallback_scene_prompt(assets, role, i, texts[i % len(texts)])
        scenes.append(
            Scene(
                id=str(uuid.uuid4()),
                start=t,
                end=t + duration,
                role=role,
                text=texts[i % len(texts)],
                visual_prompt=prompt,
                image_prompt=prompt,
            )
        )
        t += duration
    return Script(duration=t, scenes=scenes, business_category=category)


def _build_prompt(assets: BrandAssets, aspect_ratio: str, required_scenes: int | None = None) -> str:
    snippet = (assets.raw_text_snippet or "")[:500].replace("\n", " ")
    niche = assets.business_niche or infer_business_category(assets)
    natural_colors = format_natural_color_palette(assets.brand_colors, assets.primary_color)
    raw_colors = ", ".join(assets.brand_colors) if assets.brand_colors else (assets.primary_color or "modern palette")
    colors = f"{natural_colors} ({raw_colors})"
    services = ", ".join(assets.core_services) if assets.core_services else "N/A"
    products = ", ".join(assets.products) if assets.products else "N/A"
    visuals = ", ".join(assets.key_visual_elements) if assets.key_visual_elements else "N/A"

    return (
        "You are an expert commercial video director and copywriter. "
        "Write a high-converting 15-second commercial video ad script with hyper-specific visual prompts based on the brand context below.\n\n"
        "RULES:\n"
        f"1. Return exactly {required_scenes or 4} scenes.\n"
        "2. Spoken voiceover 'text' MUST be short and punchy (under 12 words / 70 characters per scene).\n"
        "3. 'image_prompt' MUST be a highly detailed, photorealistic prompt crafted for image diffusion models.\n"
        "   - Explicitly incorporate the business niche, brand color palette/ambiance (translate colors to descriptive lighting like 'deep navy blue and vibrant royal blue studio lighting'), and core product/service context.\n"
        "   - Specify photorealistic lighting (e.g., golden hour sunlight, soft studio lighting, cinematic rim light).\n"
        "   - Specify camera perspective (e.g., cinematic wide shot, shallow depth-of-field close-up, dynamic eye-level shot).\n"
        "   - Specify high detail (photorealistic, 4k, 8k resolution).\n"
        "   - MANDATE ZERO text overlays inside the image (no typography, no watermarks, no logos).\n"
        "4. DO NOT include markdown, stage directions, or metadata inside 'text'.\n\n"
        "STRICT JSON OUTPUT FORMAT ONLY:\n"
        "{\n"
        f'  "business_category": "{niche}",\n'
        '  "scenes": [\n'
        '    {\n'
        '      "scene_number": 1,\n'
        '      "role": "hook",\n'
        '      "duration": 3.5,\n'
        '      "text": "Short punchy spoken sentence under 12 words.",\n'
        '      "image_prompt": "A sleek modern tech workspace for Code Club featuring software developers working on code, illuminated by deep navy blue and vibrant royal blue studio lighting, cinematic wide shot, clean aesthetic, 8k resolution, no text"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        f"Brand Name: {assets.site_title or 'Brand'}\n"
        f"Business Niche: {niche}\n"
        f"Brand Palette & Lighting: {colors}\n"
        f"Core Services: {services}\n"
        f"Products: {products}\n"
        f"Key Visual Elements: {visuals}\n"
        f"Description: {assets.description or 'N/A'}\n"
        f"Scraped Web Context: {snippet or 'N/A'}\n"
        f"Aspect Ratio: {aspect_ratio}\n"
    )

