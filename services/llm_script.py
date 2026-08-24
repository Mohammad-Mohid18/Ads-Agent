"""Script generation via OpenRouter with business category + unique visual_prompt validation."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid

import httpx

from models import BrandAssets, Script, Scene

logger = logging.getLogger("ai_ad_engine.llm")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

SYSTEM_PROMPT = """You are an elite video-ad creative director and commercial visual designer.
Your task is to write a high-converting, dynamic video ad script for the provided brand.

Return ONLY valid JSON (no markdown fences, no explanatory text) with this exact schema:
{
  "business_category": "<concise niche label, e.g. Social Media App, E-commerce SaaS, FinTech Platform, Lifestyle Apparel, Fitness Tech>",
  "target_audience": "<key target demographic / persona, e.g. Gen-Z creators, Busy professionals, Modern shoppers>",
  "brand_core": "<unique value proposition / core differentiator>",
  "scenes": [
    {
      "role": "hook" | "problem" | "solution" | "benefit" | "cta",
      "duration": <number seconds between 2.5 and 6.0>,
      "text": "<punchy spoken narration line, max 18 words>",
      "visual_prompt": "<detailed, brand-specific commercial photography prompt for fal.ai FLUX image generation>"
    }
  ]
}

CRITICAL RULES:
1. SCENE COUNT: You MUST generate EXACTLY 5 distinct scenes in order:
   - Scene 1: "hook" (Instant visual grabber capturing the core product vibe or curiosity)
   - Scene 2: "problem" (The frustrating pain point, barrier, or boredom before using this brand)
   - Scene 3: "solution" (Hero product showcase / feature in active use solving the problem)
   - Scene 4: "benefit" (Lifestyle transformation, joyful outcome, or power-user delight)
   - Scene 5: "cta" (Compelling, inspiring final call-to-action still)

2. STRICT VISUAL PROMPT REQUIREMENTS:
   - Every single scene MUST have a completely UNIQUE, HIGHLY SPECIFIC visual_prompt tailored directly to the business.
   - NEVER use generic stock photo prompts like "a person smiling at a laptop", "analytics dashboard with graphs", "two people shaking hands in an office", or "generic modern workplace".
   - Ground each visual_prompt in the brand's authentic identity, products, and aesthetic.
     * For example, for Snapchat: "Vibrant smartphone screen showing active video chatting with colorful AR lenses, neon lighting, Gen-Z aesthetic, cinematic portrait, photorealistic, 8k"
     * For a fitness brand: "High-energy athlete pushing through an intense HIIT sprint in a moody neon-lit gym, sweat glistening, dynamic motion blur, commercial sports photography"
   - Every scene MUST specify a distinct camera angle (e.g., dynamic close-up, dramatic low-angle hero shot, wide cinematic lifestyle view), lighting style (e.g., golden hour, moody studio neon, soft natural morning light), and subject action so NO TWO SCENES LOOK ALIKE.
   - DO NOT include on-screen text, typography, UI watermarks, or logos in the visual_prompt.
"""


async def generate_script(assets: BrandAssets, aspect_ratio: str = "16:9") -> Script:
    """Call the LLM to generate a structured 5-scene script JSON with unique visual prompts."""
    if not LLM_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set; generating heuristic script")
        return _heuristic_script(assets)

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-ads-maker",
        "X-Title": "AI Ads Maker",
    }
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(assets, aspect_ratio)},
        ],
        "temperature": 0.4,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(LLM_API_URL, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or message.get("reasoning") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not str(content).strip():
            raise ValueError(f"LLM returned empty content: {payload}")
        parsed = _parse_json_content(str(content))
        return _scenes_from_parsed(parsed, assets)
    except Exception as exc:
        logger.warning("LLM generation failed; using heuristic script: %s", exc)
        return _heuristic_script(assets)


def _parse_json_content(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Extract outer JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            snippet = content[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # Clean trailing commas before closing braces/brackets
                cleaned = re.sub(r",\s*([\]}])", r"\1", snippet)
                return json.loads(cleaned)
        raise



def infer_business_category(assets: BrandAssets) -> str:
    """Heuristic niche classification from scraped brand context."""
    blob = " ".join(
        filter(
            None,
            [
                assets.site_title or "",
                assets.description or "",
                assets.raw_text_snippet or "",
            ],
        )
    ).lower()
    rules = [
        (("snapchat", "tiktok", "instagram", "social", "chat", "camera", "lens", "stories", "ar"), "Social Media & Camera Platform"),
        (("coffee", "cafe", "espresso", "bakery", "roastery"), "Artisanal Coffee Brand"),
        (("fitness", "gym", "workout", "yoga", "training", "athlete", "running"), "Fitness & Wellness Brand"),
        (("bank", "finance", "fintech", "trading", "stock", "invest", "crypto", "pay"), "Fintech Platform"),
        (("saas", "software", "cloud", "api", "developer", "automation", "ai tool"), "SaaS & AI Platform"),
        (("shop", "store", "ecommerce", "e-commerce", "retail", "fashion", "apparel", "clothing"), "E-commerce & Lifestyle Brand"),
        (("education", "school", "course", "learn", "coding", "academy", "tutor"), "EdTech Platform"),
        (("health", "clinic", "medical", "dental", "pharma", "therapy"), "Healthcare Brand"),
        (("travel", "hotel", "tour", "flight", "resort", "booking"), "Travel & Adventure Brand"),
        (("food", "restaurant", "dining", "meal", "delivery", "kitchen"), "Food & Beverage Brand"),
    ]
    for keywords, label in rules:
        if any(k in blob for k in keywords):
            return label
    brand = assets.site_title or "Business"
    return f"{brand} Brand"


def _coerce_visual_prompt(raw: object, role: str, text: str, assets: BrandAssets, idx: int = 0) -> str:
    if isinstance(raw, str) and len(raw.strip()) > 15:
        return raw.strip()
    brand = assets.site_title or "the brand"
    category = infer_business_category(assets)
    desc = (assets.description or assets.raw_text_snippet or text or brand).strip()[:160]
    
    role_angles = {
        "hook": f"Dynamic eye-catching commercial opening shot for {brand} ({category}), high-energy focal subject, vibrant colors, cinematic shallow depth of field",
        "problem": f"Expressive cinematic scene illustrating the daily challenge solved by {brand}: {desc}, dramatic moody lighting, authentic character emotion",
        "solution": f"Hero product showcase of {brand} in active vibrant use, crisp product details, modern aesthetic, professional studio commercial lighting",
        "benefit": f"Joyful and empowering lifestyle outcome using {brand}, bright optimistic lighting, authentic lifestyle setting, commercial advertising look",
        "cta": f"Inspiring call-to-action closing still for {brand} ({category}), premium aesthetic, warm inviting illumination, cinematic composition",
    }
    return role_angles.get(role, f"Cinematic {role} scene for {brand}: {desc}. Commercial lighting, 8k resolution, photorealistic.")


def _scenes_from_parsed(parsed: dict, assets: BrandAssets) -> Script:
    category = str(parsed.get("business_category") or "").strip() or infer_business_category(assets)
    raw_scenes = parsed.get("scenes", [])
    if not isinstance(raw_scenes, list) or len(raw_scenes) == 0:
        return _heuristic_script(assets)

    scenes: list[Scene] = []
    t = 0.0
    expected_roles = ["hook", "problem", "solution", "benefit", "cta"]

    for idx, item in enumerate(raw_scenes):
        role = str(item.get("role") or (expected_roles[idx] if idx < len(expected_roles) else "benefit")).lower()
        text = str(item.get("text") or "").strip() or f"Experience {assets.site_title or 'our product'} today."
        duration = float(item.get("duration") or 3.5)
        duration = max(2.0, min(duration, 8.0))
        visual_prompt = _coerce_visual_prompt(item.get("visual_prompt"), role, text, assets, idx=idx)
        scene = Scene(
            id=str(uuid.uuid4()),
            start=round(t, 3),
            end=round(t + duration, 3),
            role=role,
            text=text,
            visual_prompt=visual_prompt,
        )
        scenes.append(scene)
        t += duration

    # Guarantee minimum 5 distinct scenes
    if len(scenes) < 5:
        brand = assets.site_title or "this product"
        fallback_script = _heuristic_script(assets)
        while len(scenes) < 5 and len(scenes) < len(fallback_script.scenes):
            add_scene = fallback_script.scenes[len(scenes)]
            duration = float(add_scene.end - add_scene.start)
            add_scene.id = str(uuid.uuid4())
            add_scene.start = round(t, 3)
            add_scene.end = round(t + duration, 3)
            scenes.append(add_scene)
            t += duration

    return Script(duration=round(t, 3), scenes=scenes, business_category=category)


def _heuristic_script(assets: BrandAssets) -> Script:
    """5-scene high quality fallback script with brand-tailored prompts."""
    brand = assets.site_title or "this product"
    category = infer_business_category(assets)
    desc = (assets.description or assets.raw_text_snippet or f"The modern way to experience {brand}.").strip()
    
    durations = [3.2, 4.0, 4.5, 4.0, 3.2]
    roles = ["hook", "problem", "solution", "benefit", "cta"]
    
    texts = [
        f"Ready to revolutionize how you experience {brand}?",
        f"Tired of outdated, slow, and frustrating alternatives?",
        f"Meet {brand} — engineered for speed, power, and simplicity.",
        f"Join thousands thriving with seamless performance and results.",
        f"Get started with {brand} today. Visit our website now!",
    ]
    
    # Specific visual prompts based on category
    if "Social Media" in category or "snapchat" in brand.lower():
        visual_prompts = [
            f"Vibrant smartphone screen showing active video chatting with colorful AR face filters, neon lighting, Gen-Z aesthetic, cinematic portrait, photorealistic, 8k",
            f"A bored teenager scrolling through a cluttered dull text feed, desaturated moody ambient lighting, cinematic angle",
            f"Group of joyful friends laughing together while capturing a fun dynamic augmented reality snapshot on their phone, bright colorful aesthetic",
            f"Close-up of a stylish creator effortlessly sharing an interactive story with glowing AR lenses, golden hour lens flare, commercial quality",
            f"A glowing smartphone displaying the bright welcoming {brand} interface held in hand against a stylish urban sunset background, cinematic depth of field",
        ]
    elif "Coffee" in category:
        visual_prompts = [
            f"Hero close-up shot of rich espresso pouring smoothly into a ceramic cup with thick crema, warm morning sunlight, artisanal cafe ambiance, 8k",
            f"Frustrated morning worker drinking bland watery coffee from a cardboard cup, muted cold tones, cinematic realism",
            f"Master barista skillfully steaming silky milk and pouring intricate latte art, warm golden lighting, high-end cafe",
            f"Delighted customer taking the first aromatic sip of fresh artisan coffee in a sunlit modern aesthetic bakery, joyful expression",
            f"Stunning arrangement of roasted specialty coffee beans and a steaming cup on a rustic wooden table, cinematic commercial still",
        ]
    elif "Fitness" in category:
        visual_prompts = [
            f"High-energy athlete in athletic wear preparing for an intense training sprint, dynamic low angle, moody gym lighting, 8k commercial sports shot",
            f"Person looking fatigued and unmotivated staring at complicated fitness spreadsheets, desaturated lighting, cinematic framing",
            f"Athlete actively using {brand} equipment/app during an explosive workout, sweat glistening, dynamic motion blur, vibrant energy",
            f"Fit smiling person celebrating a personal fitness milestone outdoors during a scenic sunrise, triumphant energetic vibe",
            f"Inspiring hero shot of premium fitness gear with subtle energetic lighting, clean athletic commercial composition",
        ]
    elif "Fintech" in category:
        visual_prompts = [
            f"Sleek glowing digital smart card hovering above an ultra-modern smartphone, holographic data streams, deep blue and emerald lighting, 8k",
            f"Stressed entrepreneur dealing with piles of confusing paper receipts and rejected bank statements, moody cinematic lighting",
            f"Seamless tap-to-pay transaction on a smartphone with an instant green success confirmation, crisp modern aesthetic",
            f"Confident business owner smiling warmly while viewing exponential portfolio growth on a sleek tablet in a sunlit modern office",
            f"Hero minimalist composition of next-generation digital financial tools, premium titanium finish, elegant lighting",
        ]
    else:
        visual_prompts = [
            f"Striking hero commercial shot showcasing the core innovation of {brand} ({category}), dramatic cinematic lighting, rich colors, 8k",
            f"Cinematic scene depicting the pain point solved by {brand}: {desc[:100]}, dramatic moody shadows, authentic human emotion",
            f"Close-up of a modern professional actively using {brand} with effortless mastery, bright optimistic lighting, clean aesthetic",
            f"Excited user experiencing the breakthrough benefits of {brand}, vibrant lifestyle setting, joyful authentic expression",
            f"Clean, inspiring call-to-action visual for {brand}, inviting warm atmosphere, premium commercial advertising photography",
        ]

    scenes: list[Scene] = []
    t = 0.0
    for i, duration in enumerate(durations):
        scenes.append(
            Scene(
                id=str(uuid.uuid4()),
                start=round(t, 3),
                end=round(t + duration, 3),
                role=roles[i],
                text=texts[i],
                visual_prompt=visual_prompts[i],
            )
        )
        t += duration
    return Script(duration=round(t, 3), scenes=scenes, business_category=category)


def _build_prompt(assets: BrandAssets, aspect_ratio: str) -> str:
    return (
        "Create a 5-scene JSON video-ad script for this brand. "
        "Classify the business_category niche, and generate a UNIQUE, non-generic, brand-specific "
        "visual_prompt string for each of the 5 scenes (hook, problem, solution, benefit, cta).\n\n"
        f"Brand Name: {assets.site_title}\n"
        f"Description: {assets.description}\n"
        f"Primary Color: {assets.primary_color}\n"
        f"Scraped Site Content:\n{(assets.raw_text_snippet or '')[:600]}\n"
        f"Target Video Aspect Ratio: {aspect_ratio}\n"
    )

