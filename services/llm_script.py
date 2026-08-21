"""Script generation via OpenRouter with strict visual_prompt validation."""
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

SYSTEM_PROMPT = """You are an expert video-ad creative director.
Return ONLY valid JSON (no markdown fences) with this exact shape:
{
  "scenes": [
    {
      "role": "hook" | "problem" | "solution" | "cta",
      "duration": <number seconds>,
      "text": "<spoken narration line>",
      "visual_prompt": "<detailed image generation prompt>"
    }
  ]
}

Hard rules:
- Produce 4 scenes: hook, problem, solution, cta.
- Total duration between 15 and 30 seconds.
- Every scene MUST include a non-empty visual_prompt string.
- visual_prompt must NEVER be null, empty, or omitted.
- visual_prompt must describe a concrete cinematic advertising still grounded in the brand:
  product context, brand theme, target audience, lighting, camera angle, and mood.
- Do not put on-screen text, logos, watermarks, or UI chrome in visual_prompt.
"""


async def generate_script(assets: BrandAssets, aspect_ratio: str = "16:9") -> Script:
    """Call the LLM to generate a structured script JSON with required visual prompts."""
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
        "temperature": 0.35,
        "max_tokens": 1200,
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
        # Recover JSON object buried in prose / truncated fences.
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _coerce_visual_prompt(raw: object, role: str, text: str, assets: BrandAssets) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    brand = assets.site_title or "the product"
    desc = (assets.description or assets.raw_text_snippet or text or brand).strip()
    return (
        f"Cinematic {role} scene for {brand}: {desc[:180]}. "
        f"Professional advertising photography, shallow depth of field, dramatic lighting."
    )


def _scenes_from_parsed(parsed: dict, assets: BrandAssets) -> Script:
    scenes: list[Scene] = []
    t = 0.0
    for item in parsed.get("scenes", []):
        role = str(item.get("role") or "hook")
        text = str(item.get("text") or "").strip() or f"Learn more about {assets.site_title or 'us'}."
        duration = float(item.get("duration") or 3.5)
        duration = max(1.5, min(duration, 12.0))
        visual_prompt = _coerce_visual_prompt(item.get("visual_prompt"), role, text, assets)
        scene = Scene(
            id=str(uuid.uuid4()),
            start=t,
            end=t + duration,
            role=role,
            text=text,
            visual_prompt=visual_prompt,
        )
        scenes.append(scene)
        t += duration
    if not scenes:
        return _heuristic_script(assets)
    # Re-validate through Pydantic so null/empty visual_prompt cannot slip through.
    return Script(duration=t, scenes=scenes)


def _heuristic_script(assets: BrandAssets) -> Script:
    """Always-available fallback so an LLM outage cannot stop an ad job."""
    brand = assets.site_title or "this product"
    durations = [3.5, 6.0, 7.5, 3.0]
    roles = ["hook", "problem", "solution", "cta"]
    texts = [
        f"Discover {brand} — change your day in seconds.",
        assets.raw_text_snippet or "Many people struggle with the right tool for the job.",
        assets.description or "Our product solves it with smart automation and beautiful design.",
        "Try it today — visit the site to learn more.",
    ]
    visual_prompts = [
        f"Bold hero product shot of {brand}, dynamic angle, vibrant commercial lighting, lifestyle setting",
        f"Frustrated professional dealing with outdated tools, desaturated mood, cinematic framing for {brand}",
        f"Confident customer using {brand} successfully, bright optimistic lighting, clean modern workspace",
        f"Close-up call-to-action still featuring {brand}, premium packaging, inviting warm light, empty space for text",
    ]
    scenes: list[Scene] = []
    t = 0.0
    for i, duration in enumerate(durations):
        scenes.append(
            Scene(
                id=str(uuid.uuid4()),
                start=t,
                end=t + duration,
                role=roles[i],
                text=texts[i],
                visual_prompt=visual_prompts[i],
            )
        )
        t += duration
    return Script(duration=t, scenes=scenes)


def _build_prompt(assets: BrandAssets, aspect_ratio: str) -> str:
    return (
        "Create a JSON video-ad script for this brand. "
        "Every scene must include a concrete non-null visual_prompt string.\n\n"
        f"Brand name/title: {assets.site_title}\n"
        f"Brand description: {assets.description}\n"
        f"Primary color: {assets.primary_color}\n"
        f"Raw site snippet: {(assets.raw_text_snippet or '')[:400]}\n"
        f"Example product images: {assets.product_images[:3]}\n"
        f"Aspect ratio: {aspect_ratio}\n"
    )
