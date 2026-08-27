import os
import httpx
import logging
import uuid
import json
import re
from typing import List
from models import BrandAssets, Script, Scene

logger = logging.getLogger("ai_ad_engine.llm")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")


def _clean_text(text: str) -> str:
    """Clean stage directions, markdown, or raw JSON leaks from voiceover lines."""
    if not text:
        return ""
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
    text = re.sub(r"[\*\_\#\~]", "", text)
    return text.strip(' "\'\n\r\t')


def infer_business_category(assets: BrandAssets) -> str:
    """Dynamic universal fallback using the brand title or domain."""
    return (assets.site_title or "Commercial Business").strip()


async def generate_script(
    assets: BrandAssets,
    aspect_ratio: str = "16:9",
    required_scenes: int | None = None,
) -> Script:
    """Generate an ad script with the exact scene count required by a template."""
    if required_scenes is not None and required_scenes < 1:
        raise ValueError("required_scenes must be at least 1")
    if not LLM_API_KEY:
        logger.warning("LLM_API_KEY not set; generating heuristic script")
        return _heuristic_script(assets, required_scenes=required_scenes)

    prompt = _build_prompt(assets, aspect_ratio, required_scenes=required_scenes)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 700
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
        for s in parsed.get("scenes", []):
            dur = float(s.get("duration", 3.5))
            clean_text = _clean_text(s.get("text", ""))
            
            # Derive visual search terms directly from the scene text & brand context
            v_prompt = s.get("visual_prompt", "").strip()
            if not v_prompt:
                v_prompt = f"{clean_text[:40]} commercial"

            scene = Scene(
                id=str(uuid.uuid4()),
                start=t,
                end=t + dur,
                role=s.get("role", "scene"),
                text=clean_text,
                visual_prompt=v_prompt
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

        logger.info("Generated %d-scene ad script for category '%s'", len(scenes), business_category)
        return Script(duration=t, scenes=scenes, business_category=business_category)

    except Exception as e:
        logger.warning("LLM generation failed; using heuristic fallback: %s", e)
        return _heuristic_script(assets, required_scenes=required_scenes)


def _fit_scene_count(scenes: list[Scene], assets: BrandAssets, required_scenes: int) -> list[Scene]:
    """Trim or supplement an LLM response so template slots are always satisfied."""
    scenes = scenes[:required_scenes]
    fallback = _heuristic_script(assets, required_scenes=required_scenes).scenes
    while len(scenes) < required_scenes:
        scenes.append(fallback[len(scenes)])
    return scenes


def _heuristic_script(assets: BrandAssets, required_scenes: int | None = None) -> Script:
    """Universal fallback script for any brand type."""
    category = infer_business_category(assets)
    scene_count = required_scenes or 4
    durations = [3.5, 4.5, 5.0, 3.5]
    roles = ["hook", "problem", "solution", "cta"]
    
    brand_name = assets.site_title or "this solution"
    texts = [
        f"Discover a better way with {brand_name}.",
        assets.description[:80] if assets.description else "Stop settling for slow and complex solutions.",
        "Experience fast, reliable results tailored to your exact needs.",
        f"Get started today. Visit {brand_name} now."
    ]
    
    prompts = [
        f"{brand_name} product highlight",
        f"{category} problem concept",
        f"{category} professional solution",
        f"{category} action call"
    ]
    
    scenes = []
    t = 0.0
    for i in range(scene_count):
        duration = durations[i % len(durations)]
        scenes.append(Scene(
            id=str(uuid.uuid4()),
            start=t,
            end=t + duration,
            role=roles[i],
            text=texts[i],
            visual_prompt=prompts[i]
        ))
        t += duration
    return Script(duration=t, scenes=scenes, business_category=category)


def _build_prompt(assets: BrandAssets, aspect_ratio: str, required_scenes: int | None = None) -> str:
    snippet = (assets.raw_text_snippet or "")[:500].replace("\n", " ")
    return (
        "You are an expert commercial video copywriter. Write a ultra-short 15-second video ad script based on the brand info provided.\n\n"
        "RULES:\n"
        f"1. Return exactly {required_scenes or 4} CONCISE scenes.\n"
        "2. Voiceover 'text' MUST be short (under 12 words / 70 characters per scene).\n"
        "3. 'visual_prompt' MUST be 2-4 search terms describing a visual directly matching what is spoken in that specific scene.\n"
        "4. DO NOT include markdown, stage directions, or metadata inside 'text'.\n\n"
        "STRICT JSON OUTPUT ONLY:\n"
        "{\n"
        '  "business_category": "Dynamic 2-3 Word Niche",\n'
        '  "scenes": [\n'
        '    {\n'
        '      "role": "hook",\n'
        '      "duration": 3.5,\n'
        '      "text": "Short punchy spoken sentence.",\n'
        '      "visual_prompt": "specific image search terms matching text"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        f"Brand Name: {assets.site_title or 'Brand'}\n"
        f"Description: {assets.description or 'N/A'}\n"
        f"Scraped Web Context: {snippet or 'N/A'}\n"
        f"Aspect Ratio: {aspect_ratio}\n"
    )
