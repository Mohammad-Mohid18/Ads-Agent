import os
import uuid
import logging
import shutil
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel

from models import GenerateRequest, GenerateResponse, EditRequest, EditResponse, VideoProject, Storage
from services.scraper import scrape_site
from services.llm_script import generate_script
from services.visuals import generate_scene_visuals
from services.voice import synthesize_voice_for_script
from services.video_composer import build_and_render_video, edit_ad_component

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_ad_engine")

app = FastAPI(title="AI Video Ad Orchestrator")
storage = Storage()
RENDER_DIR = os.path.join(os.path.dirname(__file__), "data", "session_renders")


@app.on_event("startup")
async def clear_session_render_cache():
    """Completed renders are in Supabase; local files are session-only."""
    shutil.rmtree(RENDER_DIR, ignore_errors=True)
    os.makedirs(RENDER_DIR, exist_ok=True)


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


@app.get("/media/{object_path:path}")
async def get_media(object_path: str):
    """Stream renders from Supabase Storage, or a local file in demo mode."""
    if not object_path.startswith("renders/") or ".." in object_path.split("/"):
        raise HTTPException(status_code=404, detail="video not found")
    if storage.using_supabase:
        try:
            content, media_type = storage.get_media(object_path)
            return Response(content=content, media_type=media_type)
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    local_path = os.path.abspath(os.path.join(RENDER_DIR, os.path.basename(object_path)))
    if not local_path.startswith(os.path.abspath(RENDER_DIR) + os.sep) or not os.path.isfile(local_path):
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(local_path, media_type="video/mp4")


@app.post("/api/v1/ads/generate", response_model=GenerateResponse)
async def generate_ad(payload: GenerateRequest, background_tasks: BackgroundTasks):
    # Basic validation
    if not payload.url:
        raise HTTPException(status_code=400, detail="url is required")

    ad_id = str(uuid.uuid4())
    logger.info("Starting ad generation for %s (ad_id=%s)", payload.url, ad_id)

    # Create placeholder project record
    project = VideoProject(id=ad_id, source_url=payload.url, aspect_ratio=payload.aspect_ratio)
    storage.save_project(project)

    # Run full pipeline in background
    background_tasks.add_task(_run_full_pipeline, ad_id, payload)

    return GenerateResponse(ad_id=ad_id, status="processing")


async def _run_full_pipeline(ad_id: str, payload: GenerateRequest):
    project = storage.get_project(ad_id)
    try:
        # 1. Scrape
        assets = await scrape_site(payload.url)
        project.brand_assets = assets
        storage.save_project(project)

        # 2. LLM script (visual_prompt required per scene)
        script = await generate_script(assets, payload.aspect_ratio)
        project.script = script
        storage.save_project(project)

        # 3. Resolve/sanitize scene visuals (fal.ai + broken-image healing)
        await generate_scene_visuals(
            script, assets, project_id=ad_id, aspect_ratio=payload.aspect_ratio
        )
        project.script = script
        storage.save_project(project)

        # 4. ElevenLabs TTS -> measure durations -> sync scene start/end -> public audio URL
        voice = await synthesize_voice_for_script(script, project_id=ad_id)
        project.script = script  # persist voice-synced timeline
        project.voice = voice
        storage.save_project(project)

        # 5. Sanitize assets again + Creatomate render -> Supabase MP4
        preview = await build_and_render_video(project, assets, script, voice)
        project.preview_url = preview
        project.status = "ready"
        storage.save_project(project)

        logger.info("Ad generation completed for %s", ad_id)
    except Exception as e:
        logger.exception("Failed pipeline for ad %s: %s", ad_id, e)
        project.status = "failed"
        project.error = str(e)
        storage.save_project(project)


@app.get("/api/v1/ads/{ad_id}/status")
async def ad_status(ad_id: str):
    project = storage.get_project(ad_id)
    if not project:
        raise HTTPException(status_code=404, detail="ad_id not found")
    return project


@app.post("/api/v1/ads/{ad_id}/edit", response_model=EditResponse)
async def edit_ad(ad_id: str, req: EditRequest, background_tasks: BackgroundTasks):
    project = storage.get_project(ad_id)
    if not project:
        raise HTTPException(status_code=404, detail="ad_id not found")

    logger.info("Received edit request for %s: target=%s", ad_id, req.target_layer)

    # Perform targeted edit synchronously where possible, but offload heavy tasks
    try:
        result = await edit_ad_component(project, req, storage)
        # Save updated project
        storage.save_project(project)
        return EditResponse(ad_id=ad_id, status="processing", preview_url=result)
    except Exception as e:
        logger.exception("Edit failed for %s: %s", ad_id, e)
        raise HTTPException(status_code=500, detail=str(e))
