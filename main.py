import os
import uuid
import logging
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import GenerateRequest, GenerateResponse, EditRequest, EditResponse, VideoProject, Storage
from services.scraper import scrape_site
from services.video_composer import edit_ad_component, render_templates_sequentially

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_ad_engine")

# 1. Initialize FastAPI once
app = FastAPI(title="Ad Foundry & Video Ad Orchestrator")

# 2. Configure Static and HTML Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
HTML_PATH = STATIC_DIR / "index.html"
RENDER_DIR = os.path.join(BASE_DIR, "data", "session_renders")

storage = Storage()

# Mount static directory for CSS/JS assets if static folder exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 3. Serve Frontend UI at GET /
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if not HTML_PATH.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"index.html not found in static directory at: {HTML_PATH}"
        )
    return FileResponse(HTML_PATH)


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
    """Stream renders from Supabase or redirect to direct public CDN link."""
    if storage.using_supabase:
        supabase_public_url = f"{storage.supabase_url}/storage/v1/object/public/ad-assets/{object_path}"
        return RedirectResponse(url=supabase_public_url)

    local_path = os.path.abspath(os.path.join(RENDER_DIR, os.path.basename(object_path)))
    if not local_path.startswith(os.path.abspath(RENDER_DIR) + os.sep) or not os.path.isfile(local_path):
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(local_path, media_type="video/mp4")

@app.post("/api/v1/ads/generate", response_model=GenerateResponse)
async def generate_ad(payload: GenerateRequest, background_tasks: BackgroundTasks):
    if not payload.url:
        raise HTTPException(status_code=400, detail="url is required")

    ad_id = str(uuid.uuid4())
    logger.info("Starting ad generation for %s with template=%s (ad_id=%s)", payload.url, payload.template_id, ad_id)

    # Save template_id to project
    project = VideoProject(
        id=ad_id, 
        source_url=payload.url, 
        aspect_ratio=payload.aspect_ratio,
        template_id=payload.template_id
    )
    storage.save_project(project)

    background_tasks.add_task(_run_full_pipeline, ad_id, payload)

    return GenerateResponse(ad_id=ad_id, status="processing")


async def _run_full_pipeline(ad_id: str, payload: GenerateRequest):
    project = storage.get_project(ad_id)
    try:
        # 1. Scrape
        assets = await scrape_site(payload.url)
        project.brand_assets = assets
        storage.save_project(project)

        # 2. Each template gets its own constrained script, voice, and visuals.
        preview_urls = await render_templates_sequentially(project, assets)
        if not preview_urls:
            raise RuntimeError("All template renders failed; no video previews were produced")

        project.preview_urls = preview_urls
        project.preview_url = next(iter(preview_urls.values()), None)  # First variant as primary
        project.status = "ready"
        storage.save_project(project)

        logger.info("Successfully generated %d video ad variants for %s", len(preview_urls), ad_id)
    except Exception as e:
        logger.exception("Failed pipeline for ad %s: %s", ad_id, e)
        project.status = "failed"
        project.error = str(e)
        storage.save_project(project)


@app.get("/api/v1/ads/{ad_id}")
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

    try:
        result = await edit_ad_component(project, req, storage)
        storage.save_project(project)
        return EditResponse(ad_id=ad_id, status="ready", preview_url=result)
    except Exception as e:
        logger.exception("Edit failed for %s: %s", ad_id, e)
        raise HTTPException(status_code=500, detail=str(e))
