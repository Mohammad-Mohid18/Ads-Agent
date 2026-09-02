import os
import uuid
import json
import logging
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from models import GenerateRequest, GenerateResponse, EditRequest, EditResponse, VideoProject, Storage, EditUrlResponse, ReRenderRequest, ReRenderResponse
from services.scraper import scrape_site
from services.video_composer import edit_ad_component, render_templates_sequentially, re_render_with_modifications

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

from urllib.parse import quote
import json

@app.get("/api/v1/ads/{ad_id}/edit-url", response_model=EditUrlResponse)
async def get_edit_url(ad_id: str, variant: Optional[str] = Query("service_local")):
        """Returns the direct Creatomate templates dashboard link."""
        project = storage.get_project(ad_id)
        if not project:
            raise HTTPException(status_code=404, detail="Ad project not found")

        # Your updated project workspace ID
        project_space_id = "3129e5c1-7323-43c5-ac5d-0082a565522b"
        
        # Direct templates list URL
        editor_url = f"https://creatomate.com/projects/{project_space_id}/templates"

        return EditUrlResponse(
            ad_id=ad_id,
            status="success",
            editor_url=editor_url,
            template_id=project.template_id or ""
        )

@app.get("/api/v1/ads/{ad_id}/editor-session")
async def get_editor_session(
    ad_id: str,
    variant: Optional[str] = Query(None),
    template_id: Optional[str] = Query(None)
):
    """Load an ad into the Creatomate Embedded Studio Editor.
    
    Returns the template_id and modifications needed to initialize the
    interactive Creatomate Studio with the user's exact generated ad state.
    
    Optional Parameters:
    - variant: Ad style variant (service_local, news_showcase, ecommerce). Defaults to service_local.
    - template_id: Specific template ID to use. If provided, takes precedence over variant.
    
    The frontend uses this to populate the studio with the current ad content
    (text, images, audio) so users can edit from their current state.
    """
    project = storage.get_project(ad_id)
    if not project:
        raise HTTPException(status_code=404, detail="ad_id not found")

    # Default to service_local if no variant specified
    selected_variant = variant or "service_local"
    
    logger.info(
        "Editor session requested for ad %s (variant_param=%s, selected_variant=%s, project_template=%s)",
        ad_id,
        repr(variant),
        selected_variant,
        project.template_id
    )
    
    # Verify project has modifications
    if not project.creatomate_modifications:
        logger.error(
            "Cannot load editor session for ad %s: no creatomate_modifications stored",
            ad_id
        )
        raise HTTPException(
            status_code=400,
            detail="Ad not fully generated yet or modifications not available"
        )

    # Use provided template_id or fall back to project's template_id
    selected_template_id = template_id or project.template_id
    
    if not selected_template_id:
        logger.error("Cannot load editor session for ad %s: no template_id available", ad_id)
        raise HTTPException(
            status_code=400,
            detail="No template_id found for this ad"
        )

    try:
        logger.info(
            "Loading editor session for ad %s (variant=%s, template=%s)",
            ad_id,
            selected_variant,
            selected_template_id
        )
        
        return {
            "ad_id": ad_id,
            "variant": selected_variant,
            "template_id": selected_template_id,
            "modifications": project.creatomate_modifications,
            "status": "ready"
        }
    except Exception as e:
        logger.exception("Failed to load editor session for %s: %s", ad_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ads/{ad_id}/re-render", response_model=ReRenderResponse)
async def re_render_ad(ad_id: str, req: ReRenderRequest, background_tasks: BackgroundTasks):
    """Re-render an ad with updated modifications.
    
    This endpoint accepts updated text, images, and audio layer modifications,
    re-renders the ad through Creatomate with the new values, and stores the
    updated video back to Supabase.
    
    Payload example:
    {
        "template_id": "optional-override-template-id",
        "updated_modifications": {
            "Text-1": "Updated text content",
            "Image-1": "https://new-image-url.jpg",
            "Audio-1": "https://new-audio-url.mp3"
        }
    }
    """
    project = storage.get_project(ad_id)
    if not project:
        raise HTTPException(status_code=404, detail="ad_id not found")

    if not req.updated_modifications:
        raise HTTPException(
            status_code=400,
            detail="updated_modifications is required"
        )

    logger.info(
        "Re-render requested for ad %s with %d modifications",
        ad_id,
        len(req.updated_modifications)
    )

    # Run re-render in background
    background_tasks.add_task(
        _run_re_render_pipeline,
        ad_id,
        project,
        req.updated_modifications,
        req.template_id
    )

    return ReRenderResponse(
        ad_id=ad_id,
        status="processing",
        preview_url=None,
        version=project.version + 1
    )


async def _run_re_render_pipeline(
    ad_id: str,
    project: VideoProject,
    updated_modifications: dict,
    template_id: Optional[str] = None
):
    """Background task to re-render an ad with updated modifications."""
    try:
        logger.info("Starting re-render pipeline for ad %s", ad_id)
        
        # Execute re-render with Creatomate
        storage_path, public_url = await re_render_with_modifications(
            project,
            updated_modifications,
            template_id
        )
        
        if not public_url:
            logger.error("Re-render failed for ad %s: no URL returned", ad_id)
            project.status = "failed"
            project.error = "Re-render returned no video URL"
            storage.save_project(project)
            return

        # Update project with new render
        project.preview_url = public_url
        project.status = "ready"
        storage.save_project(project)

        logger.info("Re-render succeeded for ad %s (v%d): %s", ad_id, project.version, public_url)

    except Exception as e:
        logger.exception("Re-render pipeline failed for ad %s: %s", ad_id, e)
        project.status = "failed"
        project.error = f"Re-render failed: {str(e)}"
        storage.save_project(project)


# @app.get("/editor", response_class=FileResponse)
# async def serve_editor_page():
#     editor_html_path = STATIC_DIR / "editor.html"
#     if not editor_html_path.exists():
#         raise HTTPException(
#             status_code=404, 
#             detail=f"editor.html not found in static directory at: {editor_html_path}"
#         )
#     return FileResponse(editor_html_path)
