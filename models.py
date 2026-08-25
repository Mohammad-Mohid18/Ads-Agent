from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
PROJECT_DB = os.path.join(DATA_DIR, "projects.json")


class BrandAssets(BaseModel):
    site_title: Optional[str] = None
    description: Optional[str] = None
    logo_url: Optional[str] = None
    product_images: List[str] = Field(default_factory=list)
    primary_color: Optional[str] = None
    raw_text_snippet: Optional[str] = None


class Scene(BaseModel):
    id: str
    start: float = 0.0
    end: float = 0.0
    role: str = "hook"  # hook/problem/solution/cta
    text: str = ""
    visual_prompt: str = Field(..., min_length=1)  # Must NOT be null
    image_url: Optional[str] = None


class Script(BaseModel):
    duration: float = 0.0
    scenes: List[Scene] = Field(default_factory=list)
    business_category: Optional[str] = None


class VoiceSegment(BaseModel):
    scene_id: str
    audio_url: str
    duration: float = 0.0
    words: List[Dict[str, Any]] = Field(default_factory=list)  # word-level timestamps


class VoiceResult(BaseModel):
    segments: List[VoiceSegment] = Field(default_factory=list)
    total_duration: float = 0.0
    full_audio_url: Optional[str] = None


class GenerateRequest(BaseModel):
    url: str
    aspect_ratio: Optional[str] = "16:9"
    template_id: Optional[str] = None  # Captures front-end template selection

class VideoProject(BaseModel):
    id: str
    source_url: str
    aspect_ratio: Optional[str] = "16:9"
    template_id: Optional[str] = None
    version: int = 1
    status: str = "pending"
    error: Optional[str] = None
    preview_url: Optional[str] = None  # Primary featured render
    preview_urls: Optional[dict[str, str]] = Field(default_factory=dict)  # Maps template_key -> URL
    brand_assets: Optional[BrandAssets] = None
    script: Optional[Script] = None
    voice: Optional[VoiceResult] = None
    layers: Optional[dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GenerateResponse(BaseModel):
    ad_id: str
    status: str


class EditRequest(BaseModel):
    target_layer: str
    new_value: str


class EditResponse(BaseModel):
    ad_id: str
    status: str
    preview_url: Optional[str]


# Supabase-backed project storage.  A local JSON file is used only when Supabase
# has not been configured, which keeps the demo usable without cloud credentials.
class Storage:
    def __init__(self, path: str = PROJECT_DB, *, supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None, table: str = "video_projects"):
        self.path = path
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.table = table
        self.render_bucket = os.getenv("SUPABASE_RENDER_BUCKET") or os.getenv("S3_BUCKET", "ad-assets")
        self.using_supabase = bool(self.supabase_url and self.supabase_key)
        if not self.using_supabase and not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.supabase_key or "",
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }

    @property
    def _endpoint(self) -> str:
        return f"{self.supabase_url}/rest/v1/{self.table}"

    def _raise_for_supabase_error(self, response: httpx.Response, action: str) -> None:
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Supabase {action} failed ({response.status_code}): {detail}")

    def _upsert_record(self, table: str, record: Dict[str, Any], conflict: str) -> None:
        headers = {**self._headers, "Prefer": "resolution=merge-duplicates,return=minimal"}
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{self.supabase_url}/rest/v1/{table}?on_conflict={conflict}", headers=headers, json=record
            )
        if response.is_error:
            self._raise_for_supabase_error(response, f"upsert to {table}")

    def _upload_local_preview(self, project: VideoProject) -> Optional[str]:
        """Move a locally rendered MP4 to Supabase Storage and return its object path."""
        if not project.preview_url or not project.preview_url.startswith("/media/"):
            return None
        media_path = project.preview_url.removeprefix("/media/")
        if media_path.startswith("renders/"):
            return media_path
        filename = Path(media_path).name
        local_path = Path(DATA_DIR) / "session_renders" / filename
        if not local_path.is_file():
            return None
        object_path = f"renders/{filename}"
        headers = {**self._headers, "Content-Type": "video/mp4", "x-upsert": "true"}
        with local_path.open("rb") as video_file, httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.supabase_url}/storage/v1/object/{self.render_bucket}/{object_path}",
                headers=headers,
                content=video_file,
            )
        if response.is_error:
            self._raise_for_supabase_error(response, "video upload")
        local_path.unlink()
        local_path.with_suffix(".txt").unlink(missing_ok=True)
        project.preview_url = f"/media/{object_path}"
        return object_path

    def _sync_related_records(self, project: VideoProject, storage_path: Optional[str]) -> None:
        if project.brand_assets:
            self._upsert_record("ad_brand_assets", {
                "project_id": project.id,
                "asset_data": project.brand_assets.model_dump(mode="json"),
                "updated_at": project.updated_at.isoformat(),
            }, "project_id")
        if project.script:
            self._upsert_record("ad_scripts", {
                "project_id": project.id,
                "version": project.version,
                "duration": project.script.duration,
                "scenes": [scene.model_dump(mode="json") for scene in project.script.scenes],
                "updated_at": project.updated_at.isoformat(),
            }, "project_id,version")
        if project.voice:
            self._upsert_record("ad_voiceovers", {
                "project_id": project.id,
                "version": project.version,
                "total_duration": project.voice.total_duration,
                "segments": [segment.model_dump(mode="json") for segment in project.voice.segments],
                "updated_at": project.updated_at.isoformat(),
            }, "project_id,version")
        if project.preview_url:
            self._upsert_record("ad_renders", {
                "project_id": project.id,
                "version": project.version,
                "storage_path": storage_path or project.preview_url.removeprefix("/media/"),
                "preview_url": project.preview_url,
                "layers": project.layers or {},
                "updated_at": project.updated_at.isoformat(),
            }, "project_id,version")

    def get_media(self, object_path: str) -> tuple[bytes, str]:
        if not self.using_supabase:
            raise RuntimeError("Supabase Storage is not configured")
        with httpx.Client(timeout=120.0) as client:
            response = client.get(
                f"{self.supabase_url}/storage/v1/object/{self.render_bucket}/{object_path}", headers=self._headers
            )
        if response.is_error:
            self._raise_for_supabase_error(response, "video download")
        return response.content, response.headers.get("content-type", "video/mp4")

    def _read_all(self) -> dict:
        with open(self.path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}

    def _write_all(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, default=str, indent=2)

    def save_project(self, project: VideoProject):
        project.updated_at = datetime.utcnow()
        if self.using_supabase:
            storage_path = self._upload_local_preview(project)
            record = {
                "id": project.id,
                "source_url": project.source_url,
                "aspect_ratio": project.aspect_ratio,
                "status": project.status,
                "error": project.error,
                "project_data": project.model_dump(mode="json"),
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
            }
            self._upsert_record(self.table, record, "id")
            self._sync_related_records(project, storage_path)
            return
        data = self._read_all()
        data[project.id] = project.model_dump()
        self._write_all(data)

    def get_project(self, project_id: str) -> Optional[VideoProject]:
        if self.using_supabase:
            params = {"id": f"eq.{project_id}", "select": "project_data"}
            with httpx.Client(timeout=15.0) as client:
                response = client.get(self._endpoint, headers=self._headers, params=params)
            if response.is_error:
                self._raise_for_supabase_error(response, "read")
            rows = response.json()
            return VideoProject.model_validate(rows[0]["project_data"]) if rows else None
        data = self._read_all()
        item = data.get(project_id)
        if not item:
            return None
        return VideoProject.model_validate(item)

    def list_projects(self) -> List[str]:
        if self.using_supabase:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(self._endpoint, headers=self._headers, params={"select": "id"})
            if response.is_error:
                self._raise_for_supabase_error(response, "list")
            return [row["id"] for row in response.json()]
        return list(self._read_all().keys())
