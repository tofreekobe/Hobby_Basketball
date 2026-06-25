from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from hobby_basketball.clips import build_clip_intervals, merge_intervals
from hobby_basketball.detector import scan_video_for_made_shots
from hobby_basketball.ffmpeg_export import export_reel
from hobby_basketball.models import ClipInterval, MadeShotEvent
from hobby_basketball.trajectory import RimCalibration
from hobby_basketball.workspace import EXPORT_DIR, export_path, normalize_output_format, save_upload


class PlanClipsRequest(BaseModel):
    video_path: str
    events: list[MadeShotEvent]
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    merge_overlaps: bool = True


class PlanClipsResponse(BaseModel):
    clips: list[ClipInterval]


class UploadVideoResponse(BaseModel):
    video_id: str
    filename: str
    video_path: str
    preview_url: str


class ProcessVideoRequest(BaseModel):
    video_id: str
    rim: RimCalibration
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    output_format: str = "mp4"
    sample_fps: float = Field(default=15.0, gt=0)
    confidence: float = Field(default=0.15, ge=0, le=1)
    device: str = "cpu"
    model_name: str = "yolo11m.pt"


class ProcessVideoResponse(BaseModel):
    events: list[MadeShotEvent]
    clips: list[ClipInterval]
    output_path: str
    preview_url: str


app = FastAPI(title="Hobby Basketball", version="0.1.0")
VIDEO_REGISTRY: dict[str, Path] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/plan-clips", response_model=PlanClipsResponse)
def plan_clips(request: PlanClipsRequest) -> PlanClipsResponse:
    events = [
        event.model_copy(update={"video_path": event.video_path or request.video_path})
        for event in request.events
    ]
    clips = build_clip_intervals(events, request.pre_seconds, request.post_seconds)
    if request.merge_overlaps:
        clips = merge_intervals(clips)
    return PlanClipsResponse(clips=clips)


@app.post("/api/upload-video", response_model=UploadVideoResponse)
def upload_video(file: UploadFile) -> UploadVideoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="请选择视频文件")
    video_id, path = save_upload(file)
    VIDEO_REGISTRY[video_id] = path
    return UploadVideoResponse(
        video_id=video_id,
        filename=file.filename,
        video_path=str(path),
        preview_url=f"/api/videos/{video_id}",
    )


@app.get("/api/videos/{video_id}")
def video_preview(video_id: str):
    path = _get_video_path(video_id)
    return FileResponse(path)


@app.get("/api/exports/{filename}")
def export_preview(filename: str):
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(path)


@app.post("/api/process-video", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest) -> ProcessVideoResponse:
    video_path = _get_video_path(request.video_id)
    try:
        output_format = normalize_output_format(request.output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        events = scan_video_for_made_shots(
            video_path,
            request.rim,
            sample_fps=request.sample_fps,
            confidence=request.confidence,
            model_name=request.model_name,
            device=request.device,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not events:
        raise HTTPException(status_code=422, detail="未识别到进球片段，请检查篮筐标定或降低置信度")

    clips = merge_intervals(build_clip_intervals(events, request.pre_seconds, request.post_seconds))
    final_path = export_path(request.video_id, output_format)
    clip_dir = final_path.with_suffix("")
    try:
        export_reel(clips, clip_dir, final_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"FFmpeg 导出失败：{exc}") from exc

    return ProcessVideoResponse(
        events=events,
        clips=clips,
        output_path=str(final_path),
        preview_url=f"/api/exports/{final_path.name}",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    static_path = Path(__file__).with_name("static") / "index.html"
    return static_path.read_text(encoding="utf-8")


def _get_video_path(video_id: str) -> Path:
    path = VIDEO_REGISTRY.get(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="视频不存在，请重新选择文件")
    return path
