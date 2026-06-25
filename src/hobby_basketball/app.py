from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from hobby_basketball.clips import build_clip_intervals, merge_intervals
from hobby_basketball.models import ClipInterval, MadeShotEvent


class PlanClipsRequest(BaseModel):
    video_path: str
    events: list[MadeShotEvent]
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    merge_overlaps: bool = True


class PlanClipsResponse(BaseModel):
    clips: list[ClipInterval]


app = FastAPI(title="Hobby Basketball", version="0.1.0")


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    static_path = Path(__file__).with_name("static") / "index.html"
    return static_path.read_text(encoding="utf-8")
