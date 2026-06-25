from __future__ import annotations

import json
from typing import Any, Protocol

import requests

from hobby_basketball.models import ClipInterval

MICROSECONDS = 1_000_000


class HttpClient(Protocol):
    def post(self, url: str, json: dict[str, Any], timeout: float):
        ...


def clips_to_video_infos(clips: list[ClipInterval]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    timeline_start = 0
    for clip in clips:
        duration = int(round(clip.duration * MICROSECONDS))
        timeline_end = timeline_start + duration
        infos.append(
            {
                "video_url": clip.video_path,
                "start": timeline_start,
                "end": timeline_end,
                "duration": duration,
                "volume": 1.0,
            }
        )
        timeline_start = timeline_end
    return infos


class CapCutMateClient:
    def __init__(
        self,
        base_url: str = "http://localhost:30000/openapi/capcut-mate/v1",
        *,
        http: HttpClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http or requests
        self.timeout = timeout

    def create_draft(self, *, width: int = 1920, height: int = 1080) -> str:
        payload = {"width": width, "height": height}
        response = self.http.post(f"{self.base_url}/create_draft", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["draft_url"]

    def add_videos(self, draft_url: str, video_infos: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "draft_url": draft_url,
            "video_infos": json.dumps(video_infos, ensure_ascii=False),
        }
        response = self.http.post(f"{self.base_url}/add_videos", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def save_draft(self, draft_url: str) -> str:
        response = self.http.post(
            f"{self.base_url}/save_draft",
            json={"draft_url": draft_url},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("draft_url", draft_url)
