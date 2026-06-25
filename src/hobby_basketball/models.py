from __future__ import annotations

from pydantic import BaseModel, Field


class MadeShotEvent(BaseModel):
    id: str
    video_path: str
    t_make: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    t_above: float | None = None
    t_below: float | None = None
    start: float | None = None
    end: float | None = None
    kept: bool = True
    notes: str = ""


class ClipInterval(BaseModel):
    video_path: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    event_ids: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start
