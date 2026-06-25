from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from hobby_basketball.models import ClipInterval, MadeShotEvent


def build_clip_intervals(
    events: Iterable[MadeShotEvent],
    pre_seconds: float = 5.0,
    post_seconds: float = 1.5,
) -> list[ClipInterval]:
    if pre_seconds < 0:
        raise ValueError("pre_seconds must be non-negative")
    if post_seconds <= 0:
        raise ValueError("post_seconds must be greater than zero")

    intervals: list[ClipInterval] = []
    for event in events:
        if not event.kept:
            continue
        start = event.start if event.start is not None else max(0.0, event.t_make - pre_seconds)
        end = event.end if event.end is not None else event.t_make + post_seconds
        if end <= start:
            raise ValueError(f"event {event.id} has invalid clip range")
        intervals.append(
            ClipInterval(
                video_path=event.video_path,
                start=round(start, 6),
                end=round(end, 6),
                event_ids=[event.id],
            )
        )
    return intervals


def merge_intervals(intervals: Iterable[ClipInterval]) -> list[ClipInterval]:
    by_video: dict[str, list[ClipInterval]] = defaultdict(list)
    for interval in intervals:
        by_video[interval.video_path].append(interval)

    merged: list[ClipInterval] = []
    for video_path in sorted(by_video):
        current: ClipInterval | None = None
        for interval in sorted(by_video[video_path], key=lambda item: item.start):
            if current is None:
                current = interval.model_copy(deep=True)
                continue
            if interval.start <= current.end:
                current.end = max(current.end, interval.end)
                current.event_ids.extend(interval.event_ids)
                continue
            merged.append(current)
            current = interval.model_copy(deep=True)
        if current is not None:
            merged.append(current)
    return merged
