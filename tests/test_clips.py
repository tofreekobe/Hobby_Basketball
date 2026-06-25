from hobby_basketball.clips import build_clip_intervals, merge_intervals
from hobby_basketball.models import MadeShotEvent


def test_build_clip_intervals_uses_default_pre_and_post_seconds():
    event = MadeShotEvent(id="make-1", video_path="game.mp4", t_make=10.0, confidence=0.8)

    clips = build_clip_intervals([event], pre_seconds=5.0, post_seconds=1.5)

    assert clips[0].start == 5.0
    assert clips[0].end == 11.5


def test_build_clip_intervals_clamps_start_to_zero():
    event = MadeShotEvent(id="make-1", video_path="game.mp4", t_make=3.0, confidence=0.8)

    clips = build_clip_intervals([event], pre_seconds=5.0, post_seconds=1.5)

    assert clips[0].start == 0.0
    assert clips[0].end == 4.5


def test_merge_intervals_prevents_backward_replay():
    events = [
        MadeShotEvent(id="a", video_path="game.mp4", t_make=10.0, confidence=0.8),
        MadeShotEvent(id="b", video_path="game.mp4", t_make=12.0, confidence=0.8),
    ]

    merged = merge_intervals(build_clip_intervals(events, 5.0, 1.5))

    assert len(merged) == 1
    assert merged[0].start == 5.0
    assert merged[0].end == 13.5
    assert merged[0].event_ids == ["a", "b"]
