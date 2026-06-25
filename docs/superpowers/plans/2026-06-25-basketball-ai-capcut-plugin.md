# Basketball AI CapCut Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Hobby Basketball app: a local FastAPI + single-page GUI tool that turns basketball made-shot events into clipped highlight reels and CapCut/Jianying draft payloads.

**Architecture:** The repository is a clean Python project. Pure modules handle clip interval math, trajectory-based made-shot detection, FFmpeg command construction, and capcut-mate payload generation. A FastAPI app exposes a single-page GUI and JSON APIs; heavy YOLO/OpenCV detection is shaped as an adapter so core behavior remains testable without model downloads.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, pytest, optional FFmpeg CLI, optional capcut-mate HTTP API.

---

## File Structure

- Create `pyproject.toml`: package metadata, runtime dependencies, pytest config.
- Create `README.md`: setup, run, and workflow instructions.
- Create `.gitignore`: Python/cache/output ignores.
- Create `src/hobby_basketball/__init__.py`: package marker.
- Create `src/hobby_basketball/models.py`: Pydantic models for calibration, events, export options, jobs.
- Create `src/hobby_basketball/clips.py`: pure clip interval calculation and overlap merging.
- Create `src/hobby_basketball/trajectory.py`: pure made-shot detection from ball trajectory arrays.
- Create `src/hobby_basketball/ffmpeg_export.py`: FFmpeg command planning and optional execution.
- Create `src/hobby_basketball/capcut_mate.py`: adapter that maps kept clips to capcut-mate create/add/save API calls.
- Create `src/hobby_basketball/app.py`: FastAPI app and JSON routes.
- Create `src/hobby_basketball/static/index.html`: single-page workspace GUI.
- Create `tests/test_clips.py`: tests for clip windows and merging.
- Create `tests/test_trajectory.py`: tests for make/miss/bounce detection.
- Create `tests/test_ffmpeg_export.py`: tests for deterministic FFmpeg command generation.
- Create `tests/test_capcut_mate.py`: tests for payload generation using a fake HTTP client.
- Create `tests/test_app.py`: API smoke tests.

## Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/hobby_basketball/__init__.py`

- [ ] **Step 1: Write skeleton files**

```toml
[project]
name = "hobby-basketball"
version = "0.1.0"
description = "Basketball made-shot highlight reel and CapCut/Jianying draft helper"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.111",
  "pydantic>=2.7",
  "uvicorn[standard]>=0.30",
  "requests>=2.32",
  "python-multipart>=0.0.9"
]

[project.optional-dependencies]
dev = ["pytest>=8.2", "httpx>=0.27"]
vision = ["opencv-python>=4.10", "numpy>=1.26", "ultralytics>=8.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

- [ ] **Step 2: Verify package discovery baseline**

Run: `python -m pytest`
Expected: no tests collected or pass once tests exist.

## Task 2: Clip Interval Core

**Files:**
- Create: `src/hobby_basketball/models.py`
- Create: `src/hobby_basketball/clips.py`
- Test: `tests/test_clips.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_clips.py`
Expected: FAIL because `hobby_basketball.clips` does not exist.

- [ ] **Step 3: Implement minimal models and clip functions**

Implement `MadeShotEvent`, `ClipInterval`, `build_clip_intervals`, and `merge_intervals`.

- [ ] **Step 4: Run tests to verify green**

Run: `python -m pytest tests/test_clips.py`
Expected: PASS.

## Task 3: Trajectory Made-Shot Detection

**Files:**
- Create: `src/hobby_basketball/trajectory.py`
- Test: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

```python
from hobby_basketball.trajectory import BallSample, RimCalibration, detect_made_shots

def test_detects_descending_ball_crossing_rim_plane():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=100, y=80, confidence=0.8),
        BallSample(t=1.25, x=101, y=118, confidence=0.8),
    ]
    makes = detect_made_shots(samples, rim)
    assert len(makes) == 1
    assert makes[0].t_make == 1.125

def test_rejects_ball_crossing_outside_horizontal_gate():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=180, y=80, confidence=0.8),
        BallSample(t=1.25, x=180, y=118, confidence=0.8),
    ]
    assert detect_made_shots(samples, rim) == []

def test_rejects_bounce_back_above_rim():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=100, y=80, confidence=0.8),
        BallSample(t=1.25, x=101, y=118, confidence=0.8),
        BallSample(t=1.40, x=100, y=79, confidence=0.8),
    ]
    assert detect_made_shots(samples, rim) == []
```

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_trajectory.py`
Expected: FAIL because `trajectory.py` does not exist.

- [ ] **Step 3: Implement pure trajectory detector**

Port the rim-plane crossing logic from `bball-highlights` as a pure function over samples. Do not import YOLO or OpenCV in this module.

- [ ] **Step 4: Run tests to verify green**

Run: `python -m pytest tests/test_trajectory.py`
Expected: PASS.

## Task 4: FFmpeg Export Planning

**Files:**
- Create: `src/hobby_basketball/ffmpeg_export.py`
- Test: `tests/test_ffmpeg_export.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from hobby_basketball.ffmpeg_export import build_clip_command, build_concat_command
from hobby_basketball.models import ClipInterval

def test_build_clip_command_uses_source_start_and_duration():
    clip = ClipInterval(video_path="game.mp4", start=5.0, end=11.5, event_ids=["make-1"])
    cmd = build_clip_command(clip, Path("out/clip_000.mp4"))
    assert cmd[:5] == ["ffmpeg", "-y", "-ss", "5.000", "-i"]
    assert "-t" in cmd
    assert "6.500" in cmd

def test_build_concat_command_targets_final_mp4():
    cmd = build_concat_command(Path("clips.txt"), Path("final.mp4"))
    assert cmd == ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips.txt", "-c", "copy", "-movflags", "+faststart", "final.mp4"]
```

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_ffmpeg_export.py`
Expected: FAIL because `ffmpeg_export.py` does not exist.

- [ ] **Step 3: Implement command builders**

Implement deterministic command builders and an `export_reel` function that can be integration-tested later.

- [ ] **Step 4: Run tests to verify green**

Run: `python -m pytest tests/test_ffmpeg_export.py`
Expected: PASS.

## Task 5: capcut-mate Adapter

**Files:**
- Create: `src/hobby_basketball/capcut_mate.py`
- Test: `tests/test_capcut_mate.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from hobby_basketball.capcut_mate import CapCutMateClient, clips_to_video_infos
from hobby_basketball.models import ClipInterval

def test_clips_to_video_infos_places_precut_clips_contiguously():
    clips = [
        ClipInterval(video_path="clips/clip_000.mp4", start=0.0, end=6.5, event_ids=["a"]),
        ClipInterval(video_path="clips/clip_001.mp4", start=0.0, end=4.0, event_ids=["b"]),
    ]
    infos = clips_to_video_infos(clips)
    assert infos[0]["start"] == 0
    assert infos[0]["end"] == 6500000
    assert infos[1]["start"] == 6500000
    assert infos[1]["end"] == 10500000

def test_client_calls_create_and_add_videos_with_json_string():
    calls = []
    class FakeHttp:
        def post(self, url, json, timeout):
            calls.append((url, json))
            class Response:
                def raise_for_status(self): pass
                def json(self):
                    if url.endswith("/create_draft"):
                        return {"draft_url": "http://local/get_draft?draft_id=1"}
                    return {"draft_url": "http://local/get_draft?draft_id=1", "segment_ids": ["s1"]}
            return Response()
    client = CapCutMateClient("http://local/openapi/capcut-mate/v1", http=FakeHttp())
    draft_url = client.create_draft(width=1920, height=1080)
    client.add_videos(draft_url, [{"video_url": "clip.mp4", "start": 0, "end": 1000000}])
    assert calls[1][1]["video_infos"].startswith("[")
```

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_capcut_mate.py`
Expected: FAIL because `capcut_mate.py` does not exist.

- [ ] **Step 3: Implement adapter**

Implement `clips_to_video_infos` and `CapCutMateClient` methods for `create_draft`, `add_videos`, and `save_draft`.

- [ ] **Step 4: Run tests to verify green**

Run: `python -m pytest tests/test_capcut_mate.py`
Expected: PASS.

## Task 6: FastAPI App and Single-Page GUI

**Files:**
- Create: `src/hobby_basketball/app.py`
- Create: `src/hobby_basketball/static/index.html`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write failing API smoke tests**

```python
from fastapi.testclient import TestClient
from hobby_basketball.app import app

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_plan_clips_endpoint_returns_default_window():
    client = TestClient(app)
    response = client.post("/api/plan-clips", json={
        "video_path": "game.mp4",
        "events": [{"id": "make-1", "video_path": "game.mp4", "t_make": 10.0, "confidence": 0.8}]
    })
    assert response.status_code == 200
    assert response.json()["clips"][0]["start"] == 5.0
    assert response.json()["clips"][0]["end"] == 11.5
```

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_app.py`
Expected: FAIL because `app.py` does not exist.

- [ ] **Step 3: Implement app routes and static GUI**

Implement:
- `GET /api/health`
- `POST /api/plan-clips`
- `GET /` serving `index.html`

The GUI should include video path, pre/post inputs, event JSON textarea, plan button, and output preview.

- [ ] **Step 4: Run tests to verify green**

Run: `python -m pytest tests/test_app.py`
Expected: PASS.

## Task 7: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-25-basketball-ai-capcut-plugin-design.md` if needed

- [ ] **Step 1: Document setup and usage**

Include:
- install command
- test command
- server command
- capcut-mate integration note
- MVP limitations

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 3: Inspect git diff**

Run: `git status -sb` and `git diff --stat`
Expected: only intended project files.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add .
git commit -m "feat: bootstrap basketball AI clipping app"
git push -u origin main
```

Expected: branch pushed to `tofreekobe/Hobby_Basketball`.

