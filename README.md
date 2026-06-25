# Hobby Basketball

Basketball made-shot highlight reel and CapCut/Jianying draft helper.

This repository contains the first runnable MVP for the basketball AI clipping workflow:

- plan made-shot clip windows with configurable pre-roll and post-roll
- detect made shots from ball trajectory samples
- build FFmpeg commands for MP4 reels
- generate capcut-mate compatible draft payloads
- expose a local FastAPI app with a lightweight single-page GUI

The default clip window is 5 seconds before the made shot and 1.5 seconds after.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .[dev]
```

Optional vision dependencies for future YOLO/OpenCV scanning:

```powershell
.\.venv\Scripts\python -m pip install -e .[vision]
```

## Run Tests

```powershell
python -m pytest
```

## Run App

```powershell
python -m uvicorn hobby_basketball.app:app --reload
```

Open `http://127.0.0.1:8000`.

## Current Workflow

1. Start the app.
2. Paste reviewed made-shot events as JSON.
3. Adjust pre-roll and post-roll seconds.
4. Click `规划剪辑片段`.
5. Use the returned clip intervals for FFmpeg export or capcut-mate draft generation.

Example event:

```json
{
  "id": "make-1",
  "video_path": "game.mp4",
  "t_make": 10.0,
  "confidence": 0.8
}
```

## capcut-mate Integration

This project does not vendor all of capcut-mate. Instead, it prepares clip files and capcut-mate compatible `video_infos` payloads. Point `CapCutMateClient` at a running capcut-mate service to create drafts and add generated clips.

## MVP Limitations

- The GUI currently plans clips from reviewed event JSON.
- Full YOLO frame scanning is intentionally isolated for a later adapter.
- Automatic Jianying export depends on a working capcut-mate Windows environment.
