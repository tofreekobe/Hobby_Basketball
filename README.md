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

### Inference Device

The GUI and API default to `cpu` for compatibility. This avoids CUDA runtime
errors on newer GPUs when the installed PyTorch wheel does not support the
GPU's compute capability.

If you select `auto` or `cuda` and YOLO hits `CUDA error: no kernel image is
available for execution on the device`, the detector retries that prediction on
CPU. Use CUDA only after installing a PyTorch build that explicitly supports
your GPU architecture.

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
2. Click `选择篮球录像` and upload a video file.
3. Use the source preview to pause on a clear hoop frame.
4. Click the hoop center in the preview to fill rim center coordinates.
5. Adjust rim half width/height, pre-roll, post-roll, confidence, and export format.
6. Click `执行识别并剪辑`.
7. Review the exported preview video in the browser.

Full video recognition requires optional vision dependencies and FFmpeg:

```powershell
python -m pip install -e .[vision]
winget install Gyan.FFmpeg
```

The detector uses YOLO sports-ball detection near the calibrated rim and the rim-plane crossing logic from the project design.

## capcut-mate Integration

This project does not vendor all of capcut-mate. Instead, it prepares clip files and capcut-mate compatible `video_infos` payloads. Point `CapCutMateClient` at a running capcut-mate service to create drafts and add generated clips.

## MVP Limitations

- Hoop calibration is manual: click the hoop center and adjust half width/height.
- Static or mostly static camera footage works best.
- Automatic Jianying export depends on a working capcut-mate Windows environment.
