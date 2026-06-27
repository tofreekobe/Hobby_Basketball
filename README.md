# Hobby Basketball

Basketball made-shot highlight reel and CapCut/Jianying draft helper.

This repository contains the first runnable MVP for the basketball AI clipping workflow:

- plan made-shot clip windows with configurable pre-roll and post-roll
- detect made shots from fast orange-ball color/motion samples, with optional YOLO sports-ball enhancement
- build FFmpeg commands for MP4 reels
- generate capcut-mate compatible draft payloads
- expose a local FastAPI app with a lightweight single-page GUI

The default clip window is 5 seconds before the made shot and 1.5 seconds after.
The default detector profile is CPU-safe and fast: `model_name=none` at 12
sampled frames per second. Set `model_name=yolo11n.pt` only when you want the
optional YOLO sports-ball enhancer.

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
CPU and keeps using CPU for the rest of the scan. Use CUDA only after installing
a PyTorch build that explicitly supports your GPU architecture.

The `sample_fps` option controls the speed/accuracy tradeoff. The default 12 FPS
works well for the fast color/motion path; lower it only for very long videos or
slow machines.

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
4. Drag a rectangle around the hoop/net in the preview, or click the hoop center and adjust the box fields manually.
5. Adjust rim box size, pre-roll, post-roll, confidence, sample FPS, and export format.
6. Click `识别候选` to generate reviewable made-shot candidates.
7. Use each candidate's `跳转` action to inspect the source video around that moment.
8. Uncheck false positives in `候选进球复核`.
9. Click `导出已选候选` to render only the kept clips, or use `执行识别并剪辑` for the legacy one-click path.
10. Optional: seek the source video to each true make, click `添加当前时间为标注`, then click `评估候选准确率` to calculate precision, recall, and F1.

Full video recognition requires optional vision dependencies and FFmpeg:

```powershell
python -m pip install -e .[vision]
winget install Gyan.FFmpeg
```

The detector defaults to orange-ball color/motion candidates, strict rim-plane crossing, and a rim-net entry score for distant footage where the ball is only visible while entering the net. YOLO sports-ball detection near the calibrated rim is optional.

## Accuracy Evaluation

Accuracy claims require labeled made-shot times. The GUI can compare current kept candidates with manually entered labels, and `hobby_basketball.evaluation.evaluate_event_times` can do the same in code:

```python
from hobby_basketball.evaluation import evaluate_event_times

report = evaluate_event_times(
    predicted_times=[18.53, 44.66],
    truth_times=[18.40, 44.70],
    tolerance_sec=1.0,
)
print(report.precision, report.recall, report.f1)
```

Targeting 95%+ means validating both precision and recall against a representative labeled set, not a single clip.

Use `/api/evaluate-candidates` or the GUI evaluation button after entering manual
truth times to sweep candidate confidence thresholds. The report returns the
recommended threshold, whether the 95% precision and 95% recall targets were
met, and the full threshold table for tuning. In the GUI, the recommended
threshold is copied back to the confidence field and candidates below it are
unchecked before export.

## capcut-mate Integration

This project does not vendor all of capcut-mate. Instead, it prepares clip files and capcut-mate compatible `video_infos` payloads. Point `CapCutMateClient` at a running capcut-mate service to create drafts and add generated clips.

## MVP Limitations

- Hoop calibration is manual: drag a hoop/net box or edit center/size fields.
- Static or mostly static camera footage works best.
- 95%+ accuracy requires a representative labeled evaluation set and may require model fine-tuning for specific court/camera styles.
- Automatic Jianying export depends on a working capcut-mate Windows environment.
