# Basketball AI CapCut Plugin Design

Date: 2026-06-25

## Status

Approved direction from user:

- GUI direction: single-page workspace.
- Architecture direction: extend `capcut-mate` rather than building only on the older `JianYingApi`.
- Core behavior: import a basketball video, automatically detect made shots, cut each made-shot clip, concatenate/export a final video, and generate a Jianying/CapCut draft.
- Default clip window: 5 seconds before the made shot and 1.5 seconds after. Both values must be user-editable.

## References

- `Hommy-master/capcut-mate`: https://github.com/Hommy-master/capcut-mate
- `Hommy-master/capcut-mate-electron`: https://github.com/Hommy-master/capcut-mate-electron
- `owengong/bball-highlights`: https://github.com/owengong/bball-highlights
- `JianYingApi`: https://gitcode.com/gh_mirrors/ji/JianYingApi

## Problem

The user wants a practical basketball AI feature for Jianying/CapCut-style editing:

1. A non-technical user imports a basketball game video.
2. The tool detects made-shot moments.
3. The tool creates highlight clips with configurable context before and after each made shot.
4. The clips are stitched into an exported MP4.
5. The same timeline is made available as an editable Jianying/CapCut draft.

This is not a general "LLM watches video and edits creatively" feature. The hard part is deterministic event detection and reviewable timeline generation. The editing/draft layer should be delegated to an existing draft automation project.

## Scope

### MVP In Scope

- Local video input, initially one game video.
- Single hoop calibration for static or mostly static camera footage.
- YOLO-based ball detection near the calibrated rim.
- Made-shot detection using rim-plane crossing, adapted from `bball-highlights`.
- Configurable clip window:
  - `pre_seconds`, default `5.0`
  - `post_seconds`, default `1.5`
- Event list generation with confidence, source timestamps, clip start/end, and keep/delete state.
- Manual review in a single-page GUI.
- FFmpeg export to `final.mp4`.
- Jianying/CapCut draft generation through `capcut-mate` APIs:
  - create draft
  - add video segments
  - save/get draft
  - optionally submit export through existing Windows `gen_video`

### Out Of Scope For MVP

- Broadcast multi-camera support.
- Automatic multi-hoop switching.
- Dunk/block/steal/highlight action classification.
- Speech command control.
- AI music selection.
- Automatic subtitles.
- Cloud rendering unless the local `capcut-mate` setup already supports it.
- Guaranteed official CapCut/Jianying plugin integration. This is a desktop companion/workspace that generates drafts and exports, not an official in-app plugin SDK.

## Architecture

Use `capcut-mate` as the host project and add a basketball feature module.

```text
Electron single-page workspace
  -> FastAPI /openapi/capcut-mate/v1/basketball/*
    -> basketball detector service
      -> video scan cache
      -> made-shot event timeline
    -> FFmpeg reel exporter
    -> capcut-mate draft service
      -> create_draft
      -> add_videos
      -> save_draft / get_draft
      -> gen_video when available
```

The older `JianYingApi` remains a useful reference for draft JSON structure, but it should not be the first implementation base because `capcut-mate` already has better draft templates, FastAPI routing, tests, Windows export automation, and an Electron client.

## Backend Design

### New Package

Add a focused basketball module under `capcut-mate`:

```text
src/basketball/
  __init__.py
  schemas.py
  detector.py
  clips.py
  draft_export.py
  ffmpeg_export.py
  calibration.py
```

Responsibilities:

- `schemas.py`: Pydantic request/response models.
- `calibration.py`: load/save rim calibration and validate coordinates.
- `detector.py`: scan frames, run YOLO ball detection, detect made-shot events.
- `clips.py`: compute clip intervals, merge overlaps, apply keep/delete review state.
- `ffmpeg_export.py`: export reviewed intervals to `final.mp4`.
- `draft_export.py`: translate reviewed intervals into `capcut-mate` `video_infos` and call draft services.

### API Routes

Add routes under:

```text
/openapi/capcut-mate/v1/basketball
```

Endpoints:

1. `POST /calibrate`
   - Input: video path, rim box or center/size.
   - Output: calibration id/path and normalized rim config.

2. `POST /detect`
   - Input: video path, calibration, `pre_seconds`, `post_seconds`, detector options.
   - Output: job id plus current status.
   - Detection can be synchronous for tiny test files, but should be job-based for real videos.

3. `GET /jobs/{job_id}`
   - Output: status, progress, logs, detected events.

4. `POST /events/{job_id}/review`
   - Input: edited events with keep/delete and adjusted start/end.
   - Output: saved review state.

5. `POST /export/reel`
   - Input: reviewed events, output path.
   - Output: exported MP4 path and summary.

6. `POST /export/draft`
   - Input: reviewed events, draft settings.
   - Output: draft_url, segment ids, local draft path if available.

7. `POST /export/all`
   - Input: reviewed events, output MP4 path, draft settings.
   - Output: final MP4 path plus draft_url.

### Event Model

Events should be explicit and reviewable:

```json
{
  "id": "make-0001",
  "type": "made_shot",
  "video_path": "C:/videos/game.mp4",
  "t_make": 124.38,
  "t_above": 124.18,
  "t_below": 124.58,
  "start": 119.38,
  "end": 125.88,
  "pre_seconds": 5.0,
  "post_seconds": 1.5,
  "confidence": 0.82,
  "kept": true,
  "notes": "YOLO rim-plane crossing"
}
```

The default MVP end time is `t_make + post_seconds`. The `t_exit` trajectory-based endpoint is explicitly deferred to a post-MVP enhancement because fixed post-roll is simpler to verify and matches the user's requested default behavior.

### Draft Timeline Translation

For `capcut-mate`, `add_videos` accepts timeline microseconds. For each kept event:

- Source video: original basketball file.
- Timeline placement:
  - first kept clip starts at `0`.
  - next clip starts at previous timeline end.
- Source range:
  - the implementation must preserve source start/end semantics correctly.

Important implementation note:

`capcut-mate` current `add_videos` primarily interprets `start/end` as timeline positions, while `source_timerange` inside `VideoSegment` starts at `0` in the existing implementation. For true source subclips, we need either:

1. add support to `capcut-mate` for explicit `source_start` and `source_end`, or
2. pre-cut clips with FFmpeg into temporary files and add those temporary files at timeline positions.

MVP should use option 2 because it is less invasive and matches the FFmpeg final export path. Direct `source_start/source_end` support in `capcut-mate` is a post-MVP enhancement.

## GUI Design

Use a single-page workspace in the Electron client, with one new tab: `篮球 AI`.

Layout:

```text
Top bar: project name, status, settings

Left panel:
  - video picker
  - draft output directory
  - pre seconds input, default 5.0
  - post seconds input, default 1.5
  - detector device: auto / cpu / cuda
  - buttons:
    - load video
    - calibrate rim
    - detect shots
    - export MP4
    - create Jianying draft

Main panel:
  - video preview frame
  - rim calibration overlay
  - detection progress/log stream

Bottom/right panel:
  - detected events table/cards
  - each event has thumbnail, make time, confidence, start/end, keep toggle
  - batch actions: keep all, drop low confidence, export kept
```

The GUI must be honest about state:

- "未检测" before detection.
- "检测中" with progress during scan.
- "需要复核" after events are found.
- "导出失败" with error detail if FFmpeg or draft creation fails.
- "已导出" only after the file exists.

## Data Flow

1. User selects `game.mp4`.
2. GUI probes metadata with FFmpeg/OpenCV.
3. User calibrates rim by drawing a box or entering center/size.
4. Backend saves `calib.json`.
5. User starts detection.
6. Backend samples frames and runs YOLO ball detection near rim ROI.
7. Backend detects made-shot events and caches detections.
8. GUI shows event list.
9. User reviews/edits event start/end and keep flags.
10. Backend exports:
    - `events.json`
    - temporary clip files
    - `final.mp4`
    - Jianying/CapCut draft via capcut-mate

## Error Handling

- Missing video: show a blocking validation error.
- Unsupported video format: fail before detection.
- Missing FFmpeg: show install guidance and block export.
- Missing YOLO dependencies: detection unavailable, but GUI still loads.
- No calibration: block detection and ask user to calibrate.
- No makes detected: show empty state, allow retry with adjusted thresholds.
- Detection job crashes: preserve logs and partial cache.
- FFmpeg export fails: keep events and temp files for debugging.
- Draft creation fails: still keep `final.mp4` export if it succeeded.
- Jianying export unavailable: show "草稿已生成，但自动导出需要 Windows 剪映环境".

## Testing Strategy

Use TDD for implementation.

Backend unit tests:

- clip interval calculation from made-shot timestamps.
- overlap merging.
- event keep/delete filtering.
- conversion from seconds to microseconds.
- draft `video_infos` generation from pre-cut clips.
- validation errors for bad clip parameters.

Detector tests:

- small synthetic trajectory arrays for made shot vs miss vs bounce-out.
- no model required for these tests.
- YOLO integration can be marked as slow/manual.

FFmpeg tests:

- use generated tiny test video if FFmpeg is installed.
- otherwise skip with a clear reason.

GUI tests:

- component-level tests for parameter form and event table if the Electron client stack already supports test tooling.
- otherwise keep GUI thin and verify backend behavior first.

Manual acceptance test:

1. Select a short basketball sample.
2. Calibrate rim.
3. Run detection.
4. Verify events appear.
5. Change pre/post seconds.
6. Export MP4.
7. Confirm `final.mp4` exists and plays.
8. Create Jianying draft.
9. Confirm draft can be opened or downloaded via capcut-mate.

## Implementation Order

1. Create backend basketball data models and pure clip interval functions.
2. Add tests for interval calculation, overlap merging, and draft timeline conversion.
3. Port made-shot trajectory logic from `bball-highlights` into pure functions.
4. Add detector wrapper with cached scan outputs.
5. Add FFmpeg pre-cut and concat export.
6. Add `capcut-mate` draft export adapter using pre-cut clips.
7. Add FastAPI routes.
8. Add Electron single-page workspace tab.
9. Wire GUI to backend job/status/event/export APIs.
10. Run backend tests and smoke test a tiny generated video path.

## Open Technical Questions

These do not block MVP, but they should be answered during implementation:

1. Should source videos be copied into a project workspace, or referenced in place?
   - MVP: copy or pre-cut into workspace to avoid broken draft paths.
2. Should draft export use local file paths or URL-accessible material paths?
   - MVP: use local files if running desktop/local; capcut-mate can copy into draft assets.
3. Should `t_exit` be included in MVP?
   - No. MVP uses default post seconds. `t_exit` is a post-MVP enhancement.
4. Should the GUI be inside capcut-mate Electron or a standalone app?
   - MVP: inside capcut-mate Electron as a new tab.

## Acceptance Criteria

- User can select one basketball video in the GUI.
- User can set `pre_seconds` and `post_seconds`.
- Defaults are `5.0` and `1.5`.
- User can save or enter rim calibration.
- Detection produces a list of made-shot candidate events.
- User can keep/delete events before export.
- Export creates `final.mp4` from kept events.
- Export creates a Jianying/CapCut draft using the same kept event order.
- If automatic Jianying export is unavailable, the app clearly says draft creation succeeded but automatic export is unavailable.
- No success state is shown unless the output file or draft result actually exists.
