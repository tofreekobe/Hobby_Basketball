from __future__ import annotations

import json
from pathlib import Path
import uuid
from decimal import Decimal, ROUND_HALF_UP

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from hobby_basketball.clips import build_clip_intervals, merge_intervals
from hobby_basketball.detector import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SAMPLE_FPS,
    inspect_cuda_runtime,
    scan_video_for_made_shots,
)
from hobby_basketball.evaluation import (
    ConfidenceThresholdReport,
    EventEvaluationReport,
    evaluate_confidence_thresholds,
    evaluate_event_times,
)
from hobby_basketball.ffmpeg_export import export_reel
from hobby_basketball.models import ClipInterval, MadeShotEvent
from hobby_basketball.review_sheet import build_candidate_review_sheet
from hobby_basketball.trajectory import RimCalibration
from hobby_basketball.workspace import (
    EVALUATION_DIR,
    EXPORT_DIR,
    REVIEW_DIR,
    export_path,
    normalize_output_format,
    save_upload,
)


class PlanClipsRequest(BaseModel):
    video_path: str
    events: list[MadeShotEvent]
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    merge_overlaps: bool = True


class PlanClipsResponse(BaseModel):
    clips: list[ClipInterval]


class UploadVideoResponse(BaseModel):
    video_id: str
    filename: str
    video_path: str
    preview_url: str


class ProcessVideoRequest(BaseModel):
    video_id: str
    rim: RimCalibration
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    output_format: str = "mp4"
    sample_fps: float = Field(default=DEFAULT_SAMPLE_FPS, gt=0)
    confidence: float = Field(default=0.15, ge=0, le=1)
    device: str = "cpu"
    model_name: str = DEFAULT_MODEL_NAME


class DetectVideoResponse(BaseModel):
    events: list[MadeShotEvent]
    clips: list[ClipInterval]


class ExportEventsRequest(BaseModel):
    video_id: str
    events: list[MadeShotEvent]
    pre_seconds: float = Field(default=5.0, ge=0)
    post_seconds: float = Field(default=1.5, gt=0)
    output_format: str = "mp4"


class ProcessVideoResponse(BaseModel):
    events: list[MadeShotEvent]
    clips: list[ClipInterval]
    output_path: str
    preview_url: str


class EvaluateEventsRequest(BaseModel):
    predicted_times: list[float]
    truth_times: list[float]
    tolerance_sec: float = Field(default=1.0, gt=0)


class EvaluateCandidatesRequest(BaseModel):
    events: list[MadeShotEvent]
    truth_times: list[float]
    tolerance_sec: float = Field(default=1.0, gt=0)
    target_precision: float = Field(default=0.95, ge=0, le=1)
    target_recall: float = Field(default=0.95, ge=0, le=1)


class SaveEvaluationRunRequest(EvaluateCandidatesRequest):
    video_id: str
    rim: RimCalibration | None = None
    sample_fps: float = Field(default=DEFAULT_SAMPLE_FPS, gt=0)
    confidence: float = Field(default=0.15, ge=0, le=1)
    device: str = "cpu"
    model_name: str = DEFAULT_MODEL_NAME


class SaveEvaluationRunResponse(BaseModel):
    run_id: str
    evaluation_path: str
    report: ConfidenceThresholdReport


class CandidateReviewSheetRequest(BaseModel):
    video_id: str
    events: list[MadeShotEvent]
    rim: RimCalibration


class CandidateReviewSheetResponse(BaseModel):
    sheet_id: str
    review_path: str
    preview_url: str


class SaveCandidateReviewRequest(BaseModel):
    video_id: str
    events: list[MadeShotEvent]
    rim: RimCalibration | None = None
    reviewer: str = "manual"


class CandidateReviewSummary(BaseModel):
    candidate_count: int
    accepted_count: int
    rejected_count: int
    review_precision: float


class SaveCandidateReviewResponse(BaseModel):
    review_id: str
    review_path: str
    summary: CandidateReviewSummary


class EvaluationTotals(BaseModel):
    predicted_count: int = Field(default=0, ge=0)
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)


class EvaluationRunPoint(BaseModel):
    predicted_count: int = Field(default=0, ge=0)
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    precision: float = Field(default=0.0, ge=0, le=1)
    recall: float = Field(default=0.0, ge=0, le=1)
    f1: float = Field(default=0.0, ge=0, le=1)


class EvaluationDatasetSummary(BaseModel):
    run_count: int = Field(ge=0)
    target_met_count: int = Field(ge=0)
    target_precision: float = Field(ge=0, le=1)
    target_recall: float = Field(ge=0, le=1)
    target_met: bool
    totals: EvaluationTotals
    micro_precision: float = Field(ge=0, le=1)
    micro_recall: float = Field(ge=0, le=1)
    micro_f1: float = Field(ge=0, le=1)
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    run_ids: list[str]


app = FastAPI(title="Hobby Basketball", version="0.1.0")
VIDEO_REGISTRY: dict[str, Path] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/device-status")
def device_status() -> dict[str, object]:
    return inspect_cuda_runtime()


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


@app.post("/api/evaluate-events", response_model=EventEvaluationReport)
def evaluate_events(request: EvaluateEventsRequest) -> EventEvaluationReport:
    return evaluate_event_times(
        request.predicted_times,
        request.truth_times,
        tolerance_sec=request.tolerance_sec,
    )


@app.post("/api/evaluate-candidates", response_model=ConfidenceThresholdReport)
def evaluate_candidates(request: EvaluateCandidatesRequest) -> ConfidenceThresholdReport:
    return evaluate_confidence_thresholds(
        request.events,
        request.truth_times,
        tolerance_sec=request.tolerance_sec,
        target_precision=request.target_precision,
        target_recall=request.target_recall,
    )


@app.post("/api/save-evaluation-run", response_model=SaveEvaluationRunResponse)
def save_evaluation_run(request: SaveEvaluationRunRequest) -> SaveEvaluationRunResponse:
    video_path = _get_video_path(request.video_id)
    report = evaluate_confidence_thresholds(
        request.events,
        request.truth_times,
        tolerance_sec=request.tolerance_sec,
        target_precision=request.target_precision,
        target_recall=request.target_recall,
    )
    run_id = uuid.uuid4().hex
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_path = EVALUATION_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "video_id": request.video_id,
        "video_path": str(video_path),
        "rim": request.rim.model_dump() if request.rim else None,
        "settings": {
            "sample_fps": request.sample_fps,
            "confidence": request.confidence,
            "device": request.device,
            "model_name": request.model_name,
            "tolerance_sec": request.tolerance_sec,
            "target_precision": request.target_precision,
            "target_recall": request.target_recall,
        },
        "truth_times": [float(value) for value in request.truth_times],
        "events": [event.model_dump() for event in request.events],
        "report": report.model_dump(),
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return SaveEvaluationRunResponse(
        run_id=run_id,
        evaluation_path=str(evaluation_path),
        report=report,
    )


@app.get("/api/evaluation-summary", response_model=EvaluationDatasetSummary)
def evaluation_summary() -> EvaluationDatasetSummary:
    runs = _load_evaluation_runs()
    target_precision = runs[0]["target_precision"] if runs else 0.95
    target_recall = runs[0]["target_recall"] if runs else 0.95
    totals = EvaluationTotals()

    for run in runs:
        point = run["point"]
        totals.predicted_count += point.predicted_count
        totals.true_positives += point.true_positives
        totals.false_positives += point.false_positives
        totals.false_negatives += point.false_negatives

    micro_precision = _safe_ratio(totals.true_positives, totals.true_positives + totals.false_positives)
    micro_recall = _safe_ratio(totals.true_positives, totals.true_positives + totals.false_negatives)
    micro_f1 = _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    macro_precision = _average_metric([run["point"].precision for run in runs])
    macro_recall = _average_metric([run["point"].recall for run in runs])
    macro_f1 = _average_metric([run["point"].f1 for run in runs])

    return EvaluationDatasetSummary(
        run_count=len(runs),
        target_met_count=sum(1 for run in runs if run["target_met"]),
        target_precision=target_precision,
        target_recall=target_recall,
        target_met=bool(runs)
        and micro_precision >= target_precision
        and micro_recall >= target_recall,
        totals=totals,
        micro_precision=_round_metric(micro_precision),
        micro_recall=_round_metric(micro_recall),
        micro_f1=_round_metric(micro_f1),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        run_ids=[run["run_id"] for run in runs],
    )


@app.post("/api/upload-video", response_model=UploadVideoResponse)
def upload_video(file: UploadFile) -> UploadVideoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="请选择视频文件")
    video_id, path = save_upload(file)
    VIDEO_REGISTRY[video_id] = path
    return UploadVideoResponse(
        video_id=video_id,
        filename=file.filename,
        video_path=str(path),
        preview_url=f"/api/videos/{video_id}",
    )


@app.get("/api/videos/{video_id}")
def video_preview(video_id: str):
    path = _get_video_path(video_id)
    return FileResponse(path)


@app.get("/api/exports/{filename}")
def export_preview(filename: str):
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(path)


@app.get("/api/reviews/{filename}")
def review_preview(filename: str):
    safe_name = Path(filename).name
    path = REVIEW_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Review sheet not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/process-video", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest) -> ProcessVideoResponse:
    detected = detect_video(request)
    return export_events(
        ExportEventsRequest(
            video_id=request.video_id,
            events=detected.events,
            pre_seconds=request.pre_seconds,
            post_seconds=request.post_seconds,
            output_format=request.output_format,
        )
    )


@app.post("/api/detect-video", response_model=DetectVideoResponse)
def detect_video(request: ProcessVideoRequest) -> DetectVideoResponse:
    video_path = _get_video_path(request.video_id)

    try:
        events = scan_video_for_made_shots(
            video_path,
            request.rim,
            sample_fps=request.sample_fps,
            confidence=request.confidence,
            model_name=request.model_name,
            device=request.device,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not events:
        raise HTTPException(status_code=422, detail="未识别到进球片段，请检查篮筐标定或降低置信度")

    clips = merge_intervals(build_clip_intervals(events, request.pre_seconds, request.post_seconds))
    return DetectVideoResponse(events=events, clips=clips)


@app.post("/api/candidate-review-sheet", response_model=CandidateReviewSheetResponse)
def candidate_review_sheet(request: CandidateReviewSheetRequest) -> CandidateReviewSheetResponse:
    if not request.events:
        raise HTTPException(status_code=422, detail="No candidate events to review")
    video_path = _get_video_path(request.video_id)
    sheet_id = uuid.uuid4().hex
    review_path = REVIEW_DIR / f"{sheet_id}.jpg"
    try:
        build_candidate_review_sheet(video_path, request.events, request.rim, review_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CandidateReviewSheetResponse(
        sheet_id=sheet_id,
        review_path=str(review_path),
        preview_url=f"/api/reviews/{sheet_id}.jpg",
    )


@app.post("/api/save-candidate-review", response_model=SaveCandidateReviewResponse)
def save_candidate_review(request: SaveCandidateReviewRequest) -> SaveCandidateReviewResponse:
    if not request.events:
        raise HTTPException(status_code=422, detail="No candidate events to review")

    candidate_count = len(request.events)
    accepted_count = sum(1 for event in request.events if event.kept)
    rejected_count = candidate_count - accepted_count
    summary = CandidateReviewSummary(
        candidate_count=candidate_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        review_precision=round(accepted_count / candidate_count, 6),
    )

    review_id = uuid.uuid4().hex
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_path = REVIEW_DIR / f"{review_id}.json"
    payload = {
        "review_id": review_id,
        "video_id": request.video_id,
        "reviewer": request.reviewer,
        "rim": request.rim.model_dump() if request.rim else None,
        "events": [event.model_dump() for event in request.events],
        "summary": summary.model_dump(),
        "metrics_scope": "candidate_precision_only",
    }
    review_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return SaveCandidateReviewResponse(
        review_id=review_id,
        review_path=str(review_path),
        summary=summary,
    )


@app.post("/api/export-events", response_model=ProcessVideoResponse)
def export_events(request: ExportEventsRequest) -> ProcessVideoResponse:
    video_path = _get_video_path(request.video_id)
    try:
        output_format = normalize_output_format(request.output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    events = [
        event.model_copy(update={"video_path": str(video_path)})
        for event in request.events
    ]
    clips = merge_intervals(build_clip_intervals(events, request.pre_seconds, request.post_seconds))
    if not clips:
        raise HTTPException(status_code=422, detail="没有已保留的候选片段可导出")

    final_path = export_path(request.video_id, output_format)
    clip_dir = final_path.with_suffix("")
    try:
        export_reel(clips, clip_dir, final_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"FFmpeg 导出失败：{exc}") from exc

    return ProcessVideoResponse(
        events=events,
        clips=clips,
        output_path=str(final_path),
        preview_url=f"/api/exports/{final_path.name}",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    static_path = Path(__file__).with_name("static") / "index.html"
    return static_path.read_text(encoding="utf-8")


def _get_video_path(video_id: str) -> Path:
    path = VIDEO_REGISTRY.get(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="视频不存在，请重新选择文件")
    return path


def _load_evaluation_runs() -> list[dict[str, object]]:
    if not EVALUATION_DIR.exists():
        return []

    runs: list[dict[str, object]] = []
    for path in sorted(EVALUATION_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run = _evaluation_run_from_payload(path, payload)
        if run is not None:
            runs.append(run)
    return runs


def _evaluation_run_from_payload(path: Path, payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    report = payload.get("report")
    if not isinstance(report, dict):
        return None

    target_precision = _float_value(
        report.get("target_precision", _setting_value(payload, "target_precision", 0.95)),
        0.95,
    )
    target_recall = _float_value(
        report.get("target_recall", _setting_value(payload, "target_recall", 0.95)),
        0.95,
    )
    recommended = report.get("recommended")
    if isinstance(recommended, dict):
        point = EvaluationRunPoint(
            predicted_count=_int_value(recommended.get("predicted_count"), 0),
            true_positives=_int_value(recommended.get("true_positives"), 0),
            false_positives=_int_value(recommended.get("false_positives"), 0),
            false_negatives=_int_value(recommended.get("false_negatives"), 0),
            precision=_float_value(recommended.get("precision"), 0.0),
            recall=_float_value(recommended.get("recall"), 0.0),
            f1=_float_value(recommended.get("f1"), 0.0),
        )
    elif recommended is None:
        point = EvaluationRunPoint(false_negatives=_truth_time_count(payload))
    else:
        return None

    return {
        "run_id": str(payload.get("run_id") or path.stem),
        "target_met": bool(report.get("target_met")),
        "target_precision": target_precision,
        "target_recall": target_recall,
        "point": point,
    }


def _setting_value(payload: dict[str, object], name: str, default: float) -> object:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return default
    return settings.get(name, default)


def _truth_time_count(payload: dict[str, object]) -> int:
    truth_times = payload.get("truth_times")
    if not isinstance(truth_times, list):
        return 0
    return len([value for value in truth_times if isinstance(value, (int, float))])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _average_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round_metric(sum(values) / len(values))


def _round_metric(value: float) -> float:
    decimal_value = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return float(decimal_value)


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
