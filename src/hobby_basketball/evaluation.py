from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from hobby_basketball.models import MadeShotEvent


class EventEvaluationReport(BaseModel):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    matches: list[tuple[float, float]]
    unmatched_predictions: list[float]
    unmatched_truths: list[float]


class ConfidenceThresholdPoint(BaseModel):
    threshold: float = Field(ge=0, le=1)
    predicted_count: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    target_met: bool


class ConfidenceThresholdReport(BaseModel):
    target_precision: float = Field(ge=0, le=1)
    target_recall: float = Field(ge=0, le=1)
    target_met: bool
    recommended_threshold: float | None = None
    recommended: ConfidenceThresholdPoint | None = None
    points: list[ConfidenceThresholdPoint]


def evaluate_event_times(
    predicted_times: list[float],
    truth_times: list[float],
    *,
    tolerance_sec: float = 1.0,
) -> EventEvaluationReport:
    remaining_truths = sorted(float(value) for value in truth_times)
    matches: list[tuple[float, float]] = []
    unmatched_predictions: list[float] = []

    for predicted in sorted(float(value) for value in predicted_times):
        best_index: int | None = None
        best_delta = tolerance_sec
        for index, truth in enumerate(remaining_truths):
            delta = abs(predicted - truth)
            if delta <= best_delta:
                best_index = index
                best_delta = delta
        if best_index is None:
            unmatched_predictions.append(predicted)
            continue
        truth = remaining_truths.pop(best_index)
        matches.append((predicted, truth))

    true_positives = len(matches)
    false_positives = len(unmatched_predictions)
    false_negatives = len(remaining_truths)
    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return EventEvaluationReport(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        matches=matches,
        unmatched_predictions=unmatched_predictions,
        unmatched_truths=remaining_truths,
    )


def evaluate_confidence_thresholds(
    events: Iterable[MadeShotEvent],
    truth_times: list[float],
    *,
    tolerance_sec: float = 1.0,
    target_precision: float = 0.95,
    target_recall: float = 0.95,
) -> ConfidenceThresholdReport:
    ordered_events = sorted(events, key=lambda event: event.confidence, reverse=True)
    thresholds = sorted({round(float(event.confidence), 6) for event in ordered_events}, reverse=True)
    points: list[ConfidenceThresholdPoint] = []

    for threshold in thresholds:
        filtered = [event for event in ordered_events if event.confidence >= threshold]
        report = evaluate_event_times(
            [event.t_make for event in filtered],
            truth_times,
            tolerance_sec=tolerance_sec,
        )
        points.append(
            ConfidenceThresholdPoint(
                threshold=threshold,
                predicted_count=len(filtered),
                true_positives=report.true_positives,
                false_positives=report.false_positives,
                false_negatives=report.false_negatives,
                precision=report.precision,
                recall=report.recall,
                f1=report.f1,
                target_met=report.precision >= target_precision and report.recall >= target_recall,
            )
        )

    recommended = _recommend_threshold_point(points)
    return ConfidenceThresholdReport(
        target_precision=target_precision,
        target_recall=target_recall,
        target_met=bool(recommended and recommended.target_met),
        recommended_threshold=recommended.threshold if recommended else None,
        recommended=recommended,
        points=points,
    )


def _recommend_threshold_point(points: list[ConfidenceThresholdPoint]) -> ConfidenceThresholdPoint | None:
    if not points:
        return None

    target_points = [point for point in points if point.target_met]
    if target_points:
        return max(
            target_points,
            key=lambda point: (point.f1, point.recall, point.precision, point.predicted_count),
        )
    return max(
        points,
        key=lambda point: (point.f1, point.recall, point.precision, point.predicted_count),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
