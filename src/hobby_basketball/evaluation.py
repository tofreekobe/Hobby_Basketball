from __future__ import annotations

from pydantic import BaseModel, Field


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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
