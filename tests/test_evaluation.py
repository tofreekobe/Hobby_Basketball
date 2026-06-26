from hobby_basketball.evaluation import evaluate_confidence_thresholds, evaluate_event_times
from hobby_basketball.models import MadeShotEvent


def test_evaluates_event_times_with_one_to_one_matching():
    report = evaluate_event_times(
        predicted_times=[18.53, 44.66, 70.0],
        truth_times=[18.40, 44.70],
        tolerance_sec=1.0,
    )

    assert report.true_positives == 2
    assert report.false_positives == 1
    assert report.false_negatives == 0
    assert report.precision == 0.666667
    assert report.recall == 1.0
    assert report.f1 == 0.8


def test_evaluation_does_not_match_one_truth_multiple_times():
    report = evaluate_event_times(
        predicted_times=[18.1, 18.3, 18.5],
        truth_times=[18.4],
        tolerance_sec=1.0,
    )

    assert report.true_positives == 1
    assert report.false_positives == 2
    assert report.false_negatives == 0


def test_evaluates_confidence_thresholds_against_precision_recall_targets():
    events = [
        MadeShotEvent(id="make-1", video_path="game.mp4", t_make=10.0, confidence=0.92),
        MadeShotEvent(id="make-2", video_path="game.mp4", t_make=20.0, confidence=0.84),
        MadeShotEvent(id="false-1", video_path="game.mp4", t_make=45.0, confidence=0.30),
    ]

    report = evaluate_confidence_thresholds(
        events,
        truth_times=[10.2, 20.1],
        tolerance_sec=1.0,
        target_precision=0.95,
        target_recall=0.95,
    )

    assert report.target_met is True
    assert report.recommended_threshold == 0.84
    assert report.recommended is not None
    assert report.recommended.precision == 1.0
    assert report.recommended.recall == 1.0
    assert [point.threshold for point in report.points] == [0.92, 0.84, 0.3]


def test_confidence_threshold_report_uses_best_f1_when_target_is_unmet():
    events = [
        MadeShotEvent(id="make-1", video_path="game.mp4", t_make=10.0, confidence=0.90),
        MadeShotEvent(id="false-1", video_path="game.mp4", t_make=12.0, confidence=0.85),
        MadeShotEvent(id="make-2", video_path="game.mp4", t_make=20.0, confidence=0.50),
    ]

    report = evaluate_confidence_thresholds(
        events,
        truth_times=[10.1, 20.1],
        tolerance_sec=1.0,
        target_precision=0.95,
        target_recall=0.95,
    )

    assert report.target_met is False
    assert report.recommended_threshold == 0.5
    assert report.recommended is not None
    assert report.recommended.f1 > 0
