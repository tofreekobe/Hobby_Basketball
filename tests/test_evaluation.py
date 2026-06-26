from hobby_basketball.evaluation import evaluate_event_times


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
