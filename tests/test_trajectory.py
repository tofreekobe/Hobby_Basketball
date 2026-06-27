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


def test_rejects_crossing_samples_with_unrealistic_horizontal_jump():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=72, y=80, confidence=0.8),
        BallSample(t=1.25, x=128, y=118, confidence=0.8),
    ]

    assert detect_made_shots(samples, rim) == []


def test_detects_ball_entering_net_region_without_strict_upper_crossing():
    rim = RimCalibration(center_x=927, center_y=141, half_width=42, half_height=50)
    samples = [
        BallSample(t=17.60, x=882.5, y=133.5, confidence=0.62),
        BallSample(t=18.40, x=948.5, y=137.0, confidence=0.66),
        BallSample(t=18.67, x=953.0, y=210.0, confidence=0.56),
        BallSample(t=18.93, x=1044, y=71, confidence=0.47),
    ]

    makes = detect_made_shots(samples, rim)

    assert len(makes) == 1
    assert 18.0 <= makes[0].t_make <= 18.7
    assert makes[0].notes == "rim-net entry"


def test_keeps_earliest_make_when_later_net_activity_is_higher_confidence():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=20)
    samples = [
        BallSample(t=1.00, x=96, y=96, confidence=0.55),
        BallSample(t=1.20, x=98, y=124, confidence=0.55),
        BallSample(t=2.50, x=104, y=94, confidence=0.65),
        BallSample(t=2.70, x=106, y=132, confidence=0.65),
    ]

    makes = detect_made_shots(samples, rim)

    assert len(makes) == 1
    assert makes[0].t_make == 1.1


def test_replaces_early_make_when_later_cluster_candidate_is_much_stronger():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=20)
    samples = [
        BallSample(t=1.00, x=96, y=96, confidence=0.35),
        BallSample(t=1.20, x=98, y=124, confidence=0.35),
        BallSample(t=2.50, x=104, y=94, confidence=0.8),
        BallSample(t=2.70, x=106, y=132, confidence=0.8),
    ]

    makes = detect_made_shots(samples, rim)

    assert len(makes) == 1
    assert makes[0].t_make == 2.6


def test_rejects_net_entry_that_drops_outward_on_same_side_of_rim():
    rim = RimCalibration(center_x=927, center_y=141, half_width=42, half_height=50)
    samples = [
        BallSample(t=14.40, x=960.5, y=146.0, confidence=0.67),
        BallSample(t=14.60, x=989.5, y=203.5, confidence=0.58),
    ]

    assert detect_made_shots(samples, rim) == []


def test_rejects_shallow_side_entry_that_keeps_drifting_away_from_rim():
    rim = RimCalibration(center_x=927, center_y=141, half_width=42, half_height=50)
    samples = [
        BallSample(t=14.30, x=950.0, y=130.5, confidence=0.68),
        BallSample(t=14.50, x=973.5, y=169.0, confidence=0.53),
    ]

    assert detect_made_shots(samples, rim) == []
