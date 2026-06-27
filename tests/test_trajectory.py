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
