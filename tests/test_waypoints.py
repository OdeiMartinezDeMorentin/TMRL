import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tmml.core.waypoints import WaypointManager


def create_temp_waypoints(waypoints, tmpdir=None):
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "waypoints.json")
    data = {
        "track_name": "test",
        "waypoints": waypoints.tolist(),
        "spacing": 2.5,
        "source": "test",
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_find_nearest():
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [7.5, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path)

    idx = wpm.find_nearest([1.0, 0.0, 0.0])
    assert idx == 0

    idx = wpm.find_nearest([3.0, 0.0, 0.0])
    assert idx == 1

    idx = wpm.find_nearest([10.0, 0.0, 0.0])
    assert idx == 4


def test_progress_fraction():
    waypoints = np.array([
        [float(i * 2.5), 0.0, 0.0] for i in range(100)
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path)

    assert abs(wpm.progress_fraction(0) - 0.0) < 1e-6
    assert abs(wpm.progress_fraction(50) - 50 / 99) < 1e-6
    assert abs(wpm.progress_fraction(99) - 1.0) < 1e-6


def test_smooth_waypoints():
    raw = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, -0.1, 0.0],
        [3.0, 0.2, 0.0],
        [4.0, -0.2, 0.0],
        [5.0, 0.0, 0.0],
    ])
    smoothed = WaypointManager.smooth_waypoints(raw, window=3, iterations=2)
    assert smoothed.shape == raw.shape
    for i in range(1, len(smoothed) - 1):
        assert smoothed[i, 1] < 0.2

    assert abs(smoothed[0, 0] - 0.0) < 3.0


def test_respace_waypoints():
    points = np.array([
        [float(i), 0.0, 0.0] for i in range(100)
    ])
    respaced = WaypointManager.respace_waypoints(points, spacing=2.5)
    assert len(respaced) > 1

    diffs = np.diff(respaced, axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    assert np.mean(dists) > 2.0
    assert np.mean(dists) < 3.0


def test_lookahead_angles():
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [5.0, 0.0, 2.5],
        [7.5, 0.0, 5.0],
        [10.0, 0.0, 7.5],
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path, look_aheads=[1, 3])
    wpm._current_idx = 0

    angles = wpm.get_lookahead_angles([0.0, 0.0, 0.0], yaw=0.0, current_idx=0)
    assert len(angles) == 2
    for a in angles:
        assert -1.0 <= a <= 1.0


def test_lookahead_distances():
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [7.5, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path, look_aheads=[1, 2])
    wpm._current_idx = 0

    dists = wpm.get_lookahead_distances([0.0, 0.0, 0.0], current_idx=0)
    assert len(dists) == 2
    for d in dists:
        assert 0.0 <= d <= 1.0


def test_reset():
    waypoints = np.array([
        [float(i), 0.0, 0.0] for i in range(10)
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path)
    wpm._current_idx = 5
    wpm.reset()
    assert wpm._current_idx == 0


def test_total_count():
    waypoints = np.array([
        [float(i), 0.0, 0.0] for i in range(50)
    ])
    path = create_temp_waypoints(waypoints)
    wpm = WaypointManager(path)
    assert wpm.total_count == 50


if __name__ == "__main__":
    test_find_nearest()
    test_progress_fraction()
    test_smooth_waypoints()
    test_respace_waypoints()
    test_lookahead_angles()
    test_lookahead_distances()
    test_reset()
    test_total_count()
    print("All waypoint tests passed!")