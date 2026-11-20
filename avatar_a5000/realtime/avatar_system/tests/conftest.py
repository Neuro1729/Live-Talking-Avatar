import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _seg(i, clip, s, e, ss, es, conv=("idle", "listen", "think", "speak"), nods=0, gesture=False):
    return {"id": i, "key": f"{clip}@{s}-{e}", "clip": clip, "s": s, "e": e, "start_state": ss, "end_state": es,
            "conv": list(conv), "quality": 1.0, "loopable": ss == es, "speaking_ok": "speak" in conv,
            "desc": {"frames": e - s + 1, "intensity": 0.5 + 0.1 * i, "blinks": 1, "blink_frames": [s + 5], "nods": nods,
                     "gaze_up_frac": 0.0, "gaze_dir": "camera", "gesture": gesture, "head_range_deg": [1, 1, 1],
                     "smile": 0.5, "brow_up": 0.02, "closed_frac": 0.02}}


@pytest.fixture
def toy_graph():
    """6 segments over 3 clips; neutral hub + one 'tilt' state; segment 5 is a dead end (only reachable)."""
    segs = [
        _seg(0, "c1", 0, 49, "neutral", "neutral"),
        _seg(1, "c1", 50, 99, "neutral", "tilt", conv=("speak", "think")),
        _seg(2, "c2", 0, 59, "tilt", "neutral", conv=("speak",), nods=1),
        _seg(3, "c2", 60, 119, "neutral", "neutral", conv=("idle", "listen", "speak")),
        _seg(4, "c3", 0, 39, "neutral", "neutral", conv=("idle", "listen", "think", "speak"), gesture=True),
        _seg(5, "c3", 40, 89, "neutral", "neutral"),
    ]
    adj = {0: [[1, 0.1, 0], [3, 0.3, 1], [4, 0.4, 1]], 1: [[2, 0.3, 1]], 2: [[0, 0.3, 1], [3, 0.2, 0], [4, 0.5, 1]],
           3: [[0, 0.3, 1], [4, 0.3, 1], [5, 0.6, 1]], 4: [[0, 0.2, 1], [3, 0.3, 1], [5, 0.1, 0]], 5: []}
    return {"version": "toy", "fps": 25, "hub": "neutral",
            "states": {"neutral": {"members": 6}, "tilt": {"members": 2}},
            "segments": segs, "adjacency": {str(k): v for k, v in adj.items()}}


@pytest.fixture
def clip_len():
    return {"c1": 100, "c2": 120, "c3": 90}
