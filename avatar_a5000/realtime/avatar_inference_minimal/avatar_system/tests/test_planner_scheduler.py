from avatar_system.conversation.state_machine import StateMachine
from avatar_system.motion.graph_validator import validate
from avatar_system.motion.motion_planner import Context, MotionPlanner
from avatar_system.motion.motion_scheduler import MotionScheduler
from avatar_system.motion.repetition_tracker import RepetitionTracker


def runtime(g):
    g["runtime_segments"] = sorted(i for i in range(len(g["segments"]))
                                   if g["segments"][i]["key"] not in validate(g, write=False, verbose=False)["outside_main_scc"])
    return g


def test_planner_only_takes_approved_runtime_edges(toy_graph):
    g = runtime(toy_graph)
    p = MotionPlanner(g, seed=1)
    for cur in g["runtime_segments"]:
        for _ in range(20):
            nxt, cost, blend, info = p.choose(cur, Context(mode="SPEAKING"), 0)
            assert nxt in [d for d, *_ in g["adjacency"][str(cur)]]
            assert nxt in g["runtime_segments"]  # never the dead end (segment 5)


def test_scheduler_frames_are_continuous_and_inside_segments(toy_graph, clip_len):
    g = runtime(toy_graph)
    s = MotionScheduler(g, clip_len, seed=0, speed=1.0)
    prev = None
    for n in range(3000):
        r = s.next()
        assert r.n == n
        seg = g["segments"][r.seg]
        assert seg["s"] <= r.idx <= seg["e"] and r.clip == seg["clip"]
        if prev is not None and not r.seam:
            assert r.idx == prev.idx + 1 and r.clip == prev.clip  # inside a segment frames advance by one
        if r.blend_clip:
            assert 0 < r.alpha < 1 and r.blend_idx < clip_len[r.blend_clip]
        prev = r
    assert len(s.seams) > 10


def test_listening_prefers_listen_segments(toy_graph, clip_len):
    g = runtime(toy_graph)
    s = MotionScheduler(g, clip_len, seed=0, mode_source=lambda: "LISTENING")
    frames = [s.next() for _ in range(5000)]
    ok = sum("listen" in g["segments"][r.seg]["conv"] for r in frames) / len(frames)
    assert ok > 0.6


def test_repetition_penalty_orders_correctly(toy_graph):
    rt = RepetitionTracker(toy_graph["segments"])
    for t, sid in enumerate([0, 3, 0, 3]):
        rt.push(sid, t * 50)
    assert rt.penalty([3], 200) > rt.penalty([4], 200)  # 0->3 seen twice, recently played
    assert rt.penalty([4], 200) >= 0


def test_gesture_spacing_penalty(toy_graph):
    rt = RepetitionTracker(toy_graph["segments"])
    rt.push(4, 0)  # gesture segment
    assert rt.penalty([4], 100) > rt.penalty([4], 10000)


def test_state_machine_interrupt_and_recovery():
    sm = StateMachine()
    assert sm.fire("user_done") == "THINKING"
    assert sm.fire("audio_ready") == "SPEAKING"
    assert sm.fire("interrupt") == "INTERRUPTED"
    assert sm.fire("cancelled") == "RECOVERY"
    assert sm.fire("audio_ready") == "SPEAKING"  # new answer may start during recovery
    assert sm.fire("response_done") == "LISTENING"
    assert sm.fire("cancelled") == "LISTENING" and sm.ignored == 1  # invalid events are ignored, never raise


def test_early_exit_when_mode_turns_calm(toy_graph, clip_len):
    """An expressive segment (no listen tag) is left at an approved inside rest point once the mode is LISTENING."""
    g = runtime(toy_graph)
    g["exits"] = {"1": [[70, 3, 0.2, 1]]}  # segment 1 (c1@50-99, speak/think) may exit at frame 70 into segment 3
    mode = {"m": "SPEAKING"}
    s = MotionScheduler(g, clip_len, seed=0, mode_source=lambda: mode["m"], speed=1.0)
    s.cur, s.pos = 1, 50.0
    refs = [s.next() for _ in range(15)]           # frames 50..64 while speaking: no exit
    assert all(r.seg == 1 for r in refs)
    mode["m"] = "LISTENING"
    refs = [s.next() for _ in range(10)]           # 65..70 then exit
    assert refs[5].clip == "c1" and refs[5].idx == 70
    assert refs[6].seg == 3 and refs[6].seam and s.seams[-1][7] is True


def test_playback_speed_skips_frames_evenly(toy_graph, clip_len):
    g = runtime(toy_graph)
    s = MotionScheduler(g, clip_len, seed=0, speed=1.3)
    steps, prev = [], None
    for _ in range(3000):
        r = s.next()
        seg = g["segments"][r.seg]
        assert seg["s"] <= r.idx <= seg["e"]
        if prev is not None and not r.seam:
            assert r.clip == prev.clip and r.idx - prev.idx in (1, 2)
            steps.append(r.idx - prev.idx)
        prev = r
    assert abs(sum(steps) / len(steps) - 1.3) < 0.05
