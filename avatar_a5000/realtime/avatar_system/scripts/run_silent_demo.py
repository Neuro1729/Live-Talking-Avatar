"""Continuous silent motion-graph playback over a simulated conversation (Phase B deliverable).

Writes <out>.mp4 (clean), <out>_debug.mp4 (overlay: state, segment, seams), <out>_trace.json (every frame ref, seams,
decisions) for evaluation. Also supports the old GraphDirector random walk as a baseline (--planner random).

  envs/analysis/bin/python avatar_system/scripts/run_silent_demo.py --minutes 10 --out avatar_system/data/demos/silent_10min
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, path  # noqa: E402
from avatar_system.conversation.state_machine import StateMachine  # noqa: E402
from avatar_system.motion.motion_scheduler import FrameRef, MotionScheduler  # noqa: E402
from avatar_system.streaming.frame_source import ClipFrameSource  # noqa: E402

FPS = 25


def conversation_script(seconds, seed=0):
    """Deterministic simulated turns: (t_seconds, event)."""
    rng = random.Random(seed)
    ev, t = [], 4.0
    while t < seconds:
        listen = rng.uniform(8, 25)
        ev.append((t, "user_speaking"))
        t += listen
        ev.append((t, "user_done"))
        t += rng.uniform(1.5, 4)
        ev.append((t, "audio_ready"))
        speak = rng.uniform(8, 25)
        if rng.random() < 0.15:  # user barges in
            t += speak * 0.5
            ev.append((t, "interrupt"))
            ev.append((t + 0.3, "cancelled"))
            t += 0.3
        else:
            t += speak
            ev.append((t, "response_done"))
        t += rng.uniform(1, 4)
    return [e for e in ev if e[0] < seconds]


class RandomWalk:
    """Baseline: uniform random approved successor, no semantics / repetition / lookahead."""

    def __init__(self, graph, clip_len, seed=0):
        self.inner = MotionScheduler(graph, clip_len, seed=seed)
        self.rng = random.Random(seed)
        planner = self.inner.planner

        def choose(cur, ctx, t):
            d, c, b = self.rng.choice(planner.adj[cur])
            planner.last_decision = {"plan": [d], "candidates": []}
            return d, c, bool(b), planner.last_decision
        planner.choose = choose

    def __getattr__(self, k):
        return getattr(self.inner, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", default=str(data_dir("demos") / "silent_10min"))
    ap.add_argument("--planner", default="beam", choices=["beam", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--no-video", action="store_true", help="trace only (fast, for evaluation)")
    a = ap.parse_args()

    graph = json.loads((data_dir() / "motion_graph.json").read_text())
    src = ClipFrameSource()
    clips = {s["clip"] for s in graph["segments"]}
    clip_len = {c: src.length(c) for c in clips}
    sm = StateMachine()
    Sched = MotionScheduler if a.planner == "beam" else RandomWalk
    sched = Sched(graph, clip_len, seed=a.seed)
    core = sched.inner if a.planner == "random" else sched
    core.mode_source = lambda: sm.state

    def seg_end(seg):  # recovery finishes once the motion is back at the neutral hub
        if seg["end_state"] == graph["hub"]:
            sm.fire("hub_reached")
    core.on_segment_end = seg_end

    total = int(a.minutes * 60 * FPS)
    script = conversation_script(total / FPS, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = str(path("ffmpeg"))
    H = round(928 * a.width / 1024 / 2) * 2

    def pipe(name):
        return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                 f"{a.width}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                                 "-preset", "veryfast", "-pix_fmt", "yuv420p", str(name)], stdin=subprocess.PIPE)
    clean = dbg = None
    if not a.no_video:
        clean, dbg = pipe(out.with_suffix(".mp4")), pipe(out.parent / f"{out.name}_debug.mp4")
    trace, k, t0 = [], 0, time.time()
    for n in range(total):
        while k < len(script) and script[k][0] * FPS <= n:
            sm.fire(script[k][1])
            k += 1
        ref = sched.next()
        trace.append([ref.n, ref.seg, ref.clip, ref.idx, ref.blend_clip, ref.blend_idx, round(ref.alpha, 3),
                      int(ref.seam), sm.state])
        if clean:
            f = cv2.resize(src.get(ref), (a.width, H), interpolation=cv2.INTER_AREA)
            clean.stdin.write(f.tobytes())
            d = f.copy()
            cv2.putText(d, f"{sm.state}  {graph['segments'][ref.seg]['key']}  f{ref.idx}", (8, 22), 0, 0.55, (0, 0, 200), 2)
            cv2.putText(d, f"t={n / FPS:6.1f}s seams={len(sched.seams)}", (8, 44), 0, 0.55, (0, 0, 200), 2)
            if ref.seam or ref.blend_clip:
                cv2.rectangle(d, (0, 0), (a.width - 1, H - 1), (0, 0, 255) if ref.seam else (0, 160, 255), 4)
            dbg.stdin.write(d.tobytes())
        if n % (FPS * 60) == 0 and n:
            print(f"  {n / FPS / 60:.0f} min rendered ({time.time() - t0:.0f}s)", flush=True)
    for p in (clean, dbg):
        if p:
            p.stdin.close()
            p.wait()
    seams = [list(map(lambda x: x if not isinstance(x, (np.integer,)) else int(x), s)) for s in sched.seams]
    out.parent.joinpath(f"{out.name}_trace.json").write_text(json.dumps(
        {"graph_version": graph["version"], "planner": a.planner, "seed": a.seed, "fps": FPS, "frames": trace,
         "seams": seams, "script": script, "state_history": sm.history}))
    nat = sum(1 for s in seams if s[6])
    print(f"done: {total / FPS / 60:.1f} min, {len(seams)} segment changes ({nat} natural continuations, "
          f"{sum(1 for s in seams if s[4])} blended), {len({s[2] for s in seams})} distinct segments, "
          f"{time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
