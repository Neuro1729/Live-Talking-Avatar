"""Continuous lip-synced motion-graph video from prerecorded speech (Phase C deliverable). Runs the real pipeline
(same code as live), unpaced, and records every output frame + its audio into one mp4.

Script (frame-accurate hooks): idle -> Hindi sentence -> 0.4 s gap (hold, still SPEAKING) -> Hindi sentence ->
response done -> listening -> Hinglish streamed in 0.5 s chunks -> English -> fast Hindi -> interruption mid-sentence
-> recovery -> short reply -> long silence.

  cd R && CUDA_VISIBLE_DEVICES=3 envs/musetalk/bin/python avatar_system/scripts/run_lipsync_demo.py [--mock]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, load_config, path  # noqa: E402
from avatar_system.conversation.state_machine import StateMachine  # noqa: E402
from avatar_system.lipsync.audio_context import SR, AudioEvent  # noqa: E402
from avatar_system.lipsync.base import NullLipSync  # noqa: E402
from avatar_system.streaming.pipeline import FPS, AvatarPipeline  # noqa: E402


def wav(name):
    return sf.read(str(data_dir("test_audio") / f"{name}.wav"), dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="no MuseTalk: source frames only (tests the timeline)")
    ap.add_argument("--out", default=str(data_dir("demos") / "lipsync_demo"))
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--composite", default=None, help="override musetalk.yaml composite (v1|v2)")
    ap.add_argument("--restore", type=float, default=None, help="GFPGAN mouth restore weight (0 = off)")
    ap.add_argument("--v2", default=None, help='JSON overrides of musetalk.yaml v2, e.g. {"lip_w": 0.75}')
    a = ap.parse_args()
    graph = json.loads((data_dir() / "motion_graph.json").read_text())
    run = set(graph.get("runtime_segments") or range(len(graph["segments"])))
    clips = sorted({s["clip"] for s in graph["segments"] if s["id"] in run})
    if a.mock:
        backend = NullLipSync()
    else:
        from avatar_system.lipsync.musetalk_adapter import MuseTalkBackend
        cfg_mt = dict(load_config("musetalk"))
        if a.composite:
            cfg_mt["composite"] = a.composite
        if a.v2:
            cfg_mt["v2"] = {**cfg_mt["v2"], **json.loads(a.v2)}
        if a.restore is not None:
            cfg_mt["restore"] = {**cfg_mt.get("restore", {}), "enabled": a.restore > 0, "weight": a.restore}
        backend = MuseTalkBackend(clips, cfg=cfg_mt)
        print(f"MuseTalk loaded in {backend.load_s:.1f}s, {len(clips)} clip caches in {backend.cache_s:.1f}s", flush=True)
    cfg = load_config("musetalk")
    sm = StateMachine()
    pipe = AvatarPipeline(graph, backend, sm, batch=8, out_width=a.width, seed=a.seed, silence=cfg["silence"],
                          feature_shift=cfg.get("feature_shift_frames", 0),
                          encode=False)

    def push(uid, x, chunk=None):
        if chunk is None:
            pipe.push_audio(AudioEvent("demo", uid, SR, 0, x, True))
            return
        step = int(chunk * SR)
        for o in range(0, len(x), step):
            pipe.push_audio(AudioEvent("demo", uid, SR, o, x[o:o + step], o + step >= len(x)))

    events = {}  # frame -> list of actions
    def at(sec, fn):
        events.setdefault(int(sec * FPS), []).append(fn)

    log = []
    def ev(name, fn=None):
        def run(n):
            log.append((n, round(n / FPS, 2), name))
            if fn:
                fn()
        return run

    at(1, ev("user_speaking", lambda: sm.fire("user_speaking")))
    at(5, ev("user_done", lambda: sm.fire("user_done")))
    at(7, ev("hi_1 + hold", lambda: (setattr(pipe, "hold", True), push("u1", wav("hi_1")))))
    at(11.5, ev("hi_2 queued after 0.4s gap", lambda: push("u2", np.r_[np.zeros(int(0.4 * SR), np.float32), wav("hi_2")])))
    at(12, ev("answer complete (hold off)", lambda: setattr(pipe, "hold", False)))
    at(22, ev("user_speaking", lambda: sm.fire("user_speaking")))
    at(27, ev("user_done", lambda: sm.fire("user_done")))
    at(29, ev("hinglish streamed 0.5s chunks + en + fast", lambda: (setattr(pipe, "hold", True),
                                                                    push("u3", wav("hinglish_1"), chunk=0.5),
                                                                    push("u4", wav("en_1")), push("u5", wav("hi_fast")))))
    at(31, ev("hold off", lambda: setattr(pipe, "hold", False)))
    at(37, ev("INTERRUPT (user barges in)", lambda: (sm.fire("interrupt"), pipe.interrupt(), sm.fire("cancelled"))))
    at(48, ev("user_done", lambda: sm.fire("user_done")))
    at(50, ev("short reply", lambda: push("u6", wav("hi_short"))))
    total = 62 * FPS

    def hook(n):
        for fn in events.get(n, []):
            fn(n)
    pipe.frame_hooks.append(hook)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    H = round(928 * a.width / 1024 / 2) * 2
    vid = out.parent / f"{out.name}_video.mp4"
    ff = subprocess.Popen([str(path("ffmpeg")), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{a.width}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset",
                           "veryfast", "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    pcm, metas = [], []
    t0 = time.time()
    pipe.start()
    try:
        for k in range(total):
            item = pipe.read(timeout=30)
            if item is None:
                raise RuntimeError("pipeline stalled")
            f, p, meta = item
            assert meta["n"] == k, (meta["n"], k)  # continuous numbering
            ff.stdin.write(np.ascontiguousarray(f).tobytes())
            pcm.append(p)
            metas.append(meta)
    finally:
        m = pipe.metrics()
        pipe.stop()
        ff.stdin.close()
        ff.wait()
    wall = time.time() - t0
    audio = np.frombuffer(b"".join(pcm), np.int16)
    sf.write(str(out.parent / f"{out.name}.wav"), audio, SR)
    subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", str(vid), "-i",
                    str(out.parent / f"{out.name}.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    str(out.with_suffix(".mp4"))], check=True)
    vid.unlink()
    rows = [t for t in pipe.trace if t[0] < total]  # the pipeline plans ahead of what was recorded
    seams = [list(s) for s in pipe.sched.seams if s[0] < total]
    trace = {"graph_version": graph["version"], "planner": "beam", "seed": a.seed, "fps": FPS,
             "frames": [t[:9] for t in rows], "utt": [t[9] for t in rows], "seams": seams, "events": log, "metrics": m, "backend": backend.name,
             "wall_s": round(wall, 1), "state_history": sm.history}
    (out.parent / f"{out.name}_trace.json").write_text(json.dumps(trace, default=str))
    speech_seams = sum(1 for s in seams if trace["utt"][s[0]] is not None)
    print(f"done: {total / FPS:.0f}s video in {wall:.0f}s ({total / wall:.1f} fps unpaced), backend {backend.name}, "
          f"{len(seams)} segment changes ({speech_seams} during speech), "
          f"audio feature late frames {m['counters']['audio_feature_late']} -> {out.with_suffix('.mp4')}")
    print(json.dumps({k: m[k] for k in ("render_ms", "render_ms_per_frame", "encode_ms", "render_fps_capacity",
                                        "counters", "audio")}, default=str))
    for e in log:
        print("  event", e)


if __name__ == "__main__":
    main()
