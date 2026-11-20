"""Pilot test of speech-driven gesture retrieval. Two steps (two envs):

  1. plan (analysis env): Sarvam TTS for new test sentences -> retrieve the best talking-bank speech part for each
     -> data/retrieval_demo/plan.json
     envs/analysis/bin/python avatar_system/scripts/run_retrieval_demo.py plan
  2. render (musetalk env, GPU): for each sentence, left = current system (motion-graph speaking segments),
     right = retrieved talking-bank part time-warped to the sentence; both lip-synced by MuseTalk to the SAME audio
     (v2 composite + mouth restore, feature shift as live). Captions show text + what was retrieved.
     CUDA_VISIBLE_DEVICES=3 envs/musetalk/bin/python avatar_system/scripts/run_retrieval_demo.py render
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, load_config, path, root  # noqa: E402

SR, FPS = 16000, 25
TESTS = [  # answers from the live session (not in the talking-bank scripts)
    ("बहुत-बहुत बधाई! यह सुनकर मुझे सच में बहुत खुशी हुई।", "hi-IN", ["positive", "gentle_nod"]),
    ("आपकी चिंता मैं बिल्कुल समझ सकता हूँ। आप थोड़ा हिम्मत रखें और उन्हें डॉक्टर के पास ले जाएँ।", "hi-IN",
     ["empathetic", "gentle_nod"]),
    ("Blood sugar badhna thoda chinta ki baat hai. Aap apni diet mein cheeni kam karein aur roz thodi sair karein.",
     "hi-IN", ["neutral_explanation", "emphasis"]),
    ("Stay hydrated, eat fresh fruits, and try to rest during the hottest part of the day.", "en-IN",
     ["neutral_explanation"]),
]


def plan():
    sys.path.insert(0, str(root() / "app"))
    import soundfile as sf
    from sarvam_client import Sarvam
    from avatar_system.motion.speech_retrieval import SpeechRetriever
    out = data_dir("retrieval_demo")
    s = Sarvam()
    r = SpeechRetriever()
    items = []
    for k, (text, lang, hints) in enumerate(TESTS):
        pcm = asyncio.run(s.tts(text, speaker="shubh", lang=lang))
        sf.write(out / f"t{k}.wav", pcm, SR)
        best, acts = r.retrieve(text, pcm, hints=hints)
        sc, p, warp, comp = best[0]
        items.append({"k": k, "text": text, "wav": str(out / f"t{k}.wav"), "acts": acts, "hints": hints,
                      "clip": p["clip"], "s": p["s"], "e": p["e"], "part_text": p["text"], "part_act": p["act"],
                      "warp": warp, "score": sc, "components": comp,
                      "alternatives": [(b[1]["clip"], b[1]["part"], b[0]) for b in best[1:]]})
        print(f"[{k}] {text[:50]}\n     acts {acts} -> {p['clip']} part '{(p['text'] or '')[:40]}' warp {warp} score {sc} {comp}")
    (out / "plan.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))


def render():
    import cv2
    import soundfile as sf
    from PIL import Image, ImageDraw
    from avatar_system.lipsync.musetalk_adapter import MuseTalkBackend
    from avatar_system.motion.motion_scheduler import FrameRef, MotionScheduler
    from avatar_system.scripts.caption_recording import draw_mixed, fonts
    from avatar_system.streaming.frame_source import ClipFrameSource
    out = data_dir("retrieval_demo")
    items = json.loads((out / "plan.json").read_text())
    graph = json.loads((data_dir() / "motion_graph.json").read_text())
    run = set(graph["runtime_segments"])
    gclips = {s["clip"] for s in graph["segments"] if s["id"] in run}
    be = MuseTalkBackend(sorted(gclips | {it["clip"] for it in items}))
    src = ClipFrameSource()
    shift = load_config("musetalk").get("feature_shift_frames", 0)
    sched = MotionScheduler(graph, {c: src.length(c) for c in gclips}, seed=5, mode_source=lambda: "SPEAKING")
    F = fonts(20)
    W, H = 640, 580
    vid = out / "retrieval_demo_v.mp4"
    enc = subprocess.Popen([str(path("ffmpeg")), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{2 * W}x{H + 90}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    audio = []
    for it in items:
        pcm = sf.read(it["wav"], dtype="float32")[0]
        pre = int(0.4 * SR)
        pcm = np.r_[np.zeros(pre, np.float32), pcm, np.zeros(int(0.6 * SR), np.float32)]
        n = int(np.ceil(len(pcm) / 640))
        feats = be.features(pcm)
        speech = [pre / 640 <= i < (len(pcm) - 0.6 * SR) / 640 for i in range(n)]
        left = [sched.next() for _ in range(n)]
        lead = int(pre / 640)
        right = []
        for i in range(n):  # part frames, time-warped so the part spans the sentence; lead-in from its own pause
            t = it["s"] - lead + (i * it["warp"] if i >= lead else i)
            right.append(FrameRef(i, 0, it["clip"], int(min(max(round(t), 0), src.length(it["clip"]) - 1))))
        for b in range(0, n, 8):
            idx = range(b, min(b + 8, n))
            fs = [feats[max(i + shift, 0)] if speech[min(max(i + shift, 0), n - 1)] else None for i in idx]
            lo = be.process([left[i] for i in idx], fs, src)
            ro = be.process([right[i] for i in idx], fs, src)
            for i, a, c in zip(idx, lo, ro):
                if not speech[i] and not any(speech[max(i - 3, 0):i + 4]):
                    a, c = src.get(left[i]), src.get(right[i])
                im = Image.new("RGB", (2 * W, H + 90), (16, 18, 22))
                for x0, f, lab in ((0, a, "CURRENT: motion-graph speaking clips"),
                                   (W, c, f"NEW: retrieved '{it['part_act']}' gestures ({it['clip']})")):
                    im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB)), (x0, 0))
                    d = ImageDraw.Draw(im)
                    d.rectangle((x0, 0, x0 + W, 28), fill=(0, 0, 0))
                    d.text((x0 + 8, 4), lab, font=F["lat"], fill=(255, 230, 120))
                d = ImageDraw.Draw(im)
                draw_mixed(d, (12, H + 12), "says: " + it["text"][:110], F, (220, 225, 230))
                d.text((12, H + 50), f"retrieval: acts {it['acts']} -> part '{(it['part_text'] or '')[:50]}'  "
                                     f"warp {it['warp']}", font=F["lat"], fill=(150, 200, 255))
                enc.stdin.write(np.asarray(im).tobytes())
        audio.append(pcm[: n * 640])
    enc.stdin.close()
    enc.wait()
    sf.write(str(out / "retrieval_demo.wav"), np.concatenate(audio), SR)
    final = out / "retrieval_demo.mp4"
    subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", str(vid), "-i", str(out / "retrieval_demo.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)], check=True)
    vid.unlink()
    print(f"-> {final}")


if __name__ == "__main__":
    {"plan": plan, "render": render}[sys.argv[1]]()
