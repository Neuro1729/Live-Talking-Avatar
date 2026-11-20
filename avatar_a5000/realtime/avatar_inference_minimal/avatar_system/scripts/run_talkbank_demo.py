"""New-audio demo for the talking bank: a NEW answer (Sarvam TTS, never seen in the bank) is rendered by retrieving,
for each sentence, the recorded talking segment with the closest speech act / emotion / rhythm, time-warping it to the
sentence, using the clips' own natural pause frames for silences, choosing clip-change cut points inside pauses with
the transition scorer (3-frame crossfade), and re-syncing only the lips with MuseTalk (v2 composite + GFPGAN mouth).

  cd R && source avatar_system/scripts/_env.sh && CUDA_VISIBLE_DEVICES=3 \
    envs/musetalk/bin/python avatar_system/scripts/run_talkbank_demo.py --out avatar_system/data/demos/talkbank_demo
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, load_config, path, root  # noqa: E402
from avatar_system.motion.motion_scheduler import FrameRef  # noqa: E402
from avatar_system.motion.speech_retrieval import SpeechRetriever  # noqa: E402
from avatar_system.streaming.frame_source import ClipFrameSource  # noqa: E402

SR, FPS, SPF = 16000, 25, 640
ANSWER = [  # (text, lang, hints) - a new answer, not in the scripts
    ("नमस्ते! आपकी रिपोर्ट मैंने देख ली है।", "hi-IN", ["gentle_nod"]),
    ("घबराइए मत, आपकी शुगर थोड़ी ही बढ़ी हुई है।", "hi-IN", ["empathetic"]),
    ("Pehle, roz subah aadha ghanta walk kijiye. Phir, raat ka khana halka rakhiye.", "hi-IN", ["emphasis"]),
    ("Sabse zaroori baat, dawa bilkul time pe lijiye.", "hi-IN", ["emphasis"]),
    ("अपना ध्यान रखिए, कोई भी सवाल हो तो ज़रूर पूछिए।", "hi-IN", ["gentle_nod"]),
]
LEAD, GAP, TAIL = 1.0, 0.55, 1.6


async def tts_all():
    sys.path.insert(0, str(root() / "app"))
    from sarvam_client import Sarvam
    s = Sarvam()
    return [await s.tts(t, speaker="shubh", lang=l) for t, l, _ in ANSWER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(data_dir("demos") / "talkbank_demo"))
    ap.add_argument("--clips", default="", help="comma list of bank clips allowed (default: all indexed with cache)")
    ap.add_argument("--max-head", type=float, default=14.0, help="skip speech parts with head range above this")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--candidates", type=int, default=4, help="retrieved parts compared per sentence")
    ap.add_argument("--join-weight", type=float, default=0.3, help="weight of the join cost when choosing among them "
                    "(0 = best retrieval score only, the original behaviour)")
    ap.add_argument("--plan-only", action="store_true", help="retrieve + plan joins and print them, no MuseTalk render "
                    "(works before the clips' MuseTalk caches exist)")
    a = ap.parse_args()

    idx = json.loads((data_dir() / "talk_index.json").read_text())
    ok = {n for n in idx if a.plan_only or (path("musetalk_cache") / n / "DONE").exists()}
    if a.clips:
        ok &= set(a.clips.split(","))
    idx = {n: c for n, c in idx.items() if n in ok}
    for c in idx.values():
        c["parts"] = [p for p in c["parts"] if p["type"] == "pause" or max(p["head_range"]) <= a.max_head]
    if not idx:
        raise SystemExit("no indexed bank clips with MuseTalk caches")
    retr = SpeechRetriever(index=idx)
    print(f"bank: {len(idx)} clips, {len(retr.parts)} speech parts usable", flush=True)

    wavs = asyncio.run(tts_all())
    # audio timeline: lead, sentences separated by gaps, tail
    pcm, spans, t = [np.zeros(int(LEAD * SR), np.float32)], [], LEAD
    for k, w in enumerate(wavs):
        spans.append((int(round(t * FPS)), int(round((t + len(w) / SR) * FPS))))
        pcm.append(w)
        t += len(w) / SR
        g = GAP if k < len(wavs) - 1 else TAIL
        pcm.append(np.zeros(int(g * SR), np.float32))
        t += g
    audio = np.concatenate(pcm)
    total = int(np.ceil(len(audio) / SPF))
    audio = np.pad(audio, (0, total * SPF - len(audio)))

    from avatar_system.motion.clip_analyzer import analyze
    from avatar_system.motion.transition_scorer import score
    ans = {}
    getan = lambda c: ans.setdefault(c, analyze(c))
    src = ClipFrameSource()

    def join_cost(pp, gap, q):
        """Cheapest cut in the gap between part pp (from its end) and part q (into its preceding frames)."""
        clen, best = src.length(pp["clip"]), float("inf")
        for m in range(0, gap + 1, 2):
            ib = q["s"] - (gap - m)
            if ib < 0:
                continue
            c, _, reasons = score(getan(pp["clip"]), min(pp["e"] - 1 + m, clen - 1), getan(q["clip"]), ib)
            best = min(best, c + 5 * bool(reasons))
        return best

    # retrieve a part per sentence: among the top candidates take the best retrieval score + join_weight * the cost
    # of joining it to the previous sentence's part (with a big bank a near-equal match usually joins far better)
    choice = []
    for (txt, lang, hints), w, (s0, s1) in zip(ANSWER, wavs, spans):
        res, acts = retr.retrieve(txt, w, hints=hints, k=a.candidates)
        if choice and a.join_weight > 0:
            pp, _, ps1 = choice[-1]
            ranked = [(sc + a.join_weight * join_cost(pp, s0 - ps1, q), sc, q, comp) for sc, q, _, comp in res
                      if (q["clip"], q["part"]) != (pp["clip"], pp["part"])]  # never the same part twice in a row
            _, sc, p, comp = min(ranked, key=lambda x: x[0])
        else:
            sc, p, _, comp = res[0]
        choice.append((p, s0, s1))
        print(f"'{txt[:40]}' acts={acts} -> {p['clip']} part{p['part']} [{p['act']}/{p['emotion']}] "
              f"len {p['frames']}f vs {s1 - s0}f warp {(p['frames'] / max(s1 - s0, 1)):.2f} score {sc} {comp}", flush=True)

    # frame plan: each output frame -> (clip, idx); sentences warp their part, gaps continue the previous clip's
    # pause then cut (best join inside the gap) into the next part's preceding pause
    plan = [None] * total
    cuts = []

    def fill_part(p, s0, s1, limit):
        """Play the part at speed frames/n but never slower than 0.85x (no slow motion): a sentence longer than the
        part continues into the clip's following frames (its pause / next words), i.e. the body keeps talking. It
        never runs past `limit` (clip end, or where the next sentence's part in the same clip begins) - once there it
        swings back and forth near the end and still finishes at `limit`, so the gap after it continues forward
        naturally instead of freezing on the last frame, overlapping the next part or jumping out of a back-swing."""
        n = s1 - s0
        speed = max(p["frames"] / n, 0.85)
        lo, hi = p["s"], max(min(limit, src.length(p["clip"])) - 1, p["s"] + 1)
        span, extra = hi - lo, n * speed - (hi - lo)
        d = extra / (2 * np.ceil(extra / (2 * span))) if extra > 0 else 0  # depth of each back-swing

        def pos(x):
            if x <= span:
                return lo + int(round(x))
            r = (x - span) % (2 * d)
            return hi - int(round(r if r <= d else 2 * d - r))

        for k in range(n):
            plan[s0 + k] = (p["clip"], pos(k * speed))
        return pos(n * speed)

    p0, s0, _ = choice[0]
    for k in range(s0):  # lead: the frames before the first part in its clip
        plan[k] = (p0["clip"], max(p0["s"] - s0 + k, 0))
    for i, (p, s0, s1) in enumerate(choice):
        nxt = choice[i + 1] if i + 1 < len(choice) else None
        limit = src.length(p["clip"])
        if nxt and nxt[0]["clip"] == p["clip"] and nxt[0]["s"] > p["s"]:
            limit = max(nxt[0]["s"] - (nxt[1] - s1) + 1, p["e"])  # end exactly L frames before it: leave the gap frames to lead into the next part
        p = dict(p, e=fill_part(p, s0, s1, limit))  # the gap continues from where the sentence really ended
        g0, g1 = s1, (nxt[1] if nxt else total)
        L = g1 - g0
        clen = src.length(p["clip"])
        if nxt is None:
            for k in range(L):
                plan[g0 + k] = (p["clip"], min(p["e"] + k, clen - 1))
            break
        q = nxt[0]
        if q["clip"] == p["clip"] and q["s"] - p["e"] == L:  # natural continuation
            for k in range(L):
                plan[g0 + k] = (p["clip"], p["e"] + k)
            continue
        best = None
        for m in range(0, L + 1):  # m frames continue clip A, then L-m frames lead into part B
            ia, ib = min(p["e"] + m - 1, clen - 1), q["s"] - (L - m)
            if ib < 0:
                continue
            c, _, reasons = score(getan(p["clip"]), max(ia, 0), getan(q["clip"]), ib)
            c += 5 * bool(reasons)
            if best is None or c < best[0]:
                best = (c, m, ia, ib)
        c, m, ia, ib = best
        for k in range(m):
            plan[g0 + k] = (p["clip"], min(p["e"] + k, clen - 1))
        for k in range(L - m):
            plan[g0 + m + k] = (q["clip"], ib + k)
        cuts.append((g0 + m, p["clip"], ia, q["clip"], ib, round(c, 3)))
    print("cuts (frame, from, at, to, at, cost):", cuts, flush=True)
    if a.plan_only:
        return

    # refs with crossfades at cuts
    refs, cutset = [], {c[0]: c for c in cuts}
    for n, (clip, i) in enumerate(plan):
        r = FrameRef(n, 0, clip, int(i))
        for c0, c in cutset.items():
            j = n - c0
            if 0 <= j < 3:
                r.blend_clip, r.blend_idx, r.alpha = c[1], min(c[2] + 1 + j, src.length(c[1]) - 1), (j + 1) / 4
        refs.append(r)

    from avatar_system.lipsync.musetalk_adapter import MuseTalkBackend
    be = MuseTalkBackend(sorted({c for c, _ in plan} | {r.blend_clip for r in refs if r.blend_clip}))
    feats = be.features(audio)
    shift = load_config("musetalk").get("feature_shift_frames", 0)
    speech = np.zeros(total, bool)
    for s0, s1 in spans:
        speech[s0:s1] = True
    ramp = 4
    out_frames = []
    for b in range(0, total, 8):
        rr = refs[b:b + 8]
        ws, ff = [], []
        for r in rr:
            n = r.n
            d = 0 if speech[n] else next((k for k in range(1, ramp + 1) if speech[min(n + k, total - 1)] or speech[max(n - k, 0)]), None)
            ws.append(0.0 if d is None else 1 - d / (ramp + 1))
            m = min(max(n + shift, 0), len(feats) - 1)
            ff.append(feats[m] if (speech[min(max(n + shift, 0), total - 1)] or ws[-1] > 0) else None)
        lip = [k for k, w in enumerate(ws) if w > 0]
        frames = [src.get(r) for r in rr]
        if lip:
            res = be.process([rr[k] for k in lip], [ff[k] for k in lip], src)
            for k, f in zip(lip, res):
                frames[k] = f if ws[k] >= 1 else (frames[k].astype(np.float32) * (1 - ws[k]) + f.astype(np.float32) * ws[k] + 0.5).astype(np.uint8)
        out_frames += frames
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out) + ".wav", audio, SR)
    H = round(928 * a.width / 1024 / 2) * 2
    for name, dbg in ((str(out) + ".mp4", False), (str(out) + "_debug.mp4", True)):
        # video only through the pipe, audio muxed afterwards (reading a 2nd input + -shortest deadlocked the pipe)
        tmpv = name[:-4] + "_v.mp4"
        p = subprocess.Popen([str(path("ffmpeg")), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{a.width}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
                              "-pix_fmt", "yuv420p", tmpv], stdin=subprocess.PIPE)
        for n, f in enumerate(out_frames):
            f = cv2.resize(f, (a.width, H), interpolation=cv2.INTER_AREA)
            if dbg:
                si = next((i for i, (s0, s1) in enumerate(spans) if s0 <= n < s1), None)
                txt = f"{plan[n][0]} f{plan[n][1]}" + (f" | {choice[si][0]['act']}" if si is not None else " | pause")
                cv2.putText(f, txt, (8, 24), 0, 0.6, (0, 0, 200), 2)
                if any(0 <= n - c[0] < 3 for c in cuts):
                    cv2.rectangle(f, (0, 0), (a.width - 1, H - 1), (0, 0, 255), 4)
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", tmpv, "-i", str(out) + ".wav", "-c:v",
                        "copy", "-c:a", "aac", "-shortest", name], check=True)
        Path(tmpv).unlink()
    (Path(str(out) + "_plan.json")).write_text(json.dumps({"spans": spans, "cuts": cuts, "plan": plan,
        "choice": [{"clip": p["clip"], "part": p["part"], "act": p["act"], "text": t[0]} for (p, _, _), t in zip(choice, ANSWER)]},
        ensure_ascii=False))
    print(f"-> {out}.mp4 ({total / FPS:.1f}s), {out}_debug.mp4", flush=True)


if __name__ == "__main__":
    main()
