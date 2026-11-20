"""Index the talking bank: for every speech part / pause of every generated clip, what was said (text, act, emotion,
duration, rhythm) and what the body did meanwhile (hand motion, nods, brow raises, smile, mouth opening, head range),
all on the clip's frame timeline. This is what speech-driven gesture retrieval searches.

  envs/analysis/bin/python -m avatar_system.generation.talk_index [clip ...]   -> data/talk_index.json
"""
import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from avatar_system.common import clip_path, data_dir
from avatar_system.conversation.behavior_planner import prosody
from avatar_system.motion.clip_analyzer import analyze
from avatar_system.motion.feature_extractor import FI, extract, is_current

FPS, SR = 25, 16000
HAND = (int(87 * .80), 87, int(96 * .25), int(96 * .78))


def envelope(pcm, n_frames):
    x = np.pad(pcm, (0, max(n_frames * 640 - len(pcm), 0)))[: n_frames * 640].reshape(n_frames, 640)
    db = 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-5)
    return np.clip((db + 60) / 60, 0, 1)


def describe_window(an, s, e):
    f = an["feats"][s:e]
    th = an["thumb"][s:e].astype(np.float32)
    hand = np.abs(np.diff(th[:, HAND[0]:HAND[1], HAND[2]:HAND[3]], axis=0)).mean((1, 2)) if e - s > 1 else np.zeros(1)
    pitch, brow = f[:, FI["pitch"]], f[:, FI["brow_up"]]
    nods = find_peaks(pitch, prominence=2.0, distance=8)[0]
    brows = find_peaks(brow, prominence=0.08, distance=8)[0]
    return {"hand_motion": round(float(hand.mean()), 3), "hand_peak": round(float(hand.max()), 3),
            "hand_events": [int(i) for i in find_peaks(hand, height=max(1.0, hand.mean() * 2), distance=10)[0]],
            "nods": [int(i) for i in nods], "brow_raises": [int(i) for i in brows],
            "head_range": [round(float(np.ptp(f[:, FI[k]])), 1) for k in ("yaw", "pitch", "roll")],
            "smile": round(float(f[:, FI["smile"]].mean()), 3),
            "mouth_open_p95": round(float(np.nanpercentile(f[:, FI["mouth_geo"]], 95)), 3),
            "blinks": int(sum(1 for b in an["blinks"] if s <= b < e))}


def index_clip(name):
    meta = json.loads((data_dir("talk_bank") / f"{name}_parts.json").read_text())
    if not is_current(name):
        extract(name)
    an = analyze(name)
    pcm = sf.read(str(data_dir("talk_bank") / f"{name}.wav"), dtype="float32")[0]
    env = envelope(pcm, an["n"])
    parts = []
    for k, p in enumerate(meta["parts"]):
        s, e = int(round(p["start"] * FPS)), min(int(round(p["end"] * FPS)), an["n"])
        if e - s < 3:
            continue
        d = {"clip": name, "part": k, "type": p["type"], "s": s, "e": e, "frames": e - s,
             "act": meta["act"], "emotion": meta["emotion"], "behavior": meta["behavior"],
             "text": p.get("text"), "lang": p.get("lang"), **describe_window(an, s, e)}
        if p["type"] == "speech":
            seg = pcm[int(p["start"] * SR):int(p["end"] * SR)]
            pr = prosody(seg, max_s=30)
            d.update({"emphasis": pr["emphasis_at"], "pause_frac": pr["pause_frac"], "f0_median": pr.get("f0_median"),
                      "envelope": [round(float(v), 3) for v in env[s:e]]})
        parts.append(d)
    whole = describe_window(an, an["start"], an["end"])
    return {"clip": name, "frames": an["n"], "act": meta["act"], "emotion": meta["emotion"],
            "behavior": meta["behavior"], "summary": whole, "parts": parts}


def main():
    names = sys.argv[1:] or sorted(p.stem.replace("_parts", "") for p in data_dir("talk_bank").glob("*_parts.json")
                                   if clip_path(p.stem.replace("_parts", "")).exists())
    idx = {}
    for n in names:
        idx[n] = index_clip(n)
        s = idx[n]["summary"]
        print(f"{n:20s} hand {s['hand_motion']:.2f} (peak {s['hand_peak']:.1f}) nods {len(s['nods'])} brows "
              f"{len(s['brow_raises'])} head {s['head_range']} mouth_p95 {s['mouth_open_p95']} blinks {s['blinks']}")
    p = data_dir() / "talk_index.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(idx)
    p.write_text(json.dumps(old, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
