"""Repetition / diversity of a played timeline: unique clips and segments, selection distribution (entropy), repeated
2- and 3-segment sequences, salient gesture frequency (nods, gestures, tilts, gaze-up), blink-interval regularity,
and diversity in 1/5/10-minute windows."""
import json
import math
from collections import Counter

import numpy as np

from avatar_system.motion.repetition_tracker import patterns_of


def segment_sequence(trace):
    seq, last = [], None
    for f in trace["frames"]:
        if f[1] != last:
            seq.append((f[0], f[1]))
            last = f[1]
    return seq


def entropy(counter):
    tot = sum(counter.values())
    return -sum(c / tot * math.log2(c / tot) for c in counter.values()) if tot else 0.0


def report(trace, graph, cache=None):
    fps = trace["fps"]
    segs = graph["segments"]
    seq = segment_sequence(trace)
    ids = [s for _, s in seq]
    clips_frames = Counter(f[2] for f in trace["frames"])
    seg_c = Counter(ids)
    grams = {n: Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)) for n in (2, 3)}
    rep = {
        "minutes": round(len(trace["frames"]) / fps / 60, 2),
        "segment_plays": len(ids), "unique_segments": len(seg_c), "unique_clips": len(clips_frames),
        "segment_entropy_bits": round(entropy(seg_c), 2), "max_entropy_bits": round(math.log2(max(len(seg_c), 1)), 2),
        "top_segment_share": round(seg_c.most_common(1)[0][1] / len(ids), 3) if ids else 0,
        "repeated_2grams": sum(c - 1 for c in grams[2].values() if c > 1),
        "repeated_3grams": sum(c - 1 for c in grams[3].values() if c > 1),
        "max_2gram_count": max(grams[2].values(), default=0), "max_3gram_count": max(grams[3].values(), default=0),
    }
    # salient patterns per minute and minimum spacing
    pat_times = {}
    for t, s in seq:
        for p in patterns_of(segs[s]):
            pat_times.setdefault(p, []).append(t / fps)
    rep["patterns_per_min"] = {p: round(len(v) / rep["minutes"], 2) for p, v in pat_times.items()}
    rep["pattern_min_gap_s"] = {p: round(float(np.min(np.diff(v))), 1) if len(v) > 1 else None for p, v in pat_times.items()}
    # blink regularity: coefficient of variation of blink intervals (low CV = robotic periodic blinking)
    blinks = []
    for f in trace["frames"]:
        seg = segs[f[1]]
        if f[3] in set(seg["desc"]["blink_frames"]):
            blinks.append(f[0] / fps)
    iv = np.diff(blinks)
    rep["blinks_per_min"] = round(len(blinks) / rep["minutes"], 1)
    rep["blink_interval_cv"] = round(float(iv.std() / iv.mean()), 2) if len(iv) > 2 else None
    # windowed diversity
    for w in (1, 5, 10):
        n = w * 60 * fps
        if len(trace["frames"]) >= n:
            vals = []
            for st in range(0, len(trace["frames"]) - n + 1, max(n // 2, fps * 30)):
                win = trace["frames"][st:st + n]
                vals.append(len({f[1] for f in win}))
            rep[f"unique_segments_per_{w}min"] = round(float(np.mean(vals)), 1)
    # mode fit: fraction of frames whose segment suits the behaviour state
    want = {"LISTENING": "listen", "THINKING": "think", "SPEAKING": "speak", "INTERRUPTED": "listen", "RECOVERY": "idle"}
    fit = [want[f[8]] in segs[f[1]]["conv"] for f in trace["frames"]]
    rep["mode_fit_pct"] = round(100 * float(np.mean(fit)), 1)
    by_state = {}
    for f, ok in zip(trace["frames"], fit):
        by_state.setdefault(f[8], []).append(ok)
    rep["mode_fit_by_state_pct"] = {k: round(100 * float(np.mean(v)), 1) for k, v in by_state.items()}
    return rep


if __name__ == "__main__":
    import sys
    from avatar_system.common import data_dir
    tr = json.load(open(sys.argv[1]))
    g = json.loads((data_dir() / "motion_graph.json").read_text())
    print(json.dumps(report(tr, g), indent=1))
