"""Seam quality on a played timeline (trace from run_silent_demo / the live engine).

For every consecutive output-frame pair we measure the step in head pose, face position, non-mouth landmarks,
expression, gaze, hands and pixels (mouth masked; blended frames are blended in thumbnail space too). Steps that cross
a segment change are compared with the distribution of natural steps inside segments: a seam is "visible" when its
step exceeds the natural p99 of a metric. Identity and background consistency are checked over the whole timeline.
"""
import json

import numpy as np

from avatar_system.motion.clip_analyzer import analyze, thumb_mask
from avatar_system.motion.feature_extractor import FI
from avatar_system.motion.transition_scorer import STABLE_LMK

METRICS = ["head_deg", "face_px", "landmark_px", "expression", "gaze", "hand_px", "visual"]


def _frame_feats(an, idx):
    f = an["feats"][idx]
    lm = an["lmk"][idx].astype(np.float32)[STABLE_LMK] * [1024, 928]
    return f, lm, an["thumb"][idx].astype(np.float32)


def timeline_steps(trace, cache=None):
    cache = cache if cache is not None else {}
    tm = thumb_mask()
    hand = (int(87 * .80), 87, int(96 * .25), int(96 * .78))
    rows, prev = [], None
    for n, seg, clip, idx, bclip, bidx, alpha, seam, state in trace["frames"]:
        if clip not in cache:
            cache[clip] = analyze(clip)
        f, lm, th = _frame_feats(cache[clip], idx)
        if bclip:
            if bclip not in cache:
                cache[bclip] = analyze(bclip)
            fb, lmb, thb = _frame_feats(cache[bclip], bidx)
            f = fb * (1 - alpha) + f * alpha
            lm = lmb * (1 - alpha) + lm * alpha
            th = thb * (1 - alpha) + th * alpha
        cur = (f, lm, th)
        if prev is not None:
            pf, plm, pth = prev
            rows.append({
                "n": n, "seam": bool(seam), "blend": bool(bclip), "state": state, "seg": seg,
                "head_deg": float(np.abs(f[[FI["yaw"], FI["pitch"], FI["roll"]]] - pf[[FI["yaw"], FI["pitch"], FI["roll"]]]).max()),
                "face_px": float(np.hypot(f[FI["face_x"]] - pf[FI["face_x"]], f[FI["face_y"]] - pf[FI["face_y"]]) * 1024),
                "landmark_px": float(np.linalg.norm(lm - plm, axis=1).mean()),
                "expression": float(abs(f[FI["smile"]] - pf[FI["smile"]]) + abs(f[FI["brow_up"]] - pf[FI["brow_up"]])),
                "gaze": float(np.hypot(f[FI["gaze_x"]] - pf[FI["gaze_x"]], f[FI["gaze_y"]] - pf[FI["gaze_y"]])),
                "hand_px": float(np.abs(th[hand[0]:hand[1], hand[2]:hand[3]] - pth[hand[0]:hand[1], hand[2]:hand[3]]).mean()),
                "visual": float((np.abs(th - pth) * tm).sum() / tm.sum()),
                "identity": None,
            })
        prev = cur
    return rows


def seam_report(trace, cache=None, worst=10):
    rows = timeline_steps(trace, cache)
    # a "seam window" = the cut frame plus the blend frames that follow it
    nat = [r for r in rows if not r["seam"] and not r["blend"]]
    sea = [r for r in rows if r["seam"] or r["blend"]]
    p99 = {m: float(np.percentile([r[m] for r in nat], 99)) for m in METRICS}
    p50 = {m: float(np.percentile([r[m] for r in nat], 50)) for m in METRICS}
    cuts = [r for r in rows if r["seam"]]
    nat_cuts = {s[0] for s in trace["seams"] if s[6]}
    hard = [r for r in cuts if r["n"] not in nat_cuts]
    for r in sea:
        r["excess"] = max(r[m] / max(p99[m], 1e-6) for m in METRICS)
    visible = [r for r in sea if r["excess"] > 1.0]
    vis_seams = {r["n"] if r["seam"] else None for r in visible}
    worst_rows = sorted(sea, key=lambda r: -r["excess"])[:worst]
    rep = {
        "frames": len(rows) + 1, "segment_changes": len(cuts), "graph_cuts": len(hard),
        "natural_continuations": len(cuts) - len(hard), "blended_frames": sum(1 for r in rows if r["blend"]),
        "natural_step_p50": p50, "natural_step_p99": p99,
        "seam_step_mean": {m: float(np.mean([r[m] for r in sea])) if sea else 0 for m in METRICS},
        "seam_step_max": {m: float(np.max([r[m] for r in sea])) if sea else 0 for m in METRICS},
        "seam_frames_above_natural_p99": len(visible), "seam_frames": len(sea),
        "pct_seam_frames_above_p99": round(100 * len(visible) / max(len(sea), 1), 1),
        "worst": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in worst_rows],
    }
    return rep


def identity_background(trace, cache):
    ids, bgs, lum = [], [], []
    for clip in {f[2] for f in trace["frames"]}:
        an = cache[clip]
        if len(an["ident"]):
            ids.append(an["ident"].astype(np.float32).mean(0))
        bgs.append(an["feats"][:, FI["bg_luma"]].mean())
        lum.append(an["feats"][:, FI["luma"]].mean())
    ids = np.stack([i / np.linalg.norm(i) for i in ids])
    ref = ids.mean(0)
    ref /= np.linalg.norm(ref)
    return {"identity_cos_min": round(float((ids @ ref).min()), 3), "identity_cos_mean": round(float((ids @ ref).mean()), 3),
            "bg_luma_range": round(float(np.ptp(bgs)), 4), "luma_range": round(float(np.ptp(lum)), 4)}


if __name__ == "__main__":
    import sys
    tr = json.load(open(sys.argv[1]))
    cache = {}
    r = seam_report(tr, cache)
    r.update(identity_background(tr, cache))
    print(json.dumps({k: v for k, v in r.items() if k != "worst"}, indent=1))
