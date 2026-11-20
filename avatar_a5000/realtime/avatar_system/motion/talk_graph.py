"""Talking-bank motion graph: the pose-anchored motion-graph method applied to the speech-driven LTX performances.

Nodes are cut points: frames inside the clips' silent pauses (every STRIDE-th frame), where the recorded mouth is closed
and the body is between gestures, plus the calmest (lowest-velocity) moments inside talking parts, so a sentence can
leave a recorded gesture between strokes instead of only after the whole recorded sentence. Playing a clip forward from one cut point to a later one is always allowed (the
recorded performance itself, at 1x); a jump between cut points of different clips (or far apart in one clip) is an
edge only if the transition scorer approves it (boundary pose + head/face/wrist velocity over a window + expression,
gaze, hands, visual, landmarks; the motion graph's hard rejects, but a stricter accept cost and component caps -
expression varies far more between talking takes, and joins at the old 0.9 showed smile/brow pops). The runtime
planner (TalkBankScheduler) searches this graph so speech is covered by recorded talking frames and silence by pause
frames, always forward at 1x, never through an unverified join.

  envs/musetalk/bin/python -m avatar_system.motion.talk_graph [--clips a,b,...]   -> data/talk_graph.json
"""
import argparse
import json
import time

import numpy as np

from avatar_system.common import data_dir, load_config, path
from avatar_system.motion.clip_analyzer import analyze, pose_dist
from avatar_system.motion.transition_scorer import score

ACCEPT = 0.6        # max join cost (motion graph: 0.9)
CAPS = {"visual": 0.9, "expression": 0.5, "landmark": 1.5, "pose": 0.6, "hand": 1.0}  # stricter than the motion graph
STRIDE = 2          # every 2nd pause frame is a cut point
KNN = 48            # exact-score only the KNN nearest cut points by pose (cheap prefilter)
MIN_GAP = 12        # same-clip jumps must skip at least this many frames (else it is just playback)
EDGE_MARGIN = 3     # cut points need this many frames to the clip ends (velocity window)
LEAD = 10           # a pause frame this close before a talking part counts as a way into speech
CALM_PCT = 30       # talking frames slower than this percentile of the clip's talking frames are cut points too


def frame_labels(meta, n, max_head):
    """Per frame: 0 = pause (silent, mouth closed), 1 = speech, 2 = unusable (speech part with too much head motion).
    Frames after the last indexed part (the settle to the home pose) are pause. Also returns the speech part id."""
    lab = np.zeros(n, np.int8)
    part = np.full(n, -1, np.int16)
    for p in meta["parts"]:
        if p["type"] == "speech":
            lab[p["s"]:p["e"]] = 1 if max(p["head_range"]) <= max_head else 2
            part[p["s"]:p["e"]] = p["part"]
    return lab, part


def build(names=None, max_head=20.0, verbose=True):
    t0 = time.time()
    cfg = dict(load_config("motion_graph")["transition"])
    cfg["accept_cost"] = ACCEPT
    cfg["component_caps"] = dict(cfg["component_caps"], **CAPS)
    idx = json.loads((data_dir() / "talk_index.json").read_text())
    ok = {n for n in idx if (path("musetalk_cache") / n / "DONE").exists()}
    names = sorted((set(names) if names else set(idx)) & ok)
    ans = {n: analyze(n) for n in names}
    clips, cuts = {}, []
    for n in names:
        L = min(ans[n]["n"], ans[n]["end"])
        lab, part = frame_labels(idx[n], L, max_head)
        clips[n] = {"n": int(L), "lab": lab.tolist(), "part": part.tolist()}
        sp = ans[n]["speed"][:L]
        talk = np.flatnonzero(lab == 1)
        calm = np.percentile(sp[talk], CALM_PCT) if len(talk) else -1
        for i in range(EDGE_MARGIN, L - 1, STRIDE):
            if lab[i] == 0 or (lab[i] == 1 and sp[i] <= calm):  # pause frames + the calmest moments of speech
                cuts.append((n, i))
        if lab[L - 1] == 0:
            cuts.append((n, L - 1))  # the clip's last frame (home pose)
    P = np.stack([ans[c]["pose"][i] for c, i in cuts])
    D = pose_dist(P[:, None, :], P[None, :, :])
    # targets that start talking soon (talking frames, or pause frames just before a talking part): pause frames
    # resemble each other most, so a plain pose-KNN would almost never offer a way into speech
    talky = np.array([clips[c]["lab"][i] == 1 or 1 in clips[c]["lab"][i:i + LEAD] for c, i in cuts])
    clip_of = np.array([names.index(c) for c, _ in cuts])
    fr = np.array([i for _, i in cuts])
    same_near = (clip_of[:, None] == clip_of[None, :]) & (np.abs(fr[:, None] - fr[None, :]) < MIN_GAP)
    D[same_near] = np.inf
    seen = set()
    edges, tried, status = [], 0, {"approved": 0, "review": 0, "rejected": 0}
    for a in range(len(cuts)):
        order = np.argsort(D[a])
        for b in np.r_[order[:KNN], order[talky[order]][:KNN]]:
            if not np.isfinite(D[a, b]):
                continue
            (ca, ia), (cb, ib) = cuts[a], cuts[b]
            if ib >= clips[cb]["n"] - EDGE_MARGIN and ib != clips[cb]["n"] - 1:
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            tried += 1
            c, comp, reasons = score(ans[ca], ia, ans[cb], ib)
            if reasons:
                status["rejected"] += 1
            elif c <= cfg["accept_cost"] and all(comp[k] <= v for k, v in cfg["component_caps"].items()):
                status["approved"] += 1
                edges.append([ca, int(ia), cb, int(ib), round(float(c), 3)])
            elif c <= cfg["review_cost"]:
                status["review"] += 1
            else:
                status["rejected"] += 1
    g = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "max_head": max_head, "clips": clips,
         "cuts": len(cuts), "edges": edges, "tried": tried, "status": status}
    (data_dir() / "talk_graph.json").write_text(json.dumps(g))
    if verbose:
        reach = {}
        for ca, ia, cb, ib, c in edges:
            reach.setdefault(ca, set()).add(cb)
        costs = np.array([e[4] for e in edges]) if edges else np.zeros(1)
        n_speech = sum(1 for n in names for p in idx[n]["parts"] if p["type"] == "speech" and max(p["head_range"]) <= max_head)
        print(f"[talk_graph] {len(names)} clips, {n_speech} usable speech parts, {len(cuts)} cut points, "
              f"{tried} joins scored -> {status}, approved cost p50 {np.median(costs):.2f} p90 "
              f"{np.percentile(costs, 90):.2f}, clips reachable per clip: median "
              f"{np.median([len(v) for v in reach.values()]) if reach else 0:.0f}/{len(names)}, {time.time() - t0:.0f}s",
              flush=True)
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="comma list (default: every indexed clip with a MuseTalk cache)")
    ap.add_argument("--max-head", type=float, default=20.0)
    a = ap.parse_args()
    build(a.clips.split(",") if a.clips else None, a.max_head)
