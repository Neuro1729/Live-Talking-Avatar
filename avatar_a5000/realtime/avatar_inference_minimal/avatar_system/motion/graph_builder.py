"""Build the pose-anchored motion graph.

1. Canonical states: the home pose that every LTX clip starts and ends on is `neutral` (the hub). Other states are
   discovered by clustering rest-point poses (low-velocity frames) that are far from neutral. A state is a physical
   pose (head angles, face position/scale, gaze, expression, hands), not an emotion label.
2. Segments (graph edges): every clip is cut at state-assigned rest points into 1.5-5 s segments (DP for lengths).
3. Transitions: segment A -> segment B is scored when A's end pose is physically compatible with B's start pose
   (transition_scorer). approved / review / rejected by cost thresholds and hard rejects. Manual decisions from
   data/reviews.json always win, and they are keyed by the cut frames, so a rejected join stays rejected after rebuilds.
4. Output: data/motion_graph.json (runtime artifact, compact indices) + data/library.sqlite (metadata/queries).

  envs/analysis/bin/python -m avatar_system.motion.graph_builder
"""
import hashlib
import re
import json
import sqlite3
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from avatar_system.common import clip_path, data_dir, file_sha1, library_clips, load_config, path
from avatar_system.motion.clip_analyzer import POSE_W, analyze, describe, pose_dist
from avatar_system.motion.feature_extractor import FI, is_current
from avatar_system.motion.transition_scorer import score

SEMANTICS = {  # which conversation behaviours a state suits (planner uses these)
    "neutral": ["idle", "listen", "think", "speak"],
    "nod_down": ["listen", "speak"], "chin_up": ["think", "listen"], "tilt": ["think", "speak"], "turn": ["think", "speak"],
    "lean": ["speak", "listen"], "positive": ["speak", "listen"], "emphasis": ["speak"],
}


def load_reviews():
    p = data_dir() / "reviews.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    for k in ("transitions", "segments", "clips"):
        r.setdefault(k, {})
    return r


def seg_key(clip, s, e):
    return f"{clip}@{s}-{e}"


def trans_key(A, B):
    return f"{A['clip']}@{A['e']}>{B['clip']}@{B['s']}"


def _name_state(delta):
    """delta: raw feature differences centroid - neutral -> descriptive state name."""
    cand = {"nod_down": delta["pitch"] / 4, "chin_up": -delta["pitch"] / 4, "tilt": abs(delta["roll"]) / 3,
            "turn": abs(delta["yaw"]) / 3, "positive": delta["smile"] / 0.12,
            "emphasis": delta["wrist"] / 0.05, "lean": abs(delta["face_s"]) / 0.004}
    return max(cand, key=cand.get)


def discover_states(ans):
    cfg = load_config("motion_graph")["states"]
    ends = np.stack([an["pose"][i] for an in ans.values() for i in (an["start"], an["end"] - 1)])
    ends_f = np.stack([an["feats"][i] for an in ans.values() for i in (an["start"], an["end"] - 1)])
    states = {"neutral": {"centroid": ends.mean(0), "feats": ends_f.mean(0), "origin": "home image (clip ends)",
                          "members": [f"{n}@{i}" for n, an in ans.items() for i in (an["start"], an["end"] - 1)]}}
    rest = [(n, r) for n, an in ans.items() for r in an["rest"]]
    far = [(n, r) for n, r in rest if pose_dist(ans[n]["pose"][r], states["neutral"]["centroid"]) > cfg["assign_max"]]
    if len(far) >= 2:
        X = np.stack([ans[n]["pose"][r] for n, r in far])
        norm = np.sqrt(X.shape[1] * np.mean(POSE_W ** 2))
        lab = fcluster(linkage(pdist(X) / norm, "average"), cfg["cluster_dist"], "distance")
        named = {}
        for c in sorted(set(lab)):
            mem = [far[i] for i in np.flatnonzero(lab == c)]
            if len(mem) < cfg["min_members"] or len({m[0] for m in mem}) < cfg["min_clips"]:
                continue
            F = np.stack([ans[n]["feats"][r] for n, r in mem]).mean(0)
            nf = states["neutral"]["feats"]
            delta = {k: F[FI[k]] - nf[FI[k]] for k in ("pitch", "roll", "yaw", "smile", "face_s", "face_y")}
            delta["wrist"] = np.abs(F[[FI["lwr_x"], FI["lwr_y"], FI["rwr_x"], FI["rwr_y"]]]
                                    - nf[[FI["lwr_x"], FI["lwr_y"], FI["rwr_x"], FI["rwr_y"]]]).max()
            base = _name_state(delta)
            named[base] = named.get(base, 0) + 1
            name = base if named[base] == 1 else f"{base}_{named[base]}"
            states[name] = {"centroid": np.stack([ans[n]["pose"][r] for n, r in mem]).mean(0), "feats": F,
                            "members": [f"{n}@{r}" for n, r in mem], "origin": "rest-point cluster",
                            "delta": {k: round(float(v), 3) for k, v in delta.items()}}
    return states


def assign_state(states, pose):
    cfg = load_config("motion_graph")["states"]
    d = {k: float(pose_dist(pose, v["centroid"])) for k, v in states.items()}
    best = min(d, key=d.get)
    return (best, d[best]) if d[best] <= cfg["assign_max"] else (None, d[best])


def segment_clip(an, states):
    """DP over state-assigned cut candidates -> list of (s, e_inclusive, start_state, end_state)."""
    cfg = load_config("motion_graph")["segments"]
    s0, e0 = an["start"], an["end"] - 1
    cands = [(s0, assign_state(states, an["pose"][s0])[0] or "neutral")]
    for r in an["rest"]:
        st = assign_state(states, an["pose"][r])[0]
        if st and s0 + cfg["min_len"] // 2 < r < e0 - cfg["min_len"] // 2:
            cands.append((r, st))
    cands.append((e0, assign_state(states, an["pose"][e0])[0] or "neutral"))

    def pen(L):
        p = abs(L - cfg["target_len"]) / cfg["target_len"]
        if L < cfg["min_len"]:
            p += 10 + (cfg["min_len"] - L) / 10
        if L > cfg["max_len"]:
            p += 2 * (L - cfg["max_len"]) / cfg["target_len"]
        return p

    best = [(0.0, None)] + [(np.inf, None)] * (len(cands) - 1)
    for k in range(1, len(cands)):
        for j in range(k):
            L = cands[k][0] - cands[j][0] + (1 if j == 0 else 0)
            c = best[j][0] + pen(L)
            if c < best[k][0]:
                best[k] = (c, j)
    cuts, k = [], len(cands) - 1
    while k is not None:
        cuts.append(k)
        k = best[k][1]
    cuts = cuts[::-1]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        s = cands[a][0] if a == 0 else cands[a][0] + 1
        segs.append((s, cands[b][0], cands[a][1], cands[b][1]))
    return segs


def conversational_tags(desc, calm, active):
    tags = []
    clean = desc["gaze_up_frac"] < 0.1 and desc["closed_frac"] < 0.15
    if clean and desc["intensity"] <= calm and not desc["gesture"]:
        tags += ["idle", "listen"]
    if clean and desc["intensity"] <= active and not desc["gesture"] and desc["nods"] <= 1:
        tags.append("think")
    if desc["gaze_up_frac"] < 0.15 and desc["closed_frac"] < 0.2:
        tags.append("speak")
    if desc["gesture"] or (desc["nods"] >= 1 and desc["intensity"] > calm):
        tags.append("emphasis")
    if desc["nods"] >= 1 and desc["intensity"] <= active and not desc["gesture"]:
        tags.append("acknowledge")
    return tags


TALK_MAX_HEAD = 20.0  # head range allowed in talking-bank segments (speech moves the head more than listening)


def speech_frames(talk, name, n):
    """bool[n]: frames of a talking-bank clip inside a recorded spoken part (empty for silent clips)."""
    m = np.zeros(n, bool)
    for p in talk.get(name, {}).get("parts", []):
        if p["type"] == "speech":
            m[p["s"]:p["e"]] = True
    return m


def build(verbose=True):
    t0 = time.time()
    cfg = load_config("motion_graph")
    reviews = load_reviews()
    tpath = data_dir() / "talk_index.json"
    talk = json.loads(tpath.read_text()) if tpath.exists() else {}
    names = [n for n in library_clips() if is_current(n)]
    missing = [n for n in library_clips() if n not in names]
    if missing and verbose:
        print(f"[graph] {len(missing)} clips have no current features (run feature_extractor): {missing}")
    names = [n for n in names if not reviews["clips"].get(n, {}).get("low_quality")]
    ans = {n: analyze(n) for n in names}
    states = discover_states(ans)

    segs = []
    for n in names:
        for s, e, ss, es in segment_clip(ans[n], states):
            k = seg_key(n, s, e)
            rv = reviews["segments"].get(k, {})
            desc = describe(ans[n], s, e + 1)
            qg = cfg["quality_gate"]
            sp = speech_frames(talk, n, ans[n]["n"])[s:e + 1]
            max_head = TALK_MAX_HEAD if n in talk else qg["max_head_range_deg"]
            gate = [r for r, bad in (("head range", max(desc["head_range_deg"]) > max_head),
                                     ("eyes closed", desc["closed_frac"] > qg["max_closed_frac"]),
                                     ("gaze up", desc["gaze_up_frac"] > qg["max_gaze_up_frac"])) if bad]
            if rv.get("approve"):  # manual override of the automatic gate
                gate = []
            acts = sorted({p["act"] for p in talk.get(n, {}).get("parts", [])
                           if p["type"] == "speech" and p["s"] <= e and p["e"] > s})
            segs.append({"key": k, "clip": n, "s": int(s), "e": int(e), "start_state": ss, "end_state": es,
                         "desc": desc, "review": rv, "gate": gate, "talk": bool(sp.any()),
                         "speech_frac": round(float(sp.mean()), 3), "acts": acts,
                         "speed": 1.0 if n in talk else None,  # talking takes are real-time; silent LTX clips use playback_speed
                         "excluded": bool(rv.get("low_quality")) or bool(gate)})
    ints = np.array([g["desc"]["intensity"] for g in segs if not g["talk"]])  # calm/active as for the silent graph
    calm, active = np.percentile(ints, 45), np.percentile(ints, 85)
    for g in segs:
        g["conv"] = g["review"].get("labels") or conversational_tags(g["desc"], calm, active)
        if g["talk"]:  # contains recorded speech: only while the avatar speaks (its recorded mouth talks)
            g["conv"] = ["talk", "speak"] + [t for t in g["conv"] if t in ("emphasis",)]
        d = g["desc"]
        g["quality"] = round(max(0.0, 1 - 2 * d["gaze_up_frac"] - 2 * max(d["closed_frac"] - 0.08, 0)
                                 - 0.15 * max(d["intensity"] - active, 0)), 3)
        g["loopable"] = g["start_state"] == g["end_state"]
        g["speaking_ok"] = "speak" in g["conv"]
    active_segs = [g for g in segs if not g["excluded"]]
    for i, g in enumerate(active_segs):
        g["id"] = i

    T = cfg["transition"]
    trans = []
    for A in active_segs:
        pa = ans[A["clip"]]["pose"][A["e"]]
        for B in active_segs:
            if B is A:
                continue
            natural = B["clip"] == A["clip"] and B["s"] == A["e"] + 1
            compatible = A["end_state"] == B["start_state"] or \
                pose_dist(pa, ans[B["clip"]]["pose"][B["s"]]) <= cfg["states"]["assign_max"]
            if not (natural or compatible):
                continue
            cost, comp, reasons = score(ans[A["clip"]], A["e"], ans[B["clip"]], B["s"])
            if natural:
                status = "natural"
            elif reasons:
                status = "rejected"
            elif cost <= T["accept_cost"] and all(comp[k] <= v for k, v in T["component_caps"].items()):
                status = "approved"
            elif cost <= T["review_cost"]:
                status = "review"
            else:
                status = "rejected"
            k = trans_key(A, B)
            man = reviews["transitions"].get(k)
            if man:
                status = {"approve": "approved", "reject": "rejected"}.get(man.get("decision"), status)
                cost = float(man.get("cost", cost))
            trans.append({"src": A["id"], "dst": B["id"], "key": k, "cost": round(cost, 4), "status": status,
                          "manual": bool(man), "reasons": reasons,
                          "comp": {kk: round(v, 3) for kk, v in comp.items()},
                          "blend": comp["visual"] <= T["seam_blend_max"] and not natural})
    # early exits: from rest points inside a segment to calm segments (taken only when the mode turns calm)
    E = cfg["exits"]
    exits = {g["id"]: [] for g in active_segs}
    calm = [B for B in active_segs if any(t in B["conv"] for t in E["to_tags"])]
    for A in active_segs:
        an = ans[A["clip"]]
        for r in an["rest"]:
            if not (A["s"] + E["min_margin"] < r < A["e"] - E["min_margin"]):
                continue
            pa = an["pose"][r]
            for B in calm:
                if B is A or pose_dist(pa, ans[B["clip"]]["pose"][B["s"]]) > cfg["states"]["assign_max"]:
                    continue
                cost, comp, reasons = score(an, r, ans[B["clip"]], B["s"])
                if not reasons and cost <= T["accept_cost"] and all(comp[k] <= v for k, v in T["component_caps"].items()):
                    exits[A["id"]].append([int(r), B["id"], round(cost, 4), int(comp["visual"] <= T["seam_blend_max"])])
    usable = {"approved", "natural"}
    adj = {g["id"]: [] for g in active_segs}
    for t in trans:
        if t["status"] in usable:
            adj[t["src"]].append([t["dst"], t["cost"], int(t["blend"])])
    orphan = [k for k in reviews["transitions"] if k not in {t["key"] for t in trans}]

    graph = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "fps": load_config("avatar")["video"]["fps"],
        "hub": cfg["graph"]["hub_state"],
        "states": {k: {"members": len(v["members"]), "origin": v["origin"], "semantics": SEMANTICS.get(re.sub(r"_\d+$", "", k), ["speak"]),
                       "delta": v.get("delta", {}), "centroid": [round(float(x), 4) for x in v["centroid"]]}
                   for k, v in states.items()},
        "segments": [{k: g[k] for k in ("id", "key", "clip", "s", "e", "start_state", "end_state", "conv", "quality",
                                        "loopable", "speaking_ok", "desc", "talk", "speech_frac", "acts", "speed")}
                     for g in active_segs],
        "excluded_segments": [{"key": g["key"], "why": g["gate"] or ["manual low quality"]} for g in segs if g["excluded"]],
        "adjacency": {str(k): v for k, v in adj.items()},
        "exits": {str(k): v for k, v in exits.items() if v},
        "counts": {s: sum(1 for t in trans if t["status"] == s) for s in ("natural", "approved", "review", "rejected")},
        "orphan_reviews": orphan,
    }
    graph["version"] = hashlib.sha1(json.dumps([graph["segments"], graph["adjacency"]], sort_keys=True).encode()).hexdigest()[:12]
    out = data_dir() / "motion_graph.json"
    out.write_text(json.dumps(graph, indent=1))
    (data_dir() / "transitions.json").write_text(json.dumps(trans))
    write_sqlite(graph, trans, names, segs)
    if verbose:
        print(f"[graph] {len(names)} clips, {len(states)} states, {len(active_segs)} segments "
              f"({len(segs) - len(active_segs)} excluded), transitions {graph['counts']}, version {graph['version']}, "
              f"{time.time() - t0:.0f}s")
        for k, v in graph["states"].items():
            print(f"   state {k:12s} members {v['members']:3d} semantics {v['semantics']} delta {v['delta']}")
        if orphan:
            print(f"[graph] WARNING {len(orphan)} manual transition decisions no longer match any cut: {orphan[:5]}")
    return graph, trans


def write_sqlite(graph, trans, names, segs):
    p = data_dir() / "library.sqlite"
    db = sqlite3.connect(p)
    db.executescript("""
    DROP TABLE IF EXISTS clips; DROP TABLE IF EXISTS segments; DROP TABLE IF EXISTS transitions;
    DROP TABLE IF EXISTS states; DROP TABLE IF EXISTS manifest;
    CREATE TABLE clips (id TEXT PRIMARY KEY, path TEXT, sha1 TEXT, fps INT, n_frames INT, duration REAL,
                        musetalk_cache TEXT, has_cache INT, gen_params TEXT);
    CREATE TABLE segments (id INT PRIMARY KEY, key TEXT UNIQUE, clip TEXT, s INT, e INT, n_frames INT, duration REAL,
                           start_state TEXT, end_state TEXT, conv TEXT, intensity REAL, gaze_dir TEXT, gesture INT,
                           nods INT, blinks INT, loopable INT, speaking_ok INT, quality REAL, approved INT, descriptors TEXT);
    CREATE TABLE transitions (src INT, dst INT, key TEXT, cost REAL, status TEXT, manual INT, blend INT, reasons TEXT,
                              components TEXT);
    CREATE TABLE states (name TEXT PRIMARY KEY, members INT, origin TEXT, semantics TEXT, delta TEXT);
    CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT);
    """)
    fps = graph["fps"]
    cache = path("musetalk_cache")
    for n in names:
        meta = clip_path(n).with_suffix(".json")
        nfr = next(len(g["desc"]) and g for g in segs if g["clip"] == n) and max(g["e"] for g in segs if g["clip"] == n) + 1
        db.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?)",
                   (n, str(clip_path(n)), file_sha1(clip_path(n)), fps, nfr, nfr / fps, str(cache / n),
                    int((cache / n / "DONE").exists()), meta.read_text() if meta.exists() else None))
    for g in graph["segments"]:
        d = g["desc"]
        db.execute("INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (g["id"], g["key"], g["clip"], g["s"], g["e"], d["frames"], d["frames"] / fps, g["start_state"],
                    g["end_state"], ",".join(g["conv"]), d["intensity"], d["gaze_dir"], int(d["gesture"]), d["nods"],
                    d["blinks"], int(g["loopable"]), int(g["speaking_ok"]), g["quality"], 1, json.dumps(d)))
    db.executemany("INSERT INTO transitions VALUES (?,?,?,?,?,?,?,?,?)",
                   [(t["src"], t["dst"], t["key"], t["cost"], t["status"], int(t["manual"]), int(t["blend"]),
                     json.dumps(t["reasons"]), json.dumps(t["comp"])) for t in trans])
    for k, v in graph["states"].items():
        db.execute("INSERT INTO states VALUES (?,?,?,?,?)", (k, v["members"], v["origin"], ",".join(v["semantics"]),
                                                             json.dumps(v["delta"])))
    db.executemany("INSERT INTO manifest VALUES (?,?)", [("graph_version", graph["version"]), ("built", graph["built"])])
    db.commit()
    db.close()


if __name__ == "__main__":
    build()
