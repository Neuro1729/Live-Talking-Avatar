"""Validate the motion graph before it is ever played: dead ends, reachability, a safe path back to the neutral hub,
alternatives per state, and whether every behaviour mode (idle/listen/think/speak) can continue forever on its own
segments. Writes data/validation.json and stores the runtime-safe segment set (the strongly connected component that
contains the hub) in motion_graph.json, so live playback can never walk into a dead end.

  envs/analysis/bin/python -m avatar_system.motion.graph_validator
"""
import json
import sys
from collections import deque

from avatar_system.common import data_dir, load_config

MODES = ("idle", "listen", "think", "speak")


def sccs(nodes, adj):
    """Tarjan (iterative). adj: node -> iterable of successors."""
    index, low, on, stack, out, idx = {}, {}, set(), [], [], 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = idx
        idx += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            pushed = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = idx
                    idx += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    pushed = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if pushed:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = set()
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.add(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def reachable(starts, adj):
    seen, q = set(starts), deque(starts)
    while q:
        v = q.popleft()
        for w in adj.get(v, ()):
            if w not in seen:
                seen.add(w)
                q.append(w)
    return seen


def frames_to_hub(segs, adj, hub):
    """Shortest playback time (frames) from the start of each segment until a hub-ending segment has finished."""
    import heapq
    rev = {}
    for v, ws in adj.items():
        for w in ws:
            rev.setdefault(w, []).append(v)
    dist = {s["id"]: s["desc"]["frames"] for s in segs if s["end_state"] == hub}
    pq = [(d, v) for v, d in dist.items()]
    heapq.heapify(pq)
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist.get(v, 1e18):
            continue
        for u in rev.get(v, ()):
            nd = d + segs[u]["desc"]["frames"]
            if nd < dist.get(u, 1e18):
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    return dist


def validate(graph=None, write=True, verbose=True):
    p = data_dir() / "motion_graph.json"
    graph = graph or json.loads(p.read_text())
    cfg = load_config("motion_graph")["graph"]
    hub = graph["hub"]
    segs = graph["segments"]
    ids = [s["id"] for s in segs]
    adj = {int(k): [d for d, *_ in v] for k, v in graph["adjacency"].items()}
    errors, warnings = [], []

    dead = [s["key"] for s in segs if not adj.get(s["id"])]
    hub_starts = [s["id"] for s in segs if s["start_state"] == hub]
    if not hub_starts:
        errors.append("no segment starts at the hub state")
    reach = reachable(hub_starts, adj)
    unreachable = [segs[i]["key"] for i in ids if i not in reach]
    to_hub = frames_to_hub(segs, adj, hub)
    no_return = [segs[i]["key"] for i in ids if i not in to_hub]

    comps = sccs(ids, adj)
    main = max((c for c in comps if c & set(hub_starts)), key=len, default=set())
    outside = [segs[i]["key"] for i in ids if i not in main]
    if not main:
        errors.append("hub is not part of any cycle: playback cannot continue indefinitely")

    states = {}
    for s in segs:
        if s["id"] in main:
            st = states.setdefault(s["start_state"], {"out_segments": 0, "out_alt_clips": set(), "in_segments": 0})
            st["out_segments"] += 1
            st["out_alt_clips"].add(s["clip"])
            states.setdefault(s["end_state"], {"out_segments": 0, "out_alt_clips": set(), "in_segments": 0})["in_segments"] += 1
    for k, v in states.items():
        v["out_alt_clips"] = len(v["out_alt_clips"])
        if v["out_segments"] < cfg["min_alternatives"]:
            warnings.append(f"state {k}: only {v['out_segments']} outgoing segments")
    for k in graph["states"]:
        if k not in states:
            warnings.append(f"state {k}: no runtime-safe segment touches it")

    modes = {}
    for m in MODES:
        sub = {i for i in main if m in segs[i]["conv"]}
        sadj = {i: [w for w in adj.get(i, ()) if w in sub] for i in sub}
        mc = [c for c in sccs(sorted(sub), sadj) if len(c) > 1]
        big = max(mc, key=len, default=set())
        modes[m] = {"segments": len(sub), "largest_cycle_set": len(big),
                    "seconds_in_cycle_set": round(sum(segs[i]["desc"]["frames"] for i in big) / graph["fps"], 1),
                    "distinct_clips": len({segs[i]["clip"] for i in big})}
        if not big:
            (errors if m in ("idle", "listen") else warnings).append(f"mode {m}: cannot continue indefinitely on its own segments")
        elif modes[m]["distinct_clips"] < 3:
            warnings.append(f"mode {m}: only {modes[m]['distinct_clips']} distinct clips in its cycle set (repetitive)")

    speak_ret = [to_hub.get(i, 0) for i in main if "speak" in segs[i]["conv"]]
    n_edges = sum(len(adj.get(i, ())) for i in main)
    rep = {
        "graph_version": graph["version"], "hub": hub, "segments": len(segs), "runtime_segments": len(main),
        "runtime_edges": sum(1 for i in main for w in adj.get(i, ()) if w in main),
        "avg_out_degree": round(n_edges / max(len(main), 1), 1),
        "dead_ends": dead, "unreachable_from_hub": unreachable, "no_path_to_hub": no_return,
        "outside_main_scc": outside, "states": states, "modes": modes,
        "max_frames_back_to_hub_from_speaking": max(speak_ret, default=0),
        "errors": errors, "warnings": warnings, "ok": not errors,
    }
    if write:
        (data_dir() / "validation.json").write_text(json.dumps(rep, indent=1))
        graph["runtime_segments"] = sorted(main)
        graph["validation_ok"] = not errors
        p.write_text(json.dumps(graph, indent=1))
    if verbose:
        print(f"[validate] runtime {rep['runtime_segments']}/{rep['segments']} segments, {rep['runtime_edges']} edges, "
              f"avg out-degree {rep['avg_out_degree']}; dead ends {len(dead)}, unreachable {len(unreachable)}, "
              f"no path to hub {len(no_return)}; back-to-hub worst {rep['max_frames_back_to_hub_from_speaking'] / graph['fps']:.1f}s")
        for m, v in modes.items():
            print(f"   mode {m:6s} {v}")
        for k, v in states.items():
            print(f"   state {k:10s} {v}")
        for w in warnings:
            print("   WARNING", w)
        for e in errors:
            print("   ERROR", e)
    return rep


if __name__ == "__main__":
    sys.exit(0 if validate()["ok"] else 1)
