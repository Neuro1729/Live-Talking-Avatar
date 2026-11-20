"""Deterministic, cheap runtime motion planner: constrained beam search over the validated motion graph.

At each decision (the current segment is about to end) it expands the approved successors 2-4 segments deep and scores
every path on continuity (transition cost), semantic fit to the behaviour mode and hints, prosody fit (upcoming
speech energy/pauses), intensity, repetition, and whether the path ends somewhere that can get back to the neutral hub
quickly. The first step is then sampled among the best candidates with a temperature (controlled variation); a
candidate whose transition cost is clearly worse than the best is never sampled (continuity first).
"""
import math
import random
from dataclasses import dataclass, field

from avatar_system.common import load_config
from avatar_system.motion.graph_validator import frames_to_hub
from avatar_system.motion.repetition_tracker import RepetitionTracker

MODE_TAGS = {  # behaviour state -> segment tags it prefers (first = best)
    "LISTENING": ["listen", "idle"],
    "THINKING": ["think", "listen"],
    "SPEAKING": ["talk", "speak"],  # talk = recorded speech-driven takes (talking bank), preferred while speaking
    "INTERRUPTED": ["listen", "idle"],
    "RECOVERY": ["idle", "listen"],
}
HINT_TAGS = {  # behaviour hints from the LLM / heuristics -> segment tags or states that express them
    "gentle_nod": {"tags": ["acknowledge"]}, "acknowledge": {"tags": ["acknowledge"]},
    "emphasis": {"tags": ["emphasis"]}, "neutral_explanation": {"tags": ["speak"], "calm": True},
    "thoughtful": {"states": ["tilt", "tilt_2", "chin_up"]}, "positive": {"smile": True},
    "empathetic": {"tags": ["acknowledge"], "calm": True}, "reassuring": {"tags": ["acknowledge"], "calm": True},
}


LISTEN_EXPRESSION_BONUS = 0.35


@dataclass
class Context:
    mode: str = "LISTENING"
    hints: list = field(default_factory=list)       # e.g. ["gentle_nod"]
    intensity: float = 0.3                           # 0 calm .. 1 expressive (target)
    emphasis_at: list = field(default_factory=list)  # upcoming emphasis times, frames from now
    pause_frac: float = 0.0                          # fraction of the next ~3 s that is silence


class MotionPlanner:
    def __init__(self, graph, seed=0, cfg=None):
        self.cfg = cfg or load_config("motion_planner")
        self.g = graph
        self.segs = graph["segments"]
        run = set(graph.get("runtime_segments") or [s["id"] for s in self.segs])
        self.run = run
        self.adj = {int(k): [(d, c, b) for d, c, b in v if d in run] for k, v in graph["adjacency"].items() if int(k) in run}
        self.hub = graph["hub"]
        adj_ids = {k: [d for d, *_ in v] for k, v in self.adj.items()}
        self.to_hub = frames_to_hub(self.segs, adj_ids, self.hub)
        ints = sorted(self.segs[i]["desc"]["intensity"] for i in run)
        self.int_lo, self.int_hi = ints[len(ints) // 10], ints[len(ints) * 9 // 10]
        self.rng = random.Random(seed)
        self.rep = RepetitionTracker(self.segs)
        self.last_decision = {}

    # ---- scoring ----
    def _norm_int(self, seg):
        x = (seg["desc"]["intensity"] - self.int_lo) / max(self.int_hi - self.int_lo, 1e-6)
        return min(max(x, 0.0), 1.0)

    def _semantic(self, seg, ctx):
        tags = MODE_TAGS[ctx.mode]
        if tags[0] in seg["conv"]:
            pen = 0.0
        elif any(t in seg["conv"] for t in tags[1:]):
            pen = 0.6
        else:
            pen = 3.0 if ctx.mode in ("LISTENING", "RECOVERY", "INTERRUPTED") else 1.8
        if seg.get("talk") and ctx.mode != "SPEAKING":
            pen += 50.0  # its recorded mouth talks: never while silent
        if ctx.mode == "LISTENING" and not seg.get("talk") and "acknowledge" in seg["conv"]:
            pen -= LISTEN_EXPRESSION_BONUS  # small nods / expressions now and then while listening, not only stillness
        for h in ctx.hints:
            spec = HINT_TAGS.get(h, {})
            if any(t in seg["conv"] for t in spec.get("tags", [])) or \
                    seg["end_state"] in spec.get("states", []) or seg["start_state"] in spec.get("states", []):
                pen -= 0.5
            if spec.get("smile") and seg["desc"]["smile"] > 0.55:
                pen -= 0.3
            if spec.get("calm") and self._norm_int(seg) > 0.6:
                pen += 0.4
        return pen

    def _prosody(self, seg, ctx, offset):
        """Gestures/nods should coincide with emphasis inside this segment's time window; calm during pauses."""
        busy = bool(seg["desc"]["nods"] or seg["desc"]["gesture"])
        n = seg["desc"]["frames"]
        hits = [e for e in ctx.emphasis_at if offset <= e < offset + n]
        pen = 0.0
        if busy and ctx.mode == "SPEAKING":
            pen += -0.4 if hits else 0.5
        if ctx.pause_frac > 0.4 and self._norm_int(seg) > 0.5:
            pen += 0.4
        return pen

    def _path_score(self, path, costs, ctx):
        w = self.cfg["weights"]
        s, off, disc = 0.0, 0, 1.0
        for sid, c in zip(path, costs):
            seg = self.segs[sid]
            s += disc * (w["continuity"] * c + w["semantic"] * self._semantic(seg, ctx)
                         + w["prosody"] * self._prosody(seg, ctx, off)
                         + w["intensity"] * abs(self._norm_int(seg) - ctx.intensity)
                         + w["quality"] * (1 - seg["quality"]))
            off += seg["desc"]["frames"]
            disc *= self.cfg["discount"]
        s += w["repetition"] * self.rep.penalty(path, self.t)
        back = self.to_hub.get(path[-1], 10 ** 6) / self.g["fps"]
        lim = self.cfg["recovery_seconds"][ctx.mode] if isinstance(self.cfg["recovery_seconds"], dict) else 6
        s += w["return"] * max(back - lim, 0) / lim
        return s

    def choose(self, cur, ctx, t):
        """Pick the segment to play after `cur` (seg id). Returns (seg_id, transition_cost, blend, info)."""
        self.t = t
        depth, beam = self.cfg["lookahead"], self.cfg["beam_width"]
        if ctx.mode in ("INTERRUPTED", "RECOVERY"):
            depth = min(depth, 2)
        paths = [([d], [c], b) for d, c, b in self.adj.get(cur, [])]
        if not paths:
            raise RuntimeError(f"segment {cur} has no successors (graph validation should have prevented this)")
        # continuity first: drop joins clearly worse than the best one, but never below the "invisible" floor
        # (otherwise the near-zero natural continuation always wins and whole clips replay as fixed blocks)
        best_cost = max(min(c[0] for _, c, _ in paths), self.cfg["cost_floor"])
        paths = [p for p in paths if p[1][0] <= best_cost + self.cfg["max_cost_over_best"]]
        frontier = [(self._path_score(p, c, ctx), p, c, b) for p, c, b in paths]
        firsts = {}
        for _ in range(depth - 1):
            frontier.sort(key=lambda x: x[0])
            nxt = []
            for sc, p, c, b in frontier[:beam]:
                for d, cc, _ in self.adj.get(p[-1], [])[: self.cfg["max_branch"]]:
                    nxt.append((self._path_score(p + [d], c + [cc], ctx), p + [d], c + [cc], b))
            if not nxt:
                break
            frontier = nxt
        for sc, p, c, b in frontier:
            if p[0] not in firsts or sc < firsts[p[0]][0]:
                firsts[p[0]] = (sc, p, c, b)
        cands = sorted(firsts.values(), key=lambda x: x[0])[: self.cfg["sample_top_k"]]
        T = self.cfg["temperature"]
        base = cands[0][0]
        ws = [math.exp(-(sc - base) / max(T, 1e-6)) for sc, *_ in cands]
        sc, p, c, b = self.rng.choices(cands, weights=ws)[0]
        self.last_decision = {"t": t, "mode": ctx.mode, "chosen": p[0], "plan": p, "score": round(sc, 3),
                              "candidates": [(x[1][0], round(x[0], 3)) for x in cands]}
        return p[0], c[0], bool(b), self.last_decision

    def start_segment(self, ctx):
        """First segment of a session: a hub-starting segment that suits the mode."""
        opts = [i for i in self.run if self.segs[i]["start_state"] == self.hub and self.segs[i]["s"] <= 10]
        opts.sort(key=lambda i: self._semantic(self.segs[i], ctx) + abs(self._norm_int(self.segs[i]) - ctx.intensity))
        return self.rng.choice(opts[:5])
