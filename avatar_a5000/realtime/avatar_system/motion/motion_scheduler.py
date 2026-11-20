"""Source-frame scheduler: turns planner decisions into exactly one source frame per output tick.

`next()` returns a FrameRef for presentation frame n (time n / fps). Segments play frame by frame; when one ends the
planner picks the successor (decision made `decide_ahead` frames early so the plan is visible). Seams flagged `blend`
get a short crossfade between A's continuation frames and B's first frames (only allowed for joins whose visual cost
is tiny, so there is no ghosting). Every seam is logged with its cost for evaluation.
"""
from dataclasses import dataclass

from avatar_system.common import load_config
from avatar_system.motion.motion_planner import Context, MotionPlanner


@dataclass
class FrameRef:
    n: int                 # presentation frame number (global timeline)
    seg: int               # segment id
    clip: str
    idx: int               # frame index inside the clip
    blend_clip: str = None  # if set: crossfade from this clip/frame (A continuation) with weight 1-alpha
    blend_idx: int = 0
    alpha: float = 1.0
    seam: bool = False      # first frame of a new segment


class MotionScheduler:
    def __init__(self, graph, clip_lengths, seed=0, mode_source=None, speed=None):
        self.cfg_t = load_config("motion_graph")["transition"]
        self.cfg_p = load_config("motion_planner")
        self.planner = MotionPlanner(graph, seed=seed)
        self.segs = graph["segments"]
        self.clip_len = clip_lengths   # clip -> number of frames in the video
        self.ctx = Context()
        self.mode_source = mode_source  # callable -> behaviour state string (state machine)
        self.n = 0
        # playback speed: LTX "subtle" clips move slower than real people (head ~6 deg/s); >1 skips source frames at
        # a fractional rate (blinks stay natural: 160 ms closed -> ~120 ms at 1.3x). pos is fractional, idx = round
        self.base_speed = float(speed if speed is not None else self.cfg_p.get("playback_speed", 1.0))
        self.cur = self.planner.start_segment(self.ctx)
        self.speed = self.segs[self.cur].get("speed") or self.base_speed  # talking-bank takes play at 1x
        self.planner.rep.push(self.cur, 0)
        self.pos = float(self.segs[self.cur]["s"])
        self.blend_left, self.blend_from = 0, None
        self.exits = {int(k): v for k, v in graph.get("exits", {}).items()}
        self.seams = []                # (n, from_seg, to_seg, cost, blend, mode, natural, early_exit)
        self.on_segment_end = None     # callback(seg_dict) e.g. hub_reached detection

    def set_context(self, **kw):
        for k, v in kw.items():
            setattr(self.ctx, k, v)

    def _mode(self):
        if self.mode_source:
            m = self.mode_source()
            if m != self.ctx.mode:
                self.ctx.mode = m
                self.ctx.intensity = self.cfg_p["mode_intensity"][m]
        return self.ctx.mode

    def upcoming(self):
        d = self.planner.last_decision
        return {"current": self.segs[self.cur]["key"], "frames_left": int(self.segs[self.cur]["e"] - self.pos + 1),
                "plan": [self.segs[i]["key"] for i in d.get("plan", [])], "candidates": d.get("candidates", [])}

    CALM = ("LISTENING", "RECOVERY", "INTERRUPTED")

    def _early_exit(self, seg):
        """Calm mode while an expressive segment plays: leave at an approved inside rest point instead of waiting."""
        if self.ctx.mode not in self.CALM or any(t in seg["conv"] for t in ("listen", "idle")):
            return None
        last = self.pos - self.speed  # exit rest point passed since the previous output frame
        opts = [x for x in self.exits.get(self.cur, []) if last - self.speed < x[0] <= last and x[1] in self.planner.run]
        if not opts:
            return None
        return min(opts, key=lambda x: x[2] + 0.3 * self.planner.rep.penalty([x[1]], self.n))

    def _enter(self, seg, nxt, cost, blend, early):
        nseg = self.segs[nxt]
        natural = nseg["clip"] == seg["clip"] and abs(nseg["s"] - self.pos) < self.speed
        k = self.cfg_t["seam_blend_frames"] if blend and not natural else 0
        self.blend_from = (seg["clip"], int(round(self.pos))) if k else None
        self.blend_speed = self.speed
        self.blend_left, self.blend_total = k, k
        self.seams.append((self.n, self.cur, nxt, cost, bool(k), self.ctx.mode, natural, early))
        self.planner.rep.push(nxt, self.n)
        self.cur, self.pos = nxt, float(nseg["s"])
        self.speed = nseg.get("speed") or self.base_speed
        return nseg

    def next(self):
        self._mode()
        seg = self.segs[self.cur]
        seam = False
        ex = self._early_exit(seg) if self.pos <= seg["e"] else None
        if ex:
            seg, seam = self._enter(seg, ex[1], ex[2], bool(ex[3]), True), True
            self.planner.last_decision = {"t": self.n, "mode": self.ctx.mode, "chosen": ex[1], "plan": [ex[1]],
                                          "candidates": [], "early_exit": True}
        elif self.pos > seg["e"]:  # segment finished -> choose and enter the successor
            if self.on_segment_end:
                self.on_segment_end(seg)
                self._mode()
            nxt, cost, blend, info = self.planner.choose(self.cur, self.ctx, self.n)
            seg, seam = self._enter(seg, nxt, cost, blend, False), True
        idx = min(int(round(self.pos)), seg["e"])
        ref = FrameRef(self.n, self.cur, seg["clip"], idx, seam=seam)
        if self.blend_left > 0:
            j = self.blend_total - self.blend_left
            ref.blend_clip = self.blend_from[0]
            ref.blend_idx = min(self.blend_from[1] + int(round(j * self.blend_speed)), self.clip_len[self.blend_from[0]] - 1)
            ref.alpha = (j + 1) / (self.blend_total + 1)
            self.blend_left -= 1
        self.pos += self.speed
        self.n += 1
        return ref
