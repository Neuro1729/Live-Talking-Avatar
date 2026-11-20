"""Live talking-bank scheduler over the talking-bank motion graph (motion/talk_graph.py): a drop-in for MotionScheduler
(same next()/set_context()/upcoming()/seams API).

Every output frame is a recorded frame played forward at 1x; the only discontinuities are approved graph joins between
pause cut points (3-frame crossfade). Nothing is time-warped, reversed or sped up.

- Speech: when a sentence's audio is complete (AvatarPipeline.push_audio -> add_speech) a beam search over the graph
  plans the frames up to the sentence end: frames under the sentence should be recorded talking frames (cost by how
  well that recorded part's speech act / emotion / rhythm fits the sentence, SpeechRetriever), silent frames must be
  pause frames (mouth closed), and the plan must end on a pause frame. Join cost, number of joins and recent reuse
  are part of the path cost, so continuity wins over a better-fitting gesture.
- Silence (listening / thinking / between answers): the same search over a silent window - pause frames only, chained
  through approved joins (e.g. a clip's settle back to the home pose -> another clip's start).
"""
import json
import threading
from collections import deque

import numpy as np

from avatar_system.common import data_dir
from avatar_system.motion.motion_scheduler import FrameRef
from avatar_system.motion.speech_retrieval import SpeechRetriever

BLEND = 5            # crossfade frames at a join (0.2 s: residual texture/expression differences blend)
BEAM = 64            # states kept per frame
EDGES_PER_CUT = 12   # cheapest approved joins kept per cut point, for each kind (into silence / into speech)
JOIN_BASE = 1.0      # fixed cost per join (play the recording through; join only when needed)
SEM_W = 0.03         # per speech frame x (retrieval score - best score, capped at 2) of the recorded part shown
PAUSE_IN_SPEECH = 0.08  # per frame of a pause frame shown while the sentence is spoken (> worst SEM cost)
REUSE_W = 0.01       # per frame of a clip region shown in the last REUSE_WINDOW frames
REUSE_WINDOW = 750
IDLE_WINDOW = 100    # silent frames planned at a time
RECOVER_W = 0.1      # per talking frame still played in silence while finishing a gesture
MIN_RUN = 25         # frames played between two joins at least (1 s)
LOOP_W = 1.0         # join into a region this plan already joined into / just showed (same-clip replay)
IDLE_BEAM = 24       # smaller beam for silence (planned inside the frame loop)
MARGIN = 15          # frames left untouched while a sentence is planned outside the lock (~0.6 s)


class TalkBankScheduler:
    def __init__(self, index, clip_lengths, analyze=None, score=None, seed=0, mode_source=None):
        g = json.loads((data_dir() / "talk_graph.json").read_text())
        self.clips = sorted(c for c in g["clips"] if c in index)
        self.len = {c: min(g["clips"][c]["n"], clip_lengths.get(c, g["clips"][c]["n"])) for c in self.clips}
        self.lab = {c: np.array(g["clips"][c]["lab"][: self.len[c]], np.int8) for c in self.clips}
        self.part = {c: np.array(g["clips"][c]["part"][: self.len[c]], np.int16) for c in self.clips}
        # per cut point keep the cheapest joins of each kind: into silence, and into speech (talking frames or the
        # pause just before a talking part) - silent frames resemble each other most and would crowd out the rest
        self.adj, quota = {}, {}
        for ca, ia, cb, ib, cost in sorted(g["edges"], key=lambda e: e[4]):
            if ca in self.lab and cb in self.lab and ib < self.len[cb] and self.lab[cb][ib] != 2:
                kind = bool(self.lab[cb][ib] == 1 or (self.lab[cb][ib:ib + 10] == 1).any())
                q = quota.setdefault((ca, ia, kind), 0)
                if q < EDGES_PER_CUT:
                    quota[(ca, ia, kind)] = q + 1
                    self.adj.setdefault((ca, ia), []).append((cb, ib, cost))
        # Silent safety (greatest fixpoint): enter[c][i] - a join may land on pause frame i: at least MIN_RUN pause
        # frames follow it and then a join to another enterable frame; exit[c][i] - from pause frame i such a join is
        # still ahead without crossing talking frames. Silent plans only use these, so silence never dead-ends.
        self.enter = {c: self.lab[c] == 0 for c in self.clips}
        self.exit = {c: self.lab[c] == 0 for c in self.clips}
        changed = True
        while changed:
            changed = False
            for c in self.clips:
                far = None  # last frame of the current pause run with a join to an enterable frame
                for i in range(self.len[c] - 1, -1, -1):
                    if self.lab[c][i] != 0:
                        far = None
                    elif far is None and any(self.enter[cb][ib] for cb, ib, _ in self.adj.get((c, i), ())):
                        far = i
                    ex, en = far is not None, far is not None and far - i >= MIN_RUN
                    if self.exit[c][i] != ex or self.enter[c][i] != en:
                        self.exit[c][i], self.enter[c][i], changed = ex, en, True
        self.retr = SpeechRetriever(index={c: index[c] for c in self.clips})
        self.retr.retrieve("नमस्ते", np.zeros(16000, np.float32))  # warm up (first call imports/compiles ~1 s)
        self.retr.recent.clear()
        self.mode_source = mode_source
        self.lock = threading.Lock()
        self.plan = {}            # frame n -> (clip, idx, blend_clip, blend_idx, alpha)
        self.speech_until = 0     # frames before this are committed sentence plans (idle plans are replaceable)
        self.n = 0
        home = max(self.clips, key=lambda c: (self.lab[c][-1] == 0, c.startswith("talk_listen")))
        self.last = (home, self.len[home] - 1)
        self.recent = deque(maxlen=REUSE_WINDOW)
        self.hints = []
        self.seams = []           # (n, from_clip, to_clip, cost, blend, mode, natural, early) like MotionScheduler
        self.speech_log = []      # (start, end, text, parts used, joins)
        self.on_segment_end = None
        self.ctx = type("Ctx", (), {"mode": "LISTENING"})()
        self.stats = {"plans": 0, "fallbacks": 0}

    # ---------- MotionScheduler API ----------
    def set_context(self, **kw):
        if "hints" in kw:
            self.hints = list(kw["hints"] or [])

    def upcoming(self):
        with self.lock:
            left = sum(k >= self.n for k in self.plan)
        return {"current": f"{self.last[0]}#{self.last[1]}", "frames_left": left,
                "plan": [str(s[3]) for s in self.speech_log[-2:]], "candidates": []}

    def next(self):
        with self.lock:
            if self.mode_source:
                self.ctx.mode = self.mode_source()
            if self.n not in self.plan:
                self._commit(self._search(self.n, self.n + IDLE_WINDOW, None, None, self.last, beam_w=IDLE_BEAM), idle=True)
            e = self.plan.pop(self.n)
            ref = FrameRef(self.n, 0, e[0], int(e[1]), blend_clip=e[2], blend_idx=int(e[3]), alpha=e[4],
                           seam=e[0] != self.last[0] or abs(int(e[1]) - self.last[1]) != 1)
            self.last = (e[0], int(e[1]))
            self.recent.append((e[0], int(e[1]) // 25))
            if self.on_segment_end and self._settled(e[0], int(e[1])):
                self.on_segment_end({"end_state": "home"})  # at rest: silent, can stay silent (ends recovery)
            self.n += 1
            return ref

    # ---------- speech ----------
    def add_speech(self, text, pcm, start, end):
        """Plan the frames up to the end of a sentence whose audio occupies frames [start, end)."""
        res, _ = self.retr.retrieve(text, pcm, hints=self.hints, k=len(self.retr.parts))
        best = res[0][0] if res else 0.0
        sem = {(p["clip"], p["part"]): sc - best for sc, p, _, _ in res}
        with self.lock:
            h0 = max(self.n + MARGIN, self.speech_until)
            if end <= h0:
                return None
            s0 = self.plan.get(h0 - 1, (*self.last, None, 0, 1.0))[:2]
            recent = set(self.recent)
        path = self._search(h0, end, (start, end), sem, s0, recent)
        with self.lock:
            if self.n > h0:  # planning took longer than the margin: drop the frames already shown
                path = path[self.n - h0:]
                h0 = self.n
            for k in [k for k in self.plan if k >= h0]:
                del self.plan[k]
            self._commit(path, idle=False, at=h0)
            self.speech_until = end
            used = sorted({(c, int(self.part[c][i])) for c, i, *_ in path if self.part[c][i] >= 0})
            self.speech_log.append((start, end, text, used, sum(1 for x in path if x[2] is not None and x[4] < 0.3)))
        return used

    def cancel(self):
        """Interrupted: drop planned future motion; the silent planner takes over from the current frame (it first
        plays on to the next pause frame - the recorded mouth may move briefly, the lip-sync ramp covers it)."""
        with self.lock:
            self.plan = {k: v for k, v in self.plan.items() if k < self.n}
            self.speech_until = self.n

    # ---------- planning ----------
    def _settled(self, clip, i):
        """A silent frame from which silence can go on (pause frame with a silent way out)."""
        return self.lab[clip][i] == 0 and self.exit[clip][i]

    def _commit(self, path, idle, at=None):
        at = self.n if at is None else at
        for k, e in enumerate(path):
            self.plan[at + k] = e
        if path:
            self.stats["plans"] += 1

    def _search(self, t0, t1, speech, sem, s0, recent=None, beam_w=None):
        """Beam search over the graph for frames [t0, t1) starting after state s0 = (clip, idx) shown at t0 - 1.
        speech = (start, end) frames of the spoken sentence (or None: all silent). -> list of frame tuples."""
        recent = set(self.recent) if recent is None else recent
        beam_w = beam_w or BEAM
        sem = sem or {}
        worst = max(sem.values()) if sem else 0.0
        T = t1 - t0
        # beam entry: (cost, clip, idx, parent_index, joined_from, frames_since_join, join_history)
        steps = []
        beam = [(0.0, s0[0], s0[1], -1, None, MIN_RUN, ())]
        for k in range(T):
            t = t0 + k
            silent = speech is None or not (speech[0] <= t < speech[1])
            cand = {}
            for pi, (c, clip, i, _, _, sj, hist) in enumerate(beam):
                opts = []
                if i + 1 < self.len[clip]:
                    opts.append((clip, i + 1, 0.0, None))
                if sj >= MIN_RUN:  # play at least MIN_RUN frames between joins
                    for cb, ib, ec in self.adj.get((clip, i), ()):
                        opts.append((cb, ib, JOIN_BASE + ec, (clip, i, ec)))
                for cb, ib, add, jf in opts:
                    lab = self.lab[cb][ib]
                    if lab == 2:
                        continue
                    f = add
                    if silent and not (self.enter[cb][ib] if jf else self.exit[cb][ib]):
                        # silence needs safe pause frames; the only exception is finishing a gesture we are already
                        # in (after an interruption): play it on to a calm join point, at a cost so we leave early
                        if jf is not None or self._settled(clip, i):
                            continue
                        f += RECOVER_W
                    if not silent:
                        f += PAUSE_IN_SPEECH if lab == 0 else SEM_W * min(sem.get((cb, int(self.part[cb][ib])), worst), 2.0)
                    blk = (cb, ib // 25)
                    if blk in recent:
                        f += REUSE_W
                    h = hist
                    if jf is not None:  # loops: re-entering a region this path (or the last seconds) already showed
                        if blk in hist or (cb == clip and i - 60 < ib < i):
                            f += LOOP_W
                        elif blk in recent:
                            f += LOOP_W
                        h = (hist + (blk, (clip, i // 25)))[-8:]
                    key = (cb, ib)
                    if key not in cand or c + f < cand[key][0]:
                        cand[key] = (c + f, cb, ib, pi, jf, 0 if jf else sj + 1, h)
            if not cand:  # dead end (should not happen): hold the pose, count it
                self.stats["fallbacks"] += 1
                cand = {(b[1], b[2]): (b[0] + 1.0, b[1], b[2], j, None, b[5] + 1, b[6]) for j, b in enumerate(beam)}
            steps.append(beam)
            beam = sorted(cand.values(), key=lambda x: x[0])[:beam_w]
        # the frame after the plan must be able to continue silently: prefer ending on a pause frame
        final = [b for b in beam if self._settled(b[1], b[2])] or beam
        b = min(final, key=lambda x: x[0])
        path = []
        for k in range(T - 1, -1, -1):
            path.append(b)
            b = steps[k][b[3]] if b[3] >= 0 else None
            if b is None:
                break
        path.reverse()
        out = [(clip, i, None, 0, 1.0) for _, clip, i, *_ in path]
        for j, (_, clip, i, _, jf, *_) in enumerate(path):
            if jf is None:
                continue
            ca, ia, ec = jf  # a join: crossfade from A's continuation (held on A's frame where A's pause ends)
            self.seams.append((t0 + j, ca, clip, ec, True, self.ctx.mode, False, False))
            for d in range(min(BLEND, len(out) - j)):
                bi = min(ia + 1 + d, self.len[ca] - 1)
                out[j + d] = out[j + d][:2] + (ca, bi if self.lab[ca][bi] == 0 else ia, (d + 1) / (BLEND + 1))
        return out
