"""Recent-history bookkeeping for the planner: penalises replaying the same segment/clip, repeating 2- and 3-segment
sequences, and repeating salient motion patterns (nods, gestures, head tilts, upward gaze, blink-heavy segments)
too soon. Time is measured in output frames."""
from collections import deque

PATTERNS = {  # pattern -> minimum spacing (frames) before it feels repetitive
    "nod": 10 * 25, "gesture": 15 * 25, "tilt": 12 * 25, "gaze_up": 30 * 25, "blinky": 6 * 25,
}


def patterns_of(seg):
    d = seg["desc"]
    p = set()
    if d["nods"]:
        p.add("nod")
    if d["gesture"]:
        p.add("gesture")
    if max(d["head_range_deg"][2], d["head_range_deg"][0]) > 6 or seg["start_state"] != seg["end_state"]:
        p.add("tilt")
    if d["gaze_up_frac"] > 0.05:
        p.add("gaze_up")
    if d["blinks"] >= 3:
        p.add("blinky")
    return p


class RepetitionTracker:
    def __init__(self, segments, maxlen=40):
        self.segs = segments
        self.hist = deque(maxlen=maxlen)          # (seg_id, start_frame)
        self.last_pattern = {}                   # pattern -> frame it last started
        self.ngrams = {}                         # tuple -> count

    def push(self, seg_id, t):
        h = [s for s, _ in self.hist]
        for n in (2, 3):
            if len(h) >= n - 1:
                key = tuple(h[len(h) - n + 1:] + [seg_id])
                self.ngrams[key] = self.ngrams.get(key, 0) + 1
        self.hist.append((seg_id, t))
        for p in patterns_of(self.segs[seg_id]):
            self.last_pattern[p] = t

    def penalty(self, path, t):
        """Repetition penalty for appending `path` (list of seg ids) at frame t (>= 0, higher = worse)."""
        pen = 0.0
        h = [s for s, _ in self.hist]
        clips = [self.segs[s]["clip"] for s in h]
        tt = t
        seq = list(h)
        for k, sid in enumerate(path):
            seg = self.segs[sid]
            if sid in h[-12:] or sid in path[:k]:
                pen += 3.0 if sid in h[-4:] else 1.5
            if seg["clip"] in clips[-3:]:
                pen += 0.6
            for n, w in ((2, 1.0), (3, 2.0)):
                if len(seq) >= n - 1:
                    pen += w * min(self.ngrams.get(tuple(seq[len(seq) - n + 1:] + [sid]), 0), 3)
            for p in patterns_of(seg):
                last = self.last_pattern.get(p)
                if last is not None and tt - last < PATTERNS[p]:
                    pen += 1.2 * (1 - (tt - last) / PATTERNS[p])
            seq.append(sid)
            tt += seg["desc"]["frames"]
        return pen
