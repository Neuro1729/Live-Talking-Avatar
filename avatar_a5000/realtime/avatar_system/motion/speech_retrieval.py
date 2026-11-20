"""Speech-driven gesture retrieval: for a new sentence the avatar is about to say, find the recorded talking-bank
speech part (data/talk_index.json) whose speech act, emotion, length and rhythm match best. Its body/face/hand motion
is replayed (time-warped so its length and stressed words fit the new audio) while MuseTalk re-syncs only the lips.

Score (lower = better) = act/emotion mismatch + duration misfit + rhythm distance (DTW of energy envelopes) +
stressed-word count difference + recent-use penalty. Warp = part length / sentence length, limited to [0.8, 1.25].
"""
import json
import math
import re

import numpy as np

from avatar_system.common import data_dir

ACT_CUES = [  # text cue -> speech act
    (r"\b(नमस्ते|नमस्कार|hello|hi|namaste)\b", "greeting"),
    (r"(पहला|दूसरा|तीसरा|first|second|third|pehla|doosra|teesra)", "enumerate_steps"),
    (r"(बधाई|शुभकामना|badhai|congrat|wah|वाह)", "congratulate"),
    (r"(चिंता|समझ सकता|अफ़सोस|माफ़|sorry|himmat|हिम्मत)", "reassure"),
    (r"(हम्म|hmm|देखिए|dekhiye|निर्भर|depends)", "think_then_answer"),
    (r"^(जी|ji|हाँ|haan|ok|okay)\W*$", "acknowledge"),
    (r"(ज़रूरी|zaroori|ज़रूर|zaroor|important|must|सबसे)", "advise_emphasis"),
]
HINT_ACTS = {"positive": ["congratulate", "greeting"], "empathetic": ["reassure"], "thoughtful": ["think_then_answer"],
             "emphasis": ["advise_emphasis", "enumerate_steps"], "neutral_explanation": ["explain", "advise_emphasis"],
             "gentle_nod": ["acknowledge", "reassure", "greeting"]}
RELATED = {"explain": {"advise_emphasis", "enumerate_steps"}, "advise_emphasis": {"explain", "enumerate_steps"},
           "enumerate_steps": {"explain", "advise_emphasis"}, "greeting": {"congratulate"},
           "congratulate": {"greeting"}, "reassure": {"acknowledge", "think_then_answer"},
           "think_then_answer": {"explain", "reassure"}, "acknowledge": {"reassure"}}


def guess_acts(text, hints=()):
    acts = [a for pat, a in ACT_CUES if re.search(pat, text.lower())]
    for h in hints:
        acts += [a for a in HINT_ACTS.get(h, []) if a not in acts]
    return acts or ["explain"]


def resample(x, n=48):
    x = np.asarray(x, np.float32)
    if len(x) < 2:
        return np.zeros(n, np.float32)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def dtw(a, b, band=8):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


class SpeechRetriever:
    def __init__(self, index=None, weights=None):
        idx = index or json.loads((data_dir() / "talk_index.json").read_text())
        self.parts = [p for c in idx.values() for p in c["parts"] if p["type"] == "speech"]
        self.w = weights or {"act": 1.5, "emotion": 0.5, "duration": 1.2, "rhythm": 4.0, "emphasis": 0.3, "recent": 1.0}
        self.recent = []

    def score(self, p, acts, emotion, n_frames, env, n_emph):
        w = self.w
        a = 0.0 if p["act"] == acts[0] else 0.3 if p["act"] in acts else 0.6 if p["act"] in RELATED.get(acts[0], ()) else 1.0
        e = 0.0 if (emotion and p["emotion"] == emotion) else 0.5
        dur = abs(math.log(p["frames"] / max(n_frames, 1)))
        rh = dtw(resample(env), resample(p["envelope"]))
        em = abs(len(p["emphasis"]) - n_emph) / max(n_emph, 1)
        rec = self.recent[-6:].count((p["clip"], p["part"]))
        return (w["act"] * a + w["emotion"] * e + w["duration"] * dur + w["rhythm"] * rh + w["emphasis"] * em
                + w["recent"] * rec), {"act": a, "emotion": e, "duration": round(dur, 3), "rhythm": round(rh, 3),
                                      "emphasis": round(em, 3), "recent": rec}

    def retrieve(self, text, pcm, hints=(), emotion=None, k=3):
        """-> list of (score, part, warp, components) best first. pcm: float32 16 kHz of the new sentence."""
        from avatar_system.conversation.behavior_planner import prosody
        from avatar_system.generation.talk_index import envelope
        n = max(int(math.ceil(len(pcm) / 640)), 1)
        env = envelope(pcm, n)
        n_emph = len(prosody(pcm, max_s=60)["emphasis_at"])
        acts = guess_acts(text, hints)
        emotion = emotion or {"positive": "positive", "empathetic": "empathetic", "thoughtful": "thoughtful"}.get(
            next((h for h in hints if h in ("positive", "empathetic", "thoughtful")), ""), None)
        out = []
        for p in self.parts:
            s, comp = self.score(p, acts, emotion, n, env, n_emph)
            warp = min(max(p["frames"] / n, 0.8), 1.25)
            out.append((round(s, 3), p, round(warp, 3), comp))
        out.sort(key=lambda x: x[0])
        if out:
            self.recent.append((out[0][1]["clip"], out[0][1]["part"]))
        return out[:k], acts
