"""Cheap speech-conditioned behaviour cues (no extra LLM call).

- prosody(pcm): energy envelope -> emphasis peaks (frame offsets), pause fraction, speech rate proxy, pitch contour
  (autocorrelation, voiced frames only). Uses only audio that is already buffered.
- text_hints(text): keyword heuristics -> behaviour hints when the LLM gives no metadata.
- parse_behavior(meta): validate optional LLM metadata {"emotion","intensity","speaking_style","gesture_preferences"};
  everything is a suggestion, the planner may ignore it.
"""
import re

import numpy as np

SR, SPF = 16000, 640


def prosody(pcm, max_s=4.0):
    pcm = np.asarray(pcm[: int(max_s * SR)], np.float32)
    n = len(pcm) // SPF
    if n < 6:
        return {"emphasis_at": [], "pause_frac": 1.0 if n == 0 else 0.0, "energy": 0.0, "voiced_frac": 0.0}
    fr = pcm[: n * SPF].reshape(n, SPF)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-10)
    db = 20 * np.log10(rms + 1e-6)
    speech = db > max(db.max() - 35, -50)
    sm = np.convolve(np.pad(db, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    ref = np.median(sm[speech]) if speech.any() else sm.mean()
    peaks = [i for i in range(2, n - 2) if sm[i] >= sm[i - 2:i + 3].max() and sm[i] > ref + 4 and speech[i]]
    emph = []
    for p in peaks:
        if not emph or p - emph[-1] > 12:
            emph.append(int(p))
    # crude pitch on voiced frames (autocorrelation, 70-300 Hz)
    f0 = []
    for i in np.flatnonzero(speech)[::3]:
        x = fr[i] - fr[i].mean()
        ac = np.correlate(x, x, "full")[SPF - 1:]
        lo, hi = SR // 300, SR // 70
        if ac[0] > 0:
            k = lo + int(np.argmax(ac[lo:hi]))
            if ac[k] / ac[0] > 0.3:
                f0.append(SR / k)
    return {"emphasis_at": emph, "pause_frac": round(float(1 - speech.mean()), 3), "energy": round(float(ref), 1),
            "voiced_frac": round(float(speech.mean()), 3), "f0_median": round(float(np.median(f0)), 1) if f0 else None,
            "f0_range": round(float(np.ptp(f0)), 1) if len(f0) > 2 else None}


KEYWORDS = [
    (r"\b(sorry|माफ|दुख|afsos|अफ़सोस|समझ सकता)\b", ["empathetic", "gentle_nod"]),
    (r"\b(yes|haan|हाँ|हां|bilkul|बिलकुल|ज़रूर|zaroor|sure|okay|ठीक)\b", ["acknowledge"]),
    (r"\b(great|wonderful|बहुत अच्छा|badhiya|बढ़िया|शानदार|congratulations|बधाई)\b", ["positive"]),
    (r"\b(important|ज़रूरी|zaroori|must|ध्यान|dhyan|सबसे)\b", ["emphasis"]),
    (r"\b(maybe|shayad|शायद|let me think|सोच|depends|निर्भर)\b", ["thoughtful"]),
]


def text_hints(text):
    t = text.lower()
    hints = []
    for pat, h in KEYWORDS:
        if re.search(pat, t):
            hints += [x for x in h if x not in hints]
    return hints or ["neutral_explanation"]


ALLOWED = {"gentle_nod", "acknowledge", "emphasis", "neutral_explanation", "thoughtful", "positive", "empathetic",
           "reassuring"}


def parse_behavior(meta):
    if not isinstance(meta, dict):
        return {"hints": [], "intensity": None}
    hints = [g for g in meta.get("gesture_preferences", []) if g in ALLOWED]
    for k in ("emotion", "speaking_style"):
        if meta.get(k) in ALLOWED:
            hints.append(meta[k])
    inten = meta.get("intensity")
    inten = min(max(float(inten), 0.0), 1.0) if isinstance(inten, (int, float)) else None
    return {"hints": hints, "intensity": inten}
