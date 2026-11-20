"""Phase 1: shubh audio corpus for the MuseTalk fine-tune dataset.

  tts      : every sentence of configs/sentences.txt -> Sarvam bulbul:v3 (stock speaker shubh, official
             /text-to-speech) in V variants (different pace/temperature). Each segment passes our QC or is
             regenerated (up to 4 tries): stray-tail cut, clipping, words-per-second, long internal gaps,
             and a Sarvam STT round trip (saarika:v2.5) compared with the text.
  assemble : group QC'd segments into LTX-ready clips (6-12 s, lead/inner/tail pauses, padded to 8n+1 frames
             @ 25 fps), write data/audio/<clip>.wav + copy to comfy/input, data/clips.json (parts, split).

Held-out test split = 10% of sentences (seeded), never used in train clips.
  ../envs/analysis/bin/python -I scripts/gen_audio.py tts [--variants 3] [--limit N]
  ../envs/analysis/bin/python -I scripts/gen_audio.py assemble [--variant-max 0]   (incremental: appends new clips)
"""
import argparse
import asyncio
import base64
import difflib
import json
import math
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

FT = Path(__file__).resolve().parents[1]
R = FT.parent
sys.path.insert(0, str(R / "app"))
from sarvam_client import api_key  # noqa: E402

API = "https://api.sarvam.ai"
SR, FPS = 16000, 25
LANG = {"hi": "hi-IN", "hx": "hi-IN", "en": "en-IN"}
PACES = [0.85, 0.95, 1.0, 1.0, 1.1, 1.2]       # weighted towards normal speed
CACHE, AUDIO = FT / "data/tts_cache", FT / "data/audio"
MAX_FRAMES = 305                               # 12.2 s, longest clip length already proven with LTX here


def sentences():
    out = []
    for line in (FT / "configs/sentences.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "|" in line:
            tag, text = line.split("|", 1)
            out.append({"id": f"{tag}{len([s for s in out if s['tag'] == tag]):03d}", "tag": tag, "text": text.strip()})
    rng = random.Random(7)
    for tag in LANG:  # 10% test per language
        ids = [s for s in out if s["tag"] == tag]
        for s in rng.sample(ids, max(1, round(0.1 * len(ids)))):
            s["split"] = "test"
    for s in out:
        s.setdefault("split", "train")
    return out


# ---------------- QC ----------------
def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in t)  # \w misses Devanagari matras
    t = t.replace("़", "")  # nukta (ज़/ज etc. vary between TTS text and STT output)
    return re.sub(r"\s+", " ", t).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a).replace(" ", ""), norm(b).replace(" ", "")).ratio()


def loud_mask(x, win=160, thr=0.02):
    n = len(x) // win
    return np.abs(x[:n * win]).reshape(n, win).max(1) > thr


def clean(x):
    """Trim edge silence and drop a short stray burst after a long gap at the end (TTS artefact)."""
    m = loud_mask(x)
    if not m.any():
        return x[:0], {}
    i0, i1 = int(m.argmax()), len(m) - int(m[::-1].argmax())
    m = m[i0:i1]
    # stray tail: last loud run <= 250 ms preceded by >= 300 ms silence
    j = len(m)
    while j > 0 and m[j - 1]:
        j -= 1
    k = j
    while k > 0 and not m[k - 1]:
        k -= 1
    info = {}
    if k > 0 and (j - k) >= 30 and (len(m) - j) <= 25:
        info["stray_tail_ms"] = (len(m) - j) * 10
        i1 = i0 + k
    lead, tail = int(0.03 * SR), int(0.06 * SR)
    return x[max(0, i0 * 160 - lead): min(len(x), i1 * 160 + tail)], info


def longest_gap(x):
    m = loud_mask(x)
    best = cur = 0
    for v in m:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best * 0.01


NUMWORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
               "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
               "lakh crore and minus plus rupees".split())
DEVA = re.compile(r"[\u0900-\u097f]")


def stt_check(text, tag, stt):
    """Token recall of the text in the STT transcript (fuzzy per token) + token-count ratio.
    STT writes English words of Hinglish lines in Devanagari and numbers as digits, so for hx only the
    Devanagari tokens are checked, and number words are skipped when the transcript contains digits."""
    tt, st = norm(text).split(), norm(stt).split()
    digits = any(c.isdigit() for c in stt)
    roman = tag == "hx" and not DEVA.search(stt)  # STT sometimes answers code-mixed speech in Roman script
    check = [t for t in tt if (tag != "hx" or bool(DEVA.search(t)) != roman) and not (digits and (t in NUMWORDS or not tag == "en"
                                                                                    and t in HI_NUM))]
    hit = sum(any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.75 for x in st) for t in check)
    recall = hit / len(check) if check else 1.0
    ratio = len(st) / max(1, len(tt))
    return round(recall, 3), round(ratio, 2), digits


HI_NUM = set(norm("एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
                  "बाईस पच्चीस बत्तीस पैंतालीस अस्सी चालीस पचास सौ हज़ार लाख करोड़ शून्य निन्यानवे उनतालीस सैंतालीस रुपये").split())


def qc(x, text, tag, stt_text):
    dur = len(x) / SR
    words = max(1, len(norm(text).split()))
    r = {"dur": round(dur, 3), "wps": round(words / max(dur, 1e-3), 2), "clip_frac": float((np.abs(x) > 0.99).mean()),
         "gap": round(longest_gap(x), 2), "stt": stt_text, "stt_sim": round(sim(text, stt_text), 3) if stt_text else None}
    bad = []
    if dur < 0.6:
        bad.append("too_short")
    if not 0.9 <= r["wps"] <= (7.0 if tag == "en" else 5.5):  # short English words run fast
        bad.append("rate")
    if r["clip_frac"] > 0.002:
        bad.append("clipping")
    if r["gap"] > 1.0:
        bad.append("long_gap")
    if not stt_text:
        bad.append("stt_empty")
    else:
        r["stt_recall"], r["stt_ratio"], digits = stt_check(text, tag, stt_text)
        if r["stt_recall"] < (0.75 if tag == "hx" else 0.8) and (r["stt_sim"] or 0) < 0.85:
            bad.append("stt_mismatch")
        if not digits and not 0.6 <= r["stt_ratio"] <= 1.7:
            bad.append("stt_length")
    r["fail"] = bad
    return r


# ---------------- Sarvam ----------------
class Client:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0),
                                      headers={"api-subscription-key": api_key()})

    async def _post(self, what, **kw):
        for attempt in range(10):
            try:
                r = await self.http.post(f"{API}/{what}", **kw)
            except httpx.TransportError as e:
                r, err = None, str(e)
            if r is not None and r.status_code == 200:
                return r.json()
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{what} {r.status_code}: {r.text[:300]}")
            wait = min(120, 10 * 2 ** min(attempt, 4))
            print(f"  {what}: {r.status_code if r is not None else err} -> retry in {wait}s", flush=True)
            await asyncio.sleep(wait)
        raise RuntimeError(f"{what}: gave up")

    async def tts(self, text, lang, pace, temperature):
        j = await self._post("text-to-speech", json={
            "text": text, "target_language_code": lang, "model": "bulbul:v3", "speaker": "shubh", "pace": pace,
            "temperature": temperature, "speech_sample_rate": SR, "output_audio_codec": "wav"})
        raw = base64.b64decode("".join(j["audios"]))
        tmp = CACHE / f".tmp_{random.getrandbits(40):x}.wav"
        tmp.write_bytes(raw)
        x, sr = sf.read(tmp, dtype="float32")
        tmp.unlink()
        assert sr == SR, sr
        return x if x.ndim == 1 else x.mean(1)

    async def stt(self, x):
        buf = CACHE / f".stt_{random.getrandbits(40):x}.wav"
        sf.write(buf, x, SR, subtype="PCM_16")
        data = buf.read_bytes()
        buf.unlink()
        j = await self._post("speech-to-text", files={"file": ("s.wav", data, "audio/wav")},
                             data={"model": "saarika:v2.5", "language_code": "unknown"})
        return j.get("transcript", "")


async def make_segment(cl, s, v, sem, rng):
    key = f"{s['id']}_v{v}"
    meta_p = CACHE / f"{key}.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    async with sem:
        pace = rng.choice(PACES) if v else 1.0   # v0 = normal pace
        tries = []
        for attempt in range(4):  # lower temperature on retries = steadier delivery
            temp = round(rng.uniform(0.45, 0.75), 2) if attempt == 0 else round(rng.uniform(0.3, 0.55), 2)
            x = await cl.tts(s["text"], LANG[s["tag"]], pace, temp)
            x, info = clean(x)
            stt = await cl.stt(x) if len(x) > SR * 0.3 else ""
            tries.append((qc(x, s["text"], s["tag"], stt) | info | {"pace": pace, "temperature": temp}, x))
            if not tries[-1][0]["fail"]:
                break
        q, x = tries[-1]
        ok = not q["fail"]
        sf.write(CACHE / (f"{key}.wav" if ok else f"{key}.bad.wav"), x, SR, subtype="PCM_16")
        meta = s | {"key": key, "variant": v, "ok": ok, "qc": q, "tries": len(tries),
                    "failed_tries": [t[0] for t in tries[:-1]]}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        flag = "OK " if ok else "BAD"
        print(f"{flag} {key} {meta['qc']['dur']:.2f}s pace {pace} sim {meta['qc']['stt_sim']} tries {len(tries)} "
              f"{meta['qc']['fail'] or ''} | {s['text'][:50]}", flush=True)
        return meta


async def run_tts(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    cl, sem = Client(), asyncio.Semaphore(a.concurrency)
    ss = sentences()
    jobs = []
    for v in range(a.variants):  # variant-major: every sentence once before any second take
        for s in ss:
            if s["split"] == "test" and v > 0:
                continue
            jobs.append((s, v))
    jobs = jobs[:a.limit] if a.limit else jobs
    rng = random.Random(11)
    seeds = [rng.random() for _ in jobs]
    res = await asyncio.gather(*[make_segment(cl, s, v, sem, random.Random(sd)) for (s, v), sd in zip(jobs, seeds)])
    ok = [m for m in res if m["ok"]]
    print(f"TTS_DONE {len(ok)}/{len(res)} ok, speech {sum(m['qc']['dur'] for m in ok) / 60:.1f} min", flush=True)


# ---------------- assemble ----------------
def assemble(a):
    segs = [json.loads(p.read_text()) for p in sorted(CACHE.glob("*.json"))]
    segs = [m for m in segs if m["ok"] and (CACHE / f"{m['key']}.wav").exists()]
    AUDIO.mkdir(parents=True, exist_ok=True)
    cj = FT / "data/clips.json"
    clips = json.loads(cj.read_text()) if cj.exists() else []   # incremental: existing clips never change
    used = {p["key"] for c in clips for p in c["parts"]}
    rng = random.Random(3 + len(clips))
    if a.variant_max is not None:
        segs = [m for m in segs if m["variant"] <= a.variant_max]
    for split in ("train", "test"):
        pool = [m for m in segs if m["split"] == split and m["key"] not in used]
        # variant-major order keeps every sentence's first take in the earliest clips
        pool.sort(key=lambda m: (m["variant"], rng.random()))
        cur, n = [], sum(c["split"] == split for c in clips)

        def flush(cur):
            nonlocal n
            if not cur:
                return
            name = f"mtft_{split}_{n:03d}"
            n += 1
            parts, pcm, t = [], [], 0.0
            lead = rng.uniform(0.3, 0.7)
            pcm.append(np.zeros(int(lead * SR), np.float32)); t += lead
            for i, m in enumerate(cur):
                if i:
                    g = rng.uniform(0.25, 0.9)
                    pcm.append(np.zeros(int(g * SR), np.float32)); t += g
                x, _ = sf.read(CACHE / f"{m['key']}.wav", dtype="float32")
                parts.append({"key": m["key"], "text": m["text"], "tag": m["tag"], "pace": m["qc"]["pace"],
                              "start": round(t, 3), "end": round(t + len(x) / SR, 3)})
                pcm.append(x); t += len(x) / SR
            tail = rng.uniform(0.5, 1.0)
            frames = min(MAX_FRAMES, math.ceil((t + tail) * FPS / 8) * 8 + 1)
            x = np.concatenate(pcm)
            x = np.pad(x, (0, max(0, int(frames / FPS * SR) - len(x))))[:int(frames / FPS * SR)]
            peak = np.abs(x).max()
            if peak > 0:
                x = x * min(1.0, 0.9 / peak)
            sf.write(AUDIO / f"{name}.wav", x, SR, subtype="PCM_16")
            shutil.copy(AUDIO / f"{name}.wav", R / "comfy/input" / f"{name}.wav")
            clips.append({"name": name, "split": split, "frames": frames, "seconds": round(frames / FPS, 3),
                          "speech": round(sum(p["end"] - p["start"] for p in parts), 2), "parts": parts})

        target = rng.uniform(6.5, 10.0)  # speech seconds per clip
        for m in pool:
            dur = m["qc"]["dur"]
            acc = sum(c["qc"]["dur"] for c in cur) + 0.6 * len(cur)
            if cur and (acc + dur + 0.6 + 1.6 > MAX_FRAMES / FPS or acc >= target):
                flush(cur)
                cur, target = [], rng.uniform(6.5, 10.0)
            cur.append(m)
        flush(cur)
    cj.write_text(json.dumps(clips, ensure_ascii=False, indent=1))
    for split in ("train", "test"):
        cs = [c for c in clips if c["split"] == split]
        print(f"{split}: {len(cs)} clips, {sum(c['seconds'] for c in cs) / 60:.1f} min video, "
              f"{sum(c['speech'] for c in cs) / 60:.1f} min speech")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tts", "assemble"])
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--variant-max", type=int, default=None, help="assemble: only use takes <= this variant")
    a = ap.parse_args()
    if a.cmd == "tts":
        asyncio.run(run_tts(a))
    else:
        assemble(a)


if __name__ == "__main__":
    main()
