"""Local speech-to-text candidates vs Sarvam on our own held-out sentences (true text known).

Test set: every test-split sentence of the fine-tune corpus (data/tts_cache/*_v0.wav + json: Hindi, Hinglish, English,
shubh voice, 16 kHz) - its json also holds Sarvam saarika's transcript from the dataset QC. Accuracy = the QC's fuzzy
token recall (gen_audio.stt_check: Devanagari-aware, number words tolerant); speed = wall time per clip on one GPU.

  CUDA_VISIBLE_DEVICES=0 ../envs/musetalk_trainer/bin/python scripts/eval_local_stt.py --models openai/whisper-large-v3-turbo
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT / "scripts"))
from gen_audio import stt_check  # noqa: E402

LANG = {"hi": "hi", "hx": "hi", "en": "en"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/whisper-large-v3-turbo"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = []
    for j in sorted((FT / "data/tts_cache").glob("*_v0.json")):
        d = json.loads(j.read_text())
        if d.get("split") == "test" and d.get("ok"):
            items.append(d)
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} test sentences: " + ", ".join(f"{t} {sum(d['tag'] == t for d in items)}" for t in ("hi", "hx", "en")))

    res = {"sarvam_saarika": {}}
    for d in items:
        res["sarvam_saarika"][d["key"]] = d["qc"].get("stt", "")
    times = {}
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    for name in a.models:
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
        res[name], times[name] = {}, []
        for k, d in enumerate(items):
            x, sr = sf.read(FT / f"data/tts_cache/{d['key']}.wav", dtype="float32")
            feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features.cuda().half()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                try:
                    ids = model.generate(feats, language=LANG[d["tag"]], task="transcribe", max_new_tokens=128)
                except ValueError:  # older fine-tunes: language via forced decoder ids
                    ids = model.generate(feats, forced_decoder_ids=proc.get_decoder_prompt_ids(
                        language=LANG[d["tag"]], task="transcribe"), max_new_tokens=128)
            torch.cuda.synchronize()
            if k:  # first call includes warm-up
                times[name].append(time.time() - t0)
            res[name][d["key"]] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        del model
        torch.cuda.empty_cache()

    print(f"\n{'model':38s} {'hi':>6s} {'hx':>6s} {'en':>6s} {'all':>6s}  s/clip")
    for name, tr in res.items():
        rec = {t: [] for t in ("hi", "hx", "en")}
        for d in items:
            r, _, _ = stt_check(d["text"], d["tag"], tr.get(d["key"], ""))
            rec[d["tag"]].append(r)
        allr = [v for vs in rec.values() for v in vs]
        sp = f"{np.mean(times[name]):.3f}" if name in times else "api"
        print(f"{name:38s} " + " ".join(f"{np.mean(rec[t]):6.3f}" for t in ("hi", "hx", "en")) + f" {np.mean(allr):6.3f}  {sp}")
    out = FT / "eval/local_stt.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"items": items, "transcripts": res, "times": times}, ensure_ascii=False, indent=1), encoding="utf-8")
    for d in items[:6] + [d for d in items if d["tag"] == "hx"][:4]:
        print(f"\n[{d['tag']}] TRUE  {d['text']}")
        for name, tr in res.items():
            print(f"     {name.split('/')[-1][:22]:22s} {tr.get(d['key'], '')}")


if __name__ == "__main__":
    main()
