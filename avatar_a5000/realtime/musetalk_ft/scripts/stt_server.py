"""Local speech-to-text service for the live avatar (replaces the Sarvam STT round trip).

Routing by conversation language (picked on our 37 held-out sentences, scripts/eval_local_stt.py):
  hi-IN (Hindi + Hinglish)  ARTPARK-IISc/whisper-large-v3-vaani-hindi  recall hi 0.992 / hx 1.000 (Sarvam 0.985 / 1.000)
  en-IN                     openai/whisper-large-v3-turbo              recall en 0.983 (= Sarvam)
Models load one after the other at startup (host-RAM rule), fp16 on one GPU.

  CUDA_VISIBLE_DEVICES=0 ../envs/musetalk_trainer/bin/python scripts/stt_server.py --port 8310
  POST /stt {"wav_b64": <16 kHz mono wav>, "lang": "hi-IN"} -> {"text": ..., "lang": ..., "stt_s": ...}
"""
import argparse
import base64
import io
import threading
import time

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODELS = {"hi": "ARTPARK-IISc/whisper-large-v3-vaani-hindi", "en": "openai/whisper-large-v3-turbo"}
app = FastAPI()
M = {}
LOCK = threading.Lock()  # one GPU, one request at a time


def load():
    for lang, name in MODELS.items():  # sequential loads
        t0 = time.time()
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
        M[lang] = (proc, model)
        transcribe(torch.zeros(16000).numpy(), lang)  # warm-up
        print(f"[stt] {lang}: {name} ready in {time.time() - t0:.0f}s", flush=True)


def transcribe(x, lang):
    proc, model = M[lang]
    feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with LOCK, torch.no_grad():
        try:
            ids = model.generate(feats, language=lang, task="transcribe", max_new_tokens=128)
        except ValueError:  # older fine-tunes: language via forced decoder ids
            ids = model.generate(feats, forced_decoder_ids=proc.get_decoder_prompt_ids(language=lang, task="transcribe"),
                                 max_new_tokens=128)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


@app.post("/stt")
def stt(body: dict):
    t0 = time.time()
    x, sr = sf.read(io.BytesIO(base64.b64decode(body["wav_b64"])), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    lang = "en" if body.get("lang", "hi-IN").startswith("en") else "hi"
    text = transcribe(x, lang)
    return {"text": text, "lang": body.get("lang", "hi-IN"), "stt_s": round(time.time() - t0, 3)}


@app.get("/health")
def health():
    return {"ready": sorted(M)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8310)
    a = ap.parse_args()
    load()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
