"""Real-time talking avatar server.

Browser <-WebSocket-> server: server pushes 25 fps JPEG frames + matching 40 ms PCM audio; browser sends questions
(text, or mic audio via POST /stt). Pipeline per question: Sarvam LLM (streamed, split into sentences) -> Sarvam TTS
per sentence (parallel, played in order) -> MuseTalk lip-sync on the currently playing base clip.

Run (musetalk env, from realtime/):
  CUDA_VISIBLE_DEVICES=3 envs/musetalk/bin/python app/server.py --port 8765
Then on your PC:  ssh -L 8765:127.0.0.1:8765 vishal@10.12.0.213   and open http://localhost:8765
"""
import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_engine import FPS, AvatarEngine, R  # noqa: E402
from sarvam_client import Sarvam  # noqa: E402

FF = "/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/bin/ffmpeg"
SYSTEM = {
    "hi-IN": "You are a friendly, warm Indian digital assistant who talks to people as a video avatar of a man. "
             "You are male: always use masculine Hindi verb forms about yourself (e.g. करता हूँ, सकता हूँ, रहा हूँ). "
             "You are speaking out loud, so answer in 1-3 short, natural, conversational sentences. "
             "ALWAYS reply in Hindi written in Devanagari script (common English words like phone, doctor may be "
             "written in Devanagari too). Never use Roman/Latin script for Hindi. "
             "Start with a short first sentence. No lists, markdown, emojis or bullet points.",
    "en-IN": "You are a friendly, warm Indian digital assistant speaking to people as a video avatar of a man. "
             "You are speaking out loud: answer in 1-3 short, natural spoken sentences in simple Indian English. "
             "Start with a short first sentence. No lists, markdown, emojis or bullet points.",
}

app = FastAPI()
engine: AvatarEngine = None
sarvam: Sarvam = None
clients: set = set()
history: list = []
turn_id = 0


async def broadcast_json(msg):
    for ws in list(clients):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            clients.discard(ws)


async def frame_pump():
    """Take frames from the engine and send them to all clients at a steady 25 fps."""
    loop = asyncio.get_running_loop()
    k, t0 = 0, time.time()
    while True:
        jpg, pcm, meta = await loop.run_in_executor(None, engine.out_q.get)
        header = struct.pack("<II", k, len(pcm))
        msg = header + pcm + jpg
        for ws in list(clients):
            try:
                await ws.send_bytes(msg)
            except Exception:
                clients.discard(ws)
        k += 1
        delay = t0 + k / FPS - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.5:  # fell behind (engine slower than real time) -> reset the pacing clock
            t0, k0 = time.time(), k
            t0 -= k0 / FPS


async def answer(text, lang, speaker):
    global turn_id
    turn_id += 1
    my_turn = turn_id
    engine.clear_speech()
    engine.hold = True
    engine.set_mode("think")
    history.append({"role": "user", "content": text})
    msgs = [{"role": "system", "content": SYSTEM.get(lang, SYSTEM["hi-IN"])}] + history[-10:]
    await broadcast_json({"type": "user", "text": text})
    t_start = time.time()
    loop = asyncio.get_running_loop()
    tts_q: asyncio.Queue = asyncio.Queue()
    full = []

    async def feeder():
        first = True
        while True:
            task = await tts_q.get()
            if task is None or my_turn != turn_id:
                return
            try:
                wav = await task
            except Exception as e:
                await broadcast_json({"type": "error", "text": f"TTS: {e}"})
                continue
            if my_turn != turn_id:
                return
            await loop.run_in_executor(None, engine.add_speech, wav)
            if first:
                first = False
                await broadcast_json({"type": "latency", "first_audio_s": round(time.time() - t_start, 2)})

    feed = asyncio.create_task(feeder())
    try:
        async for sent in sarvam.sentences(msgs):
            if my_turn != turn_id:
                break
            full.append(sent)
            await tts_q.put(asyncio.create_task(sarvam.tts(sent, speaker=speaker, lang=lang)))
            await broadcast_json({"type": "assistant", "text": sent})
    except Exception as e:
        await broadcast_json({"type": "error", "text": f"LLM: {e}"})
    await tts_q.put(None)
    await feed
    if my_turn == turn_id:
        engine.hold = False
        while engine.speech:  # wait until the last buffered word has been rendered
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.6)  # ~frames still in the output queue
        await broadcast_json({"type": "done"})
    if full:
        history.append({"role": "assistant", "content": " ".join(full)})


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/stt")
async def stt(file: UploadFile, lang: str = "unknown"):
    raw = await file.read()
    wav = subprocess.run([FF, "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                         input=raw, capture_output=True).stdout
    if len(wav) < 16000:  # < 0.5 s
        return JSONResponse({"text": ""})
    try:
        text, code = await sarvam.stt(wav, lang)
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)})
    return JSONResponse({"text": text, "language_code": code})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "ask" and msg.get("text", "").strip():
                asyncio.create_task(answer(msg["text"].strip(), msg.get("lang", "hi-IN"), msg.get("speaker", "shubh")))
            elif msg.get("type") == "stop":
                global turn_id
                turn_id += 1
                engine.hold = False
                engine.clear_speech()
            elif msg.get("type") == "listening":
                engine.set_mode("listen")
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.on_event("startup")
async def startup():
    asyncio.create_task(frame_pump())


def main():
    global engine, sarvam
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--clips", default="", help="comma list; default = every prepared clip in avatar_cache/")
    ap.add_argument("--out-width", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--exclude", default="bank_talkslow_s11,bank_talkslow_s13",
                    help="clips rejected in review (sleepy eyes / big head tilts)")
    a = ap.parse_args()
    names = a.clips.split(",") if a.clips else sorted(p.parent.name for p in (R / "avatar_cache").glob("*/DONE"))
    names = [n for n in names if n not in a.exclude.split(",")]
    engine = AvatarEngine(names, batch=a.batch, out_width=a.out_width)
    sarvam = Sarvam()
    engine.start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
