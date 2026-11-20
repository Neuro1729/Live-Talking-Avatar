"""Persistent avatar server (FastAPI). One avatar playback session runs continuously from start-up; browsers attach
to it over one WebSocket that is never recreated per answer.

Transport: WebSocket binary messages = header(<IIII: frame n, pcm bytes, meta bytes, jpeg bytes>) + PCM16 (40 ms) +
meta JSON + JPEG. Audio and video of frame n travel in the same message, so they cannot drift. WebRTC (aiortc) would
plug in as another transport reading pipeline.read(); it is not used here because the demo is reached through an SSH
TCP tunnel, where WebRTC's UDP media path does not work without a TURN server.

  cd R && CUDA_VISIBLE_DEVICES=3 envs/musetalk/bin/python avatar_system/streaming/server.py [--mock] [--tts sarvam]
  then: ssh -L 8780:127.0.0.1:8780 vishal@<gpu28>  and open http://localhost:8780
"""
import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, load_config, path  # noqa: E402
from avatar_system.conversation.engine import ConversationEngine  # noqa: E402
from avatar_system.conversation.state_machine import StateMachine  # noqa: E402
from avatar_system.lipsync.base import NullLipSync  # noqa: E402
from avatar_system.streaming.pipeline import FPS, AvatarPipeline  # noqa: E402

app = FastAPI()
S = {"pipe": None, "engine": None, "clients": set(), "rec": None, "args": None, "graph": None,
     "mic": bytearray(), "stt": None}  # mic: user speech (int16 16 kHz) still to be laid into the recording


def rec_event(kind, text="", **extra):
    """Caption events for recordings: the frame number being output when the event happened."""
    if S["rec"] is not None:
        S["rec"]["events"].append({"n": S["pipe"].last_meta.get("n", 0), "t": time.time(), "type": kind,
                                   "text": text, **extra})


async def broadcast(msg):
    if msg.get("type") in ("user", "assistant", "latency", "behavior", "ack"):
        rec_event(msg["type"], msg.get("text", ""), **{k: v for k, v in msg.items() if k in ("first_audible_s", "turn",
                                                                                               "hints")})
    data = json.dumps(msg, ensure_ascii=False)
    for ws in list(S["clients"]):
        try:
            await ws.send_text(data)
        except Exception:
            S["clients"].discard(ws)


async def frame_pump():
    """Paced 25 fps consumer of the pipeline; the pacing clock is reset (and counted) if the renderer falls behind."""
    loop = asyncio.get_running_loop()
    pipe, eng = S["pipe"], S["engine"]
    t0, k, last = time.time(), 0, time.time()
    resets = 0
    while True:
        item = await loop.run_in_executor(None, pipe.read, 1.0)
        if item is None:
            if pipe.error:
                await broadcast({"type": "error", "text": pipe.error})
                await asyncio.sleep(1)
            continue
        jpg, pcm, meta = item
        lat = eng.on_frame_out(meta)
        if lat:
            await broadcast({"type": "latency", **{k2: v for k2, v in lat.items() if k2 != "t0"}})
        mb = json.dumps(meta).encode()
        msg = struct.pack("<IIII", meta["n"], len(pcm), len(mb), len(jpg)) + pcm + mb + jpg
        for ws in list(S["clients"]):
            try:
                await ws.send_bytes(msg)
            except Exception:
                S["clients"].discard(ws)
        if S["mic"]:  # the user's voice (/api/voice) plays in real time alongside the avatar's audio
            import numpy as np
            take = bytes(S["mic"][:len(pcm)])
            del S["mic"][:len(pcm)]
            a = np.frombuffer(pcm, np.int16).astype(np.int32)
            u = np.frombuffer(take, np.int16).astype(np.int32)
            a[:len(u)] += u
            pcm_rec = np.clip(a, -32768, 32767).astype(np.int16).tobytes()
        else:
            pcm_rec = pcm
        if S["rec"]:
            S["rec"].setdefault("n_first", meta["n"])
            S["rec"]["video"].stdin.write(jpg)
            S["rec"]["pcm"].append(pcm_rec)
        now = time.time()
        pipe.stats["out_interval_ms"].add((now - last) * 1000)
        last = now
        k += 1
        delay = t0 + k / FPS - now
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.5:
            resets += 1
            pipe.counters["pacing_resets"] = resets
            t0, k = now, 0


@app.on_event("startup")
async def _startup():
    asyncio.create_task(frame_pump())
    if S["args"] is not None and S["args"].ack_delay > 0 and S["args"].tts != "mock":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
        from sarvam_client import Sarvam
        asyncio.create_task(S["engine"].prepare_acks(Sarvam(), S["engine"].speaker))


@app.get("/")
async def index():
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S["clients"].add(ws)
    eng = S["engine"]
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "ask" and msg.get("text", "").strip():
                asyncio.create_task(eng.ask(msg["text"].strip(), msg.get("lang", "hi-IN")))
            elif t == "interrupt":
                eng.interrupt()
            elif t == "user_speaking":
                eng.user_speaking()
    except WebSocketDisconnect:
        pass
    finally:
        S["clients"].discard(ws)


@app.post("/api/message")
async def api_message(body: dict):
    asyncio.create_task(S["engine"].ask(body["text"], body.get("lang", "hi-IN")))
    return {"ok": True}


async def local_stt(wav_b64, lang):
    import httpx
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0)) as c:
        r = await c.post(f"{S['args'].stt_url}/stt", json={"wav_b64": wav_b64, "lang": lang})
        r.raise_for_status()
        return r.json()["text"], lang


async def stream_stt(x, lang, dur, chunk_s=0.25):
    """Sarvam streaming speech-to-text (wss /speech-to-text/ws, saaras:v3; protocol from the official sarvamai SDK):
    int16 16 kHz audio is sent in real time as the user speaks, then a flush forces the final transcript."""
    import base64
    import urllib.parse
    import websockets
    from sarvam_client import api_key
    q = {"language-code": lang, "model": "saaras:v3", "mode": "transcribe", "sample_rate": "16000",
         "input_audio_codec": "pcm_s16le", "flush_signal": "true", "vad_signals": "true"}
    url = "wss://api.sarvam.ai/speech-to-text/ws?" + urllib.parse.urlencode(q)
    t0 = time.time()
    parts, det = [], None
    async with websockets.connect(url, additional_headers={"Api-Subscription-Key": api_key()}, open_timeout=5) as ws:
        got = asyncio.Queue()  # (arrival time, message), read continuously so nothing goes stale

        async def reader():
            async for raw in ws:
                await got.put((time.time(), json.loads(raw)))
        rd = asyncio.create_task(reader())
        try:
            n = int(chunk_s * 16000)
            for i in range(0, len(x), n):  # backlog from the connect time goes out at once, then real time
                await ws.send(json.dumps({"audio": {"data": base64.b64encode(x[i:i + n].tobytes()).decode(),
                                                    "sample_rate": 16000, "encoding": "audio/wav"}}))
                delay = t0 + (i + n) / 16000 - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            await asyncio.sleep(max(0.0, t0 + dur - time.time()))
            t_flush = time.time()
            await ws.send(json.dumps({"type": "flush"}))
            while True:
                try:
                    t, m = await asyncio.wait_for(got.get(), timeout=4.0)
                except asyncio.TimeoutError:
                    break
                if m.get("type") == "error":
                    raise RuntimeError(str(m.get("data"))[:200])
                if m.get("type") == "data":
                    parts.append(m["data"].get("transcript", ""))
                    det = m["data"].get("language_code")
                    if t > t_flush:  # answer to the flush: everything said is now final
                        break
        finally:
            rd.cancel()
    if not parts:
        raise RuntimeError("no transcript")
    return " ".join(p for p in parts if p).strip(), det


@app.post("/api/voice")
async def api_voice(body: dict):
    """Spoken user turn: {"wav_b64": 16 kHz mono wav, "lang": "hi-IN"}. The audio plays in real time (avatar listening,
    user voice in the recording); when it ends, Sarvam speech-to-text (saarika) runs and its transcript is asked like
    a typed message - every gap in the conversation is real processing time."""
    import base64
    import io
    import numpy as np
    import soundfile as sf
    wav = base64.b64decode(body["wav_b64"])
    x, sr = sf.read(io.BytesIO(wav), dtype="int16")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != 16000:
        return JSONResponse({"error": f"need 16 kHz audio, got {sr}"}, status_code=400)
    lang = body.get("lang", "hi-IN")
    S["engine"].user_speaking()
    tts = S["engine"].tts
    if hasattr(tts, "prepare"):  # open the TTS socket while the user is still talking
        asyncio.create_task(tts.prepare(lang, S["engine"].speaker))
    S["mic"].extend(x.astype(np.int16).tobytes())
    dur = len(x) / sr
    rec_event("user_audio", "", duration_s=round(dur, 2))

    async def finish():
        if S["stt"] is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
            from sarvam_client import Sarvam
            S["stt"] = Sarvam()
        t_end = time.time() + dur  # when the user stops talking
        mode = S["args"].stt
        try:
            if mode == "local":  # local Whisper service (musetalk_ft/scripts/stt_server.py) on a spare GPU
                await asyncio.sleep(max(0.0, t_end - time.time()))
                text, det = await local_stt(body["wav_b64"], lang)
            else:  # Sarvam streaming STT runs WHILE the user talks; only the last bit is left when they stop
                text, det = await stream_stt(x, lang, dur)
        except Exception as e:
            print(f"[voice] streaming STT failed ({e!r}); batch fallback", flush=True)
            mode = "batch"
            await asyncio.sleep(max(0.0, t_end - time.time()))
            text, det = await S["stt"].stt(wav, lang)
        stt_s = round(time.time() - t_end, 2)
        rec_event("stt", text, stt_s=stt_s, detected=det, mode=mode)
        print(f"[voice] {dur:.1f}s audio -> STT ({mode}) {stt_s}s after speech end: {text}", flush=True)
        if text.strip():
            await S["engine"].ask(text.strip(), lang)
        else:
            S["engine"].voice_turn_failed()
    if S["args"].ack_delay > 0:  # acknowledgement this long after the user stops, unless the answer is already there
        S["engine"].expect_answer(lang, time.time() + dur + S["args"].ack_delay)
    asyncio.create_task(finish())
    return {"ok": True, "duration_s": round(dur, 2)}


@app.post("/api/interrupt")
async def api_interrupt():
    rec_event("interrupt")
    return {"cancelled": S["engine"].interrupt()}


@app.post("/api/user_speaking")
async def api_user_speaking():
    S["engine"].user_speaking()
    return {"state": S["pipe"].sm.state}


@app.get("/api/state")
async def api_state():
    return S["pipe"].sm.snapshot() | {"current": S["pipe"].last_meta, "plan": S["pipe"].sched.upcoming()}


@app.get("/api/metrics")
async def api_metrics():
    m = S["pipe"].metrics()
    m["latency_turns"] = [{k: v for k, v in x.items() if k != "t0"} for x in S["engine"].latency_log[-20:]]
    m["clients"] = len(S["clients"])
    m["backend"] = S["pipe"].backend.name
    try:
        import torch
        m["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    except Exception:
        m["gpu_peak_mem_gb"] = None
    import resource
    m["rss_peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
    return JSONResponse(json.loads(json.dumps(m, default=str)))


@app.get("/api/clips")
async def api_clips():
    g = S["graph"]
    return {"graph_version": g["version"], "states": g["states"],
            "segments": [{k: s[k] for k in ("id", "key", "clip", "start_state", "end_state", "conv", "quality")}
                         for s in g["segments"] if s["id"] in set(g["runtime_segments"])]}


@app.get("/api/avatar")
async def api_avatar():
    return {"image": str(path("avatar_image")), "graph_version": S["graph"]["version"],
            "clips": len({s["clip"] for s in S["graph"]["segments"]}), "backend": S["pipe"].backend.name}


@app.post("/api/record/start")
async def rec_start():
    out = data_dir("recordings") / time.strftime("rec_%Y%m%d_%H%M%S")
    p = subprocess.Popen([str(path("ffmpeg")), "-y", "-loglevel", "error", "-f", "mjpeg", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out) + "_v.mp4"],
                         stdin=subprocess.PIPE)
    S["rec"] = {"video": p, "pcm": [], "out": out, "n0": len(S["pipe"].trace), "events": []}
    return {"recording": str(out)}


@app.post("/api/record/stop")
async def rec_stop():
    r, S["rec"] = S["rec"], None
    if not r:
        return {"error": "not recording"}
    r["video"].stdin.close()
    r["video"].wait()
    import numpy as np
    import soundfile as sf
    sf.write(str(r["out"]) + ".wav", np.frombuffer(b"".join(r["pcm"]), np.int16), 16000)
    subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", str(r["out"]) + "_v.mp4", "-i",
                    str(r["out"]) + ".wav", "-c:v", "copy", "-c:a", "aac", str(r["out"]) + ".mp4"], check=True)
    Path(str(r["out"]) + "_v.mp4").unlink()
    trace = list(S["pipe"].trace)[r["n0"]:]
    Path(str(r["out"]) + "_trace.json").write_text(json.dumps(
        {"fps": FPS, "frames": [t[:9] for t in trace], "utt": [t[9] for t in trace],
         "graph_version": S["graph"]["version"], "seams": [list(s) for s in S["pipe"].sched.seams],
         "latency": [{k: v for k, v in x.items() if k != "t0"} for x in S["engine"].latency_log],
         "n_first": r.get("n_first", 0), "events": r["events"]}, default=str, ensure_ascii=False))
    return {"video": str(r["out"]) + ".mp4"}


@app.get("/api/trace")
async def api_trace(last: int = 500):
    return {"frames": list(S["pipe"].trace)[-last:], "seams": [list(s) for s in S["pipe"].sched.seams[-50:]]}


def talkbank_scheduler(args):
    """Body motion from the talking-bank motion graph (data/talk_graph.json, motion/talk_graph.py) instead of the
    silent-clip motion graph. --talkbank all = every clip in the graph, else a comma list (subset of the graph)."""
    from avatar_system.motion.talkbank_scheduler import TalkBankScheduler
    from avatar_system.streaming.frame_source import ClipFrameSource
    idx = json.loads((data_dir() / "talk_index.json").read_text())
    tg = json.loads((data_dir() / "talk_graph.json").read_text())
    names = sorted(tg["clips"]) if args.talkbank == "all" else args.talkbank.split(",")
    missing = [n for n in names if n not in tg["clips"]]
    if missing:
        raise SystemExit(f"not in talk_graph.json (python -m avatar_system.motion.talk_graph): {missing}")
    src = ClipFrameSource()
    sched = TalkBankScheduler({n: idx[n] for n in names}, {n: src.length(n) for n in names})
    # one placeholder segment: the pipeline's frame metadata / API endpoints expect FrameRef.seg to index a segment
    seg = {"id": 0, "key": "talkbank", "clip": "talkbank", "start_state": "home", "end_state": "home",
           "conv": ["talk"], "quality": 1.0}
    graph = {"version": f"talkgraph:{tg['built']}:{len(names)}", "states": ["home"], "hub": "home",
             "segments": [seg], "runtime_segments": [0]}
    print(f"[server] talking-bank graph {tg['built']}: {len(sched.clips)} clips, {len(sched.retr.parts)} speech parts, "
          f"{sum(len(v) for v in sched.adj.values())} joins in use ({tg['status']['approved']} approved)", flush=True)
    return graph, sched.clips, sched


def build(args):
    sched = None
    if args.talkbank:
        graph, clips, sched = talkbank_scheduler(args)
    else:
        graph = json.loads((data_dir() / args.graph).read_text())
        if not graph.get("validation_ok", False):
            raise SystemExit("motion graph not validated: run graph_validator first")
        clips = sorted({s["clip"] for i, s in enumerate(graph["segments"]) if i in set(graph["runtime_segments"])})
    if args.mock:
        backend = NullLipSync()
    else:
        from avatar_system.lipsync.musetalk_adapter import MuseTalkBackend
        missing = [c for c in clips if not (path("musetalk_cache") / c / "DONE").exists()]
        if missing:
            raise SystemExit(f"MuseTalk caches missing for {missing}: run scripts/prep_avatar.py first")
        backend = MuseTalkBackend(clips)
        print(f"[server] MuseTalk loaded {backend.load_s:.1f}s, caches {backend.cache_s:.1f}s", flush=True)
    sm = StateMachine()
    cfg = load_config("musetalk")
    talk_frames = None
    if not args.talkbank:  # talking-bank clips inside the motion graph: which of their frames are recorded speech
        from avatar_system.motion.graph_builder import speech_frames
        from avatar_system.streaming.frame_source import ClipFrameSource
        tidx = json.loads((data_dir() / "talk_index.json").read_text())
        src_len = ClipFrameSource()
        talk_frames = {c: speech_frames(tidx, c, src_len.length(c)) for c in clips if c in tidx}
    pipe = AvatarPipeline(graph, backend, sm, batch=args.batch, out_width=args.width, silence=cfg["silence"],
                          feature_shift=cfg.get("feature_shift_frames", 0),
                          playback_buffer=args.buffer, scheduler=sched, talk_frames=talk_frames)
    if args.tts == "mock" or args.llm == "mock":
        from avatar_system.conversation.llm_adapter import MockLLM
        from avatar_system.conversation.tts_interface import MockTTS
    if args.llm == "mock":
        llm = MockLLM()
    else:
        from avatar_system.conversation.llm_adapter import SarvamLLM
        client = None
        if args.llm == "local":  # e.g. Sarvam-30B FP8 on vLLM (OpenAI-compatible) on a spare GPU
            from avatar_system.conversation.llm_adapter import local_client
            client = local_client(args.llm_url, args.llm_model)
        llm = SarvamLLM(client=client, behavior_tags=not args.no_behavior_tags)
    if args.tts == "mock":
        tts = MockTTS()
    else:
        from avatar_system.conversation.tts_interface import make_tts
        tts = make_tts(args.tts)
    S.update(pipe=pipe, graph=graph, engine=ConversationEngine(pipe, llm, tts, broadcast, speaker=args.speaker))
    pipe.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--stt", default="sarvam-stream", choices=["sarvam-stream", "local"],
                    help="speech-to-text for /api/voice; local = musetalk_ft/scripts/stt_server.py")
    ap.add_argument("--stt-url", default="http://127.0.0.1:8310")
    ap.add_argument("--ack-delay", type=float, default=0.0, help="s after a spoken question ends before a short "
                    "acknowledgement (जी…/हम्म…) if the answer has not started yet; 0 = off")
    ap.add_argument("--graph", default="motion_graph.json", help="motion graph file in data/ (e.g. the silent-bank "
                    "snapshot motion_graph_silent_7f1cea162602.json used for live_conversation_captioned)")
    ap.add_argument("--mock", action="store_true", help="no MuseTalk (source frames only)")
    ap.add_argument("--tts", default="sarvam", choices=["sarvam", "sarvam-stream", "sarvam-ws", "mock"])
    ap.add_argument("--llm", default="sarvam", choices=["sarvam", "local", "mock"])
    ap.add_argument("--llm-url", default="http://127.0.0.1:8300", help="--llm local: OpenAI-compatible server")
    ap.add_argument("--llm-model", default="sarvam-30b", help="--llm local: served model name")
    ap.add_argument("--no-behavior-tags", action="store_true")
    ap.add_argument("--speaker", default="shubh")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=4, help="playback buffer (frames)")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--talkbank", default="", help="'all' or a comma list of talking-bank clips: body motion "
                    "planned over the talking-bank graph (data/talk_graph.json) instead of the motion graph")
    a = ap.parse_args()
    S["args"] = a
    build(a)
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    finally:
        S["pipe"].stop()


if __name__ == "__main__":
    main()
