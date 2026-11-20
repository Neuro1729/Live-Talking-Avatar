"""TTS providers behind one interface. Every provider yields AudioEvents (PCM float32 16 kHz mono, sample offsets,
end-of-utterance flag); codecs are decoded here at the boundary. Credentials come from the existing .env reader
(app/sarvam_client.api_key) and are never logged.

  SarvamTTS          existing integration: bulbul:v3 via /text-to-speech/stream, whole utterance in one event
  SarvamStreamingTTS same endpoint read incrementally: PCM chunks are emitted as they arrive
  SarvamWSTTS        bulbul:v3 over the TTS WebSocket (wss /text-to-speech/ws, protocol from the official sarvamai
                     SDK); prepare() opens + configures the connection while the user is still talking, sentences
                     reuse it in order -> first audio ~0.4 s after the sentence instead of ~1 s
  MockTTS            deterministic synthetic speech-like audio (tests, no API key)
  WavTTS             prerecorded WAVs (evaluation)
"""
import sys
import time

import numpy as np

from avatar_system.common import root
from avatar_system.lipsync.audio_context import SR, AudioEvent

sys.path.insert(0, str(root() / "app"))


class TTSProvider:
    name = "base"

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh"):
        raise NotImplementedError
        yield  # pragma: no cover


class SarvamTTS(TTSProvider):
    name = "sarvam"

    def __init__(self, client=None, pace=1.0):
        from sarvam_client import Sarvam
        self.client = client or Sarvam()
        self.pace = pace

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh"):
        t0 = time.time()
        pcm = await self.client.tts(text, speaker=speaker, lang=lang, pace=self.pace)
        yield AudioEvent(session_id, utterance_id, SR, 0, pcm, True,
                         {"tts_first_chunk_s": round(time.time() - t0, 3), "tts_done_s": round(time.time() - t0, 3)})


class SarvamStreamingTTS(SarvamTTS):
    name = "sarvam-stream"

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh", min_chunk=0.25):
        from sarvam_client import API
        body = {"text": text, "target_language_code": lang, "model": self.client.tts_model, "speaker": speaker,
                "speech_sample_rate": SR, "pace": self.pace, "output_audio_codec": "wav"}
        t0, first = time.time(), None
        buf, header_done, off = b"", False, 0
        async with self.client.http.stream("POST", f"{API}/text-to-speech/stream", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"tts {r.status_code}: {(await r.aread()).decode()[:200]}")
            async for chunk in r.aiter_bytes():
                buf += chunk
                if not header_done:
                    i = buf.find(b"data")
                    if buf[:4] != b"RIFF" or i < 0 or len(buf) < i + 8:
                        continue
                    sr = int.from_bytes(buf[24:28], "little")
                    if sr != SR:
                        raise RuntimeError(f"unexpected sample rate {sr}")
                    buf, header_done = buf[i + 8:], True
                usable = len(buf) - len(buf) % 2
                if usable >= int(min_chunk * SR) * 2:
                    pcm = np.frombuffer(buf[:usable], np.int16).astype(np.float32) / 32768
                    buf = buf[usable:]
                    first = first or time.time() - t0
                    yield AudioEvent(session_id, utterance_id, SR, off, pcm, False, {"tts_first_chunk_s": round(first, 3)})
                    off += len(pcm)
        pcm = np.frombuffer(buf[: len(buf) - len(buf) % 2], np.int16).astype(np.float32) / 32768
        yield AudioEvent(session_id, utterance_id, SR, off, pcm, True,
                         {"tts_first_chunk_s": round(first or time.time() - t0, 3), "tts_done_s": round(time.time() - t0, 3)})


class SarvamWSTTS(TTSProvider):
    name = "sarvam-ws"

    def __init__(self, pace=1.0, model="bulbul:v3", min_buffer_size=30):
        self.pace, self.model, self.min_buffer_size = pace, model, min_buffer_size
        self.ws, self.key = None, None          # open connection and its (lang, speaker) config
        self.lock = None                        # sentences of a turn share the socket, in order

    async def prepare(self, lang="hi-IN", speaker="shubh"):
        """Open and configure the socket ahead of the first sentence (call when the user starts speaking)."""
        import asyncio
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            await self._ensure(lang, speaker)

    async def _ensure(self, lang, speaker):
        import json
        import websockets
        from sarvam_client import api_key
        if self.ws is not None and self.key == (lang, speaker) and self.ws.state.name == "OPEN":
            return
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        url = f"wss://api.sarvam.ai/text-to-speech/ws?model={self.model}&send_completion_event=true"
        self.ws = await websockets.connect(url, additional_headers={"Api-Subscription-Key": api_key()}, open_timeout=5,
                                           max_size=None)
        await self.ws.send(json.dumps({"type": "config", "data": {
            "language_code": lang, "speaker": speaker, "speech_sample_rate": SR, "output_audio_codec": "linear16",
            "min_buffer_size": self.min_buffer_size, "pace": self.pace}}))
        self.key = (lang, speaker)

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh"):
        import asyncio
        import base64
        import json
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            for attempt in range(2):  # a socket the server closed while idle is reopened once
                try:
                    await self._ensure(lang, speaker)
                    t0, first, off = time.time(), None, 0
                    await self.ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                    await self.ws.send(json.dumps({"type": "flush"}))
                    break
                except Exception:
                    self.ws = None
                    if attempt:
                        raise
            while True:
                m = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15))
                if m.get("type") == "audio":
                    raw = base64.b64decode(m["data"]["audio"])
                    pcm = np.frombuffer(raw[: len(raw) - len(raw) % 2], np.int16).astype(np.float32) / 32768
                    first = first or time.time() - t0
                    yield AudioEvent(session_id, utterance_id, SR, off, pcm, False,
                                     {"tts_first_chunk_s": round(first, 3)})
                    off += len(pcm)
                elif m.get("type") == "event":  # completion of this sentence
                    break
                elif m.get("type") == "error":
                    self.ws = None
                    raise RuntimeError(f"tts ws: {str(m.get('data'))[:200]}")
        yield AudioEvent(session_id, utterance_id, SR, off, np.zeros(0, np.float32), True,
                         {"tts_first_chunk_s": round(first or time.time() - t0, 3), "tts_done_s": round(time.time() - t0, 3)})


class MockTTS(TTSProvider):
    name = "mock"

    def __init__(self, chars_per_s=14, chunk_s=None):
        self.cps, self.chunk_s = chars_per_s, chunk_s

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh"):
        n = int(max(len(text) / self.cps, 0.4) * SR)
        t = np.arange(n) / SR
        env = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t)) * (np.sin(np.pi * t / t[-1]) if n > 1 else 1)
        pcm = (0.2 * env * np.sin(2 * np.pi * 140 * t)).astype(np.float32)
        step = int((self.chunk_s or 1e9) * SR)
        for o in range(0, n, step):
            yield AudioEvent(session_id, utterance_id, SR, o, pcm[o:o + step], o + step >= n, {"tts_first_chunk_s": 0.0})


class WavTTS(TTSProvider):
    name = "wav"

    def __init__(self, mapping):
        self.mapping = mapping  # text -> wav path

    async def synthesize(self, text, session_id, utterance_id, lang="hi-IN", speaker="shubh"):
        import soundfile as sf
        pcm, sr = sf.read(self.mapping[text], dtype="float32")
        assert sr == SR
        yield AudioEvent(session_id, utterance_id, SR, 0, pcm, True, {"tts_first_chunk_s": 0.0})


def make_tts(kind):
    return {"sarvam": SarvamTTS, "sarvam-stream": SarvamStreamingTTS, "sarvam-ws": SarvamWSTTS, "mock": MockTTS}[kind]()
