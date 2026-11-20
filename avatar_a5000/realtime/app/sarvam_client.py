"""Async Sarvam AI client: streaming chat (LLM), text-to-speech (bulbul:v3), speech-to-text (saarika)."""
import base64
import io
import json
import re
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

API = "https://api.sarvam.ai"
ENV = Path(__file__).resolve().parents[1] / ".env"  # Aaryan/avatar/.env (SARVAM_API_KEY)


def api_key():
    for line in ENV.read_text().splitlines():
        if line.split("=", 1)[0].strip().lower() == "sarvam_api_key":
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"SARVAM_API_KEY missing in {ENV}")


SENT_END = re.compile(r"(?<=[.!?।])\s+|\n+")


def clean_for_speech(text):
    text = re.sub(r"[*_#`>|\[\]]", " ", text)                  # markdown
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)  # emoji
    return re.sub(r"\s+", " ", text).strip()


class Sarvam:
    def __init__(self, llm_model="sarvam-105b-conversations", tts_model="bulbul:v3"):
        self.key = api_key()
        self.llm_model, self.tts_model = llm_model, tts_model
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0),
                                      headers={"api-subscription-key": self.key})

    async def chat_stream(self, messages, max_tokens=300, temperature=0.6):
        """Yield text deltas from the LLM."""
        body = {"model": self.llm_model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True}
        async with self.http.stream("POST", f"{API}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"chat {r.status_code}: {(await r.aread()).decode()[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]  # final usage chunk has no choices
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta

    async def sentences(self, messages, min_chars=12):
        """Group the LLM stream into speakable sentences. The very first chunk may also end at a comma, so the
        first TTS request is short (faster first audio)."""
        buf, full, first = "", "", True
        async for d in self.chat_stream(messages):
            buf += d
            full += d
            if first:
                m = re.search(r"^(.{15,}?[,،])\s", buf)
                if m and not SENT_END.search(m.group(1)):
                    yield clean_for_speech(m.group(1))
                    buf, first = buf[m.end():], False
                    continue
            parts = SENT_END.split(buf)
            if len(parts) > 1:
                first = False
            for p in parts[:-1]:
                p = clean_for_speech(p)
                if len(p) >= min_chars:
                    yield p
                elif p:
                    parts[-1] = p + " " + parts[-1]
            buf = parts[-1]
        if clean_for_speech(buf):
            yield clean_for_speech(buf)

    async def tts(self, text, speaker="aditya", lang="hi-IN", pace=1.0):
        """-> float32 mono 16 kHz numpy array. Uses the streaming endpoint (measured ~2.1 s total vs 2-4 s REST)."""
        body = {"text": text, "target_language_code": lang, "model": self.tts_model, "speaker": speaker,
                "speech_sample_rate": 16000, "pace": pace, "output_audio_codec": "wav"}
        r = await self.http.post(f"{API}/text-to-speech/stream", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"tts {r.status_code}: {r.text[:300]}")
        raw = r.content
        i = raw.find(b"data")  # streamed WAV headers may carry a bogus length -> read PCM after the data chunk
        if raw[:4] != b"RIFF" or i < 0:
            raise RuntimeError(f"tts: unexpected payload {raw[:16]!r}")
        sr = int.from_bytes(raw[24:28], "little")
        pcm = np.frombuffer(raw[i + 8:len(raw) - (len(raw) - i - 8) % 2], dtype=np.int16).astype(np.float32) / 32768
        assert sr == 16000, sr
        return pcm

    async def stt(self, wav_bytes, lang="unknown"):
        """wav bytes -> (transcript, language_code)."""
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        r = await self.http.post(f"{API}/speech-to-text", files=files,
                                 data={"model": "saarika:v2.5", "language_code": lang})
        if r.status_code != 200:
            raise RuntimeError(f"stt {r.status_code}: {r.text[:300]}")
        j = r.json()
        return j.get("transcript", ""), j.get("language_code")
