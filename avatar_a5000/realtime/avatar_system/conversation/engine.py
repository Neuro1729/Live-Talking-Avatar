"""Conversation orchestration: user turn -> THINKING -> LLM sentences -> TTS (parallel, played in order) -> audio
timeline -> SPEAKING -> LISTENING. Each turn owns a CancelToken; a new message or an interrupt cancels the previous
turn, and every await point checks the token, so stale LLM/TTS output can never be pushed after cancellation (the
audio timeline additionally drops chunks of cancelled utterances)."""
import asyncio
import itertools
import time

from avatar_system.lipsync.audio_context import FPS, SR, AudioEvent


class CancelToken:
    def __init__(self, turn):
        self.turn, self.cancelled = turn, False

    def cancel(self):
        self.cancelled = True


class ConversationEngine:
    def __init__(self, pipeline, llm, tts, broadcast, session_id="main", speaker="shubh"):
        self.pipe, self.llm, self.tts, self.broadcast = pipeline, llm, tts, broadcast
        self.session_id, self.speaker = session_id, speaker
        self.history = []
        self.token = None
        self.turns = itertools.count(1)
        self.latency_log = []                 # one dict per turn
        self.first_audible = {}               # utterance id -> callback when its first frame is output
        # spoken acknowledgement ("जी…") played ~1 s after the user stops if the answer has not started by then
        self.acks = {}                        # lang -> [(text, pcm float32 16 kHz)], filled by prepare_acks()
        self.ack_count = itertools.count(1)
        self.answer_started = True            # the current voice turn's answer audio already went to lip-sync
        self._push_lock = None
        self.ack_log = []                     # one dict per acknowledgement (for latency reports)

    ACK_TEXTS = {"hi-IN": ["जी…", "हम्म…", "अच्छा…"], "en-IN": ["Sure…", "Hmm…", "Okay…"]}

    async def prepare_acks(self, tts_client, speaker):
        """Synthesize the acknowledgement clips once (same voice as the answers), trimmed to the spoken part."""
        import numpy as np
        for lang, texts in self.ACK_TEXTS.items():
            out = []
            for t in texts:
                try:
                    pcm = np.asarray(await tts_client.tts(t, speaker=speaker, lang=lang), np.float32)
                except Exception as e:
                    print(f"[ack] {lang} {t}: {e}", flush=True)
                    continue
                env = np.abs(pcm) > 0.02
                if env.any():
                    a, b = np.argmax(env), len(env) - np.argmax(env[::-1])
                    pcm = pcm[max(0, a - SR // 50): min(len(pcm), b + SR // 10)]
                out.append((t, pcm))
            self.acks[lang] = out
        print("[ack] ready: " + ", ".join(f"{k}: {[f'{t} {len(p) / SR:.2f}s' for t, p in v]}" for k, v in self.acks.items()),
              flush=True)

    def _lock(self):
        if self._push_lock is None:
            self._push_lock = asyncio.Lock()
        return self._push_lock

    def expect_answer(self, lang, ack_at):
        """A spoken question ended: schedule the acknowledgement for `ack_at` (epoch s) unless the answer is faster."""
        self.answer_started = False
        if self.acks.get(lang) or self.acks.get("hi-IN"):
            asyncio.create_task(self._ack(lang, ack_at))

    async def _ack(self, lang, at):
        await asyncio.sleep(max(0.0, at - time.time()))
        async with self._lock():
            if self.answer_started:
                return
            choices = self.acks.get(lang) or self.acks["hi-IN"]
            k = next(self.ack_count)
            text, pcm = choices[k % len(choices)]
            uid = f"ack{k}"
            self.pipe.sm.fire("user_done")
            self.pipe.hold = True        # stay SPEAKING from the acknowledgement into the answer
            ev = AudioEvent(self.session_id, uid, SR, 0, pcm, True, {"text": text})
            start = await asyncio.get_running_loop().run_in_executor(None, self.pipe.push_audio, ev)
            self.ack_log.append({"uid": uid, "text": text, "t_push": time.time(), "start_frame": start})
            await self.broadcast({"type": "ack", "text": text})

    def voice_turn_failed(self):
        """No usable transcript: release the hold an acknowledgement may have set."""
        self.answer_started = True
        self.pipe.hold = False

    def user_speaking(self):
        if self.pipe.sm.state == "SPEAKING":
            self.interrupt()
        self.pipe.sm.fire("user_speaking")

    def interrupt(self):
        if self.token:
            self.token.cancel()
        self.pipe.sm.fire("interrupt")
        ids = self.pipe.interrupt(self.session_id)
        self.pipe.sm.fire("cancelled")
        return ids

    async def ask(self, text, lang="hi-IN"):
        if self.token and not self.token.cancelled and not getattr(self.token, "finished", False):
            if self.pipe.sm.state in ("SPEAKING", "THINKING"):
                self.interrupt()
            self.token.cancel()
        tok = self.token = CancelToken(next(self.turns))
        t0 = time.time()
        lat = {"turn": tok.turn, "text": text, "t0": t0}
        self.latency_log.append(lat)
        self.pipe.sm.fire("user_done")
        self.pipe.hold = True
        self.history.append({"role": "user", "content": text})
        await self.broadcast({"type": "user", "text": text, "turn": tok.turn})
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        spoken = []
        tag = {"raw": ""}

        async def tts_task(uid, sent, out):
            """Stream the sentence's audio chunks into `out` as they arrive (None = done, Exception = failed)."""
            try:
                async for ev in self.tts.synthesize(sent, self.session_id, uid, lang=lang, speaker=self.speaker):
                    ev.meta["text"] = sent  # speech-driven schedulers retrieve body motion by the sentence
                    await out.put(ev)
            except Exception as e:
                await out.put(e)
            await out.put(None)

        async def feeder():
            k = 0
            while True:
                item = await q.get()
                if item is None or tok.cancelled:
                    return
                uid, out = item
                first = True
                while True:  # sentences stay in order; each audio chunk goes to the lip-sync pipeline on arrival
                    ev = await out.get()
                    if ev is None:
                        break
                    if isinstance(ev, Exception):  # TTS failure: stay idle/listening, report
                        await self.broadcast({"type": "error", "text": f"TTS: {ev}"})
                        lat["tts_errors"] = lat.get("tts_errors", 0) + 1
                        break
                    if tok.cancelled:
                        return
                    if k == 0 and first:  # serialised with a pending acknowledgement (keeps their order)
                        async with self._lock():
                            self.answer_started = True
                            start = await loop.run_in_executor(None, self.pipe.push_audio, ev)
                    else:
                        start = await loop.run_in_executor(None, self.pipe.push_audio, ev)
                    if k == 0 and first:
                        lat["first_tts_done_s"] = round(time.time() - t0, 3)  # first audio handed to lip-sync
                        lat["tts_first_chunk_s"] = ev.meta.get("tts_first_chunk_s")
                        lat["first_utt"] = uid
                        lat["first_speech_frame"] = start
                        self.first_audible[uid] = lat
                    first = False
                k += 1

        feed = asyncio.create_task(feeder())
        try:
            k = 0
            async for kind, x in self.llm.respond(self.history, lang=lang, timing=lat):
                if tok.cancelled:
                    break
                if kind == "behavior":
                    lat["behavior"] = {k: v for k, v in x.items() if k != "raw_tag"}
                    tag["raw"] = x.get("raw_tag", "")
                    self.pipe.sched.set_context(hints=x.get("hints", []))
                    await self.broadcast({"type": "behavior", **lat["behavior"]})
                    continue
                spoken.append(x)
                uid, out = f"t{tok.turn}u{k}", asyncio.Queue()
                asyncio.create_task(tts_task(uid, x, out))
                await q.put((uid, out))
                k += 1
                await self.broadcast({"type": "assistant", "text": x, "turn": tok.turn})
        except Exception as e:
            await self.broadcast({"type": "error", "text": f"LLM: {e}"})
        await q.put(None)
        await feed
        if tok is self.token:
            self.pipe.hold = False
            if not spoken or tok.cancelled:
                self.pipe.sm.fire("response_done")
            self.pipe.sched.set_context(hints=[])
        lat["total_s"] = round(time.time() - t0, 3)
        tok.finished = True
        self.answer_started = True
        if spoken and not tok.cancelled:
            # the hidden tag stays in history (never in TTS), otherwise the model copies its untagged replies
            self.history.append({"role": "assistant", "content": (tag["raw"] + " " + " ".join(spoken)).strip()})

    def on_frame_out(self, meta):
        """Called by the frame pump for every output frame: records time to first audible speech."""
        uid = meta.get("utt")
        if uid and uid in self.first_audible:
            lat = self.first_audible.pop(uid)
            lat["first_audible_s"] = round(time.time() - lat["t0"], 3)
            lat["video_frames_ahead"] = round(self.pipe.out_q.qsize() / FPS, 2)
            return lat
        return None
