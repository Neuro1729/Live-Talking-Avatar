"""The single continuous audio-video timeline.

  plan thread    : motion scheduler -> FrameRef n, audio timeline -> PCM + lip-sync feature of frame n   -> q_plan
  render thread  : batches of q_plan -> lip-sync backend (MuseTalk or mock)                              -> q_render
  encode thread  : resize + JPEG                                                                         -> out_q
  consumer       : transport / recorder pulls out_q (paced at 25 fps live, unpaced offline)

Frame n always carries audio samples [n*640, (n+1)*640): audio and video cannot drift. Clip changes do not touch
frame numbering, timestamps, the audio clock or the model. All queues are bounded and their sizes are exported as
metrics; a slow stage shows up as a growing upstream wait / output underruns, never as unbounded memory.
"""
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from avatar_system.conversation.behavior_planner import prosody
from avatar_system.lipsync.audio_context import SPF, AudioTimeline
from avatar_system.motion.motion_scheduler import MotionScheduler
from avatar_system.streaming.frame_source import ClipFrameSource

FPS = 25


class Stat:
    def __init__(self, n=250):
        self.v = deque(maxlen=n)

    def add(self, x):
        self.v.append(x)

    def summary(self):
        if not self.v:
            return None
        a = np.array(self.v)
        return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


class AvatarPipeline:
    def __init__(self, graph, backend, state_machine, batch=4, out_width=768, jpeg_q=82, playback_buffer=6,
                 seed=0, silence="ramp", encode=True, ramp=4, feature_shift=0, scheduler=None, talk_frames=None):
        self.graph, self.backend, self.sm = graph, backend, state_machine
        self.batch, self.out_width, self.jpeg_q, self.encode = batch, out_width, jpeg_q, encode
        self.silence, self.ramp = silence, ramp
        self.source = ClipFrameSource()
        if scheduler is not None:  # e.g. TalkBankScheduler: speech-driven, plans each sentence when its audio arrives
            clips = scheduler.clips
        else:
            run = set(graph.get("runtime_segments") or range(len(graph["segments"])))
            clips = sorted({s["clip"] for s in graph["segments"] if s["id"] in run})  # only playable segments matter
        missing = [c for c in clips if not backend.has_clip(c)]
        if missing:
            raise RuntimeError(f"lip-sync cache missing for clips {missing}")
        if scheduler is not None:
            self.sched = scheduler
            scheduler.mode_source = lambda: self.sm.state
        else:
            self.sched = MotionScheduler(graph, {c: self.source.length(c) for c in clips}, seed=seed,
                                         mode_source=lambda: self.sm.state)
        self.sched.on_segment_end = self._segment_end
        # speech starts `ramp` frames after the render point so the pre-speech crossfade frames are planned with it
        self.audio = AudioTimeline(backend.features, lead_frames=max(ramp, 2) + max(feature_shift, 0),
                                   feature_shift=feature_shift)
        # small queues = low speech latency: speech can only start after everything already planned (~lead +
        # q_plan + q_render + out_q frames); render capacity (~70 fps) keeps them full without underruns
        self.q_plan = queue.Queue(maxsize=batch)
        self.q_render = queue.Queue(maxsize=1)
        self.out_q = queue.Queue(maxsize=playback_buffer)
        self.hold = False                 # an answer is still being produced: stay SPEAKING through sentence gaps
        # talking-bank frames whose recorded mouth speaks: when one is shown without our speech (a talking segment
        # finishing after the answer), lip-sync it with silence instead of letting the recorded mouth mouth words
        self.talk_frames = talk_frames or {}
        self.silent_feat = None
        if self.talk_frames and backend.features is not None:
            f = backend.features(np.zeros(SPF * 25, np.float32))
            self.silent_feat = f[len(f) // 2] if f is not None and len(f) else None
        self.frame_hooks = []             # callables(n) run in the plan thread before frame n (offline scripts)
        self.running = False
        self.t_start = None
        self.stats = {k: Stat() for k in ("plan_ms", "render_ms", "render_ms_per_frame", "encode_ms", "out_interval_ms")}
        self.counters = {"frames_planned": 0, "frames_rendered": 0, "frames_out": 0, "output_underruns": 0,
                         "audio_feature_late": 0, "speech_frames": 0}
        self.last_meta = {}
        self.trace = deque(maxlen=200000)
        self.on_speech_done = None
        self._prosody_n = -1
        self.error = None

    # ---------- control ----------
    def push_audio(self, ev):
        start = self.audio.push(ev)
        if ev.end_of_utterance and hasattr(self.sched, "add_speech"):  # whole sentence known: plan its body motion
            u = self.audio.by_id.get(ev.utterance_id)
            if u is not None and not u.cancelled:
                self.sched.add_speech(ev.meta.get("text", ""), u.pcm, u.start, u.end)
        return start

    def interrupt(self, session_id=None):
        ids = self.audio.cancel(session_id)
        self.hold = False
        if hasattr(self.sched, "cancel"):
            self.sched.cancel()
        return ids

    def _segment_end(self, seg):
        if seg["end_state"] == self.graph["hub"]:
            self.sm.fire("hub_reached")

    # ---------- stages ----------
    def _plan_loop(self):
        n = 0
        while self.running:
            t0 = time.time()
            for h in self.frame_hooks:
                h(n)
            speaking = self.audio.speaking_at(n)
            if speaking and self.sm.state not in ("SPEAKING",):
                self.sm.fire("audio_ready")
            elif not speaking and self.sm.state == "SPEAKING" and not self.hold and self.audio.pending_frames(n) == 0:
                self.sm.fire("response_done")
                if self.on_speech_done:
                    self.on_speech_done()
            if self.sm.state == "SPEAKING" and n - self._prosody_n >= 10:
                self._prosody_n = n
                self._update_prosody(n)
            ref = self.sched.next()
            pcm, feat, uid, final = self.audio.frame(ref.n)
            # lip-sync weight: 1 during speech, ramping to 0 over `ramp` frames before/after it ("ramp" silence mode)
            if uid is not None or feat is not None:  # speech audio, or speech features (shifted ahead)
                w = 1.0
            else:
                d = next((k for k in range(1, self.ramp + 1)
                          if self.audio.speaking_at(ref.n + k) or self.audio.speaking_at(ref.n - k)), None)
                w = 0.0 if d is None else 1 - d / (self.ramp + 1)
            tf = self.talk_frames.get(ref.clip)
            if w < 1 and self.silent_feat is not None and tf is not None and tf[min(ref.idx, len(tf) - 1)]:
                w, feat = 1.0, self.silent_feat
                self.counters["muted_talk_frames"] = self.counters.get("muted_talk_frames", 0) + 1
            if uid is not None:
                self.counters["speech_frames"] += 1
            if feat is not None and not final:
                    self.counters["audio_feature_late"] += 1
            self.stats["plan_ms"].add((time.time() - t0) * 1000)
            self.counters["frames_planned"] += 1
            self._put(self.q_plan, (ref, pcm, feat, uid, self.sm.state, w))
            n += 1

    def _update_prosody(self, n):
        with self.audio.lock:
            pcm = [u.pcm[max(n - u.start, 0) * SPF:] for u in self.audio.utts if u.end > n]
        pcm = np.concatenate(pcm) if pcm else np.zeros(0, np.float32)
        p = prosody(pcm)
        self.sched.set_context(emphasis_at=p["emphasis_at"], pause_frac=p["pause_frac"])

    def _render_loop(self):
        while self.running:
            items = [self._get(self.q_plan)]
            if items[0] is None:
                continue
            while len(items) < self.batch:
                try:
                    items.append(self.q_plan.get(timeout=0.02))
                except queue.Empty:
                    break
            refs = [it[0] for it in items]
            feats = [it[2] for it in items]
            t0 = time.time()
            if self.silence == "always":
                frames = self.backend.process(refs, feats, self.source)
            else:  # raw source frames when nobody speaks; lip-sync (crossfaded in "ramp" mode) around speech
                ws = [it[5] if self.silence == "ramp" else float(it[2] is not None) for it in items]
                lip = [k for k, w in enumerate(ws) if w > 0]
                frames = [self.source.get(r) if ws[k] < 1 else None for k, r in enumerate(refs)]
                if lip:
                    out = self.backend.process([refs[k] for k in lip], [feats[k] for k in lip], self.source)
                    for k, f in zip(lip, out):
                        w = ws[k]
                        frames[k] = f if w >= 1 else (frames[k].astype(np.float32) * (1 - w)
                                                      + f.astype(np.float32) * w + 0.5).astype(np.uint8)
            dt = (time.time() - t0) * 1000
            self.stats["render_ms"].add(dt)
            self.stats["render_ms_per_frame"].add(dt / len(items))
            self.counters["frames_rendered"] += len(items)
            self._put(self.q_render, (items, frames))

    def _encode_loop(self):
        while self.running:
            got = self._get(self.q_render)
            if got is None:
                continue
            items, frames = got
            for (ref, pcm, feat, uid, state, w), f in zip(items, frames):
                t0 = time.time()
                if self.out_width and f.shape[1] != self.out_width:
                    h = round(f.shape[0] * self.out_width / f.shape[1] / 2) * 2
                    f = cv2.resize(f, (self.out_width, h), interpolation=cv2.INTER_AREA)
                payload = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes() if self.encode else f
                self.stats["encode_ms"].add((time.time() - t0) * 1000)
                seg = self.graph["segments"][ref.seg]
                meta = {"n": ref.n, "t": round(ref.n / FPS, 3), "clip": ref.clip, "idx": ref.idx, "seg": seg["key"],
                        "pose": seg["end_state"], "state": state, "utt": uid, "seam": ref.seam,
                        "blend": ref.blend_clip is not None, "audio_sample": ref.n * SPF, "lip_w": round(w, 2)}
                self.trace.append([ref.n, ref.seg, ref.clip, ref.idx, ref.blend_clip, ref.blend_idx, round(ref.alpha, 3),
                                   int(ref.seam), state, uid])
                self._put(self.out_q, (payload, pcm, meta))

    def _put(self, q, item):
        while self.running:
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _get(self, q):
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            return None

    # ---------- lifecycle ----------
    def start(self):
        self.running = True
        self.t_start = time.time()
        self.threads = [threading.Thread(target=self._guard, args=(f,), daemon=True, name=f.__name__)
                        for f in (self._plan_loop, self._render_loop, self._encode_loop)]
        for t in self.threads:
            t.start()

    def _guard(self, fn):
        """A crashed stage must not leave a silently frozen stream: record it, stop everything, re-raise on read()."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            self.error = f"{fn.__name__}: {type(e).__name__}: {e}"
            traceback.print_exc()
            self.running = False

    def stop(self):
        self.running = False
        for t in getattr(self, "threads", []):
            t.join(timeout=2)
        self.backend.close()

    def read(self, timeout=1.0):
        """Consumer side: next (payload, pcm_bytes, meta) or None on underrun."""
        if self.error:
            raise RuntimeError(f"pipeline stage crashed: {self.error}")
        try:
            item = self.out_q.get(timeout=timeout)
        except queue.Empty:
            self.counters["output_underruns"] += 1
            return None
        self.counters["frames_out"] += 1
        self.last_meta = item[2]
        return item

    def metrics(self):
        el = max(time.time() - (self.t_start or time.time()), 1e-6)
        m = {"state": self.sm.snapshot(), "queues": {"plan": self.q_plan.qsize(), "render": self.q_render.qsize(),
                                                     "out": self.out_q.qsize()},
             "audio_buffer_frames": self.audio.pending_frames(self.last_meta.get("n", 0)),
             "counters": dict(self.counters), "audio": dict(self.audio.stats),
             "render_fps_capacity": None, "uptime_s": round(el, 1), "current": self.last_meta,
             "plan": self.sched.upcoming(), "transitions": len(self.sched.seams),
             "av_offset_samples": 0, "error": self.error}  # by construction: frame n carries samples [n*640,(n+1)*640)
        for k, s in self.stats.items():
            m[k] = s.summary()
        if m["render_ms_per_frame"]:
            m["render_fps_capacity"] = round(1000 / m["render_ms_per_frame"]["mean"], 1)
        return m
