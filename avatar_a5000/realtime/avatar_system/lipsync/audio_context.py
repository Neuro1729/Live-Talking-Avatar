"""One continuous audio timeline in samples (16 kHz mono float32), the master clock for speaking playback.

Output frame n covers samples [n*SPF, (n+1)*SPF). Utterances are placed on the timeline (never overlapping, never
before the next frame still to be rendered) and may arrive as several chunks. Lip-sync features are computed by the
backend over the *whole utterance so far* (whisper needs left/right context), and a frame's feature is only final once
`right_ctx` frames of audio after it exist or the utterance has ended, so chunk boundaries never cut the audio context.
Cancelled utterances are removed from the not-yet-rendered part of the timeline and can never be emitted later.
"""
import threading
from dataclasses import dataclass, field

import numpy as np

SR, FPS = 16000, 25
SPF = SR // FPS


@dataclass
class AudioEvent:
    """Provider-independent audio chunk (PCM float32 16 kHz mono, decoded at the TTS boundary)."""
    session_id: str
    utterance_id: str
    sample_rate: int
    sample_offset: int          # offset of this chunk inside its utterance
    audio: np.ndarray
    end_of_utterance: bool = False
    meta: dict = field(default_factory=dict)


class Utterance:
    def __init__(self, uid, start_frame, session_id):
        self.uid, self.session_id = uid, session_id
        self.start = start_frame            # timeline frame of the first sample
        self.pcm = np.zeros(0, np.float32)
        self.ended = False
        self.cancelled = False
        self.closed = False                 # truncated by an interruption: later chunks are stale
        self.feats = None                   # [k, ...] per-frame lip-sync features (backend specific)
        self.final_frames = 0               # frames whose features are final
        self.meta = {}

    @property
    def n_frames(self):
        return int(np.ceil(len(self.pcm) / SPF))

    @property
    def end(self):
        return self.start + self.n_frames


class AudioTimeline:
    def __init__(self, feature_fn, right_ctx=3, lead_frames=2, feature_shift=0):
        self.feature_fn = feature_fn        # pcm float32 -> [n_frames, ...] features (or None for mock)
        # lip-sync model latency compensation: the mouth of frame n is driven by the features of frame n+shift
        # (SyncNet: unshifted MuseTalk 1.5 mouth led the audio by 2 frames -> shift -2); audio samples never shift
        self.shift = feature_shift
        self.right_ctx, self.lead = right_ctx, lead_frames
        self.utts = []                      # ordered, non-overlapping
        self.by_id = {}
        self.render_frame = 0               # next frame the renderer will ask for
        self.lock = threading.Lock()
        self.stats = {"chunks": 0, "underruns": 0, "cancelled": 0}

    def push(self, ev: AudioEvent):
        """Add an audio chunk. Returns the utterance's start frame."""
        assert ev.sample_rate == SR, f"resample to {SR} at the provider boundary (got {ev.sample_rate})"
        with self.lock:
            u = self.by_id.get(ev.utterance_id)
            if u is None:
                prev_end = self.utts[-1].end if self.utts else 0
                u = Utterance(ev.utterance_id, max(prev_end, self.render_frame + self.lead), ev.session_id)
                u.meta = dict(ev.meta)
                self.utts.append(u)
                self.by_id[ev.utterance_id] = u
            if u.cancelled or u.closed:
                self.stats["stale_chunks"] = self.stats.get("stale_chunks", 0) + 1
                return u.start
            if ev.sample_offset != len(u.pcm):  # keep continuity: pad gaps / drop overlaps
                if ev.sample_offset > len(u.pcm):
                    u.pcm = np.concatenate([u.pcm, np.zeros(ev.sample_offset - len(u.pcm), np.float32)])
                else:
                    ev.audio = ev.audio[len(u.pcm) - ev.sample_offset:]
            u.pcm = np.concatenate([u.pcm, ev.audio.astype(np.float32)])
            u.ended = u.ended or ev.end_of_utterance
            self.stats["chunks"] += 1
        self._featurize(u)
        return u.start

    def _featurize(self, u):
        if self.feature_fn is None:
            u.final_frames = u.n_frames if u.ended else max(u.n_frames - self.right_ctx, 0)
            return
        feats = self.feature_fn(u.pcm)
        with self.lock:
            u.feats = feats
            n = len(feats)
            u.final_frames = n if u.ended else max(n - self.right_ctx, 0)

    def frame(self, n):
        """-> (pcm int16 bytes for frame n, lip-sync feature (of frame n+shift) or None, utterance id of the audio
        at frame n or None, final: bool)."""
        with self.lock:
            self.render_frame = max(self.render_frame, n + 1 + self.shift)
            pcm, uid, feat, final = bytes(SPF * 2), None, None, True
            m = n + self.shift
            for u in self.utts:
                if u.cancelled:
                    continue
                if u.start <= n < u.end:
                    k = n - u.start
                    x = u.pcm[k * SPF:(k + 1) * SPF]
                    if len(x) < SPF:
                        x = np.pad(x, (0, SPF - len(x)))
                    pcm, uid = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(), u.uid
                if u.start <= m < u.end:
                    k = m - u.start
                    final = k < u.final_frames
                    if not final:
                        self.stats["underruns"] += 1
                    feat = u.feats[min(k, len(u.feats) - 1)] if u.feats is not None and len(u.feats) else None
            if uid is None and feat is None:
                self.utts = [u for u in self.utts if u.end > n - FPS]  # drop utterances that are fully played
        return pcm, feat, uid, final

    def cancel(self, session_id=None, keep_until=None):
        """Cancel all not-yet-rendered speech (optionally only one session's). Returns cancelled utterance ids."""
        with self.lock:
            out = []
            cut = self.render_frame if keep_until is None else keep_until
            for u in self.utts:
                if session_id and u.session_id != session_id:
                    continue
                if u.end > cut and not u.cancelled:
                    if u.start >= cut:
                        u.cancelled = True
                    else:  # truncate at the cut so already-rendered audio stays consistent
                        u.pcm = u.pcm[:(cut - u.start) * SPF]
                        u.ended = u.closed = True
                        if u.feats is not None:
                            u.feats = u.feats[:cut - u.start]
                    out.append(u.uid)
            self.utts = [u for u in self.utts if not u.cancelled]
            self.stats["cancelled"] += len(out)
            return out

    def pending_frames(self, n):
        with self.lock:
            return sum(max(u.end - max(u.start, n), 0) for u in self.utts if not u.cancelled)

    def speaking_at(self, n):
        with self.lock:
            return any(u.start <= n < u.end and not u.cancelled for u in self.utts)
