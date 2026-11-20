"""Real-time avatar engine: picks base clips (director) + MuseTalk 1.5 lip-sync on the fly + 25 fps frame stream.

Only the UNet + VAE decoder run live; face boxes, latents and blend masks come from avatar_cache/ (prep_avatar.py).
Every base clip starts/ends on the same home pose, so the director can chain clips seamlessly and may cut mid-clip
at "near-home" frames when the conversation state changes (listen -> think -> speak).
"""
import json
import math
import pickle
import queue
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
MT = R / "MuseTalk"
sys.path.insert(0, str(MT))
from transformers import AutoFeatureExtractor, WhisperModel  # noqa: E402
from musetalk.models.unet import PositionalEncoding, UNet  # noqa: E402
from musetalk.models.vae import VAE  # noqa: E402

FPS = 25
SR = 16000
SPF = SR // FPS  # audio samples per video frame (640)
# which clip categories (name prefix after bank_/ltx_) each conversation mode may play, with weights
MODE_CATS = {
    "listen": {"idle": 1.0},
    "think": {"think": 1.0},
    "speak": {"steady": 0.45, "steadywarm": 0.25, "nod": 0.12, "talkslow": 0.08, "onehand": 0.10},
}
BUSY = {"nod", "talkslow", "onehand"}


def clip_category(name):
    if name.startswith("bank_"):
        return name.split("_")[1]
    return "idle" if "idle" in name else "steady"


class Clip:
    def __init__(self, name, device):
        d = R / "avatar_cache" / name
        self.name, self.cat = name, clip_category(name)
        self.frames = np.load(d / "frames.npy", mmap_mode="r")  # page-cache backed, not process RAM
        self.boxes = np.load(d / "boxes.npy")
        self.latents = torch.load(d / "latents.pt").to(device)
        m = pickle.load(open(d / "masks.pkl", "rb"))
        self.masks, self.crop_boxes = m["masks"], m["crop_boxes"]
        self.n = len(self.boxes)
        # near-home frames = safe cut points (small difference to frame 0, which is the home pose)
        small = lambda f: cv2.resize(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY), (64, 58),
                                     interpolation=cv2.INTER_AREA).astype(np.float32)
        sm = [small(self.frames[i]) for i in range(self.n)]
        dist = np.array([np.abs(s - sm[0]).mean() for s in sm])
        self.cut_ok = dist < 2.0
        # skip the near-frozen "settle" frames at both ends so chained clips don't show a pause (keep 2)
        d = np.array([np.abs(sm[i + 1] - sm[i]).mean() for i in range(self.n - 1)])
        s, e = 0, len(d)
        while s < e and d[s] < 0.15:
            s += 1
        while e > s and d[e - 1] < 0.15:
            e -= 1
        self.start, self.end = max(s - 2, 0), min(e + 3, self.n)


class Director:
    """Decides which (clip, frame) plays next. Conversation-like: listen=idle, think, speak=mostly steady."""

    def __init__(self, clips, seed=0):
        self.clips = clips
        self.by_cat = {}
        for c in clips.values():
            self.by_cat.setdefault(c.cat, []).append(c)
        self.rng = random.Random(seed)
        self.mode = "listen"
        self.last_busy = False
        self.cur = self._pick("listen")
        self.idx = self.cur.start

    def _pick(self, mode, avoid=None):
        weights = {k: v for k, v in MODE_CATS[mode].items() if k in self.by_cat}
        if mode == "speak" and getattr(self, "last_busy", False):
            weights = {k: v for k, v in weights.items() if k not in BUSY} or weights
        if not weights:  # fall back: speak->steady clips, others->idle
            fb = "steady" if mode == "speak" else "idle"
            weights = {fb if fb in self.by_cat else next(iter(self.by_cat)): 1.0}
        cat = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        pool = [c for c in self.by_cat[cat] if c is not avoid] or self.by_cat[cat]
        clip = self.rng.choice(pool)
        self.last_busy = cat in BUSY
        return clip

    def set_mode(self, mode):
        self.mode = mode

    def next(self):
        """Return (clip, frame_idx) for the next output frame."""
        fits = self.cur.cat in MODE_CATS[self.mode]
        at_end = self.idx >= self.cur.end
        early_cut = (not fits) and self.idx > self.cur.start + 2 and self.cur.cut_ok[min(self.idx, self.cur.n - 1)]
        if at_end or early_cut:
            self.cur = self._pick(self.mode, avoid=self.cur)
            self.idx = self.cur.start  # new clip begins at the home pose ~= the frame we left on
        out = (self.cur, self.idx)
        self.idx += 1
        return out


SUB_MODE_CATS = {  # subtle bank (sub_<cat>_s<seed>): calm by default, nods are rare
    "listen": {"listen": 0.7, "still": 0.3},
    "think": {"think": 1.0},
    "speak": {"still": 0.2, "blink": 0.3, "talk": 0.25, "warm": 0.15, "nod": 0.10},
}


class GraphDirector:
    """Random walk on the motion graph (avatar_cache/graph.json): jumps between clips at frames that look and move
    alike, so playback never repeats in fixed blocks. Weights: conversation mode, recency, nod spacing, match cost."""

    def __init__(self, clips, graph, seed=0, dwell_s=4.5):
        self.clips, self.rng = clips, random.Random(seed)
        self.cat = {n: (n.split("_")[1] if n.startswith("sub_") else clip_category(n)) for n in clips}
        self.edges = {n: {} for n in clips}
        for a, info in graph["clips"].items():
            if a not in clips:
                continue
            for i, b, j, cost in info["edges"]:
                if b in clips:
                    self.edges[a].setdefault(i, []).append((b, j, cost))
        self.p_jump = graph["step"] / (dwell_s * FPS)
        self.mode, self.recent, self.t, self.last_nod_t = "listen", deque(maxlen=6), 0, -1e9
        self.cur, self.idx = self._choose_start()

    def set_mode(self, mode):
        self.mode = mode

    def _cats(self):
        return SUB_MODE_CATS[self.mode] if any(c.startswith("sub_") for c in self.clips) else MODE_CATS[self.mode]

    def _weight(self, name, cost=0.0):
        w = self._cats().get(self.cat[name], 0.0)
        if name in self.recent:
            w *= 0.15
        if self.cat[name] == "nod" and self.t - self.last_nod_t < 12 * FPS:
            w = 0.0
        return w * math.exp(-cost)

    def _choose_start(self):
        names = list(self.clips)
        ws = [self._weight(n) for n in names]
        if sum(ws) == 0:
            ws = [1.0] * len(names)
        n = self.rng.choices(names, weights=ws)[0]
        return self._enter(n, self.clips[n].start)

    def _enter(self, name, idx):
        self.recent.append(name)
        if self.cat[name] == "nod":
            self.last_nod_t = self.t
        return self.clips[name], idx

    def next(self):
        self.t += 1
        c = self.cur
        if self.idx >= c.end:  # clip finished -> any clip start (all start on the home pose)
            self.cur, self.idx = self._choose_start()
        else:
            opts = self.edges[c.name].get(self.idx)
            fits = self.cat[c.name] in self._cats()
            if opts and (not fits or self.rng.random() < self.p_jump):
                ws = [self._weight(b, cost) for b, j, cost in opts]
                if sum(ws) > 0:
                    b, j, _ = self.rng.choices(opts, weights=ws)[0]
                    self.cur, self.idx = self._enter(b, j)
        out = (self.cur, self.idx)
        self.idx += 1
        return out


class AvatarEngine:
    def __init__(self, clip_names, device="cuda", batch=4, out_width=768, jpeg_q=82, lipsync_silence=True, sharpen=0.6):
        self.device = torch.device(device)
        self.sharpen = sharpen
        self.batch, self.out_width, self.jpeg_q, self.lipsync_silence = batch, out_width, jpeg_q, lipsync_silence
        t0 = time.time()
        self.vae = VAE(model_path=str(MT / "models/sd-vae"), use_float16=True)
        self.unet = UNet(unet_config=str(MT / "models/musetalkV15/musetalk.json"),
                         model_path=str(MT / "models/musetalkV15/unet.pth"), device=self.device)
        self.unet.model = self.unet.model.half().to(self.device).eval()
        self.pe = PositionalEncoding(d_model=384).half().to(self.device)
        self.fe = AutoFeatureExtractor.from_pretrained(str(MT / "models/whisper"))
        self.whisper = WhisperModel.from_pretrained(str(MT / "models/whisper")).to(self.device, torch.float16).eval()
        self.timesteps = torch.tensor([0], device=self.device)
        print(f"[engine] models loaded in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        self.clips = {n: Clip(n, self.device) for n in clip_names}
        print(f"[engine] {len(self.clips)} clips loaded in {time.time() - t0:.1f}s: "
              f"{sorted({c.cat for c in self.clips.values()})}", flush=True)
        gpath = R / "avatar_cache" / "graph.json"
        graph = json.loads(gpath.read_text()) if gpath.exists() else None
        if graph and set(self.clips) <= set(graph["clips"]):
            self.director = GraphDirector(self.clips, graph)
            print(f"[engine] graph director: {sum(len(v) for v in self.director.edges.values())} jump points", flush=True)
        else:
            self.director = Director(self.clips)
            print("[engine] simple director (graph.json missing or does not cover all clips)", flush=True)
        self.silence_feat = self.audio_features(np.zeros(SR * 2, dtype=np.float32))[0][0]
        self.speech = deque()  # per-frame (whisper_feat [50,384] on GPU, pcm int16[640])
        self.speech_lock = threading.Lock()
        self.out_q = queue.Queue(maxsize=4)  # (jpeg bytes, pcm int16 bytes, meta)
        self.pool = ThreadPoolExecutor(4)
        self.running = False
        self.hold = False  # True while an answer is still being produced: keep "speak" mode through sentence gaps
        self.on_speech_done = None

    # ---------- audio ----------
    @torch.no_grad()
    def audio_features(self, wav):
        """wav float32 16 kHz -> per-frame whisper prompts [T, 50, 384] (MuseTalk v1.5 recipe, padding 2/2)."""
        segs = [wav[i:i + 30 * SR] for i in range(0, len(wav), 30 * SR)] or [wav]
        feats = []
        for s in segs:
            x = self.fe(s, return_tensors="pt", sampling_rate=SR).input_features.to(self.device, torch.float16)
            hs = self.whisper.encoder(x, output_hidden_states=True).hidden_states
            feats.append(torch.stack(hs, dim=2))
        wf = torch.cat(feats, dim=1)
        n_frames = math.floor(len(wav) / SR * FPS)
        wf = wf[:, :math.floor(len(wav) / SR * 50)]
        pad = 2
        wf = torch.cat([torch.zeros_like(wf[:, :pad * 2]), wf, torch.zeros_like(wf[:, :pad * 3 * 2])], 1)
        L = 2 * (2 + 2 + 1)
        prompts = torch.stack([wf[0, i * 2:i * 2 + L] for i in range(n_frames)])  # [T, 10, 5, 384]
        return prompts.reshape(n_frames, L * 5, 384), n_frames

    def add_speech(self, wav):
        """Queue one utterance (float32 mono 16 kHz). Lip-synced frames follow the already-buffered frames."""
        feats, n = self.audio_features(wav)
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        with self.speech_lock:
            for i in range(n):
                self.speech.append((feats[i], pcm[i * SPF:(i + 1) * SPF].tobytes()))
            self.director.set_mode("speak")
        return n / FPS

    def set_mode(self, mode):
        with self.speech_lock:
            if not self.speech:
                self.director.set_mode(mode)

    def clear_speech(self):
        """Barge-in: drop whatever is still queued to be spoken."""
        with self.speech_lock:
            self.speech.clear()

    # ---------- rendering ----------
    @staticmethod
    def _blend(frame, face, box, mask, crop_box):
        """numpy version of MuseTalk get_image_blending (same result, ~5x faster): paste the generated face into
        the expanded crop, then alpha-blend that crop back with the face-parsing mask."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        xs, ys, xe, ye = crop_box
        large = np.zeros((ye - ys, xe - xs, 3), np.uint8)  # crop may extend past the frame (PIL pads with black)
        fx0, fy0, fx1, fy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)
        large[fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs] = frame[fy0:fy1, fx0:fx1]
        orig = large.copy()
        large[y1 - ys:y2 - ys, x1 - xs:x2 - xs] = face
        m = (mask.astype(np.float32) / 255.0)[..., None]
        out = (large * m + orig * (1 - m)).astype(np.uint8)
        frame[fy0:fy1, fx0:fx1] = out[fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs]
        return frame

    def _compose(self, clip, idx, face):
        frame = np.array(clip.frames[idx])  # copy out of mmap
        if face is not None:
            x1, y1, x2, y2 = map(int, clip.boxes[idx])
            face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            if self.sharpen > 0:  # MuseTalk works at 256 px; restore a bit of skin/stubble crispness
                face = cv2.addWeighted(face, 1 + self.sharpen, cv2.GaussianBlur(face, (0, 0), 1.6), -self.sharpen, 0)
            frame = self._blend(frame, face, (x1, y1, x2, y2), clip.masks[idx], clip.crop_boxes[idx])
        if self.out_width and frame.shape[1] != self.out_width:
            h = round(frame.shape[0] * self.out_width / frame.shape[1] / 2) * 2
            frame = cv2.resize(frame, (self.out_width, h), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes()

    def _render_batch(self):
        return self._finish(*self._gpu_batch())

    def _finish(self, items, faces):
        jpgs = list(self.pool.map(lambda a: self._compose(*a), [(it[0], it[1], faces[k]) for k, it in enumerate(items)]))
        return [(jpgs[k], it[3], {"clip": it[0].name, "idx": int(it[1]), "talking": it[4]}) for k, it in enumerate(items)]

    @torch.no_grad()
    def _gpu_batch(self):
        items = []
        with self.speech_lock:
            for _ in range(self.batch):
                clip, idx = self.director.next()
                if self.speech:
                    feat, pcm = self.speech.popleft()
                    talking = True
                else:
                    feat, pcm, talking = self.silence_feat, bytes(SPF * 2), False
                items.append((clip, idx, feat, pcm, talking))
            if not self.speech and self.director.mode == "speak" and not self.hold:
                self.director.set_mode("listen")
                if self.on_speech_done:
                    self.on_speech_done()
        need = [i for i, it in enumerate(items) if it[4] or self.lipsync_silence]
        faces = [None] * len(items)
        if need:
            lat = torch.cat([items[i][0].latents[items[i][1]:items[i][1] + 1] for i in need])
            aud = self.pe(torch.stack([items[i][2] for i in need]))
            pred = self.unet.model(lat, self.timesteps, encoder_hidden_states=aud).sample
            dec = self.vae.decode_latents(pred)
            for j, i in enumerate(need):
                faces[i] = dec[j]
        return items, faces

    def _loop(self):
        # 2-stage pipeline: CPU compose/JPEG of batch k runs in a worker while the GPU renders batch k+1
        stage = ThreadPoolExecutor(1)
        pending = None
        while self.running:
            nxt = self._gpu_batch()
            if pending is not None:
                self._emit(pending.result())
            pending = stage.submit(self._finish, *nxt)
        stage.shutdown()

    def _emit(self, rendered):
        for item in rendered:
            while self.running:
                try:
                    self.out_q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def benchmark(self, seconds=8.0, wav=None):
        """Run the real pipelined loop unpaced for `seconds`; returns achieved fps (lip-synced) + stage timings."""
        if wav is not None:
            self.add_speech(wav)
        t = time.time()
        for _ in range(5):
            torch.cuda.synchronize()
            items, faces = self._gpu_batch()
        torch.cuda.synchronize()
        gpu_ms = (time.time() - t) / 5 * 1000
        t = time.time()
        for _ in range(5):
            self._finish(items, faces)
        cpu_ms = (time.time() - t) / 5 * 1000
        n, t0 = 0, time.time()
        self.running = True
        th = threading.Thread(target=self._loop, daemon=True)
        th.start()
        while time.time() - t0 < seconds:
            self.out_q.get()
            n += 1
        self.running = False
        th.join()
        while not self.out_q.empty():
            self.out_q.get()
        return n / (time.time() - t0), gpu_ms, cpu_ms
