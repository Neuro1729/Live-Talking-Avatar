"""MuseTalk 1.5 backend (runs in envs/musetalk). Model-specific details stay in this file.

Per clip, everything frame-dependent is precomputed by R/scripts/prep_avatar.py into avatar_cache/<clip>/ (same idea
as MuseTalk's realtime_inference.Avatar.prepare_material, generalised to a library): smoothed face boxes, masked+ref
VAE latents, face-parsing blend masks and crop boxes. Live work per frame = UNet (single step) + VAE decode + blend.
MuseTalk is frame-wise (no temporal state), so changing clips at a seam cannot reset the mouth; audio features are
indexed by the audio timeline, not by the clip.
"""
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from avatar_system.common import load_config, path, root
from avatar_system.lipsync.base import LipSyncBackend

SR, FPS = 16000, 25


class ClipCache:
    def __init__(self, name, device):
        d = path("musetalk_cache") / name
        if not (d / "DONE").exists():
            raise FileNotFoundError(f"MuseTalk cache missing for {name}: run scripts/prep_avatar.py {name}")
        self.boxes = np.load(d / "boxes.npy")
        self.latents = torch.load(d / "latents.pt").to(device)
        m = pickle.load(open(d / "masks.pkl", "rb"))
        self.masks, self.crop_boxes = m["masks"], m["crop_boxes"]
        f = path("data_dir") / "features" / f"{name}.npz"  # MediaPipe landmarks (normalised) for composite v2
        self.lmk = np.load(f)["lmk"].astype(np.float32) if f.exists() else None


LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32) * 512


def mouth_geom(L, cfg):
    """Generated-mouth ellipse (centre, axes, mouth width) from the frame's MediaPipe landmarks (pixels)."""
    lips = L[LIPS]
    (cx, cy), mw = lips.mean(0), float(np.ptp(lips[:, 0]))
    lh = max(float(np.ptp(lips[:, 1])), mw * cfg["lip_min_h"])
    return (cx, cy + lh * cfg["lip_drop"]), (mw * cfg["lip_w"], lh * cfg["lip_h"]), mw


class MouthRestorer:
    """GFPGAN 1.4 (ONNX, CUDA) applied only inside the generated-mouth ellipse: sharper lips/teeth than MuseTalk's
    256 px output, while skin/stubble outside the mouth stay the original frame's (GFPGAN would smooth them)."""

    def __init__(self, onnx_path, device_id=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=[("CUDAExecutionProvider", {"device_id": device_id}),
                                                                         "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("GFPGAN fell back to CPU (~375 ms/face): CUDA 11 libs missing, source scripts/_env.sh")
        x = np.zeros((1, 3, 512, 512), np.float32)
        for _ in range(3):  # first calls autotune cuDNN (~0.6 s stall) - never let that hit a live stream
            self.sess.run(None, {self.inp: x})

    def __call__(self, frame, L, cfg, weight):
        H, W = frame.shape[:2]
        eyes = sorted([L[468], L[473]], key=lambda p: p[0])
        mouth = sorted([L[61], L[291]], key=lambda p: p[0])
        src = np.array([eyes[0], eyes[1], L[1], mouth[0], mouth[1]], np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, FFHQ_512, method=cv2.LMEDS)
        crop = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = (crop[:, :, ::-1].astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.inp: x})[0][0]
        rest = ((y.transpose(1, 2, 0)[:, :, ::-1] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        (cx, cy), (ax, ay), mw = mouth_geom(L, cfg)
        x0, y0 = int(max(cx - ax * 1.4, 0)), int(max(cy - ay * 1.4, 0))
        x1, y1 = int(min(cx + ax * 1.4, W)), int(min(cy + ay * 1.4, H))
        if x1 <= x0 or y1 <= y0:
            return frame
        Minv = cv2.invertAffineTransform(M)
        Minv[:, 2] -= [x0, y0]
        back = cv2.warpAffine(rest, Minv, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.ellipse(m, (int(cx - x0), int(cy - y0)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
        m = (cv2.GaussianBlur(m, (0, 0), mw * 0.08) * weight)[..., None]
        reg = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (reg * (1 - m) + back.astype(np.float32) * m + 0.5).astype(np.uint8)
        return frame


class MuseTalkBackend(LipSyncBackend):
    name = "musetalk1.5"

    def __init__(self, clips, device="cuda", cfg=None):
        self.cfg = cfg or load_config("musetalk")
        mt = root() / "MuseTalk"
        sys.path.insert(0, str(mt))
        from transformers import AutoFeatureExtractor, WhisperModel
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        self.device = torch.device(device)
        t0 = time.time()
        self.vae = VAE(model_path=str(mt / "models/sd-vae"), use_float16=True)
        self.unet = UNet(unet_config=str(mt / "models/musetalkV15/musetalk.json"),
                         model_path=str(mt / self.cfg.get("unet", "models/musetalkV15/unet.pth")), device=self.device)
        self.unet.model = self.unet.model.half().to(self.device).eval()
        self.pe = PositionalEncoding(d_model=384).half().to(self.device)
        self.fe = AutoFeatureExtractor.from_pretrained(str(mt / "models/whisper"))
        self.whisper = WhisperModel.from_pretrained(str(mt / "models/whisper")).to(self.device, torch.float16).eval()
        self.t0 = torch.tensor([0], device=self.device)
        self.load_s = time.time() - t0
        t0 = time.time()
        self.clips = {c: ClipCache(c, self.device) for c in clips}
        self.cache_s = time.time() - t0
        self.silence_feature = self.features(np.zeros(SR * 2, np.float32))[0]
        self.pool = ThreadPoolExecutor(self.cfg["compose_threads"])
        self.restorer = None
        rcfg = self.cfg.get("restore", {})
        if rcfg.get("enabled"):
            mp = root() / rcfg["model"]
            if not mp.exists():
                raise FileNotFoundError(f"face restore model missing: {mp}")
            self.restorer = MouthRestorer(mp)
        self.timing = {"unet_vae_ms": 0.0, "compose_ms": 0.0, "frames": 0}

    def has_clip(self, clip):
        return clip in self.clips

    @torch.no_grad()
    def features(self, pcm):
        """float32 16 kHz -> [T, 50, 384] whisper prompts (MuseTalk 1.5 recipe: layer stack, 2 frames context each side)."""
        segs = [pcm[i:i + 30 * SR] for i in range(0, len(pcm), 30 * SR)] or [pcm]
        feats = []
        for s in segs:
            x = self.fe(s, return_tensors="pt", sampling_rate=SR).input_features.to(self.device, torch.float16)
            hs = self.whisper.encoder(x, output_hidden_states=True).hidden_states
            feats.append(torch.stack(hs, dim=2)[:, : math.ceil(len(s) / SR * 50)])
        wf = torch.cat(feats, dim=1)
        n = math.ceil(len(pcm) / SR * FPS)
        wf = wf[:, : math.ceil(len(pcm) / SR * 50)]
        pad = 2
        wf = torch.cat([torch.zeros_like(wf[:, :pad * 2]), wf, torch.zeros_like(wf[:, :pad * 3 * 2])], 1)
        L = 2 * (2 + 2 + 1)
        need = n * 2 + L
        if wf.shape[1] < need:
            wf = torch.cat([wf, torch.zeros_like(wf[:, : need - wf.shape[1]])], 1)
        prompts = torch.stack([wf[0, i * 2:i * 2 + L] for i in range(n)])
        return prompts.reshape(n, L * 5, 384)

    def _blend(self, frame, face, box, mask, crop_box):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        xs, ys, xe, ye = crop_box
        large = np.zeros((ye - ys, xe - xs, 3), np.uint8)
        fx0, fy0, fx1, fy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)
        large[fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs] = frame[fy0:fy1, fx0:fx1]
        orig = large.copy()
        large[y1 - ys:y2 - ys, x1 - xs:x2 - xs] = face
        m = (mask.astype(np.float32) / 255.0)[..., None]
        out = (large * m + orig * (1 - m)).astype(np.uint8)
        frame[fy0:fy1, fx0:fx1] = out[fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs]
        return frame

    def set_variant(self, v):
        self.cfg = dict(self.cfg, composite=v)

    def _compose(self, source, clip, idx, face):
        if self.cfg.get("composite", "v1") == "v2" and self.clips[clip].lmk is not None:
            return self._compose_v2(source, clip, idx, face)
        frame = np.array(source.clip(clip)[idx])
        c = self.clips[clip]
        x1, y1, x2, y2 = map(int, c.boxes[idx])
        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        s = self.cfg["sharpen"]
        if s > 0:
            face = cv2.addWeighted(face, 1 + s, cv2.GaussianBlur(face, (0, 0), 1.6), -s, 0)
        return self._blend(frame, face, (x1, y1, x2, y2), c.masks[idx], c.crop_boxes[idx])

    def _compose_v2(self, source, clip, idx, face):
        """Detail-preserving composite. MuseTalk generates the face at 256 px, so pasting its whole jaw/cheek region
        wipes stubble and skin texture (measured: lower-face Laplacian variance -15..-27%). Here:
        1. paste only a feathered ellipse around mouth + jaw (from the frame's own landmarks) inside the parsing mask,
        2. colour-match the generated patch to the original (mean shift over the skin ring),
        3. outside the lips, keep the generated low frequencies (jaw/mouth shape) but put back the original frame's
           high frequencies (stubble, pores); inside the lips/teeth area use the generated face, mildly sharpened."""
        cfg = self.cfg["v2"]
        frame = np.array(source.clip(clip)[idx])
        H, W = frame.shape[:2]
        c = self.clips[clip]
        x1, y1, x2, y2 = map(int, c.boxes[idx])
        xs, ys, xe, ye = c.crop_boxes[idx]
        fx0, fy0, fx1, fy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)
        O = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        G = O.copy()
        gface = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        gx0, gy0 = max(x1, fx0), max(y1, fy0)
        gx1, gy1 = min(x2, fx1), min(y2, fy1)
        G[gy0 - fy0:gy1 - fy0, gx0 - fx0:gx1 - fx0] = gface[gy0 - y1:gy1 - y1, gx0 - x1:gx1 - x1]
        m = c.masks[idx][fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs].astype(np.float32) / 255.0
        L = c.lmk[idx] * [W, H] - [fx0, fy0]
        (lcx, lcy), (lax, lay), mw = mouth_geom(L, cfg)
        cx = lcx
        nose_y, chin_y = float(L[2, 1]), float(L[152, 1])
        h, w = O.shape[:2]
        tight = np.zeros((h, w), np.float32)
        jcy = (nose_y + chin_y) / 2 + cfg["jaw_drop"] * mw
        cv2.ellipse(tight, (int(cx), int(jcy)), (int(mw * cfg["jaw_w"]), int((chin_y - nose_y) / 2 * cfg["jaw_h"])),
                    0, 0, 360, 1.0, -1)
        tight = cv2.GaussianBlur(tight, (0, 0), mw * cfg["feather"])
        m2 = (m * tight)[..., None]
        lipm = np.zeros((h, w), np.float32)
        cv2.ellipse(lipm, (int(lcx), int(lcy)), (int(lax), int(lay)), 0, 0, 360, 1.0, -1)
        lipm = cv2.GaussianBlur(lipm, (0, 0), mw * 0.06)[..., None]
        ring = (m2[..., 0] > 0.5) & (lipm[..., 0] < 0.2)
        if ring.sum() > 50:
            G += (O[ring].mean(0) - G[ring].mean(0)) * cfg["color_match"]
        sig = cfg["detail_sigma"]
        hp_O = O - cv2.GaussianBlur(O, (0, 0), sig)
        lp_G = cv2.GaussianBlur(G, (0, 0), sig)
        sk = cfg["lip_sharpen"]
        lipG = G + sk * (G - lp_G) if sk > 0 else G
        skinG = lp_G + cfg["detail_gain"] * hp_O
        Gf = lipm * lipG + (1 - lipm) * skinG
        out = O * (1 - m2) + Gf * m2
        frame[fy0:fy1, fx0:fx1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        rcfg = self.cfg.get("restore", {})
        if self.restorer is not None and rcfg.get("enabled"):
            frame = self.restorer(frame, c.lmk[idx] * [W, H], cfg, rcfg["weight"])
        return frame

    @torch.no_grad()
    def process(self, refs, feats, source):
        """refs: FrameRefs; feats[i]: whisper prompt [50,384] tensor or None (silence -> closed mouth)."""
        jobs = []  # (ref index, clip, idx, feat)
        for i, (r, f) in enumerate(zip(refs, feats)):
            f = self.silence_feature if f is None else f
            jobs.append((i, r.clip, r.idx, f))
            if r.blend_clip is not None and r.alpha < 1:
                jobs.append((i, r.blend_clip, r.blend_idx, f))
        t0 = time.time()
        lat = torch.cat([self.clips[c].latents[k:k + 1] for _, c, k, _ in jobs])
        aud = self.pe(torch.stack([f for *_, f in jobs]).to(self.device, torch.float16))
        pred = self.unet.model(lat, self.t0, encoder_hidden_states=aud).sample
        faces = self.vae.decode_latents(pred)
        t1 = time.time()
        composed = list(self.pool.map(lambda j: self._compose(source, j[1][1], j[1][2], faces[j[0]]), enumerate(jobs)))
        out = [None] * len(refs)
        for (i, *_), img in zip(jobs, composed):
            if out[i] is None:
                out[i] = img
            else:  # second job of a seam frame = A continuation; crossfade
                a = refs[i].alpha
                out[i] = (img.astype(np.float32) * (1 - a) + out[i].astype(np.float32) * a + 0.5).astype(np.uint8)
        self.timing["unet_vae_ms"] += (t1 - t0) * 1000
        self.timing["compose_ms"] += (time.time() - t1) * 1000
        self.timing["frames"] += len(refs)
        return out
