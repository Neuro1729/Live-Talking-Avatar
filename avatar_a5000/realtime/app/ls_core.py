"""Lean LatentSync 1.6 core for live use (runs in the ComfyUI venv).

Everything that depends only on the base frame is computed once per frame (prep): face affine, reference latents,
masked-face latents, paste-back alpha. Live work per 16-frame chunk is only: whisper-tiny features of the sentence,
a few DDIM UNet steps, a VAE decode of the mouth rows of the latent, and a GPU inverse-warp + alpha blend.
"""
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[1]
LSW = Path("/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper")
CWD = R / "latentsync" / "cwd"  # holds symlinks checkpoints/ configs/ latentsync/ (LatentSync uses cwd-relative paths)
sys.path.insert(0, str(LSW))

FPS, SR, NF, RES = 25, 16000, 16, 512
LAT_TOP = 16  # decode latent rows LAT_TOP..63 only: the mouth mask starts at pixel row ~190 of 512


class LatentSyncCore:
    def __init__(self, device="cuda", steps=4, guidance=1.5, deepcache=False, seed=1247):
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.whisper.whisper import load_model

        os.chdir(CWD)
        self.device, self.dtype = torch.device(device), torch.float16
        self.steps, self.guidance = steps, guidance
        cfg = OmegaConf.load(str(LSW / "configs/unet/stage2_512.yaml"))
        self.vae = AutoencoderKL.from_pretrained(str(LSW / "checkpoints/vae"), torch_dtype=self.dtype,
                                                 local_files_only=True).to(self.device).eval()
        unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model),
                                                       str(LSW / "checkpoints/latentsync_unet.pt"), device="cpu")
        self.unet = unet.to(self.device, self.dtype).eval()
        self.whisper = load_model(str(LSW / "checkpoints/whisper/tiny.pt"), self.device)
        self.sched = DDIMScheduler(beta_end=0.012, beta_schedule="scaled_linear", beta_start=0.00085,
                                   clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon",
                                   set_alpha_to_one=False, steps_offset=1)
        self.deepcache = None
        if deepcache:
            from DeepCache import DeepCacheSDHelper

            class _P:  # DeepCacheSDHelper only needs .unet
                pass
            p = _P()
            p.unet = self.unet
            self.deepcache = DeepCacheSDHelper(pipe=p)
            self.deepcache.set_params(cache_interval=3, cache_branch_id=0)
            self.deepcache.enable()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.noise = torch.randn((1, 4, 1, 64, 64), generator=g).to(self.device, self.dtype)  # one noise for all frames
        m = cv2.imread(str(LSW / "latentsync/utils/mask.png"))
        m = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_LANCZOS4) / 255.0
        self.mask = torch.from_numpy(m).permute(2, 0, 1).float()  # 1 keep, 0 = mouth region to generate
        self.mask_lat = F.interpolate(self.mask[None, :1], size=(64, 64)).to(self.device, self.dtype)
        # paste-back alpha in aligned-face space: generated mouth region, feathered, faded out near the crop border
        mouth = 1 - self.mask[0].numpy().astype(np.float32)
        mouth = cv2.GaussianBlur(cv2.dilate(mouth, np.ones((9, 9), np.uint8)), (0, 0), 7)
        border = np.zeros((RES, RES), np.float32)
        border[24:-24, 24:-24] = 1
        border = cv2.GaussianBlur(border, (0, 0), 9)
        self.alpha = torch.from_numpy(mouth * border)[None, None].to(self.device, self.dtype)
        self._det = None

    # ---------- prep (per base frame, offline) ----------
    def _detector(self):
        if self._det is None:
            from latentsync.utils.affine_transform import AlignRestore
            from latentsync.utils.face_detector import FaceDetector
            self._det = (FaceDetector(device="cuda"), AlignRestore(resolution=RES, device="cuda"))
        return self._det

    @torch.no_grad()
    def prep_frames(self, frames_rgb):
        """frames_rgb: iterable of HxWx3 uint8 RGB. Returns dict of per-frame tensors (CPU fp16 / fp32)."""
        det, ar = self._detector()
        ar.p_bias = None
        mats, ref_l, msk_l = [], [], []
        batch = []

        def flush():
            x = torch.stack(batch).to(self.device, self.dtype)  # [b,3,512,512] in [-1,1]
            ref_l.append(self._encode(x).cpu())
            msk_l.append(self._encode(x * self.mask.to(self.device, self.dtype)).cpu())
            batch.clear()

        for fr in frames_rgb:
            bbox, lm = det(fr)
            if bbox is None:
                raise RuntimeError("face not detected")
            pts = np.round([lm[[43, 48, 49, 51, 50]].mean(0), lm[101:106].mean(0), lm[[74, 77, 83, 86]].mean(0)])
            face, M = ar.align_warp_face(fr.copy(), landmarks3=pts, smooth=True)  # face: (560,420,3)
            fh, fw = face.shape[:2]
            S = np.array([[RES / fw, 0, 0], [0, RES / fh, 0], [0, 0, 1]], np.float64)
            A = S @ np.vstack([M[0].float().cpu().numpy(), [0, 0, 1]])  # frame -> 512 aligned face
            mats.append(A[:2].astype(np.float32))
            face = cv2.resize(face, (RES, RES), interpolation=cv2.INTER_LANCZOS4)
            batch.append(torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1)
            if len(batch) == 16:
                flush()
        if batch:
            flush()
        return {"mats": torch.from_numpy(np.stack(mats)), "ref": torch.cat(ref_l), "masked": torch.cat(msk_l)}

    def _encode(self, x):
        return self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

    # ---------- audio ----------
    @torch.no_grad()
    def audio_features(self, wav):
        """wav float32 16 kHz -> per-video-frame whisper features [T, 50, 384] (LatentSync audio_feat_length [2,2])."""
        from latentsync.whisper.whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
        mel = log_mel_spectrogram(torch.from_numpy(wav.astype(np.float32)))
        embs = []
        for seek in range(0, mel.shape[-1], 3000):
            end = min(seek + 3000, mel.shape[-1])
            seg = pad_or_trim(mel[:, seek:end], N_FRAMES).to(self.device).half()[None]
            x = F.gelu(self.whisper.encoder.conv1(seg))
            x = F.gelu(self.whisper.encoder.conv2(x)).permute(0, 2, 1)
            x = (x + self.whisper.encoder.positional_embedding).to(x.dtype)
            hs = [x]
            for blk in self.whisper.encoder.blocks:
                x = blk(x)
                hs.append(x)
            e = torch.stack(hs, dim=2)[0]  # [1500, 5, 384]
            embs.append(e[: (end - seek) // 2])
        feat = torch.cat(embs)  # [T50, 5, 384]
        n = math.floor(len(wav) / SR * FPS)
        L = len(feat)
        idx = torch.stack([torch.arange(int(i * 2) - 4, int(i * 2) + 6) for i in range(n)]).clamp(0, L - 1)
        return feat[idx.to(feat.device)].reshape(n, 50, 384)  # [n, 10*5, 384]

    # ---------- live ----------
    @torch.no_grad()
    def render(self, ref, masked, mats, frames, audio):
        """ref/masked: [b,4,64,64] latents, mats: [b,2,3] frame->face affine, frames: [b,H,W,3] uint8 RGB on GPU,
        audio: [b,50,384]. Returns lip-synced frames [b,H,W,3] uint8 on GPU. b <= 16 (one temporal window)."""
        import kornia
        b = ref.shape[0]
        dev, dt = self.device, self.dtype
        cfg = self.guidance > 1.0
        to5 = lambda t: t.permute(1, 0, 2, 3)[None]  # [b,c,h,w] -> [1,c,b,h,w]
        cond = torch.cat([self.mask_lat.expand(b, -1, -1, -1), masked.to(dev, dt), ref.to(dev, dt)], 1)
        cond = to5(cond)
        aud = audio.to(dev, dt)
        if cfg:
            cond, aud = torch.cat([cond] * 2), torch.cat([torch.zeros_like(aud), aud])
        self.sched.set_timesteps(self.steps, device=dev)
        lat = self.noise.repeat(1, 1, b, 1, 1) * self.sched.init_noise_sigma
        for t in self.sched.timesteps:
            x = torch.cat([lat] * 2) if cfg else lat
            x = torch.cat([self.sched.scale_model_input(x, t), cond], 1)
            eps = self.unet(x, t, encoder_hidden_states=aud).sample
            if cfg:
                u, c = eps.chunk(2)
                eps = u + self.guidance * (c - u)
            lat = self.sched.step(eps, t, lat).prev_sample
        lat = lat[0].permute(1, 0, 2, 3)[:, :, LAT_TOP:] / self.vae.config.scaling_factor
        dec = self.vae.decode(lat).sample  # [b,3,(64-LAT_TOP)*8,512] in [-1,1]
        face = torch.zeros((b, 3, RES, RES), device=dev, dtype=dt)
        face[:, :, LAT_TOP * 8:] = dec
        H, W = frames.shape[1:3]
        inv = kornia.geometry.transform.invert_affine_transform(mats.to(dev).float())
        src = torch.cat([face.float(), self.alpha.float().expand(b, -1, -1, -1)], 1)
        warped = kornia.geometry.transform.warp_affine(src, inv, (H, W), mode="bilinear", padding_mode="zeros")
        fr = frames.permute(0, 3, 1, 2).float()
        rgb = (warped[:, :3] / 2 + 0.5).clamp(0, 1) * 255
        a = warped[:, 3:4]
        out = fr + a * (rgb - fr)
        return out.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
