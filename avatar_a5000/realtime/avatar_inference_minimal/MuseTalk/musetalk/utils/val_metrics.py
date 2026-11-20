"""Held-out validation losses for fine-tuning (added for the shubh fine-tune; upstream only saves a sample image).

build_val_set() draws a FIXED list of samples from the val dataset (seeded, every val clip equally often, global
RNG state restored afterwards), so every evaluation sees exactly the same frames/crops/audio and the curve is
comparable across steps and runs. compute_val_metrics() returns
  l1          same L1 as the training loss (whole 256x256 crop, inference-style input: GT upper half + ref)
  l1_lower    L1 on the generated lower half only (what MuseTalk actually paints)
  l1_mouth    L1 inside the mouth mask (dataset's pixel_values_face_mask box, resized to 256 like the mouth GAN)
  vgg         the training VGG perceptual loss (pyramid, same layer weights)
  sharp_ratio Laplacian variance of pred / GT on the lower half (1.0 = as sharp as GT, < 1 = blurry)
  sync        SyncNet cosine loss of the predicted 16-frame window vs its audio (only with a SyncNet and T == 16)
  loss        l1_loss * l1 + vgg_loss * vgg (+ sync_loss * sync), i.e. the training objective without GAN terms
"""
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import default_collate

from musetalk.loss.syncnet import get_sync_loss
from musetalk.utils.utils import get_image_pred, get_mouth_region, process_audio_features

_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def build_val_set(val_dataset, per_clip, seed=1234):
    if hasattr(val_dataset, "datasets"):  # PortraitDataset returns a ConcatDataset
        val_dataset = val_dataset.datasets[0]
    paths = sorted(set(val_dataset.meta_paths))
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_paths, samples = val_dataset.meta_paths, []
    try:
        for p in paths:
            with open(p) as f:  # FaceDataset needs >= T * 10 frames (16-frame stage 2: >= 160)
                if json.load(f)["frames"] < val_dataset.T * 10:
                    continue
            val_dataset.meta_paths = [p]
            for _ in range(per_clip):
                try:
                    s = val_dataset[0]
                except ValueError:  # no valid sample in this clip
                    break
                samples.append({k: v for k, v in s.items() if k != 'wav_path'})
    finally:
        val_dataset.meta_paths = all_paths
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    return samples


def _lap_var(x):
    g = x.mean(1, keepdim=True)
    return F.conv2d(g, _LAPLACE.to(g), padding=1).flatten(1).var(1)


@torch.no_grad()
def compute_val_metrics(cfg, samples, net, vae, wav2vec, vgg_IN, pyramid, downsampler, syncnet, device,
                        weight_dtype, bs=8):
    was_training = net.training
    net.eval()
    sums, n = defaultdict(float), 0
    T = cfg.data.n_sample_frames
    bs = max(1, bs // T) if T > 1 else bs
    for i in range(0, len(samples), bs):
        batch = default_collate(samples[i:i + bs])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pixel_values = batch["pixel_values_vid"].to(weight_dtype)
        ref = batch["pixel_values_ref_img"].to(weight_dtype)
        b, f, _, h, _ = pixel_values.shape
        audio = process_audio_features(cfg, batch, wav2vec, b, f, weight_dtype)
        audio = rearrange(rearrange(audio, 'b f c h w -> (b f) c h w'), '(b f) c h w -> (b f) (c h) w', b=b)
        pred = get_image_pred(pixel_values, ref, audio, vae, net, weight_dtype).float()
        gt = rearrange(pixel_values, 'b f c h w -> (b f) c h w').float()
        mask = rearrange(batch["pixel_values_face_mask"], 'b f c h w -> (b f) c h w')

        m = {"l1": (gt - pred).abs().mean(),
             "l1_lower": (gt[:, :, h // 2:] - pred[:, :, h // 2:]).abs().mean()}
        gm, pm = get_mouth_region(gt, pred, mask)
        if gm is not None and len(gm):
            m["l1_mouth"] = (gm - pm).abs().mean()
        m["sharp_ratio"] = (_lap_var(pred[:, :, h // 2:]) / _lap_var(gt[:, :, h // 2:]).clamp_min(1e-8)).mean()
        if vgg_IN is not None:
            pr, pg = pyramid(downsampler(gt)), pyramid(downsampler(pred))
            v = 0
            for scale in cfg.loss_params.pyramid_scale:
                xv, yv = vgg_IN(pg['prediction_' + str(scale)]), vgg_IN(pr['prediction_' + str(scale)])
                for j, w in enumerate(cfg.loss_params.vgg_layer_weight):
                    v += w * (xv[j] - yv[j]).abs().mean()
            m["vgg"] = v / sum(cfg.loss_params.vgg_layer_weight)
        if syncnet is not None and f == 16:
            audio_embed = syncnet.get_audio_embed(batch["mel"].to(weight_dtype))
            gt_s = rearrange(pixel_values, 'b f c h w -> b (f c) h w')[:, :, h // 2:]
            pred_s = rearrange(pred, '(b f) c h w -> b (f c) h w', f=f)[:, :, h // 2:]
            m["sync"], _ = get_sync_loss(audio_embed, gt_s, pred_s.to(weight_dtype), syncnet, 1, 0, 16)
        for k, v in m.items():
            sums[k] += float(v) * b
        n += b
    out = {k: v / n for k, v in sums.items()}
    out["loss"] = cfg.loss_params.l1_loss * out["l1"] + cfg.loss_params.vgg_loss * out.get("vgg", 0.0)
    if "sync" in out:
        out["loss"] += cfg.loss_params.sync_loss * out["sync"]
    if was_training:
        net.train()
    return out
