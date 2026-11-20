"""Val losses (musetalk/utils/val_metrics.py, same fixed held-out samples as train.py) for a given UNet, written to
<output_dir>/<exp_name>/val_baseline.json; train.py then plots them as val_base/* next to val/*.
Default UNet = the config's unet_init_path, i.e. the weights the run started from (stock MuseTalk 1.5 for stage 1).

  cd MuseTalk && ../envs/musetalk_trainer/bin/python ../musetalk_ft/scripts/val_baseline.py \
      --config configs/training/shubh_stage1.yaml [--unet path.pth] [--device cpu|cuda] [--out file.json]
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())  # MuseTalk repo root
from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402
from transformers import WhisperModel  # noqa: E402

from musetalk.utils.training_utils import Net, initialize_dataloaders, initialize_syncnet, initialize_vgg  # noqa: E402
from musetalk.utils.val_metrics import build_val_set, compute_val_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unet")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = OmegaConf.load(a.config)
    dev = torch.device(a.device)
    acc = SimpleNamespace(device=dev)
    unet_path = a.unet or cfg.unet_init_path
    out = a.out or os.path.join(cfg.output_dir, cfg.exp_name, "val_baseline.json")

    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder=cfg.vae_type).to(dev).eval()
    unet = UNet2DConditionModel(**json.load(open(os.path.join(cfg.pretrained_model_name_or_path,
                                                              cfg.unet_sub_folder, "musetalk.json"))))
    unet.load_state_dict(torch.load(unet_path, map_location="cpu"))
    net = Net(unet).to(dev).eval()
    wav2vec = WhisperModel.from_pretrained(cfg.whisper_path).to(dev).eval()
    vgg_IN, pyramid, downsampler = initialize_vgg(cfg, acc)
    syncnet = initialize_syncnet(cfg, acc, torch.float32)
    samples = build_val_set(initialize_dataloaders(cfg)["val_dataset"],
                            cfg.get("val_per_clip", 4 if cfg.data.n_sample_frames == 1 else 1))
    vm = compute_val_metrics(cfg, samples, net, vae, wav2vec, vgg_IN, pyramid, downsampler, syncnet, dev,
                             torch.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(vm, open(out, "w"), indent=1)
    print(f"{unet_path}: {len(samples)} samples in {time.time() - t0:.0f}s -> {out}")
    print(" ".join(f"{k}={v:.4f}" for k, v in vm.items()))


if __name__ == "__main__":
    main()
