"""Phase 7: fine-tune the SD-VAE DECODER on the shubh data with the stage-2 UNet frozen (beard/stubble/teeth detail).

Why: the 8x VAE already smooths individual stubble hairs (VAE-only reconstruction of real frames loses them), so even
a perfect UNet cannot draw them. Training only decoder (+ post_quant_conv) keeps the latent space, so the frozen UNet
and the encoder are untouched and inference cost is identical (same architecture, new weights).

Each step decodes two kinds of latents against the real 256 crop:
  * the frozen UNet's prediction (masked GT upper half + reference + audio) -> learn to render *our* latents sharply
  * the encoder's latent of the GT crop                                   -> stay a faithful autoencoder
Losses: L1 + VGG pyramid perceptual (like MuseTalk) + LSGAN with a face and a 64x128 mouth-crop discriminator
(start after gan_start steps, D trained from step 0, every step). Val: musetalk/utils/val_metrics.py on the stage-2
val set with the decoder under training (val/*) vs the original decoder (val_base/*), every --val-every steps.
Resume: newest <out>/state-N.pt (decoder, optimizers, discriminators, step) is loaded automatically; STOP file ->
save + exit. One process, one GPU, models loaded once (host-RAM rule).

  cd MuseTalk && CUDA_VISIBLE_DEVICES=1 WANDB_BASE_URL=https://api.forge.coreweave.com \
    ../envs/musetalk_trainer/bin/python ../musetalk_ft/scripts/train_vae_decoder.py \
    --unet exp_out/shubh/shubh_s2/unet-10000.pth --out exp_out/shubh/vae_dec/run1
Output: <out>/vae-N/ (diffusers AutoencoderKL dir; use with inference --vae_type ../<out>/vae-N or copy to models/)
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())  # MuseTalk repo root
from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402
from transformers import WhisperModel  # noqa: E402

from musetalk.loss.discriminator import DiscriminatorFullModel, MultiScaleDiscriminator  # noqa: E402
from musetalk.utils.training_utils import Net, initialize_dataloaders, initialize_syncnet, initialize_vgg  # noqa: E402
from musetalk.utils.utils import get_mouth_crop, process_audio_features  # noqa: E402
from musetalk.utils.val_metrics import build_val_set, compute_val_metrics  # noqa: E402


def vgg_loss(cfg, vgg_IN, pyramid, downsampler, gt, pred):
    pr, pg = pyramid(downsampler(gt)), pyramid(downsampler(pred))
    v = 0
    for scale in cfg.loss_params.pyramid_scale:
        xv, yv = vgg_IN(pg['prediction_' + str(scale)]), vgg_IN(pr['prediction_' + str(scale)])
        for j, w in enumerate(cfg.loss_params.vgg_layer_weight):
            v += w * (xv[j] - yv[j].detach()).abs().mean()
    return v / sum(cfg.loss_params.vgg_layer_weight)


def g_adv(D, x):
    return ((1 - D(D_in(x))['prediction_map_1']) ** 2).mean()


def D_in(x):
    return {'prediction_1': x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True, help="frozen stage-2 UNet checkpoint")
    ap.add_argument("--config", default="configs/training/shubh_stage2.yaml", help="data/crop/val settings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8, help="clips per step (x 1 frame)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--w-vgg", type=float, default=0.02)
    ap.add_argument("--w-gan", type=float, default=0.05)
    ap.add_argument("--w-mouth-gan", type=float, default=0.05)
    ap.add_argument("--gan-start", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=5, help="vae-N dirs kept")
    ap.add_argument("--wandb-project", default="musetalk-shubh")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    cfg = OmegaConf.load(a.config)

    # data: same crops as stage 2 but single frames (T=1) for many different faces per batch
    dcfg = OmegaConf.merge(cfg, {"data": {"n_sample_frames": 1, "train_bs": a.bs, "num_workers": 8}})
    dl = initialize_dataloaders(dcfg)
    train_dl = dl["train_dataloader"]

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder=cfg.vae_type).to(dev)
    vae.requires_grad_(False)
    dec_params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    for p in dec_params:
        p.requires_grad_(True)
    vae_base = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder=cfg.vae_type).to(dev).eval()
    vae_base.requires_grad_(False)
    unet = UNet2DConditionModel(**json.load(open(os.path.join(cfg.pretrained_model_name_or_path, cfg.unet_sub_folder,
                                                              "musetalk.json"))))
    unet.load_state_dict(torch.load(a.unet, map_location="cpu"))
    net = Net(unet).to(dev).eval()
    net.requires_grad_(False)
    wav2vec = WhisperModel.from_pretrained(cfg.whisper_path).to(dev).eval()
    acc = type("A", (), {"device": dev})()
    vgg_IN, pyramid, downsampler = initialize_vgg(cfg, acc)
    syncnet = initialize_syncnet(cfg, acc, torch.float32)
    dp = dict(cfg.model_params.discriminator_params)
    D, Dm = MultiScaleDiscriminator(**dp).to(dev), MultiScaleDiscriminator(**dp).to(dev)
    D_full, Dm_full = DiscriminatorFullModel(D), DiscriminatorFullModel(Dm)
    opt = torch.optim.AdamW(dec_params, lr=a.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    opt_D = torch.optim.AdamW(list(D.parameters()) + list(Dm.parameters()), lr=a.d_lr, betas=(0.5, 0.999),
                              weight_decay=1e-2)

    step = 0
    states = sorted((f for f in os.listdir(a.out) if re.fullmatch(r"state-\d+\.pt", f)),
                    key=lambda f: int(f[6:-3]))
    if states:
        st = torch.load(os.path.join(a.out, states[-1]), map_location="cpu")
        vae.decoder.load_state_dict(st["decoder"])
        vae.post_quant_conv.load_state_dict(st["post_quant_conv"])
        D.load_state_dict(st["D"])
        Dm.load_state_dict(st["Dm"])
        opt.load_state_dict(st["opt"])
        opt_D.load_state_dict(st["opt_D"])
        step = st["step"]
        print(f"resumed from {states[-1]} (step {step})", flush=True)

    import wandb
    run_id_file = os.path.join(a.out, "wandb_run_id.txt")
    kw = dict(project=a.wandb_project, dir=a.out, config={**vars(a), "stage2_config": a.config})
    if os.path.exists(run_id_file) and states:
        kw.update(id=open(run_id_file).read().strip(), resume="allow")
    else:
        kw["name"] = a.name or f"vae_dec-{os.path.basename(a.out.rstrip('/'))}"
    run = wandb.init(**kw)
    open(run_id_file, "w").write(run.id)

    # val on the stage-2 val set (16-frame clips -> sync metric too), not the single-frame training loader's
    val_samples = build_val_set(initialize_dataloaders(cfg)["val_dataset"], cfg.get("val_per_clip", 1))

    def val():
        vm = compute_val_metrics(cfg, val_samples, net, vae, wav2vec, vgg_IN, pyramid, downsampler, syncnet, dev,
                                 torch.float32)
        return vm

    base = compute_val_metrics(cfg, val_samples, net, vae_base, wav2vec, vgg_IN, pyramid, downsampler, syncnet, dev,
                               torch.float32)
    del vae_base
    torch.cuda.empty_cache()
    print("val_base " + " ".join(f"{k}={v:.4f}" for k, v in base.items()), flush=True)

    def save(step, final=False):
        tmp = os.path.join(a.out, f"state-{step}.pt.tmp")
        torch.save({"decoder": vae.decoder.state_dict(), "post_quant_conv": vae.post_quant_conv.state_dict(),
                    "D": D.state_dict(), "Dm": Dm.state_dict(), "opt": opt.state_dict(), "opt_D": opt_D.state_dict(),
                    "step": step}, tmp)
        os.replace(tmp, tmp[:-4])
        for f in sorted((f for f in os.listdir(a.out) if re.fullmatch(r"state-\d+\.pt", f)),
                        key=lambda f: int(f[6:-3]))[:-2]:
            os.remove(os.path.join(a.out, f))
        vae.save_pretrained(os.path.join(a.out, f"vae-{step}"))
        dirs = sorted((d for d in os.listdir(a.out) if re.fullmatch(r"vae-\d+", d)), key=lambda d: int(d[4:]))
        for d in dirs[:-a.keep]:
            shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        print(f"saved step {step}{' (final)' if final else ''}", flush=True)

    sf = vae.config.scaling_factor
    t0 = time.time()
    while step < a.max_steps:
        for batch in train_dl:
            gt = batch["pixel_values_vid"].to(dev).float()          # b,1,3,256,256
            ref = batch["pixel_values_ref_img"].to(dev).float()
            mask = rearrange(batch["pixel_values_face_mask"], 'b f c h w -> (b f) c h w').to(dev)
            b, f, _, h, _ = gt.shape
            with torch.no_grad():
                audio = process_audio_features(cfg, {k: v.to(dev) if torch.is_tensor(v) else v
                                                     for k, v in batch.items()}, wav2vec, b, f, torch.float32)
                audio = rearrange(rearrange(audio, 'b f c h w -> (b f) c h w'), '(b f) c h w -> (b f) (c h) w', b=b)
                gt2 = rearrange(gt, 'b f c h w -> (b f) c h w')
                masked = gt2.clone()
                masked[:, :, h // 2:] = -1
                z_m = vae.encode(masked).latent_dist.mode() * sf
                z_r = vae.encode(rearrange(ref, 'b f c h w -> (b f) c h w')).latent_dist.mode() * sf
                z_unet = net(torch.cat([z_m, z_r], 1), torch.tensor([0], device=dev), audio) / sf
                z_gt = vae.encode(gt2).latent_dist.mode()
            z = torch.cat([z_unet, z_gt])
            target = torch.cat([gt2, gt2])
            mk = torch.cat([mask, mask])
            pred = vae.decode(z).sample.float()
            gan_on = step >= a.gan_start
            l1 = (pred - target).abs().mean()
            lv = vgg_loss(cfg, vgg_IN, pyramid, downsampler, target, pred)
            rm, pm = get_mouth_crop(target, pred, mk, (64, 128))
            D.requires_grad_(False)
            Dm.requires_grad_(False)
            lg, lgm = g_adv(D, pred), g_adv(Dm, pm)
            loss = l1 + a.w_vgg * lv + (a.w_gan * lg + a.w_mouth_gan * lgm) * gan_on
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
            opt.step()
            D.requires_grad_(True)
            Dm.requires_grad_(True)
            ld = D_full(target, pred.detach()) + Dm_full(rm, pm.detach())
            opt_D.zero_grad(set_to_none=True)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(list(D.parameters()) + list(Dm.parameters()), 1.0)
            opt_D.step()
            step += 1
            logs = {"vae/l1": l1.item(), "vae/l1_unet_lat": (pred[:b] - target[:b]).abs().mean().item(),
                    "vae/l1_gt_lat": (pred[b:] - target[b:]).abs().mean().item(), "vae/vgg": lv.item(),
                    "vae/gan": lg.item(), "vae/gan_mouth": lgm.item(), "vae/loss_D": ld.item(),
                    "vae/loss": loss.item()}
            if step % a.val_every == 0 or step == 1:
                vm = val()
                logs.update({f"val/{k}": v for k, v in vm.items()})
                logs.update({f"val_base/{k}": v for k, v in base.items()})
                print(f"step {step} ({(time.time() - t0) / 60:.1f} min) val " +
                      " ".join(f"{k}={v:.4f}" for k, v in vm.items()), flush=True)
            wandb.log(logs, step=step)
            stop = os.path.exists(os.path.join(a.out, "STOP"))
            if step % a.save_every == 0 or stop:
                save(step, final=stop)
            if stop:
                print("STOP file found, exiting", flush=True)
                wandb.finish()
                return
            if step >= a.max_steps:
                break
    save(step, final=True)
    wandb.finish()


if __name__ == "__main__":
    main()
