import argparse
import diffusers
import logging
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import warnings
import json
import random
import re
import shutil

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from datetime import datetime
from datetime import timedelta

from diffusers.utils import check_min_version
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from musetalk.utils.utils import (
    delete_additional_ckpt, 
    seed_everything, 
    get_mouth_region,
    get_mouth_crop,
    process_audio_features,
    save_models
)
from musetalk.loss.basic_loss import set_requires_grad
from musetalk.loss.syncnet import get_sync_loss
from musetalk.utils.val_metrics import build_val_set, compute_val_metrics
from musetalk.utils.training_utils import (
    initialize_models_and_optimizers,
    initialize_dataloaders,
    initialize_loss_functions,
    initialize_syncnet,
    initialize_vgg,
    validation
)

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

def find_resume_checkpoint(save_dir):
    """Newest complete full-state checkpoint dir (save_resume_state only renames checkpoint-N.tmp -> checkpoint-N
    once every file is written, so a crash mid-save can never be picked up)"""
    dirs = [d for d in os.listdir(save_dir) if re.fullmatch(r"checkpoint-\d+", d)]
    return os.path.join(save_dir, max(dirs, key=lambda d: int(d.split("-")[1]))) if dirs else None


def save_resume_state(accelerator, save_dir, global_step, keep):
    """Full training state (UNet, optimizer, LR scheduler, discriminators + their optimizers, RNG) for resuming.
    Called on every rank; keeps the newest `keep` checkpoint dirs."""
    tmp = os.path.join(save_dir, f"checkpoint-{global_step}.tmp")
    t0 = time.time()
    accelerator.save_state(tmp)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = tmp[:-len(".tmp")]
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)
        done = sorted((d for d in os.listdir(save_dir) if re.fullmatch(r"checkpoint-\d+", d)),
                      key=lambda d: int(d.split("-")[1]))
        for d in done[:-keep]:
            shutil.rmtree(os.path.join(save_dir, d), ignore_errors=True)
        print(f"Resume state saved: {final} in {time.time() - t0:.0f}s")
    accelerator.wait_for_everyone()


def one_rank_at_a_time(accelerator, what, fn):
    """Run fn on one rank after the other (host-RAM safety: never load model weights on all GPUs at once).
    fn must not contain collectives."""
    out = None
    for r in range(accelerator.num_processes):
        if accelerator.process_index == r:
            t0 = time.time()
            out = fn()
            avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 2**20
            print(f"[rank {r}] {what} done in {time.time() - t0:.0f}s, host RAM available {avail} GiB", flush=True)
        accelerator.wait_for_everyone()
    return out


def main(cfg):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    kwargs = DistributedDataParallelKwargs()
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=cfg.get("log_with", "tensorboard"),
        project_dir=os.path.join(save_dir, "./tensorboard"),
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        print('cfg.seed', cfg.seed, accelerator.process_index)
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32

    def load_all():
        model_dict = initialize_models_and_optimizers(cfg, accelerator, weight_dtype)
        dataloader_dict = initialize_dataloaders(cfg)
        loss_dict = initialize_loss_functions(cfg, accelerator, model_dict['scheduler_max_steps'])
        syncnet = initialize_syncnet(cfg, accelerator, weight_dtype)
        return model_dict, dataloader_dict, loss_dict, syncnet, initialize_vgg(cfg, accelerator)
    model_dict, dataloader_dict, loss_dict, syncnet, (vgg_IN, pyramid, downsampler) = one_rank_at_a_time(
        accelerator, "model loading", load_all)

    # Prepare everything with our `accelerator`.
    model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader'] = accelerator.prepare(
        model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader']
    )
    print("length train/val", len(dataloader_dict['train_dataloader']), len(dataloader_dict['val_dataloader']))
    # the discriminators and their optimizers/schedulers are not prepared by accelerate: register them so that
    # save_state/load_state carry them too (same order on save and load since it only depends on the config)
    for k in ('discriminator', 'optimizer_D', 'scheduler_D',
              'mouth_discriminator', 'mouth_optimizer_D', 'mouth_scheduler_D'):
        if loss_dict[k] is not None:
            accelerator.register_for_checkpointing(loss_dict[k])
    resume_path = find_resume_checkpoint(save_dir) if cfg.resume_from_checkpoint else None
    if accelerator.is_main_process:  # leftovers of saves interrupted by a crash
        for d in os.listdir(save_dir):
            if re.fullmatch(r"checkpoint-\d+\.tmp", d):
                shutil.rmtree(os.path.join(save_dir, d), ignore_errors=True)
    accelerator.wait_for_everyone()
    
    # Calculate training steps and epochs
    num_update_steps_per_epoch = math.ceil(
        len(dataloader_dict['train_dataloader']) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # Initialize trackers on the main process
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        if cfg.get("log_with") == "wandb":
            # a resumed run keeps logging into the same W&B run
            wandb_kwargs = {"name": f"{cfg.exp_name}-{run_time}", "dir": save_dir}
            run_id_file = os.path.join(save_dir, "wandb_run_id.txt")
            if resume_path and os.path.exists(run_id_file):
                wandb_kwargs = {"id": open(run_id_file).read().strip(), "resume": "allow", "dir": save_dir}
            accelerator.init_trackers(
                cfg.get("wandb_project", "musetalk"),
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_kwargs},
            )
            with open(run_id_file, "w") as f:
                f.write(accelerator.get_tracker("wandb", unwrap=True).id)
        else:
            accelerator.init_trackers(
                cfg.exp_name,
                init_kwargs={"mlflow": {"run_name": run_time}},
            )

    # Calculate total batch size
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"Num Epochs = {num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Load checkpoint if resuming training
    if resume_path:
        one_rank_at_a_time(accelerator, f"load_state {resume_path}", lambda: accelerator.load_state(resume_path))
        global_step = int(resume_path.rsplit("-", 1)[1])
        first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"Resuming from {resume_path} (step {global_step})")
    stop_training = False
    # discriminators step every d_update_every micro-steps (upstream: once per generator step = every grad-accum
    # cycle, which left a fresh D with ~10 updates per 250 steps and near-chance loss)
    d_update_every = cfg.discriminator_train_params.get("update_every", cfg.solver.gradient_accumulation_steps)
    if resume_path:  # the config's D lr wins over the one stored in the checkpoint
        for o, sch in (('optimizer_D', 'scheduler_D'), ('mouth_optimizer_D', 'mouth_scheduler_D')):
            if loss_dict[o] is not None:
                for g in loss_dict[o].param_groups:
                    g['lr'] = g['initial_lr'] = cfg.discriminator_train_params.lr
                loss_dict[sch].base_lrs = [cfg.discriminator_train_params.lr] * len(loss_dict[o].param_groups)
    # fixed held-out samples for val losses (built lazily on the main process); optional reference numbers of the
    # starting weights from musetalk_ft/scripts/val_baseline.py, plotted as val_base/*
    val_samples, launch_step = None, global_step
    val_base_file = os.path.join(save_dir, "val_baseline.json")

    # Initialize progress bar
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # Log model types
    print("log type of models")
    print("unet", model_dict['unet'].dtype)
    print("vae", model_dict['vae'].dtype)
    print("wav2vec", model_dict['wav2vec'].dtype)

    def get_ganloss_weight(step):
        """Calculate GAN loss weight based on training step"""
        if step < cfg.discriminator_train_params.start_gan:
            return 0.0
        else:
            return 1.0

    def get_dloss_weight(step):
        """Discriminators may start learning before the generator feels them (start_d < start_gan), so the generator
        is never judged by an untrained discriminator. Defaults to start_gan (upstream behaviour)."""
        return 0.0 if step < cfg.discriminator_train_params.get("start_d", cfg.discriminator_train_params.start_gan) else 1.0
        
    # Training loop
    for epoch in range(first_epoch, num_train_epochs):
        # Set models to training mode
        model_dict['unet'].train()
        if cfg.loss_params.gan_loss > 0:
            loss_dict['discriminator'].train()
        if cfg.loss_params.mouth_gan_loss > 0:
            loss_dict['mouth_discriminator'].train()

        # Initialize loss accumulators
        train_loss = 0.0
        train_loss_D = 0.0
        train_loss_D_mouth = 0.0
        l1_loss_accum = 0.0
        vgg_loss_accum = 0.0
        gan_loss_accum = 0.0
        gan_loss_accum_mouth = 0.0
        fm_loss_accum = 0.0
        sync_loss_accum = 0.0
        adapted_weight_accum = 0.0

        t_data_start = time.time()
        for step, batch in enumerate(dataloader_dict['train_dataloader']):
            t_data = time.time() - t_data_start
            t_model_start = time.time()

            with torch.no_grad():
                # Process input data
                pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                bsz, num_frames, c, h, w = pixel_values.shape
                
                # Process reference images
                ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                
                # Get face mask for GAN
                pixel_values_face_mask = batch['pixel_values_face_mask']
                
                # Process audio features
                audio_prompts = process_audio_features(cfg, batch, model_dict['wav2vec'], bsz, num_frames, weight_dtype)
                
                # Initialize adapted weight
                adapted_weight = 1
                
                # Process sync loss if enabled
                if cfg.loss_params.sync_loss > 0:
                    mels = batch['mel']
                    # Prepare frames for latentsync (combine channels and frames)
                    gt_frames = rearrange(pixel_values, 'b f c h w-> b (f c) h w')
                    # Use lower half of face for latentsync
                    height = gt_frames.shape[2]
                    gt_frames = gt_frames[:, :, height // 2:, :]
                    
                    # Get audio embeddings
                    audio_embed = syncnet.get_audio_embed(mels)
                    
                    # Calculate adapted weight based on audio-visual similarity
                    if cfg.use_adapted_weight:
                        vision_embed_gt = syncnet.get_vision_embed(gt_frames)
                        image_audio_sim_gt = F.cosine_similarity(
                            audio_embed, 
                            vision_embed_gt, 
                            dim=1
                        )[0]
                        
                        if image_audio_sim_gt < 0.05 or image_audio_sim_gt > 0.65:
                            if cfg.adapted_weight_type == "cut_off":
                                adapted_weight = 0.0  # Skip this batch
                                print(
                                    f"\nThe i-a similarity in step {global_step} is {image_audio_sim_gt}, set adapted_weight to {adapted_weight}.")
                            elif cfg.adapted_weight_type == "linear":
                                adapted_weight = image_audio_sim_gt
                            else:
                                print(f"unknown adapted_weight_type: {cfg.adapted_weight_type}")
                                adapted_weight = 1
                    
                    # Random frame selection for memory efficiency
                    max_start = 16 - cfg.num_backward_frames
                    frames_left_index = random.randint(0, max_start) if max_start > 0 else 0
                    frames_right_index = frames_left_index + cfg.num_backward_frames         
                else:
                    frames_left_index = 0
                    frames_right_index = cfg.data.n_sample_frames

                # Extract frames for backward pass
                pixel_values_backward = pixel_values[:, frames_left_index:frames_right_index, ...]
                ref_pixel_values_backward = ref_pixel_values[:, frames_left_index:frames_right_index, ...]
                pixel_values_face_mask_backward = pixel_values_face_mask[:, frames_left_index:frames_right_index, ...]
                audio_prompts_backward = audio_prompts[:, frames_left_index:frames_right_index, ...]
                
                # Encode target images
                frames = rearrange(pixel_values_backward, 'b f c h w-> (b f) c h w')
                latents = model_dict['vae'].encode(frames).latent_dist.mode()
                latents = latents * model_dict['vae'].config.scaling_factor
                latents = latents.float()

                # Create masked images
                masked_pixel_values = pixel_values_backward.clone()
                masked_pixel_values[:, :, :, h//2:, :] = -1
                masked_frames = rearrange(masked_pixel_values, 'b f c h w -> (b f) c h w')
                masked_latents = model_dict['vae'].encode(masked_frames).latent_dist.mode()
                masked_latents = masked_latents * model_dict['vae'].config.scaling_factor
                masked_latents = masked_latents.float()

                # Encode reference images
                ref_frames = rearrange(ref_pixel_values_backward, 'b f c h w-> (b f) c h w')
                ref_latents = model_dict['vae'].encode(ref_frames).latent_dist.mode()
                ref_latents = ref_latents * model_dict['vae'].config.scaling_factor
                ref_latents = ref_latents.float()

                # Prepare face mask and audio features
                pixel_values_face_mask_backward = rearrange(
                    pixel_values_face_mask_backward, 
                    "b f c h w -> (b f) c h w"
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    'b f c h w-> (b f) c h w'
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    '(b f) c h w -> (b f) (c h) w', 
                    b=bsz
                )

            # Apply reference dropout (currently inactive)
            dropout = nn.Dropout(p=cfg.ref_dropout_rate)
            ref_latents = dropout(ref_latents)

            # Prepare model inputs
            input_latents = torch.cat([masked_latents, ref_latents], dim=1)
            input_latents = input_latents.to(weight_dtype)
            timesteps = torch.tensor([0], device=input_latents.device)

            # Forward pass
            latents_pred = model_dict['net'](
                input_latents,
                timesteps,
                audio_prompts_backward,
            )
            latents_pred = (1 / model_dict['vae'].config.scaling_factor) * latents_pred
            image_pred = model_dict['vae'].decode(latents_pred).sample
            
            # Convert to float
            image_pred = image_pred.float()
            frames = frames.float()
            
            # Calculate L1 loss
            l1_loss = loss_dict['L1_loss'](frames, image_pred)
            l1_loss_accum += l1_loss.item()
            loss = cfg.loss_params.l1_loss * l1_loss * adapted_weight

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                if cfg.get("mouth_d_crop"):  # paper-style fixed window, no resizing (see get_mouth_crop)
                    frames_mouth, image_pred_mouth = get_mouth_crop(
                        frames, image_pred, pixel_values_face_mask_backward, tuple(cfg.mouth_d_crop))
                    pyramide_real_mouth = pyramid(frames_mouth)
                    pyramide_generated_mouth = pyramid(image_pred_mouth)
                else:
                    frames_mouth, image_pred_mouth = get_mouth_region(
                        frames, 
                        image_pred, 
                        pixel_values_face_mask_backward
                    )
                    pyramide_real_mouth = pyramid(downsampler(frames_mouth))
                    pyramide_generated_mouth = pyramid(downsampler(image_pred_mouth))

            # Process VGG loss if enabled
            if cfg.loss_params.vgg_loss > 0:
                pyramide_real = pyramid(downsampler(frames))
                pyramide_generated = pyramid(downsampler(image_pred))

                loss_IN = 0
                for scale in cfg.loss_params.pyramid_scale:
                    x_vgg = vgg_IN(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = vgg_IN(pyramide_real['prediction_' + str(scale)])
                    for i, weight in enumerate(cfg.loss_params.vgg_layer_weight):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        loss_IN += weight * value
                loss_IN /= sum(cfg.loss_params.vgg_layer_weight)
                loss += loss_IN * cfg.loss_params.vgg_loss * adapted_weight
                vgg_loss_accum += loss_IN.item()

            # Process GAN loss if enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], False)
                loss_G = 0.
                discriminator_maps_generated = loss_dict['discriminator'](pyramide_generated)
                discriminator_maps_real = loss_dict['discriminator'](pyramide_real)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum += loss_G.item()

                loss += loss_G * cfg.loss_params.gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss if enabled
                if cfg.loss_params.fm_loss[0] > 0:
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], False)
                loss_G = 0.
                mouth_discriminator_maps_generated = loss_dict['mouth_discriminator'](pyramide_generated_mouth)
                mouth_discriminator_maps_real = loss_dict['mouth_discriminator'](pyramide_real_mouth)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - mouth_discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum_mouth += loss_G.item()

                loss += loss_G * cfg.loss_params.mouth_gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss for mouth if enabled
                if cfg.loss_params.fm_loss[0] > 0:
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(mouth_discriminator_maps_real[key], mouth_discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()
        
            # Process sync loss if enabled
            if cfg.loss_params.sync_loss > 0:
                pred_frames = rearrange(
                    image_pred, '(b f) c h w-> b (f c) h w', f=pixel_values_backward.shape[1])
                pred_frames = pred_frames[:, :, height // 2 :, :]
                sync_loss, image_audio_sim_pred = get_sync_loss(
                    audio_embed, 
                    gt_frames, 
                    pred_frames, 
                    syncnet, 
                    adapted_weight,
                    frames_left_index=frames_left_index,
                    frames_right_index=frames_right_index,
                )
                sync_loss_accum += sync_loss.item()
                loss += sync_loss * cfg.loss_params.sync_loss * adapted_weight

            # Backward pass
            avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
            train_loss += avg_loss.item()
            accelerator.backward(loss)

            # Train discriminator if GAN loss is enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], True)
                loss_D = loss_dict['discriminator_full'](frames, image_pred.detach())
                avg_loss_D = accelerator.gather(loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D += avg_loss_D.item() / 1
                loss_D = loss_D * get_dloss_weight(global_step) * adapted_weight
                accelerator.backward(loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % d_update_every == 0:
                    loss_dict['optimizer_D'].step()
                    loss_dict['scheduler_D'].step()
                    loss_dict['optimizer_D'].zero_grad()

            # Train mouth discriminator if mouth GAN loss is enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], True)
                mouth_loss_D = loss_dict['mouth_discriminator_full'](
                    frames_mouth, image_pred_mouth.detach())
                avg_mouth_loss_D = accelerator.gather(
                    mouth_loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D_mouth += avg_mouth_loss_D.item() / 1
                mouth_loss_D = mouth_loss_D * get_dloss_weight(global_step) * adapted_weight
                accelerator.backward(mouth_loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['mouth_discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % d_update_every == 0:
                    loss_dict['mouth_optimizer_D'].step()
                    loss_dict['mouth_scheduler_D'].step()
                    loss_dict['mouth_optimizer_D'].zero_grad()

            # Update main model
            if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model_dict['trainable_params'],
                        cfg.solver.max_grad_norm,
                    )
                model_dict['optimizer'].step()
                model_dict['lr_scheduler'].step()
                model_dict['optimizer'].zero_grad()

            # Update progress and log metrics
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "train_loss_D": train_loss_D,
                    "train_loss_D_mouth": train_loss_D_mouth,
                    "l1_loss": l1_loss_accum,
                    "vgg_loss": vgg_loss_accum,
                    "gan_loss": gan_loss_accum,
                    "fm_loss": fm_loss_accum,
                    "sync_loss": sync_loss_accum,
                    "adapted_weight": adapted_weight_accum,
                    "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                }, step=global_step)

                # Reset loss accumulators
                train_loss = 0.0
                l1_loss_accum = 0.0
                vgg_loss_accum = 0.0
                gan_loss_accum = 0.0
                fm_loss_accum = 0.0
                sync_loss_accum = 0.0
                adapted_weight_accum = 0.0
                train_loss_D = 0.0
                train_loss_D_mouth = 0.0

                # Run validation if needed
                if global_step % cfg.val_freq == 0 or global_step == 10:
                    try:
                        validation(
                            cfg,
                            dataloader_dict['val_dataloader'],
                            model_dict['net'],
                            model_dict['vae'],
                            model_dict['wav2vec'],
                            accelerator,
                            save_dir,
                            global_step,
                            weight_dtype,
                            syncnet_score=adapted_weight,
                        )
                        if accelerator.is_main_process and cfg.get("log_with") == "wandb":
                            import glob
                            import wandb
                            samples = sorted(glob.glob(f"{save_dir}/samples/sample_{global_step}_*.jpg"))
                            if samples:
                                accelerator.get_tracker("wandb").log(
                                    {"val/sample": wandb.Image(samples[0], caption="rows: masked | ref | pred(train) | gt | pred(infer)")},
                                    step=global_step)
                    except Exception as e:
                        print(f"An error occurred during validation: {e}")

                # Held-out val losses (main process only; the other rank waits in its next all-reduce)
                if accelerator.is_main_process and (global_step % cfg.get("val_loss_every", cfg.val_freq) == 0
                                                    or global_step == launch_step + 1):
                    try:
                        t_val = time.time()
                        if val_samples is None:
                            val_samples = build_val_set(dataloader_dict['val_dataset'],
                                                        cfg.get("val_per_clip", 4 if cfg.data.n_sample_frames == 1 else 1))
                        vm = compute_val_metrics(cfg, val_samples, accelerator.unwrap_model(model_dict['net']),
                                                 model_dict['vae'], model_dict['wav2vec'], vgg_IN, pyramid,
                                                 downsampler, syncnet, accelerator.device, weight_dtype)
                        logs_val = {f"val/{k}": v for k, v in vm.items()}
                        if os.path.exists(val_base_file):
                            logs_val.update({f"val_base/{k}": v for k, v in json.load(open(val_base_file)).items()})
                        accelerator.log(logs_val, step=global_step)
                        print(f"\nval step {global_step} ({len(val_samples)} samples, {time.time() - t_val:.0f}s): "
                              + " ".join(f"{k}={v:.4f}" for k, v in vm.items()))
                    except Exception as e:
                        print(f"An error occurred during val loss: {e}")

                # Save checkpoint if needed
                if global_step % cfg.checkpointing_steps == 0:
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    try:
                        start_time = time.time()
                        if accelerator.is_main_process:
                            save_models(
                                accelerator, 
                                model_dict['net'],
                                save_dir, 
                                global_step, 
                                cfg, 
                                logger=logger
                            )
                            delete_additional_ckpt(save_dir, cfg.total_limit)
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 300:
                            print(f"Skipping storage as it took too long in step {global_step}.")
                        else:
                            print(f"Resume states saved at {save_dir} successfully in {elapsed_time}s.")
                    except Exception as e:
                        print(f"Error when saving model in step {global_step}:", e)

                # Full resume state every resume_every steps, or now if someone touched <save_dir>/STOP
                # (all ranks must agree, hence the reduce)
                stop = torch.tensor(float(os.path.exists(os.path.join(save_dir, "STOP"))), device=accelerator.device)
                stop_training = accelerator.reduce(stop, reduction="max").item() > 0
                if stop_training and global_step % cfg.checkpointing_steps != 0 and accelerator.is_main_process:
                    save_models(accelerator, model_dict['net'], save_dir, global_step, cfg, logger=logger)
                if global_step % cfg.get("resume_every", cfg.checkpointing_steps) == 0 or stop_training:
                    save_resume_state(accelerator, save_dir, global_step, cfg.get("resume_keep", 2))
                if stop_training:
                    accelerator.print(f"STOP file found: state saved at step {global_step}, exiting")
                    break

            # Update progress bar
            t_model = time.time() - t_model_start
            logs = {
                "step_loss": loss.detach().item(),
                "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                "td": f"{t_data:.2f}s",
                "tm": f"{t_model:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # Save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0:
            try:
                start_time = time.time()
                if accelerator.is_main_process:
                    save_models(accelerator, model_dict['net'], save_dir, global_step, cfg)
                    accelerator.save_state(save_path)
                elapsed_time = time.time() - start_time
                if elapsed_time > 120:
                    print(f"Skipping storage as it took too long in step {global_step}.")
                else:
                    print(f"Model saved successfully in {elapsed_time}s.")
            except Exception as e:
                print(f"Error when saving model in step {global_step}:", e)
        accelerator.wait_for_everyone()
        if stop_training:
            break

    # End training
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
