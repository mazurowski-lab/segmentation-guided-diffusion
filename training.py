"""
training utils
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import diffusers

from eval import evaluate, add_segmentations_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline

@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume_epoch: int = None

    # EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2 # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3 # w in the paper
    trans_noise_level: float = 0.5 # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    use_cfg_for_eval_conditioning: bool = True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    cfg_maskguidance_condmodel_only: bool = True  # if using mask guidance AND cfg, only give mask to conditional network
    # ^ this is because giving mask to both uncond and cond model make class guidance not work 
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    # logging
    run_name = '{}-{}-{}'.format(config.model_type.lower(), config.dataset, config.image_size)
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # for loading segs to condition on:
    eval_dataloader = iter(eval_dataloader)

    # Now you train the model
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            clean_images = clean_images.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(noisy_images, batch, config, device)

            # Predict the noise residual
            if config.class_conditional:
                class_labels = torch.ones(noisy_images.size(0)).long().to(device)
                # classifier-free guidance
                a = np.random.uniform()
                if a <= config.cfg_p_uncond:
                    class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
            else:
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # also train on target domain images if conditional
            # (we don't have masks for this domain, so we can't do segmentation-guided; just use blank masks)
            if config.class_conditional:
                target_domain_images = batch['images_target']
                target_domain_images = target_domain_images.to(device)

                # Sample noise to add to the images
                noise = torch.randn(target_domain_images.shape).to(target_domain_images.device)
                bs = target_domain_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=target_domain_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(target_domain_images, noise, timesteps)

                if config.segmentation_guided:
                    # no masks in target domain so just use blank masks
                    noisy_images = torch.cat((noisy_images, torch.zeros_like(noisy_images)), dim=1)

                # Predict the noise residual
                class_labels = torch.full([noisy_images.size(0)], 2).long().to(device)
                # classifier-free guidance
                a = np.random.uniform()
                if a <= config.cfg_p_uncond:
                    class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
                loss_target_domain = F.mse_loss(noise_pred, noise)
                loss_target_domain.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            if config.class_conditional:
                logs = {"loss": loss.detach().item(), "loss_target_domain": loss_target_domain.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                writer.add_scalar("loss_target_domain", loss.detach().item(), global_step)
            else: 
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            writer.add_scalar("loss", loss.detach().item(), global_step)

            progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                    )
            else:
                if config.class_conditional:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDPM")
                else:
                    pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
        elif config.model_type == "DDIM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                    )
            else:
                if config.class_conditional:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDIM")
                else:
                    pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)

        model.eval()

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            if config.segmentation_guided:
                seg_batch = next(eval_dataloader)
                evaluate(config, epoch, pipeline, seg_batch)
            else:
                evaluate(config, epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)
