import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from aid_attention import (
    OuterConvergedAttnProcessor_SDPA,
    OuterConvergedAttnProcessor_SDPA2,
    OuterInterpolatedAttnProcessor_SDPA,
)
from aid_utils import (
    fourier_filter,
    generate_beta_tensor,
    linear_interpolation,
    load_im_from_path,
    spherical_interpolation,
)

import argparse, sys, os, math, re

@torch.no_grad()
def aid_inversion(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncon: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale=7,
):
    warmup_step = int(steps * 0.3)
    warmup_step2 = int(steps * 0.6)
    iter_latent = latent.clone()
    for i, t in enumerate(timesteps):
        if i > warmup_step and i < warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
        elif i > warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
        else:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(AttnProcessor2_0())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(AttnProcessor2_0())
        for _ in range(5):
            noise_pred_cond = unet(iter_latent, t, encoder_hidden_states=text_input_con).sample
            iter_latent = invert_scheduler.step(sample=latent, model_output=noise_pred_cond, timestep=t).prev_sample
        latent = iter_latent.clone()
    return latent


@torch.no_grad()
def aid_forward(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncond: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale: float = 3,
):
    attn_processor_dict = {
        "original": {
            "self_attn": AttnProcessor2_0(),
            "cross_attn": AttnProcessor2_0(),
        },
    }
    warmup_step1 = int(len(timesteps) * 0.2)
    warmup_step2 = int(len(timesteps) * 0.6)
    for i, t in enumerate(timesteps):
        if i < warmup_step1:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_cross_attn),
            }
        elif i < warmup_step2:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_cross_attn),
            }
        else:
            interpolate_attn_proc = {
                "self_attn": AttnProcessor2_0(),
                "cross_attn": AttnProcessor2_0(),
            }
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(interpolate_attn_proc["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(interpolate_attn_proc["cross_attn"])
        noise_pred_cond = unet(latent, t, encoder_hidden_states=text_input_con).sample
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(attn_processor_dict["original"]["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(attn_processor_dict["original"]["cross_attn"])
        noise_pred_uncond = unet(latent, t, encoder_hidden_states=text_input_uncond).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(sample=latent, model_output=noise_pred, timestep=t).prev_sample
    return latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path', required=True, type=str, help="path to the captioned json")
    args, extras = parser.parse_known_args()
    
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)
    model_name = "stabilityai/stable-diffusion-2-1"
    image_resolution = 768
    dtype_weight = torch.float16
    steps = 50
    edit_strength = 0.8
    guidance_scale = 7.5
    accelerater = Accelerator()
    this_gpu = accelerater.local_process_index
    # class_name = args.class_name
    # exp_name = args.class_name
    # save_dir = f"./eval_results/{exp_name}/freemorph"
    save_dir = "./eval_results/freemorph"
    os.makedirs(save_dir, exist_ok=True)
    device = accelerater.device
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device, dtype_weight)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device, dtype_weight)
    forward_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    forward_scheduler.set_timesteps(steps)
    invert_scheduler = DDIMInverseScheduler.from_pretrained(model_name, subfolder="scheduler")
    invert_scheduler.set_timesteps(steps)
    injection_noise = torch.randn((1, 4, 96, 96), device=device, dtype=dtype_weight)
    with open(args.json_path) as f:
        for idx, i in enumerate(tqdm(f, disable=not accelerater.is_local_main_process)):
            i = json.loads(i)
            image_paths = i["image_paths"]
            print(image_paths)
            prompts = i["prompts"]
            exp_id = i["exp_id"]
            latent_x_list = []
            text_input_con_list = []
            text_input_uncond_list = []
            original_images = []
            for img_idx in range(2):
                image_path_idx = image_paths[img_idx]
                image = load_im_from_path(image_path_idx, [768, 768]).to(device, dtype_weight)
                original_images.append(image)
                input_ids = tokenizer(
                    [prompts[img_idx], ""],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                ).input_ids.to(device)
                text_input = text_encoder(input_ids).last_hidden_state.to(device, dtype_weight)
                text_input_con, text_input_uncond = text_input.chunk(2, dim=0)
                text_input_con_list.append(text_input_con)
                text_input_uncond_list.append(text_input_uncond)
                latent_x = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent_x)

            interpolation_size = 7
            coef_attn = generate_beta_tensor(interpolation_size, alpha=20, beta=20)
            latent_x = spherical_interpolation(latent_x_list[0], latent_x_list[1], interpolation_size).squeeze(0)
            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(steps * edit_strength)]
            text_input_con_all = linear_interpolation(
                text_input_con_list[0], text_input_con_list[1], interpolation_size
            ).squeeze(0)
            text_input_uncond_all = linear_interpolation(
                text_input_uncond_list[0], text_input_uncond_list[1], interpolation_size
            ).squeeze(0)
            reverted_latent = aid_inversion(
                latent=latent_x,
                timesteps=invert_timesteps,
                text_input_con=text_input_con_all,
                text_input_uncon=text_input_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
            )
            thresholds = [48] + [44] + [42] * 2 + [42] + [42] * 2 + [44] + [48]
            new_input_latents_list = []
            for k in range(1, 7 - 1):
                new_input_latent = fourier_filter(
                    x=reverted_latent[k].unsqueeze(0),
                    y=injection_noise.clone(),
                    threshold=int(thresholds[k]),
                )
                new_input_latents_list.append(new_input_latent.clone())
            latent = torch.cat(
                [reverted_latent[0].unsqueeze(0)] + new_input_latents_list + [reverted_latent[-1].unsqueeze(0)]
            )
            forward_timesteps = forward_scheduler.timesteps - 1
            forward_timesteps = forward_timesteps[steps - int(steps * edit_strength) :]
            coef_attn = generate_beta_tensor(interpolation_size, alpha=20, beta=20)
            morphing_latent = aid_forward(
                forward_timesteps,
                latent=latent,
                text_input_con=text_input_con_all,
                text_input_uncond=text_input_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
                guidance_scale=guidance_scale,
            )
            images = []
            for x in morphing_latent:
                x = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                x = (x / 2 + 0.5).clamp(0, 1)
                x = x.detach().to(torch.float).cpu()
                images.append(x)
            images[0] = (original_images[0] / 2 + 0.5).detach().to(torch.float).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float).cpu()
            save_image(torch.cat(images), f"{save_dir}/{exp_id}.png")
