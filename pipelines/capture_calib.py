#!/usr/bin/env python3
"""Capture real UNet activations for INT8 / FP8 calibration.

Random calibration (torch.randn) gives meaningless INT8 scales -> degraded quality.
This runs fp16 SDXL on a curated, diverse prompt set and records the UNet's *actual*
inputs (sample, timestep, encoder_hidden_states, text_embeds, time_ids) at a spread
of denoise timesteps -- activation ranges vary a lot early vs late in the schedule,
so timestep coverage matters. The INT8 entropy calibrator and ModelOpt FP8 PTQ then
calibrate on these real batches.

GPU-only; runs on the Modal build box. Matches the build/serve model choices
(fp16-fix VAE, Euler). Output is a directory of batch_*.pt files (CFG batch = 2).

    python pipelines/capture_calib.py --out /work/calib --steps 30 --every 3
"""

from __future__ import annotations

import argparse
import pathlib

# ~30 diverse prompts across subjects, media, and lighting for broad activation
# coverage. Edit freely -- more diversity = better-calibrated scales.
PROMPTS = [
    "a lighthouse at dusk, dramatic sky",
    "portrait of an elderly fisherman, weathered face, natural light",
    "a bustling tokyo street at night, neon signs, rain reflections",
    "a bowl of ramen, steam rising, close-up food photography",
    "a red fox in a snowy forest, wildlife photography",
    "cyberpunk city skyline, flying cars, purple haze",
    "an oil painting of a sunflower field, impressionist style",
    "a cozy library interior, warm lamplight, wooden shelves",
    "a majestic lion on the savanna at golden hour",
    "abstract geometric pattern, vibrant colors, minimalist",
    "a vintage red sports car on a coastal road",
    "a fantasy castle on a floating island, cascading waterfalls",
    "macro photo of a dew-covered spider web at dawn",
    "an astronaut floating in space, earth in the background",
    "a watercolor landscape of rolling green hills",
    "a steaming cup of coffee on a rustic wooden table",
    "a futuristic humanoid robot in a laboratory, highly detailed",
    "a field of lavender under a bright blue sky",
    "black and white street photography, a lone figure walking",
    "a tropical beach with turquoise water and palm trees",
    "a detailed pencil sketch of a galloping horse",
    "a plate of colorful macarons on white marble",
    "a mountain peak at sunrise, alpine glow, dramatic clouds",
    "a golden retriever puppy playing in the grass",
    "a neon-lit arcade interior, retro 80s aesthetic",
    "a bouquet of roses in a glass vase, soft focus",
    "an ancient temple overgrown with vines, deep jungle",
    "a chef plating a gourmet dish, professional kitchen",
    "a starry night sky over a desert, the milky way",
    "a stack of old leather books beside a candle",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for batch_*.pt files")
    ap.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--every", type=int, default=3, help="capture every Nth denoise step")
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--limit-prompts", type=int, default=0, help="cap prompt count (0 = all)")
    args = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    state = {"step": 0, "saved": 0}

    def _pre_hook(_module, pos, kw):
        step = state["step"]
        state["step"] += 1
        if step % args.every != 0:
            return
        sample = pos[0] if len(pos) > 0 else kw["sample"]
        timestep = pos[1] if len(pos) > 1 else kw["timestep"]
        ehs = pos[2] if len(pos) > 2 else kw["encoder_hidden_states"]
        ack = kw.get("added_cond_kwargs", {})
        text_embeds, time_ids = ack["text_embeds"], ack["time_ids"]

        ts = torch.as_tensor(timestep, device=sample.device).reshape(-1)
        if ts.numel() == 1:
            ts = ts.expand(sample.shape[0])

        batch = {
            "sample": sample.detach().to("cpu", torch.float16),
            "timestep": ts.detach().to("cpu", torch.float16),
            "encoder_hidden_states": ehs.detach().to("cpu", torch.float16),
            "text_embeds": text_embeds.detach().to("cpu", torch.float16),
            "time_ids": time_ids.detach().to("cpu", torch.float16),
        }
        torch.save(batch, out / f"batch_{state['saved']:05d}.pt")
        state["saved"] += 1

    handle = pipe.unet.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    prompts = PROMPTS[: args.limit_prompts] if args.limit_prompts else PROMPTS
    try:
        for i, prompt in enumerate(prompts):
            state["step"] = 0
            gen = torch.Generator(device="cuda").manual_seed(1000 + i)
            with torch.inference_mode():
                pipe(
                    prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                    width=1024, height=1024, generator=gen, output_type="latent",
                )
            print(f"[{i + 1}/{len(prompts)}] {prompt[:44]!r} -> {state['saved']} batches", flush=True)
    finally:
        handle.remove()
    print(f"captured {state['saved']} calibration batches into {out}")


if __name__ == "__main__":
    main()
