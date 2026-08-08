"""Offline engine build on Modal (L40S) - the GPU build box for quantized variants.

Captures real calibration activations, builds a quantized TensorRT engine with them,
and writes the bundle straight into the serving Volume (fastdiff-engines) - so no S3
write is needed; the serving app picks it up on the next deploy. Optionally also
publishes to S3 (needs a write-capable secret).

  # INT8 base: capture calibration -> build with real scales -> write to the Volume
  modal run serverless/modal_build.py::build_variant --engine int8-base --precision int8

  # later: FP8 (needs ModelOpt wired in) / LoRA variants
  #   modal run serverless/modal_build.py::build_variant --engine fp8-base --precision fp8

First run is slow (builds the ~15 GB image once, downloads SDXL ~7 GB into the HF
cache Volume, then capture + calibrated build). Re-runs are much faster.
"""

from __future__ import annotations

import modal

app = modal.App("fastdiff-build")

engines = modal.Volume.from_name("fastdiff-engines", create_if_missing=True)   # serving reads this
hf_cache = modal.Volume.from_name("fastdiff-hf-cache", create_if_missing=True)  # SDXL weights cache

# NGC TensorRT 10.3 (must match the serving TRT major) + the offline build stack.
build_image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.08-py3")
    # Versions PINNED to the NGC tensorrt:24.08 base (CUDA 12.6, TensorRT 10.3).
    # Loose `>=` pins previously dragged in torch 2.13 + CUDA 13 libs on top of a
    # CUDA-12 base, which hung the build. Keep the whole stack in the torch-2.4 era.
    .pip_install(
        "torch==2.4.1",                  # cu124 -> matches the base's CUDA 12
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate>=0.33,<0.35",
        "onnx>=1.16,<1.18",
        "onnxscript>=0.1",
        "onnx-graphsurgeon>=0.5,<0.6",   # FP8 zero-point conversion
        "nvidia-modelopt[torch]>=0.19,<0.25",  # [torch] extra pulls pulp + quant deps
        "pulp",                          # modelopt searcher dep (belt-and-suspenders)
        "safetensors>=0.4",
        "boto3>=1.34",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # expandable_segments trims the fragmentation that pushed the folding-heavy UNet
    # export just over the L40S's 44 GB (it left ~750 MB reserved-but-unallocated).
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/root/pipelines",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("pipelines", "/root/pipelines", copy=True)
    # LoRA weights for the *-lora variants (build_bundle fuses them before export).
    .add_local_dir("inference/loras", "/root/pipelines/loras", copy=True)
)


@app.function(image=build_image, gpu="L40S", volumes={"/cache/hf": hf_cache}, timeout=1800)
def smoke_quant(base_model: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
    """Minimal repro of the ModelOpt calibration hang: load UNet -> one un-quantized
    forward (baseline) -> mtq.quantize with a single-forward loop. Reaches the exact
    hang in ~5 min (no capture/export/TRT), so fixes can be iterated cheaply."""
    import sys

    import torch

    sys.path.insert(0, "/root/pipelines")
    from diffusers import StableDiffusionXLPipeline

    print("[smoke] loading SDXL UNet ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    unet = pipe.unet.to("cuda").eval()
    print("[smoke] UNet loaded", flush=True)

    def batch():
        return dict(
            sample=torch.randn(2, 4, 128, 128, dtype=torch.float16, device="cuda"),
            timestep=torch.tensor([999.0, 999.0], dtype=torch.float16, device="cuda"),
            encoder_hidden_states=torch.randn(2, 77, 2048, dtype=torch.float16, device="cuda"),
            text_embeds=torch.randn(2, 1280, dtype=torch.float16, device="cuda"),
            time_ids=torch.randn(2, 6, dtype=torch.float16, device="cuda"),
        )

    def fwd(m):
        b = batch()
        print("[smoke] running ONE forward ...", flush=True)
        with torch.no_grad():
            m(b["sample"], b["timestep"], b["encoder_hidden_states"],
              added_cond_kwargs={"text_embeds": b["text_embeds"], "time_ids": b["time_ids"]})
        torch.cuda.synchronize()
        print("[smoke] forward DONE", flush=True)

    print("[smoke] baseline un-quantized forward:", flush=True)
    fwd(unet)
    print("[smoke] baseline OK -> now mtq.quantize (INT8) ...", flush=True)
    import modelopt.torch.quantization as mtq
    mtq.quantize(unet, mtq.INT8_DEFAULT_CFG, fwd)
    print("[smoke] mtq.quantize DONE -- no hang!", flush=True)


@app.function(
    image=build_image,
    gpu="L40S",
    volumes={"/engines": engines, "/cache/hf": hf_cache},
    timeout=10800,  # 3 h: SDXL download + capture + calibrated quantize + slow TRT build
)
def build_variant(
    engine: str = "int8-base",
    precision: str = "int8",
    lora: str = "",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 30,
    every: int = 3,
    calib_size: int = 0,        # 0 = ModelOpt default (int8:64, fp8:128); small = fast diagnostic
    capture_prompts: int = 0,   # 0 = all 30; small = fast diagnostic capture
    publish: bool = False,
) -> None:
    import pathlib
    import shutil
    import subprocess
    import sys

    sys.path.insert(0, "/root/pipelines")
    from build_engines import EngineSpec, build_bundle, publish_s3

    # 1. Capture real calibration activations (INT8 / FP8 only).
    calib_dir = None
    if precision in ("int8", "fp8"):
        calib_dir = pathlib.Path("/tmp/calib")
        print(f"[build] capturing calibration: {steps} steps, every {every} ...", flush=True)
        cap_cmd = [sys.executable, "/root/pipelines/capture_calib.py",
                   "--out", str(calib_dir), "--steps", str(steps), "--every", str(every),
                   "--base-model", base_model]
        if capture_prompts:
            cap_cmd += ["--limit-prompts", str(capture_prompts)]
        subprocess.run(cap_cmd, check=True)
        hf_cache.commit()  # persist SDXL weights so later builds skip the ~7 GB download

    # 2. Build the engine bundle with those scales.
    spec = EngineSpec(engine=engine, precision=precision, lora=(lora or None))
    out_root = pathlib.Path("/tmp/out")
    print(f"[build] building {engine} ({precision}) ...", flush=True)
    bundle = build_bundle(spec, base_model, out_root, pathlib.Path("/root/pipelines/loras"),
                          calib_dir=calib_dir, calib_size=(calib_size or None))

    # 3. Write into the serving Volume (skip the heavy onnx/ dir).
    dest = pathlib.Path("/engines") / engine
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in bundle.iterdir():
        if item.name == "onnx":
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, dest / item.name)
    engines.commit()
    print(f"[build] wrote {engine} into the serving Volume (fastdiff-engines)", flush=True)

    # 4. Optionally publish to S3 (needs a write-capable secret attached).
    if publish:
        publish_s3(bundle, "s3://quant-studio-engine-bucket")
        print(f"[build] also published {engine} to S3", flush=True)
