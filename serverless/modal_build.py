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
    .pip_install(
        "torch>=2.4",
        "diffusers>=0.31",
        "transformers>=4.44",
        "accelerate>=0.33",
        "onnx>=1.16",
        "onnxscript>=0.1",
        "safetensors>=0.4",
        "boto3>=1.34",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/root/pipelines"})
    .add_local_dir("pipelines", "/root/pipelines", copy=True)
)


@app.function(
    image=build_image,
    gpu="L40S",
    volumes={"/engines": engines, "/cache/hf": hf_cache},
    timeout=5400,  # 90 min: first run downloads SDXL + captures + calibrated build
)
def build_variant(
    engine: str = "int8-base",
    precision: str = "int8",
    lora: str = "",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 30,
    every: int = 3,
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
        subprocess.run(
            [sys.executable, "/root/pipelines/capture_calib.py",
             "--out", str(calib_dir), "--steps", str(steps), "--every", str(every),
             "--base-model", base_model],
            check=True,
        )

    # 2. Build the engine bundle with those scales.
    spec = EngineSpec(engine=engine, precision=precision, lora=(lora or None))
    out_root = pathlib.Path("/tmp/out")
    print(f"[build] building {engine} ({precision}) ...", flush=True)
    bundle = build_bundle(spec, base_model, out_root, pathlib.Path("/root/pipelines/loras"),
                          calib_dir=calib_dir)

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
