"""fastdiff diffusers baseline on Modal - the framework-overhead reference the
TensorRT plane is measured against, and the place flash-attention actually runs
(torch SDPA / xFormers).

Same warm L40S + SSE contract as modal_app.py, but STUDIO_BACKEND=diffusers, so the
Registry runs the un-quantised PyTorch pipeline. Attention is switchable via the
ATTENTION constant (redeploy to A/B):
    sdpa    - torch SDPA -> flash-attention kernel   (flash ON, the default)
    vanilla - naive matmul-softmax-matmul            (flash OFF, the baseline)
    flash   - force the flash SDPA backend
    xformers- xFormers memory-efficient attention    (needs xformers in the image)

    modal deploy serverless/modal_diffusers.py
    python3 serverless/bench_compare.py --url <this-url> --label diffusers-sdpa \
        --variant fp16-base --n 20 --steps 30 --gpu L40S --price-per-hour 1.95

Then compare against the TRT engine (serverless/modal.json) to see the TRT win, and
against a `vanilla` redeploy to see the flash-attention win.
"""

from __future__ import annotations

import modal

APP_NAME = "fastdiff-diffusers"
ATTENTION = "sdpa"  # sdpa | vanilla | flash | xformers  -> redeploy to A/B

app = modal.App(APP_NAME)

# SDXL weights (~7 GB) download once into this Volume, not on every cold start.
hf_cache = modal.Volume.from_name("fastdiff-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "diffusers>=0.31",
        "transformers>=4.44",
        "accelerate>=0.33",
        "safetensors>=0.4",
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.30",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "pillow>=10.3",
        "numpy>=1.26,<2.3",
        "httpx>=0.27",  # in-container warm-up via TestClient
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env(
        {
            "PYTHONPATH": "/root/inference",
            "STUDIO_BACKEND": "diffusers",
            "STUDIO_ATTENTION": ATTENTION,
            "STUDIO_MAX_RESIDENT": "1",
            "HF_HOME": "/cache/huggingface",
            "STUDIO_CORS_ORIGINS": "https://fastdiff.vercel.app,http://localhost:3000",
        }
    )
    .workdir("/root/inference")
    .add_local_dir("inference", "/root/inference", copy=True)
)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/cache/huggingface": hf_cache},
    min_containers=1,      # keep one warm for stable p95
    max_containers=1,      # pin to one container (no cold scale-out)
    scaledown_window=300,
    timeout=1200,          # first cold start downloads SDXL (~7 GB)
)
@modal.concurrent(max_inputs=1)  # one gen at a time
class Diffusers:
    @modal.enter()
    def warm(self) -> None:
        from fastapi.testclient import TestClient

        from main import app as fastapi_app  # inference/main.py, picks DiffusersBackend via env

        self._app = fastapi_app
        try:
            client = TestClient(fastapi_app)
            with client.stream("POST", "/generate", json={
                "prompt": "warmup", "variantId": "fp16-base",
                "steps": 2, "seed": 1, "width": 1024, "height": 1024,
            }) as r:
                for _ in r.iter_lines():
                    pass
            hf_cache.commit()  # persist the downloaded weights
            print(f"diffusers warm-up complete (attention={ATTENTION})")
        except Exception as exc:  # noqa: BLE001
            print(f"warm-up skipped: {exc}")

    @modal.asgi_app()
    def serve(self):
        return self._app
