"""fastdiff on Modal - serverless L40S, kept warm, TensorRT engines from a Volume.

Reuses inference/main.py's FastAPI app verbatim. Modal only supplies the L40S GPU,
a warm container (min_containers=1) for a stable p95, and the prebuilt engine
bundles on a persistent Volume. HF-free at serve time - engines run as .plan files.

Run everything from the REPO ROOT (paths below are repo-relative):

  # 0. one-time: store the read-only S3 key as a Modal secret
  modal secret create aws-s3-readonly \
      AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-southeast-5

  # 1. one-time per engine: pull the bundle from S3 into the Volume
  modal run serverless/modal_app.py::sync_engines --engine fp16-base

  # 2. deploy the warm endpoint (prints the https URL)
  modal deploy serverless/modal_app.py
"""

from __future__ import annotations

import modal

APP_NAME = "fastdiff-inference"
ENGINE_MOUNT = "/engines"
S3_BUCKET = "quant-studio-engine-bucket"

app = modal.App(APP_NAME)

# Persistent Volume holding the engine bundles (one dir per serving.engine).
engines = modal.Volume.from_name("fastdiff-engines", create_if_missing=True)

# Read-only S3 credentials (created in phase 0 from the IAM key). Used only by the
# one-time sync function, never by the serving container.
s3_secret = modal.Secret.from_name("aws-s3-readonly")

# Serving image: NGC TensorRT 10.3 (must match the engines' build-time TRT major)
# + torch + the API stack. transformers ships only for the offline CLIP tokenizer.
serve_image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.08-py3")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.44",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.30",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "pillow>=10.3",
        "numpy>=1.26,<2.3",
        "httpx>=0.27",  # for the in-container warm-up via TestClient
    )
    .env(
        {
            "PYTHONPATH": "/root/inference",  # make main/pipelines importable
            "STUDIO_ENGINE_DIR": ENGINE_MOUNT,
            "STUDIO_MAX_RESIDENT": "1",
            "STUDIO_CORS_ORIGINS": "https://fastdiff.vercel.app,http://localhost:3000",
        }
    )
    .workdir("/root/inference")
    .add_local_dir("inference", "/root/inference", copy=True)
)

# Lightweight CPU image for the S3 -> Volume sync (no GPU, no torch).
sync_image = modal.Image.debian_slim().pip_install("boto3>=1.34")


@app.function(
    image=sync_image,
    volumes={ENGINE_MOUNT: engines},
    secrets=[s3_secret],
    timeout=3600,
)
def sync_engines(engine: str = "fp16-base") -> None:
    """Download one engine bundle from S3 into the Volume. Run once per engine."""
    import pathlib

    import boto3

    s3 = boto3.client("s3")
    prefix = f"{engine}/"
    dest_root = pathlib.Path(ENGINE_MOUNT)
    paginator = s3.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. "fp16-base/unet.plan"
            out = dest_root / key  # -> /engines/fp16-base/unet.plan
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(out))
            count += 1
            print(f"  {key} -> {out}")
    engines.commit()
    print(f"synced {count} objects for engine '{engine}' into the Volume")


@app.cls(
    image=serve_image,
    gpu="L40S",
    volumes={ENGINE_MOUNT: engines},
    min_containers=1,      # keep one warm -> stable warm p95 (the benchmark's point)
    scaledown_window=300,  # if we ever allow >1, idle extras die after 5 min
    timeout=600,
)
class Inference:
    @modal.enter()
    def warm(self) -> None:
        # The engine bundle is already on the Volume. Do a tiny generation so the
        # TRT engine is deserialized and resident in VRAM before real traffic.
        from fastapi.testclient import TestClient

        from main import app as fastapi_app  # inference/main.py

        self._app = fastapi_app
        try:
            client = TestClient(fastapi_app)
            with client.stream(
                "POST",
                "/generate",
                json={
                    "prompt": "warmup",
                    "variantId": "fp16-base",
                    "steps": 2,
                    "seed": 1,
                    "width": 1024,
                    "height": 1024,
                },
            ) as r:
                for _ in r.iter_lines():
                    pass
            print("warm-up generation complete; engine resident")
        except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
            print(f"warm-up skipped: {exc}")

    @modal.asgi_app()
    def serve(self):
        return self._app
