"""Runtime configuration, sourced from the environment.

The service has two execution planes:

* **real** — prebuilt TensorRT engines on a CUDA GPU, measuring honest metrics.
  No Hugging Face / diffusers at serving time: engine bundles are fetched from
  ``STUDIO_ENGINE_S3_URI`` into ``STUDIO_ENGINE_DIR`` (by an init container).
* **demo** — a CPU-friendly fallback (no engines) that simulates the pipeline so
  the frontend is fully exercisable in local dev. Demo metrics are derived from
  the registry benchmark and are clearly logged as simulated.

The real plane is selected automatically when CUDA + TensorRT are available and
``STUDIO_DEMO`` is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# Force the demo plane even if a GPU is present (useful for UI work).
FORCE_DEMO: bool = _flag("STUDIO_DEMO")

# Path to the variant catalog.
VARIANTS_FILE: Path = Path(os.getenv("STUDIO_VARIANTS", str(HERE / "variants.yaml")))

# Local directory holding the prebuilt TensorRT engine bundles (one dir per
# `serving.engine`). Populated from S3 by an init container in production.
ENGINE_DIR: Path = Path(os.getenv("STUDIO_ENGINE_DIR", str(HERE / "engines")))

# S3 (or s3-compatible) URI the engine bundles are published to by the build
# pipeline. Used by the init container; surfaced here for clear error messages.
ENGINE_S3_URI: str | None = os.getenv("STUDIO_ENGINE_S3_URI") or None

# Browser origins allowed to call the API (the Next.js dev + preview servers).
CORS_ORIGINS: list[str] = _list(
    "STUDIO_CORS_ORIGINS", "http://localhost:3000,http://localhost:3100"
)

# Generation guard-rails (also clamped per-request).
MAX_STEPS: int = int(os.getenv("STUDIO_MAX_STEPS", "60"))
MAX_DIM: int = int(os.getenv("STUDIO_MAX_DIM", "1024"))

# Override variants.yaml's `max_resident` (engine bundles kept hot in VRAM). Set to
# 1 on a 24GB card (A10G/L4) so a second bundle can't OOM the GPU; None = use YAML.
MAX_RESIDENT: int | None = int(os.getenv("STUDIO_MAX_RESIDENT")) if os.getenv("STUDIO_MAX_RESIDENT") else None

# Serving backend on the GPU plane. Default (unset) = TensorRT engines. Set to
# "diffusers" to run the un-quantised PyTorch reference pipeline -- the
# framework-overhead baseline the TRT plane is measured against.
BACKEND: str | None = os.getenv("STUDIO_BACKEND")

# Attention implementation for the diffusers backend:
#   sdpa    - AttnProcessor2_0 -> torch SDPA (dispatches to the flash-attention
#             kernel on SDXL); the default and the "flash on" case.
#   flash   - like sdpa but forces the flash SDPA backend (efficient fallback).
#   xformers- xFormers memory-efficient attention.
#   vanilla - naive matmul-softmax-matmul (no fusion); the "flash off" baseline.
ATTENTION: str = os.getenv("STUDIO_ATTENTION", "sdpa")
