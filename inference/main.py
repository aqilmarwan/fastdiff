"""FastAPI entrypoint for the fastdiff inference service.

Exposes exactly the two endpoints the frontend contract in `web/lib/types.ts`
depends on, plus a health probe for Kubernetes:

* ``GET  /variants``  -- the precision x style catalog with registry metrics.
* ``POST /generate``  -- a generation, streamed back as Server-Sent Events.
* ``GET  /healthz``   -- liveness/readiness for the k8s deployment.

All the heavy lifting lives in :class:`pipelines.Registry`, which picks the real
CUDA backend or the CPU demo backend automatically. `Registry.run` is a blocking,
synchronous call whose ``emit`` callback fires from the worker thread; this module
bridges that to an async SSE stream via a thread + :class:`asyncio.Queue`.

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
from pipelines import Registry
from schemas import GenerationParams, Variant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("studio.main")

app = FastAPI(
    title="fastdiff -- Inference",
    version="1.0.0",
    summary="Quantised SDXL variants, served with honest, server-side metrics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# One registry per process: owns the variant catalog and the (single) backend.
registry = Registry()
log.info("Serving %d variants on the %r plane", len(registry.variants()), registry.plane)


# --------------------------------------------------------------------------- #
# Meta / health
# --------------------------------------------------------------------------- #


@app.get("/")
def root() -> dict:
    return {"service": "fastdiff-inference", "plane": registry.plane}


@app.get("/healthz")
def healthz() -> dict:
    """Liveness + readiness. Cheap and dependency-free so probes stay honest."""
    return {"status": "ok", "plane": registry.plane, "variants": len(registry.variants())}


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


@app.get("/variants", response_model=list[Variant], response_model_by_alias=True)
def variants() -> list[Variant]:
    """The servable precision x style matrix. Emitted in camelCase for the UI."""
    return registry.variants()


# --------------------------------------------------------------------------- #
# Generation (SSE)
# --------------------------------------------------------------------------- #


def _sse(event: dict) -> str:
    """Encode one event dict as an SSE frame (``data:`` line + blank separator)."""
    return f"data: {json.dumps(event)}\n\n"


def _clamp(params: GenerationParams) -> GenerationParams:
    """Enforce server-side guard-rails on top of the per-field schema limits."""
    return params.model_copy(
        update={
            "steps": min(params.steps, config.MAX_STEPS),
            "width": min(params.width, config.MAX_DIM),
            "height": min(params.height, config.MAX_DIM),
        }
    )


async def _stream(params: GenerationParams) -> AsyncGenerator[str, None]:
    """Run the (blocking) generation in a worker thread, yielding SSE frames.

    Events emitted by the backend -- ``status`` / ``progress`` -- flow straight
    through; the terminal ``done`` (with the full result) or ``error`` frame is
    appended once the worker finishes.
    """
    if not registry.has(params.variant_id):
        yield _sse({"type": "error", "message": f"unknown variant: {params.variant_id!r}"})
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def emit(event: dict) -> None:
        # Called from the worker thread -- hop back onto the event loop safely.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        try:
            result = registry.run(params, emit)
            emit({"type": "done", "result": result.model_dump(by_alias=True)})
        except Exception as exc:  # surface as an SSE error, never a 500 mid-stream
            log.exception("generation failed for variant %s", params.variant_id)
            emit({"type": "error", "message": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    task = loop.run_in_executor(None, worker)
    try:
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            yield _sse(event)
    finally:
        await task


@app.post("/generate")
async def generate(params: GenerationParams) -> StreamingResponse:
    return StreamingResponse(
        _stream(_clamp(params)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so frames flush live
        },
    )
