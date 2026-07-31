"""RunPod serverless handler for fastdiff - L40S, TensorRT engines from a volume.

Mirrors the SSE /generate contract as a RunPod generator handler: it yields the
same status / progress / done / error events, and reuses inference/pipelines.py's
Registry so the measured metrics match the Modal plane exactly.

Design note: ALL failable work (S3 sync, torch/TRT import, Registry init) happens
lazily on the first job, not at module import. That way the serverless worker
always starts and logs, and any failure comes back as a clear job error instead
of a silent `worker exited with exit code 1`.

Engines live on the RunPod network volume (mounted at /runpod-volume). On the
first job the worker syncs the bundle from S3 if it's missing, using the endpoint's
env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION).
"""

from __future__ import annotations

import os
import pathlib
import queue
import sys
import threading
import traceback

# Only env + sys.path at import time - nothing that can fail.
ENGINE_DIR = os.environ.setdefault("STUDIO_ENGINE_DIR", "/runpod-volume/engines")
os.environ.setdefault("STUDIO_MAX_RESIDENT", "1")
os.environ.setdefault(
    "STUDIO_CORS_ORIGINS", "https://fastdiff.vercel.app,http://localhost:3000"
)
sys.path.insert(0, "/app/inference")

S3_BUCKET = os.environ.get("ENGINE_S3_BUCKET", "quant-studio-engine-bucket")

_registry = None  # cached across invocations on a warm worker


def _ensure_engine(engine: str) -> None:
    """Sync one engine bundle from S3 into the volume if it isn't already there."""
    dest = pathlib.Path(ENGINE_DIR) / engine
    if dest.exists() and any(dest.glob("*.plan")):
        return
    import boto3  # lazy: only needed on a cold, empty volume

    print(f"[bootstrap] syncing {engine} from s3://{S3_BUCKET} into {dest} ...", flush=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    root = pathlib.Path(ENGINE_DIR)
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{engine}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. "fp16-base/unet.plan"
            out = root / key
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(out))
            count += 1
    print(f"[bootstrap] synced {count} objects for {engine}", flush=True)


def _get_registry():
    """Lazily build the Registry once per worker. Raises with a clear message."""
    global _registry
    if _registry is None:
        _ensure_engine(os.environ.get("WARM_ENGINE", "fp16-base"))
        from pipelines import Registry  # imports torch/tensorrt - do it lazily

        _registry = Registry()
        plane = getattr(getattr(_registry, "backend", None), "plane", "?")
        print(f"[init] registry ready on the '{plane}' plane", flush=True)
    return _registry


def handler(job: dict):
    """RunPod generator handler: yields the same events as the SSE endpoint."""
    # Lazy init - surface any setup failure as a job error, not a worker crash.
    try:
        registry = _get_registry()
        import config
        from schemas import GenerationParams
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield {"type": "error", "message": f"worker init failed: {type(exc).__name__}: {exc}"}
        return

    inp = job.get("input") or {}
    try:
        params = GenerationParams.model_validate(inp)
        params = params.model_copy(
            update={
                "steps": min(params.steps, config.MAX_STEPS),
                "width": min(params.width, config.MAX_DIM),
                "height": min(params.height, config.MAX_DIM),
            }
        )
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"bad input: {exc}"}
        return

    if not registry.has(params.variant_id):
        yield {"type": "error", "message": f"unknown variant: {params.variant_id!r}"}
        return

    try:
        _ensure_engine(params.variant_id)  # idempotent; covers non-default variants
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"engine sync failed: {exc}"}
        return

    events: queue.Queue = queue.Queue()
    sentinel = object()

    def emit(event: dict) -> None:
        events.put(event)

    def worker() -> None:
        try:
            result = registry.run(params, emit)
            emit({"type": "done", "result": result.model_dump(by_alias=True)})
        except Exception as exc:  # noqa: BLE001 - surface as an error event, not a crash
            emit({"type": "error", "message": str(exc)})
        finally:
            events.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        event = events.get()
        if event is sentinel:
            break
        yield event


if __name__ == "__main__":
    import runpod

    print("[boot] starting RunPod serverless worker (lazy init on first job)", flush=True)
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
