#!/usr/bin/env python3
"""Warm-latency + cost benchmark for a fastdiff /generate SSE endpoint.

Platform-agnostic: point it at the Modal web URL (or any host serving the same
SSE contract) and it runs a warm distribution over N generations, reporting every
timing component the server measures plus a derived per-generation price.

    python serverless/bench_compare.py --url https://<app>.modal.run \
        --label modal --variant fp16-base --n 20 --steps 30 --price-per-hour 1.95

Concurrency is intentionally 1: the TRT execution context is single-threaded, so
concurrent requests would serialize and pollute the latency distribution.

Server metrics (from each `done` event, camelCase aliases):
    latencyMs.coldLoad   engine deserialize from disk (0 when warm)
    latencyMs.denoise    the UNet denoise loop
    latencyMs.vaeDecode  VAE decode of the final latent
    latencyMs.total      coldLoad + denoise + vaeDecode
    throughputStepsPerSec
    vramPeakGB
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import time
import urllib.request

# Rough per-hour GPU rates ($/hr) for the derived cost; override with --price-per-hour.
# Modal on-demand, mid-2026 (confirm against the live pricing page).
DEFAULT_RATES = {"L40S": 1.95, "A100": 2.50, "A100-80GB": 3.95, "H100": 4.90, "L4": 0.80}


def _payload(variant: str, steps: int, seed: int) -> dict:
    return {
        "prompt": "a lighthouse at dusk, dramatic sky",
        "variantId": variant,
        "steps": steps,
        "seed": seed,
        "width": 1024,
        "height": 1024,
    }


def one_generation_sse(url: str, variant: str, steps: int, seed: int) -> tuple[float, dict]:
    """Modal / any SSE endpoint: stream /generate, return (wall_s, metrics)."""
    body = json.dumps(_payload(variant, steps, seed)).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    metrics: dict = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            evt = json.loads(data)
            if evt.get("type") == "error":
                raise RuntimeError(f"server error: {evt.get('message')}")
            if evt.get("type") == "done":
                metrics = evt["result"].get("metrics", {})
    return time.time() - t0, metrics


def one_generation_runpod(endpoint_id: str, api_key: str, variant: str, steps: int, seed: int) -> tuple[float, dict]:
    """RunPod serverless: submit /run, poll /status, return (wall_s, metrics)."""
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.time()

    submit = urllib.request.Request(
        base + "/run", data=json.dumps({"input": _payload(variant, steps, seed)}).encode(), headers=headers
    )
    with urllib.request.urlopen(submit, timeout=60) as r:
        job_id = json.loads(r.read())["id"]

    status: dict = {}
    while True:
        poll = urllib.request.Request(base + f"/status/{job_id}", headers=headers)
        with urllib.request.urlopen(poll, timeout=60) as r:
            status = json.loads(r.read())
        st = status.get("status")
        if st == "COMPLETED":
            break
        if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"runpod job {st}: {status.get('error')}")
        if time.time() - t0 > 600:
            raise TimeoutError("runpod job exceeded 600s")
        time.sleep(0.5)

    out = status.get("output")
    events = out if isinstance(out, list) else [out]
    metrics: dict = {}
    for evt in events:
        if isinstance(evt, dict) and evt.get("type") == "done":
            metrics = evt.get("result", {}).get("metrics", {})
    return time.time() - t0, metrics


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def summarize(values: list[float]) -> dict:
    if not values:
        return {k: float("nan") for k in ("p50", "p95", "p99", "mean", "min", "max")}
    return {
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="SSE endpoint base URL (Modal). Omit if using --runpod-endpoint-id")
    ap.add_argument("--runpod-endpoint-id", help="RunPod serverless endpoint id (uses the RunPod API)")
    ap.add_argument("--runpod-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key (or RUNPOD_API_KEY env)")
    ap.add_argument("--variant", default="fp16-base")
    ap.add_argument("--n", type=int, default=20, help="timed samples")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2, help="discarded warm-up gens")
    ap.add_argument("--label", default="endpoint", help="name for the printout")
    ap.add_argument("--gpu", default="L40S", help="GPU name (for the default rate)")
    ap.add_argument(
        "--price-per-hour",
        type=float,
        default=None,
        help="GPU $/hr for cost derivation (defaults by --gpu)",
    )
    args = ap.parse_args()

    rate_hr = args.price_per_hour if args.price_per_hour is not None else DEFAULT_RATES.get(args.gpu, 1.95)
    rate_s = rate_hr / 3600.0

    # Pick the transport: RunPod endpoint id, else SSE URL.
    if args.runpod_endpoint_id:
        if not args.runpod_key:
            ap.error("--runpod-key (or RUNPOD_API_KEY env) is required with --runpod-endpoint-id")
        gen = functools.partial(one_generation_runpod, args.runpod_endpoint_id, args.runpod_key)
    elif args.url:
        gen = functools.partial(one_generation_sse, args.url)
    else:
        ap.error("provide --url (SSE) or --runpod-endpoint-id (RunPod)")

    print(f"[{args.label}] warming up ({args.warmup} gens, discarded)...")
    for i in range(args.warmup):
        dt, _ = gen(args.variant, args.steps, seed=1000 + i)
        print(f"  warmup {i + 1}: {dt:.2f}s")

    print(f"[{args.label}] timing {args.n} gens @ {args.steps} steps, 1024x1024, conc=1...")
    series: dict[str, list[float]] = {
        "wall_s": [], "total_ms": [], "denoise_ms": [], "vae_ms": [],
        "cold_ms": [], "throughput": [], "vram_gb": [],
    }
    for i in range(args.n):
        dt, m = gen(args.variant, args.steps, seed=i + 1)
        lm = m.get("latencyMs") or m.get("latency_ms") or {}
        series["wall_s"].append(dt)
        if isinstance(lm, dict):
            if lm.get("total") is not None:
                series["total_ms"].append(float(lm["total"]))
            if lm.get("denoise") is not None:
                series["denoise_ms"].append(float(lm["denoise"]))
            if lm.get("vaeDecode") is not None:
                series["vae_ms"].append(float(lm["vaeDecode"]))
            if lm.get("coldLoad") is not None:
                series["cold_ms"].append(float(lm["coldLoad"]))
        if m.get("throughputStepsPerSec") is not None:
            series["throughput"].append(float(m["throughputStepsPerSec"]))
        if m.get("vramPeakGB") is not None:
            series["vram_gb"].append(float(m["vramPeakGB"]))
        print(f"  {i + 1:>2}/{args.n}: wall={dt:.2f}s")

    # ---- timing table --------------------------------------------------------
    rows = [
        ("wall total (s)", "wall_s", 1.0, "s"),
        ("server total (s)", "total_ms", 1000.0, "s"),
        ("  denoise (s)", "denoise_ms", 1000.0, "s"),
        ("  vae decode (s)", "vae_ms", 1000.0, "s"),
        ("  cold load (s)", "cold_ms", 1000.0, "s"),
        ("throughput (steps/s)", "throughput", 1.0, ""),
        ("VRAM peak (GB)", "vram_gb", 1.0, ""),
    ]
    print(f"\n===== {args.label} :: {args.variant} @ {args.steps} steps, N={args.n}, conc=1 =====")
    hdr = f"{'metric':<22}{'p50':>10}{'p95':>10}{'p99':>10}{'mean':>10}{'min':>10}{'max':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, key, div, _unit in rows:
        vals = [v / div for v in series[key]]
        if not vals:
            print(f"{label:<22}{'n/a':>10}")
            continue
        s = summarize(vals)
        print(
            f"{label:<22}{s['p50']:>10.2f}{s['p95']:>10.2f}{s['p99']:>10.2f}"
            f"{s['mean']:>10.2f}{s['min']:>10.2f}{s['max']:>10.2f}"
        )

    # ---- cost ----------------------------------------------------------------
    wall = summarize(series["wall_s"])
    print(f"\n----- cost ({args.gpu} @ ${rate_hr:.2f}/hr = ${rate_s * 1000:.4f}/1000 GPU-s) -----")
    print("  (GPU is busy one request at a time, so cost/gen = gen wall-time x rate)")
    print(f"  $ / generation   p50 : ${wall['p50'] * rate_s:.4f}   (p95 ${wall['p95'] * rate_s:.4f})")
    print(f"  $ / 1000 gens    p50 : ${wall['p50'] * rate_s * 1000:.2f}   (p95 ${wall['p95'] * rate_s * 1000:.2f})")
    print(f"  images / GPU-hour p50: {3600 / wall['p50']:.0f}   (mean {3600 / wall['mean']:.0f})")
    print(f"  keep-warm idle cost  : ${rate_hr:.2f}/hr = ${rate_hr * 24:.2f}/day per warm container")
    print("\n  note: --price-per-hour to set the exact live rate; server total is the honest")
    print("  GPU-busy time, wall total includes network RTT to the endpoint.")


if __name__ == "__main__":
    main()
