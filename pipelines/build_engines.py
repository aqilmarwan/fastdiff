"""Offline: SDXL (+ optional LoRA) -> ONNX -> TensorRT engine bundle -> S3.

This is the *build* side of the TensorRT migration. HF is used here, once, to
pull the base weights and export them to ONNX; the resulting `.plan` engines are
what the serving plane runs (no HF/diffusers at request time). Invoked per
variant by pipelines/build_flow.py's build step on the real plane.

A bundle directory (published to ``s3://.../<engine>/``) contains:
    text_encoder.plan  text_encoder_2.plan  unet.plan  vae_decoder.plan
    tokenizer/  tokenizer_2/  metadata.json

Only the UNet is quantised; the VAE and text encoders stay FP16.
Precision -> TensorRT builder flags:
    fp16 -> FP16
    int8 -> INT8, entropy-calibrated by `_UNetCalibrator` (representative inputs)
    fp8  -> builds FP16 for now + warns; real FP8 needs ModelOpt Q/DQ at export

GPU-only and heavy. Everything is imported lazily. The engine-build specifics
(opset, dynamic axes, calibration) follow NVIDIA's SDXL demo conventions but must
be validated on-device before production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("studio.build_engines")

# Static input shapes for a 1024x1024 SDXL at CFG batch 2 (uncond+cond).
_B, _SEQ, _LAT, _H, _W = 2, 77, 128, 1024, 1024


@dataclass
class EngineSpec:
    engine: str            # bundle key (== serving.engine)
    precision: str         # fp16 | int8 | fp8
    lora: Optional[str] = None  # LoRA name to fuse before export


def _normalise_lora(lora_path: Path) -> dict:
    """Load a LoRA safetensors and normalise its keys for diffusers.

    Community LoRAs vary in layout; the two mismatches we handle:
      * a doubled UNet prefix ``unet.unet.`` -> ``unet.`` (what breaks target-module
        matching with a "Target modules not found" ValueError), and
      * the ``.lora.down`` / ``.lora.up`` naming -> diffusers' ``.lora_A`` / ``.lora_B``.
    Keys already in diffusers form pass through untouched.
    """
    from safetensors.torch import load_file

    state = load_file(str(lora_path))
    out: dict = {}
    for key, val in state.items():
        k = key
        while k.startswith("unet.unet."):
            k = k.replace("unet.unet.", "unet.", 1)
        k = k.replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight", ".lora_B.weight")
        out[k] = val
    if out != state:
        log.info("  normalised %d LoRA keys for diffusers", sum(1 for a, b in zip(out, state) if a != b))
    return out


def build_bundle(spec: EngineSpec, base_model: str, out_root: Path, lora_dir: Path,
                 calib_dir: Optional[Path] = None) -> Path:
    """Build a full engine bundle for one variant. Returns its local directory."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    bundle = out_root / spec.engine
    bundle.mkdir(parents=True, exist_ok=True)
    log.info("Building %s bundle (%s%s)", spec.engine, spec.precision,
             f", lora={spec.lora}" if spec.lora else "")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    # SDXL's stock VAE overflows in fp16 and decodes to a solid black image. Swap
    # in the fp16-fix VAE so the fp16 VAE engine decodes correctly.
    from diffusers import AutoencoderKL
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    if spec.lora:
        lora_path = lora_dir / f"{spec.lora}.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA {lora_path} missing -- train it first (build_flow train step).")
        pipe.load_lora_weights(_normalise_lora(lora_path))
        pipe.fuse_lora()  # bake weights in so the exported engine carries the style

    onnx_dir = bundle / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    # Only the UNet is quantised (it dominates compute + VRAM); the VAE and text
    # encoders stay FP16, which is standard for SDXL and matches variants.yaml's
    # VRAM figures. So precision is per-component, not per-bundle.
    components = {
        "text_encoder": (_export_text_encoder(pipe.text_encoder, onnx_dir, "text_encoder", pooled=False), "fp16"),
        "text_encoder_2": (_export_text_encoder(pipe.text_encoder_2, onnx_dir, "text_encoder_2", pooled=True), "fp16"),
        "unet": (_export_unet(pipe.unet, onnx_dir), spec.precision),
        "vae_decoder": (_export_vae_decoder(pipe.vae, onnx_dir), "fp16"),
    }
    for name, (onnx_path, precision) in components.items():
        calibrator = (_UNetCalibrator(onnx_dir / f"{name}.calib", calib_dir=calib_dir)
                      if (name == "unet" and precision == "int8") else None)
        _build_engine(onnx_path, bundle / f"{name}.plan", precision, calibrator)

    # Tokenizers travel with the bundle so serving loads them offline.
    pipe.tokenizer.save_pretrained(str(bundle / "tokenizer"))
    pipe.tokenizer_2.save_pretrained(str(bundle / "tokenizer_2"))
    (bundle / "metadata.json").write_text(json.dumps({
        "engine": spec.engine, "precision": spec.precision, "lora": spec.lora,
        "base_model": base_model, "resolution": _H,
    }, indent=2))

    del pipe
    torch.cuda.empty_cache()
    return bundle


# --------------------------------------------------------------------------- #
# ONNX export (one function per component; dynamic batch on the token dim)
# --------------------------------------------------------------------------- #


def _export_text_encoder(model, onnx_dir: Path, name: str, pooled: bool) -> Path:
    import torch

    path = onnx_dir / f"{name}.onnx"
    model = model.eval()
    ids = torch.zeros((1, _SEQ), dtype=torch.int32, device=model.device)
    outputs = ["hidden_states"] + (["pooled"] if pooled else [])

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids):
            out = self.m(input_ids, output_hidden_states=True)
            hs = out.hidden_states[-2]                 # SDXL uses the penultimate layer
            return (hs, out[0]) if pooled else (hs,)

    torch.onnx.export(
        _Wrap(model), (ids,), str(path),
        input_names=["input_ids"], output_names=outputs,
        dynamic_axes={"input_ids": {0: "B"}, **{o: {0: "B"} for o in outputs}},
        opset_version=18,
    )
    return path


def _export_unet(unet, onnx_dir: Path) -> Path:
    import torch

    path = onnx_dir / "unet.onnx"
    unet = unet.eval()
    dev, dt = unet.device, torch.float16
    dummies = (
        torch.randn(_B, 4, _LAT, _LAT, dtype=dt, device=dev),          # sample
        torch.tensor([1.0, 1.0], dtype=dt, device=dev),                # timestep
        torch.randn(_B, _SEQ, 2048, dtype=dt, device=dev),             # encoder_hidden_states
    )
    added = {"text_embeds": torch.randn(_B, 1280, dtype=dt, device=dev),
             "time_ids": torch.randn(_B, 6, dtype=dt, device=dev)}

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.m(sample, timestep, encoder_hidden_states,
                          added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}).sample

    torch.onnx.export(
        _Wrap(unet), (*dummies, added["text_embeds"], added["time_ids"]), str(path),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["noise_pred"],
        # timestep must share the batch axis too, or the engine is inconsistent at
        # the profile's min-batch corner and build_serialized_network returns None.
        dynamic_axes={k: {0: "B"} for k in
                      ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "noise_pred"]},
        opset_version=18,
    )
    return path


def _export_vae_decoder(vae, onnx_dir: Path) -> Path:
    import torch

    path = onnx_dir / "vae_decoder.onnx"
    vae = vae.eval()
    latent = torch.randn(1, 4, _LAT, _LAT, dtype=torch.float16, device=vae.device)

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, latent):
            return self.m.decode(latent).sample

    torch.onnx.export(
        _Wrap(vae), (latent,), str(path),
        input_names=["latent"], output_names=["images"],
        dynamic_axes={"latent": {0: "B"}, "images": {0: "B"}},
        opset_version=18,
    )
    return path


# --------------------------------------------------------------------------- #
# TensorRT engine build
# --------------------------------------------------------------------------- #


def _UNetCalibrator(cache_path: Path, calib_dir: Optional[Path] = None, num_batches: int = 16):
    """INT8 entropy calibrator for the UNet, fed via torch device tensors.

    If ``calib_dir`` holds ``batch_*.pt`` files captured by
    ``pipelines/capture_calib.py`` (real UNet activations from actual denoising),
    it calibrates on those -- the correct way, giving faithful INT8 scales.
    Otherwise it falls back to random tensors: a *working* engine but poor scales
    and noisy quality (this is what made INT8 a "fallback"). TRT's computed scales
    are cached to ``cache_path``.

    Built inside a factory so ``tensorrt`` stays a lazy import.
    """
    import tensorrt as trt
    import torch

    shapes = {
        "sample": (_B, 4, _LAT, _LAT),
        "timestep": (_B,),
        "encoder_hidden_states": (_B, _SEQ, 2048),
        "text_embeds": (_B, 1280),
        "time_ids": (_B, 6),
    }
    files = sorted(Path(calib_dir).glob("batch_*.pt")) if calib_dir else []
    if files:
        log.info("INT8 calibrating on %d real activation batches from %s", len(files), calib_dir)
    else:
        log.warning("INT8 calibrating on RANDOM tensors (no calib_dir) -- scales will be poor")

    class _Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.cache_path = Path(cache_path)
            self.idx = 0
            self._buffers = {}  # keep refs so the device memory outlives get_batch

        def get_batch_size(self):
            return _B

        def get_batch(self, names):
            if files:
                if self.idx >= len(files):
                    return None
                data = torch.load(files[self.idx], map_location="cpu")
                self._buffers = {
                    n: data[n].to("cuda", torch.float16).contiguous() for n in names if n in data
                }
            else:
                if self.idx >= num_batches:
                    return None
                self._buffers = {
                    n: torch.randn(shapes[n], device="cuda", dtype=torch.float16).contiguous()
                    for n in names if n in shapes
                }
            self.idx += 1
            if len(self._buffers) != len(names):
                return None
            return [int(self._buffers[n].data_ptr()) for n in names]

        def read_calibration_cache(self):
            return self.cache_path.read_bytes() if self.cache_path.exists() else None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return _Calib()


def _build_engine(onnx_path: Path, plan_path: Path, precision: str, calibrator=None) -> None:
    import tensorrt as trt

    class _Logger(trt.ILogger):
        # Collect ERROR/INTERNAL_ERROR messages so a failed (None) build reports
        # the real reason instead of a bare "returned None".
        def __init__(self):
            trt.ILogger.__init__(self)
            self.errors: list[str] = []

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.ERROR:
                self.errors.append(msg)
            if severity <= trt.ILogger.Severity.INFO:
                print(f"[TRT] {msg}", flush=True)

    logger = _Logger()
    builder = trt.Builder(logger)
    # TRT 10 networks are always explicit-batch (the flag was removed); TRT 8
    # needs it set. Handle both so the same build box works across versions.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file (not parse(read_bytes())) so TRT can resolve weights that
    # torch.onnx.export stored as external data sidecar files next to the .onnx.
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path.name}: {errs}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GiB

    def _set(flag_name: str) -> bool:
        # BuilderFlag member names vary slightly across TensorRT versions; resolve
        # by name and skip (with a warning) rather than crash on a missing attr.
        flag = getattr(trt.BuilderFlag, flag_name, None)
        if flag is None:
            log.warning("TensorRT BuilderFlag.%s unavailable (trt %s) -- skipping", flag_name, trt.__version__)
            return False
        cfg.set_flag(flag)
        return True

    _set("FP16")                                    # fp16 fallback for unsupported layers

    if precision == "int8":
        _set("INT8")
        if calibrator is not None:
            cfg.int8_calibrator = calibrator        # entropy-calibrated scales
            # Dynamic-shape INT8 needs a FIXED calibration profile matching the
            # calibrator's batch (2); without it TRT calibrates at the min shape
            # and reads past the batch-2 buffers -> illegal memory access.
            calib_profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                fixed = [2 if d == -1 else d for d in inp.shape]
                calib_profile.set_shape(inp.name, fixed, fixed, fixed)
            cfg.set_calibration_profile(calib_profile)
        else:
            log.warning("INT8 requested for %s without a calibrator -- scales will be poor", onnx_path.name)
    elif precision == "fp8":
        # Post-training FP8 for SDXL needs Q/DQ nodes inserted at export time via
        # NVIDIA ModelOpt; a bare FP8 flag can't calibrate scales, so TRT falls
        # back to FP16 for most layers. Build FP16 and say so rather than ship a
        # silently-degraded "FP8" engine. Wire ModelOpt to make FP8 real.
        log.warning("FP8 not yet calibrated (needs ModelOpt Q/DQ) -- building FP16 for %s", onnx_path.name)

    _add_dynamic_profile(builder, network, cfg, trt)

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        detail = " | ".join(logger.errors[-6:]) or "no TRT error captured (see [TRT] log above)"
        raise RuntimeError(f"engine build failed for {onnx_path.name}: {detail}")
    # serialized is an IHostMemory (no len); .nbytes is its size, and write_bytes
    # accepts its buffer view.
    plan_path.write_bytes(bytes(serialized))
    log.info("  built %s (%.0f MB, %s)", plan_path.name, serialized.nbytes / 1e6, precision)


def _add_dynamic_profile(builder, network, cfg, trt) -> None:
    """Fix the batch dim to 2 (CFG); text encoders run batch 1..2."""
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        lo = [1 if d == -1 else d for d in shape]
        opt = [2 if d == -1 else d for d in shape]
        hi = [2 if d == -1 else d for d in shape]
        profile.set_shape(inp.name, lo, opt, hi)
    cfg.add_optimization_profile(profile)


# --------------------------------------------------------------------------- #
# Publish
# --------------------------------------------------------------------------- #


def publish_s3(bundle: Path, s3_uri: str) -> None:
    """Upload a bundle to ``<s3_uri>/<bundle-name>/`` (skips the local onnx/ dir)."""
    import boto3

    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    key_root = f"{prefix.rstrip('/')}/{bundle.name}" if prefix else bundle.name
    s3 = boto3.client("s3")
    for path in bundle.rglob("*"):
        if path.is_dir() or "onnx" in path.relative_to(bundle).parts:
            continue
        key = f"{key_root}/{path.relative_to(bundle).as_posix()}"
        s3.upload_file(str(path), bucket, key)
    log.info("  published %s -> s3://%s/%s", bundle.name, bucket, key_root)
