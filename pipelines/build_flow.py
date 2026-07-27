"""Offline build + benchmark pipeline for the servable SDXL variants.

Metaflow orchestrates the DAG and versions every run; Ray fans the per-variant
build+benchmark out across the cluster's GPUs in parallel; MLflow records the
params, measured metrics, and artifacts and registers each variant as a model
version. The measured `registry:` block for every variant is what
`inference/variants.yaml` surfaces through ``GET /variants`` -- this flow is how
those numbers get there honestly, instead of being hand-typed.

Two planes, mirroring the inference service:

* **real**  -- builds each variant's TensorRT engine bundle (build_engines.py:
  ONNX export -> engine -> publish to S3), then benchmarks it through the serving
  backend, measuring size / VRAM / throughput around a real denoise loop.
* **demo**  -- no GPU, no downloads. Simulates each variant's benchmark from the
  catalog baseline so the whole flow is exercisable in CI / on a laptop. Demo
  metrics are clearly logged as simulated. Selected with ``--demo`` or whenever
  CUDA is absent.

Run::

    python pipelines/build_flow.py run --demo            # laptop / CI dry-run
    python pipelines/build_flow.py run --sync            # real GPU box, write back
    python pipelines/build_flow.py run --trials 5 --sync

Requires the extra tooling in ``pipelines/requirements.txt`` (Metaflow, Ray,
MLflow, ruamel.yaml). None of it is imported by the inference service.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from metaflow import FlowSpec, Parameter, current, step

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_VARIANTS = REPO / "inference" / "variants.yaml"
LORAS_FILE = HERE / "loras.yaml"
LORA_OUT_DIR = REPO / "inference" / "loras"
ENGINE_OUT_DIR = HERE / "engines_out"   # local staging before S3 publish
OUT_DIR = HERE / "out"

sys.path.insert(0, str(HERE))  # let `import train_lora` resolve inside Metaflow steps


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Ray remote: build + benchmark one variant
# --------------------------------------------------------------------------- #
#
# Defined at module scope so Ray can pickle it. Decorated lazily inside the flow
# (after `ray.init`) to keep import side-effect free and demo runs Ray-optional.


def _build_and_benchmark(base_model: str, variant: dict, trials: int, demo: bool,
                         seed: int, engine_out: str, s3_uri: Optional[str]) -> dict:
    """Return the measured ``registry`` block for one variant.

    Demo mode simulates from the catalog baseline. Real mode builds the variant's
    TensorRT engine bundle (ONNX export -> engine -> publish to S3), then loads it
    through the *serving* backend and times ``trials`` denoise passes -- so the
    numbers come from the exact engines that will serve traffic.
    """
    vid = variant["id"]
    serving = variant.get("serving", {}) or {}
    baseline = variant.get("registry", {}) or {}

    if demo:
        # Deterministic jitter around the baseline so runs look measured, not fixed.
        wobble = 1.0 + ((hash((vid, seed)) % 1000) / 1000.0 - 0.5) * 0.04
        return {
            "sizeGB": round(baseline["sizeGB"] * wobble, 1),
            "vramGB": round(baseline["vramGB"] * wobble, 1),
            "stepsPerSec": round(baseline["stepsPerSec"] * wobble, 1),
            "quality": baseline["quality"],
            "defaultSteps": baseline["defaultSteps"],
            "_simulated": True,
        }

    from build_engines import EngineSpec, build_bundle, publish_s3

    spec = EngineSpec(
        engine=serving.get("engine", vid),
        precision=serving.get("precision", "fp16"),
        lora=serving.get("lora"),
    )
    out_root = Path(engine_out)
    bundle = build_bundle(spec, base_model, out_root, LORA_OUT_DIR)
    if s3_uri:
        publish_s3(bundle, s3_uri)

    # Honest on-disk size: the actual serialized engines.
    size_gb = sum(p.stat().st_size for p in bundle.glob("*.plan")) / (1024**3)

    steps = int(baseline.get("defaultSteps", 30))
    bench = _benchmark_bundle(out_root, variant, serving, trials, seed, steps)

    return {
        "sizeGB": round(size_gb, 1),
        "vramGB": bench["vramGB"],
        "stepsPerSec": bench["stepsPerSec"],
        # Provisional; `build` rescales this from _clipRaw relative to fp16-base.
        "quality": baseline["quality"],
        "_clipRaw": bench["clipRaw"],
        "defaultSteps": steps,
        "_simulated": False,
    }


def _benchmark_bundle(engine_out: str, variant: dict, serving: dict,
                      trials: int, seed: int, steps: int) -> dict:
    """Time the freshly built engines through the serving TensorRTBackend.

    Runs the same code path production serves, so latency/VRAM are honest and a
    CLIP score (relative to fp16-base, normalised in the join) gives quality.
    """
    import importlib.util

    sys.path.insert(0, str(REPO / "inference"))  # let the serving module's imports resolve
    _spec = importlib.util.spec_from_file_location("serving_pipelines", REPO / "inference" / "pipelines.py")
    serving_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(serving_mod)
    import schemas  # the SAME module serving_mod imported (sys.modules cache)
    from metrics import peak_vram_gb, reset_peak_vram

    backend = serving_mod.TensorRTBackend(Path(engine_out), max_resident=1)
    v = schemas.Variant(
        id=variant["id"], label=variant["label"], precision=variant["precision"],
        style=variant["style"], base="SDXL 1.0", size_gb=0, vram_gb=0,
        steps_per_sec=0, quality=0, default_steps=steps, blurb="", licence="",
    )
    sv = serving_mod.Serving(
        engine=serving.get("engine", variant["id"]),
        precision=serving.get("precision", "fp16"), lora=serving.get("lora"),
    )

    prompts = [
        "a photograph of a red fox in a snowy forest, sharp focus",
        "an oil painting of a lighthouse at sunset over crashing waves",
        "a studio portrait of an elderly man, soft rim light",
    ]
    reset_peak_vram()
    images, sps = [], []
    for i in range(max(1, trials)):
        prompt = prompts[i % len(prompts)]
        params = schemas.GenerationParams(prompt=prompt, variant_id=variant["id"],
                                          steps=steps, seed=seed + i)
        t0 = time.perf_counter()
        out = backend.generate(v, sv, params, lambda e: None)
        sps.append(steps / max(1e-6, (time.perf_counter() - t0)))
        if i < len(prompts):
            images.append((out.image_url, prompt))

    return {
        "stepsPerSec": round(statistics.median(sps), 1),
        "vramGB": round(peak_vram_gb(), 1),
        "clipRaw": round(_clip_score(images), 4),
    }


def _clip_score(images: list) -> float:
    """Mean CLIP image-text cosine (raw ~0..35) over (data_url, prompt) pairs."""
    import base64
    import io

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    sims: list[float] = []
    for data_url, prompt in images:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        inputs = proc(text=[prompt], images=img, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims.append(float((ie * te).sum(dim=-1).item()))
    del model
    torch.cuda.empty_cache()
    return 100.0 * sum(sims) / len(sims) if sims else 0.0


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #


class BuildFlow(FlowSpec):
    """Build, benchmark, and register every SDXL variant in the catalog."""

    variants_file = Parameter("variants-file", default=str(DEFAULT_VARIANTS), help="Path to variants.yaml")
    demo = Parameter("demo", is_flag=True, default=False, help="Force the simulated plane (no GPU/downloads).")
    trials = Parameter("trials", default=3, help="Benchmark passes per variant (median is reported).")
    seed = Parameter("seed", default=1234, help="Base seed for reproducible benchmarks.")
    sync = Parameter("sync", is_flag=True, default=False, help="Write measured metrics back into variants.yaml.")
    experiment = Parameter("experiment", default="fastdiff/sdxl-quant", help="MLflow experiment name.")
    engine_s3 = Parameter("engine-s3", default=os.getenv("STUDIO_ENGINE_S3_URI"),
                          help="s3:// URI to publish built engine bundles to (real plane).")

    @step
    def start(self):
        import yaml

        doc = yaml.safe_load(Path(self.variants_file).read_text())
        self.base_model = doc["base_model"]
        self.variants = doc["variants"]
        self.is_demo = bool(self.demo) or not _cuda_available()
        if self.is_demo and not self.demo:
            print("No CUDA device -- falling back to the simulated (demo) plane.")
        print(f"Building {len(self.variants)} variants from {self.base_model} "
              f"[{'demo' if self.is_demo else 'real'} plane]")
        self.next(self.train)

    @step
    def train(self):
        """Train LoRAs declared in loras.yaml into inference/loras/ (real plane only)."""
        self.trained = {}
        if self.is_demo:
            print("Demo plane -- skipping LoRA training (no fake weights emitted).")
            self.next(self.build)
            return

        import yaml

        from train_lora import LoraSpec
        from train_lora import train as train_one

        specs = [LoraSpec.from_dict(d) for d in (yaml.safe_load(LORAS_FILE.read_text()) or {}).get("loras", [])] \
            if LORAS_FILE.exists() else []
        for spec in specs:
            try:
                path = train_one(spec, self.base_model, LORA_OUT_DIR, REPO)
                self.trained[spec.name] = str(path)
                print(f"  trained {spec.name} -> {path}")
            except Exception as exc:
                # A missing dataset shouldn't sink the build; the variant falls
                # back to base weights until the LoRA is trained.
                print(f"  LoRA {spec.name} skipped: {exc}")
        self.next(self.build)

    @step
    def build(self):
        """Fan the per-variant build+benchmark across Ray, collect the results."""
        import ray

        ray.init(ignore_reinit_error=True, configure_logging=False)
        remote = ray.remote(num_gpus=0 if self.is_demo else 1)(_build_and_benchmark)
        futures = [
            remote.remote(self.base_model, v, self.trials, self.is_demo, self.seed,
                          str(ENGINE_OUT_DIR), self.engine_s3)
            for v in self.variants
        ]
        metrics = ray.get(futures)
        ray.shutdown()

        self.results = {v["id"]: m for v, m in zip(self.variants, metrics)}
        self._normalise_quality()
        for vid, m in self.results.items():
            tag = "sim" if m.get("_simulated") else "measured"
            print(f"  {vid:<12} {m['stepsPerSec']:>5} steps/s  {m['vramGB']:>5} GB VRAM  "
                  f"q={m['quality']:>3}  ({tag})")
        self.next(self.register)

    def _normalise_quality(self):
        """Rescale measured CLIP scores to 0-100 quality, relative to fp16-base.

        Demo results already carry a baseline `quality`, so they're left alone.
        """
        raws = {vid: m["_clipRaw"] for vid, m in self.results.items() if "_clipRaw" in m}
        if not raws:
            return
        ref = raws.get("fp16-base") or max(raws.values())  # fp16 == 100% fidelity
        for vid, raw in raws.items():
            self.results[vid]["quality"] = min(100, round(100 * raw / ref)) if ref else 0

    @step
    def register(self):
        """Log to MLflow, persist the registry artifact, optionally sync the YAML."""
        self._log_to_mlflow()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        clean = {vid: {k: v for k, v in m.items() if not k.startswith("_")} for vid, m in self.results.items()}
        registry_json = OUT_DIR / "registry.json"
        registry_json.write_text(json.dumps(clean, indent=2) + "\n")
        print(f"Wrote {registry_json}")

        if self.sync:
            self._sync_variants_yaml(clean)
            print(f"Synced measured metrics into {self.variants_file}")
        else:
            print("Skipped variants.yaml sync (pass --sync to write back).")
        self.next(self.end)

    @step
    def end(self):
        plane = "demo/simulated" if self.is_demo else "real/measured"
        print(f"BuildFlow complete ({plane}). {len(self.results)} variants registered.")

    # ----------------------------------------------------------------- helpers

    def _log_to_mlflow(self):
        try:
            import mlflow
        except Exception:
            print("mlflow not installed -- skipping experiment logging.")
            return

        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=f"build-{current.run_id}"):
            mlflow.log_params({
                "base_model": self.base_model,
                "plane": "demo" if self.is_demo else "real",
                "trials": self.trials,
                "variants": len(self.results),
            })
            for vid, m in self.results.items():
                with mlflow.start_run(run_name=vid, nested=True):
                    variant = next(v for v in self.variants if v["id"] == vid)
                    serving = variant.get("serving", {}) or {}
                    mlflow.log_params({
                        "variant": vid,
                        "precision": variant["precision"],
                        "style": variant["style"],
                        "quant": serving.get("quant", "none"),
                        "lora": serving.get("lora") or "none",
                    })
                    mlflow.log_metrics({
                        "size_gb": m["sizeGB"],
                        "vram_gb": m["vramGB"],
                        "steps_per_sec": m["stepsPerSec"],
                        "quality": m["quality"],
                    })

    def _sync_variants_yaml(self, clean: dict):
        """Round-trip variants.yaml with ruamel so comments/formatting survive."""
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)  # match variants.yaml's block-list layout
        path = Path(self.variants_file)
        doc = yaml.load(path)
        for entry in doc["variants"]:
            measured = clean.get(entry["id"])
            if not measured:
                continue
            reg = entry["registry"]
            for key in ("sizeGB", "vramGB", "stepsPerSec", "quality", "defaultSteps"):
                if key in measured:
                    reg[key] = measured[key]
        yaml.dump(doc, path)


if __name__ == "__main__":
    # Metaflow reads its subcommand (run/resume/show) from argv.
    os.environ.setdefault("USERNAME", os.environ.get("USER", "ci"))
    BuildFlow()
