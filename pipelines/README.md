# pipelines — build & benchmark flow

`build_flow.py` is the offline pipeline that turns the base SDXL checkpoint into
the servable **TensorRT engine bundles** (`build_engines.py`: ONNX export →
engine build → publish to S3) and measures the `registry:` metrics that
`inference/variants.yaml` surfaces through `GET /variants`. This is the only
place Hugging Face is used — the serving plane runs the engines HF-free.

- **Metaflow** orchestrates: train LoRA → build engines → benchmark → register.
- **Ray** fans the per-variant engine build + benchmark across the cluster's GPUs.
- **MLflow** logs params, measured metrics, and registers each variant.
- Benchmarks run the freshly built engines through the *serving* backend, so the
  numbers come from the exact engines that will serve traffic.

It shares the inference service's two-plane design: a real CUDA plane that builds
and measures honestly, and a `--demo` plane that simulates from the catalog
baseline so the flow runs on a laptop or in CI with no GPU or downloads.

## Run

```bash
pip install -r pipelines/requirements.txt

# laptop / CI — simulated, no GPU, does not touch variants.yaml
python pipelines/build_flow.py run --demo

# real L40S GPU box — build + publish to S3 (engines are device-specific)
pip install -r inference/requirements.txt -r inference/requirements-gpu.txt
python pipelines/build_flow.py run --trials 5 --sync --engine-s3 s3://<bucket>
```

To skip training and reuse a pre-trained LoRA, grab one with
`./pipelines/fetch_lora.sh` before building. To train your own, add instance
images under `pipelines/data/` (see `pipelines/data/README.md`).

## Outputs

- `pipelines/out/registry.json` — measured metrics per variant (always written).
- `inference/variants.yaml` — `registry:` blocks updated in place, comments
  preserved (only with `--sync`).
- MLflow experiment `fastdiff/sdxl-quant` — one parent run per build, one nested
  run per variant.

## What's measured (real plane)

- **size / VRAM / throughput** — measured around a real denoise loop.
- **quality** — a CLIP image-text score over a fixed prompt set, normalised
  against `fp16-base` so it reads as *fidelity retained vs FP16* (FP16 ≈ 100).
- **LoRAs** — the `train` step trains adapters declared in `loras.yaml` into
  `inference/loras/` before benchmarking (real plane only; `--demo` skips it).

The `--demo` plane simulates all of the above from the catalog baseline.
