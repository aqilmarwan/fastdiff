# web — fastdiff frontend

The Next.js frontend for fastdiff. It calls the FastAPI inference
service and **never bundles model code** — it only hits two endpoints:

| Endpoint            | Used for                                             |
| ------------------- | ---------------------------------------------------- |
| `GET  /variants`    | the precision × style matrix + registered metrics    |
| `POST /generate`    | a generation, streamed back as SSE progress events   |

If the API isn't reachable, the UI **transparently falls back to in-browser demo
data** (mock variants + a seeded procedural renderer) so the studio is fully
demoable before the backend exists. A badge in the nav shows `API live` vs
`demo data`. Nothing changes when FastAPI comes up.

## Pages

- **`/` Studio** — prompt → pick a variant/preset → stream progress → result card
  with honest, server-side metrics (latency breakdown, throughput, peak VRAM,
  cold/warm flag).
- **`/compare` Compare** — one prompt, one seed, two variants side by side, with a
  head-to-head verdict (speed / VRAM / size / quality deltas).

## Run

```bash
pnpm install
pnpm dev            # http://localhost:3000
```

Point it at your inference service:

```bash
cp .env.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Stack

Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS v4 ·
`motion` (Framer Motion) · `lucide-react`. No other UI dependencies.

## Backend contract

`POST /generate` should stream `text/event-stream` frames, each `data:` line a
JSON `GenEvent` (see `lib/types.ts`):

```jsonc
{ "type": "status",   "stage": "load|denoise|decode", "message": "…", "cold": true }
{ "type": "progress", "step": 12, "totalSteps": 30, "previewUrl": "data:…" }
{ "type": "done",     "result": { "imageUrl": "…", "variantId": "…", "params": {}, "metrics": {} } }
{ "type": "error",    "message": "…" }
```

`GET /variants` returns the `Variant[]` shape from `lib/types.ts`. Match those and
the frontend works unchanged.
