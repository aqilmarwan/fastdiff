"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ChevronDown,
  Dices,
  Download,
  GitCompareArrows,
  Lock,
  RotateCcw,
  Sparkles,
  Unlock,
  Wand2,
} from "lucide-react";
import { getVariants } from "@/lib/api";
import { EXAMPLE_PROMPTS, MOCK_PRESETS, MOCK_VARIANTS } from "@/lib/variants";
import type { GenerationParams, Preset, Variant } from "@/lib/types";
import { useGeneration } from "@/lib/useGeneration";
import { cn, fmtMs, randomSeed } from "@/lib/utils";
import { Slider } from "./Slider";
import { VariantPicker } from "./VariantPicker";
import { GenerationCanvas } from "./GenerationCanvas";
import { MetricsPanel } from "./MetricsPanel";
import { PrecisionTag } from "./PrecisionTag";
import { writeHandoff } from "@/lib/handoff";

export function Studio() {
  const router = useRouter();
  const [variants, setVariants] = useState<Variant[]>(MOCK_VARIANTS);
  const [variantId, setVariantId] = useState<string>(MOCK_VARIANTS[0].id);

  const [prompt, setPrompt] = useState(EXAMPLE_PROMPTS[0]);
  const [negative, setNegative] = useState("blurry, low detail, watermark, extra fingers");
  const [showNegative, setShowNegative] = useState(false);
  const [steps, setSteps] = useState(30);
  const [guidance, setGuidance] = useState(6.5);
  const [seed, setSeed] = useState(742183);
  const [lockSeed, setLockSeed] = useState(true);
  const [activePreset, setActivePreset] = useState<Preset["id"] | null>("quality");

  const { state, run, cancel } = useGeneration();
  const variant = useMemo(() => variants.find((v) => v.id === variantId) ?? variants[0], [variants, variantId]);

  useEffect(() => {
    getVariants().then(({ variants }) => {
      setVariants(variants);
      setVariantId((id) => (variants.some((v) => v.id === id) ? id : variants[0].id));
    });
  }, []);

  const estDenoiseMs = (steps / variant.stepsPerSec) * 1000;

  const applyPreset = (p: Preset) => {
    setActivePreset(p.id);
    setVariantId(p.variantId);
    setSteps(p.steps);
    setGuidance(p.guidance);
  };

  const handleGenerate = () => {
    if (!prompt.trim() || state.running) return;
    const nextSeed = lockSeed ? seed : randomSeed();
    if (!lockSeed) setSeed(nextSeed);
    const params: GenerationParams = {
      prompt: prompt.trim(),
      negativePrompt: negative.trim(),
      variantId: variant.id,
      steps,
      guidance,
      seed: nextSeed,
      width: 1024,
      height: 1024,
    };
    run(params, variant);
  };

  const sendToCompare = () => {
    writeHandoff({ prompt: prompt.trim(), negativePrompt: negative.trim(), steps, guidance, seed, variantId: variant.id });
    router.push("/compare");
  };

  return (
    <div className="mx-auto grid max-w-[1400px] gap-5 px-5 py-6 lg:grid-cols-[390px_1fr]">
      {/* ----------------------------------------------------------- controls */}
      <aside className="card h-fit space-y-5 p-4 lg:sticky lg:top-[4.5rem]">
        <Section label="Prompt" hint={`${prompt.length} chars`}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Describe the image you want to render…"
            className="w-full resize-none rounded-xl border border-line bg-ink-2/60 p-3 text-[13px] leading-relaxed text-fg outline-none transition-colors placeholder:text-faint focus:border-line-strong"
          />
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLE_PROMPTS.map((p, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setPrompt(p)}
                className="rounded-full border border-line px-2.5 py-1 text-[11px] text-muted transition-colors hover:border-line-strong hover:text-fg"
              >
                {p.split(",")[0].slice(0, 22)}…
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setShowNegative((s) => !s)}
            className="flex items-center gap-1 text-[11px] text-faint transition-colors hover:text-muted"
          >
            <ChevronDown className={cn("h-3 w-3 transition-transform", showNegative && "rotate-180")} />
            Negative prompt
          </button>
          {showNegative && (
            <textarea
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              rows={2}
              className="w-full resize-none rounded-xl border border-line bg-ink-2/60 p-3 text-[12px] text-muted outline-none focus:border-line-strong"
            />
          )}
        </Section>

        <Divider />

        <Section label="Presets">
          <div className="grid grid-cols-2 gap-2">
            {MOCK_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p)}
                className={cn(
                  "rounded-xl border p-2.5 text-left transition-all",
                  activePreset === p.id ? "border-transparent bg-white/[0.06] ring-1 ring-line-strong" : "border-line hover:border-line-strong",
                )}
              >
                <div className="flex items-center gap-1.5 text-[13px] font-semibold">
                  {p.id === "fast" ? <Sparkles className="h-3.5 w-3.5 text-fp8" /> : <Wand2 className="h-3.5 w-3.5 text-fp16" />}
                  {p.label}
                </div>
                <p className="mt-1 text-[11px] leading-snug text-faint">{p.blurb}</p>
              </button>
            ))}
          </div>
        </Section>

        <Section label="Variant" hint="precision × style">
          <VariantPicker
            variants={variants}
            value={variantId}
            disabled={state.running}
            onChange={(id) => {
              setVariantId(id);
              setActivePreset(null);
            }}
          />
        </Section>

        <Divider />

        <Section label="Parameters">
          <div className="space-y-4">
            <Slider label="Steps" value={steps} min={1} max={50} suffix=" steps" disabled={state.running} onChange={(v) => { setSteps(v); setActivePreset(null); }} />
            <Slider label="Guidance" value={guidance} min={0} max={15} step={0.5} disabled={state.running} onChange={(v) => { setGuidance(v); setActivePreset(null); }} />
            <SeedField seed={seed} locked={lockSeed} disabled={state.running} onSeed={setSeed} onToggleLock={() => setLockSeed((l) => !l)} onRandomize={() => setSeed(randomSeed())} />
          </div>
        </Section>

        <button
          type="button"
          onClick={state.running ? cancel : handleGenerate}
          disabled={!state.running && !prompt.trim()}
          className={cn(
            "flex h-12 w-full items-center justify-center gap-2 rounded-xl text-[14px] font-semibold transition-all disabled:opacity-40",
            state.running ? "border border-line bg-ink-2 text-fg hover:border-line-strong" : "btn-glow text-ink",
          )}
        >
          {state.running ? (
            <>Cancel</>
          ) : (
            <>
              <Sparkles className="h-4 w-4" strokeWidth={2.5} />
              Generate
              <span className="tabular ml-1 rounded-md bg-black/20 px-1.5 py-0.5 text-[11px] font-normal">
                ~{fmtMs(estDenoiseMs)}
              </span>
            </>
          )}
        </button>
      </aside>

      {/* ------------------------------------------------------------- canvas */}
      <section className="space-y-4">
        <div className="mx-auto w-full max-w-[640px] space-y-4">
          <GenerationCanvas state={state} variant={variant} />

          {state.stage === "error" ? (
            <div className="card border-bad/30 p-4 text-[13px] text-bad">Generation failed: {state.error}</div>
          ) : state.result ? (
            <ResultCard
              variant={variants.find((v) => v.id === state.result!.variantId) ?? variant}
              result={state.result}
              onRegenerate={handleGenerate}
              onCompare={sendToCompare}
            />
          ) : (
            <IdleHint variant={variant} steps={steps} />
          )}
        </div>
      </section>
    </div>
  );
}

/* --------------------------------------------------------------- subviews -- */

function ResultCard({
  variant,
  result,
  onRegenerate,
  onCompare,
}: {
  variant: Variant;
  result: NonNullable<ReturnType<typeof useGeneration>["state"]["result"]>;
  onRegenerate: () => void;
  onCompare: () => void;
}) {
  return (
    <div className="card space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <PrecisionTag precision={variant.precision} />
          <span className="text-[13px] font-medium">{variant.label}</span>
          <span className="tabular rounded-md bg-ink-2 px-1.5 py-0.5 text-[11px] text-faint">
            seed {result.params.seed}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <IconBtn title="Regenerate" onClick={onRegenerate}><RotateCcw className="h-3.5 w-3.5" /></IconBtn>
          <a
            href={result.imageUrl}
            download={`fastdiff-${variant.id}-${result.params.seed}.svg`}
            className="grid h-8 w-8 place-items-center rounded-lg border border-line text-muted transition-colors hover:border-line-strong hover:text-fg"
            title="Download"
          >
            <Download className="h-3.5 w-3.5" />
          </a>
          <button
            type="button"
            onClick={onCompare}
            className="ml-1 flex h-8 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[12px] font-medium text-muted transition-colors hover:border-line-strong hover:text-fg"
          >
            <GitCompareArrows className="h-3.5 w-3.5" />
            Compare
          </button>
        </div>
      </div>
      <MetricsPanel metrics={result.metrics} />
      <div className="tabular flex flex-wrap gap-x-4 gap-y-1 border-t border-line pt-3 text-[11px] text-faint">
        <span>{result.params.steps} steps</span>
        <span>guidance {result.params.guidance}</span>
        <span>{result.params.width}×{result.params.height}</span>
        <span>{variant.base}</span>
      </div>
    </div>
  );
}

function IdleHint({ variant, steps }: { variant: Variant; steps: number }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 text-[12px] text-muted">
        <PrecisionTag precision={variant.precision} />
        <span>
          Expecting <span className="tabular text-fg">~{fmtMs((steps / variant.stepsPerSec) * 1000)}</span> denoise ·{" "}
          <span className="tabular text-fg">{variant.vramGB.toFixed(1)} GB</span> VRAM ·{" "}
          <span className="tabular text-fg">{variant.quality}/100</span> quality.
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        Switch precision to watch the tradeoff, or open Compare to render two variants on one seed. Metrics are measured
        server-side around the real denoising loop — cold starts are flagged honestly.
      </p>
    </div>
  );
}

function SeedField({
  seed,
  locked,
  disabled,
  onSeed,
  onToggleLock,
  onRandomize,
}: {
  seed: number;
  locked: boolean;
  disabled?: boolean;
  onSeed: (n: number) => void;
  onToggleLock: () => void;
  onRandomize: () => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-[12px] font-medium text-muted">Seed</label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={seed}
          disabled={disabled}
          onChange={(e) => onSeed(Number(e.target.value))}
          className="tabular h-9 w-full rounded-lg border border-line bg-ink-2/60 px-3 text-[13px] text-fg outline-none focus:border-line-strong disabled:opacity-50"
        />
        <IconBtn title={locked ? "Seed locked (reused each run)" : "Seed roams (random each run)"} onClick={onToggleLock} active={locked}>
          {locked ? <Lock className="h-3.5 w-3.5" /> : <Unlock className="h-3.5 w-3.5" />}
        </IconBtn>
        <IconBtn title="Randomize seed" onClick={onRandomize} disabled={disabled}>
          <Dices className="h-3.5 w-3.5" />
        </IconBtn>
      </div>
    </div>
  );
}

function IconBtn({
  children,
  title,
  onClick,
  active,
  disabled,
}: {
  children: React.ReactNode;
  title: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "grid h-8 w-8 shrink-0 place-items-center rounded-lg border transition-colors disabled:opacity-40",
        active ? "border-accent/50 bg-accent/10 text-accent" : "border-line text-muted hover:border-line-strong hover:text-fg",
      )}
    >
      {children}
    </button>
  );
}

function Section({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-faint">{label}</h3>
        {hint ? <span className="text-[10px] text-faint">{hint}</span> : null}
      </div>
      {children}
    </div>
  );
}

function Divider() {
  return <div className="h-px bg-line" />;
}
