"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Cpu, GitCompareArrows, Sparkles } from "lucide-react";
import { getVariants } from "@/lib/api";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/", label: "Studio", icon: Sparkles },
  { href: "/compare", label: "Compare", icon: GitCompareArrows },
];

export function Nav() {
  const pathname = usePathname();
  const [source, setSource] = useState<"live" | "mock" | "checking">("checking");

  useEffect(() => {
    let alive = true;
    getVariants().then((r) => alive && setSource(r.source));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 glass">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-4 px-5">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg btn-glow text-ink">
            <Cpu className="h-4 w-4" strokeWidth={2.5} />
          </span>
          <span className="text-[15px] font-semibold tracking-tight">
            Fast<span className="text-grad">Diff</span>
          </span>
        </Link>

        <nav className="ml-2 flex items-center gap-1 rounded-full border border-line bg-ink-2/60 p-1">
          {TABS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[13px] font-medium transition-colors",
                  active ? "bg-white/10 text-fg" : "text-muted hover:text-fg",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <span className="hidden items-center gap-1.5 rounded-full border border-line px-3 py-1 text-[11px] text-muted sm:flex">
            <span className="text-faint">base</span>
            <span className="font-medium text-fg">SDXL 1.0</span>
            <span className="text-faint">·</span>
            <span className="text-faint">24GB GPU</span>
          </span>
          <ApiBadge source={source} />
        </div>
      </div>
    </header>
  );
}

function ApiBadge({ source }: { source: "live" | "mock" | "checking" }) {
  const map = {
    checking: { dot: "bg-faint", label: "connecting", title: "Probing inference API…" },
    live: { dot: "bg-good", label: "API live", title: "FastAPI inference reachable" },
    mock: { dot: "bg-warn", label: "demo data", title: "FastAPI not reachable — using in-browser mock" },
  } as const;
  const s = map[source];
  return (
    <span
      title={s.title}
      className="flex items-center gap-1.5 rounded-full border border-line bg-ink-2/60 px-2.5 py-1 text-[11px] font-medium text-muted"
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot, source !== "checking" && "animate-pulse")} />
      {s.label}
    </span>
  );
}
