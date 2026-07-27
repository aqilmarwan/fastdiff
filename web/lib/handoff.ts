/** Small sessionStorage bridge so "Compare" inherits the Studio's prompt/seed. */

export interface Handoff {
  prompt: string;
  negativePrompt: string;
  steps: number;
  guidance: number;
  seed: number;
  variantId: string;
}

const KEY = "fastdiff:handoff";

export function writeHandoff(h: Handoff): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(h));
  } catch {
    /* storage unavailable — ignore */
  }
}

export function readHandoff(): Handoff | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    sessionStorage.removeItem(KEY);
    return JSON.parse(raw) as Handoff;
  } catch {
    return null;
  }
}
