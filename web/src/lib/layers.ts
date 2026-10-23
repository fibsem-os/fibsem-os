// How a layer is drawn: the maths of fibsem.fm.composite, and each channel's look
// kept in this browser by its name, so GFP keeps its contrast from stack to stack.

import type { Values } from "./planes";

/** The channel colours, as fibsem.fm.composite tints them (napari's colormap ends). */
export const TINTS: Record<string, [number, number, number]> = {
  violet: [0.933, 0.51, 0.933],
  blue: [0, 0, 1],
  cyan: [0, 1, 1],
  green: [0, 1, 0],
  yellow: [1, 1, 0],
  red: [1, 0, 0],
  magenta: [1, 0, 1],
  gray: [1, 1, 1],
};
export const COLORS = Object.keys(TINTS);

export const tint = (color: string): [number, number, number] => TINTS[color] ?? TINTS.gray;
export const cssColor = (color: string) =>
  `rgb(${tint(color)
    .map((v) => Math.round(v * 255))
    .join(",")})`;

/** What the operator set for a channel; contrast is auto until set by hand. */
export interface Look {
  color: string;
  opacity: number;
  gamma: number;
  clim: [number, number] | null;
}

const KEY = "fibsem-layer:";

export function storedLook(name: string, color: string): Look {
  const look: Look = { color, opacity: 1, gamma: 1, clim: null };
  try {
    const saved = JSON.parse(localStorage.getItem(KEY + name) ?? "null");
    if (saved && typeof saved === "object") return { ...look, ...saved };
  } catch {
    // storage off or unreadable: the defaults
  }
  return look;
}

export function storeLook(name: string, look: Look, color: string) {
  try {
    const plain = look.opacity === 1 && look.gamma === 1 && look.clim == null && look.color === color;
    if (plain) localStorage.removeItem(KEY + name);
    else localStorage.setItem(KEY + name, JSON.stringify(look));
  } catch {
    // storage off: for this page only
  }
}

function percentile(sorted: Float64Array, q: number) {
  // numpy's default (linear) interpolation
  const rank = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

/** fibsem.fm.composite.auto_clim: the 1st and 99th percentiles of a (h, w) plane,
 * strided to about 250k values as it is; its min and max when they meet. */
export function autoClim(values: Values, width: number, maxSamples = 250_000): [number, number] {
  const height = values.length / width;
  const step = values.length > maxSamples ? Math.ceil(Math.sqrt(values.length / maxSamples)) : 1;
  const sample: number[] = [];
  for (let y = 0; y < height; y += step) for (let x = 0; x < width; x += step) sample.push(values[y * width + x]);
  const sorted = Float64Array.from(sample).sort();
  let lo = percentile(sorted, 1);
  let hi = percentile(sorted, 99);
  if (hi <= lo) {
    lo = sorted[0];
    hi = sorted[sorted.length - 1];
    if (hi <= lo) hi = lo + 1;
  }
  return [lo, hi];
}

/** A plane's values in *bins* bins from its least to its most, and that range. */
export function histogram(values: Values, bins = 128): { counts: number[]; range: [number, number] } {
  let lo = Infinity;
  let hi = -Infinity;
  const step = Math.max(1, Math.floor(values.length / 500_000));
  for (let i = 0; i < values.length; i += step) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) hi = lo + 1;
  const counts = new Array(bins).fill(0);
  const k = bins / (hi - lo);
  for (let i = 0; i < values.length; i += step) counts[Math.min(bins - 1, Math.floor((values[i] - lo) * k))]++;
  return { counts, range: [lo, hi] };
}
