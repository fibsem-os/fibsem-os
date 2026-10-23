import { ChevronDown, Eye, EyeOff, GripHorizontal, Layers, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { COLORS, type Look, cssColor, histogram, tint } from "@/lib/layers";
import type { Values } from "@/lib/planes";
import { cn } from "@/lib/utils";

/** One row of the panel: a channel or a detector, and what it shows now. */
export interface PanelLayer {
  name: string;
  /** Its colour before anyone changed it. */
  color: string;
  visible: boolean;
  look: Look;
  /** The contrast drawn: the operator's, else auto. */
  clim: [number, number];
  /** Auto contrast held from the projection while stepping planes. */
  held: boolean;
  /** The values on screen, for the histogram. */
  values: Values | null;
}

const ROUND = (v: number) => (Math.abs(v) >= 100 ? Math.round(v) : +v.toPrecision(3));

/** The histogram of the shown values, the contrast range over it, and the display
 * curve: where black and white fall, bent by gamma. */
function Histogram({ layer, range }: { layer: PanelLayer; range: [number, number] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const counts = useHistogram(layer.values, range);
  useEffect(() => {
    const c = ref.current;
    if (!c || !counts) return;
    const dpr = devicePixelRatio || 1;
    c.width = c.clientWidth * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext("2d")!;
    const pad = 6 * dpr;
    const W = c.width - 2 * pad;
    const H = c.height;
    const [r0, r1] = range;
    const fx = (v: number) => pad + Math.max(0, Math.min(1, (v - r0) / (r1 - r0))) * W;
    const [lo, hi] = layer.clim;
    g.fillStyle = "rgba(0, 122, 204, 0.18)";
    g.fillRect(fx(lo), 0, fx(hi) - fx(lo), H);
    const top = Math.max(...counts.map((n) => Math.log1p(n))) || 1;
    const t = tint(layer.look.color);
    g.fillStyle = `rgba(${t.map((v) => Math.round(80 + v * 150)).join(",")}, 0.85)`;
    counts.forEach((n, i) => {
      const bh = (Math.log1p(n) / top) * (H - 3 * dpr);
      g.fillRect(pad + (i * W) / counts.length, H - bh, W / counts.length - 0.5 * dpr, bh);
    });
    g.strokeStyle = "rgba(255, 255, 255, 0.7)";
    g.lineWidth = dpr;
    g.beginPath();
    for (let x = 0; x <= W; x += 2) {
      const v = r0 + (x / W) * (r1 - r0);
      const n = Math.max(0, Math.min(1, (v - lo) / (hi - lo))) ** layer.look.gamma;
      const y = H - 1 - n * (H - 3 * dpr);
      if (x === 0) g.moveTo(pad + x, y);
      else g.lineTo(pad + x, y);
    }
    g.stroke();
  });
  return <canvas ref={ref} className="mt-2 block h-10 w-full rounded-sm bg-[#17191d]" />;
}

function useHistogram(values: Values | null, range: [number, number]) {
  const [lo, hi] = range;
  return useMemo(() => (values ? binned(values, [lo, hi]) : null), [values, lo, hi]);
}

function binned(values: Values, [lo, hi]: [number, number], bins = 96) {
  const counts = new Array(bins).fill(0);
  const step = Math.max(1, Math.floor(values.length / 300_000));
  const k = bins / (hi - lo);
  for (let i = 0; i < values.length; i += step) {
    const b = Math.floor((values[i] - lo) * k);
    if (b >= 0 && b < bins) counts[b]++;
  }
  return counts;
}

/** Two handles on one groove: the contrast's black and white points. */
function RangeSlider({
  range,
  clim,
  auto,
  set,
}: {
  range: [number, number];
  clim: [number, number];
  auto: boolean;
  set: (end: 0 | 1, value: number) => void;
}) {
  const groove = useRef<HTMLDivElement>(null);
  const [r0, r1] = range;
  const left = (v: number) => `calc(6px + (100% - 12px) * ${Math.max(0, Math.min(1, (v - r0) / (r1 - r0)))})`;
  const valueAt = (clientX: number) => {
    const r = groove.current!.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (clientX - r.left - 6) / (r.width - 12)));
    return r0 + f * (r1 - r0);
  };
  const grab = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  return (
    <div ref={groove} className="relative mt-1.5 h-4 touch-none">
      <div className="absolute top-1.5 right-1.5 left-1.5 h-1 rounded-sm bg-border" />
      <div
        className={cn("absolute top-1.5 h-1", auto ? "bg-[#414851]" : "bg-primary")}
        style={{ left: left(clim[0]), width: `calc(${left(clim[1])} - ${left(clim[0])})` }}
      />
      {([0, 1] as const).map((end) => (
        <div
          key={end}
          role="slider"
          tabIndex={0}
          aria-label={end ? "Contrast maximum" : "Contrast minimum"}
          aria-valuenow={ROUND(clim[end])}
          aria-valuemin={ROUND(r0)}
          aria-valuemax={ROUND(r1)}
          className={cn(
            "absolute top-px -ml-[6.5px] size-[13px] cursor-ew-resize rounded-full border focus-visible:outline-2 focus-visible:outline-link",
            auto ? "border-[#5a626c] bg-[#5a626c]" : "border-primary bg-foreground",
          )}
          style={{ left: left(clim[end]) }}
          onPointerDown={grab}
          onPointerMove={(e) => e.currentTarget.hasPointerCapture(e.pointerId) && set(end, valueAt(e.clientX))}
          onKeyDown={(e) => {
            if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
            e.preventDefault();
            e.stopPropagation();
            const step = ((r1 - r0) / 100) * (e.shiftKey ? 10 : 1);
            set(end, clim[end] + (e.key === "ArrowRight" ? step : -step));
          }}
        />
      ))}
    </div>
  );
}

function Swatch({ color, className }: { color: string; className?: string }) {
  return (
    <span
      className={cn("inline-block size-[13px] shrink-0 rounded-[3px] border border-white/25", className)}
      style={{ background: cssColor(color) }}
    />
  );
}

/** A labelled slider with its value. */
function Slider({
  label,
  value,
  shown,
  min,
  max,
  step,
  set,
}: {
  label: string;
  value: number;
  shown: string;
  min: number;
  max: number;
  step: number;
  set: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex justify-between text-[12px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{shown}</span>
      </span>
      <input
        type="range"
        className="h-3.5 w-full accent-[var(--primary)]"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => set(Number(e.target.value))}
      />
    </label>
  );
}

/**
 * The floating layers panel, as the Qt FM view's: the layers with their eyes, then
 * the selected one's colour, opacity, gamma and contrast (Auto, or set by hand on
 * the histogram), and Reset. Dragged by its header inside the image.
 */
export function LayersPanel({
  title,
  layers,
  selected,
  select,
  toggle,
  setLook,
  close,
}: {
  title: string;
  layers: PanelLayer[];
  selected: number;
  select: (i: number) => void;
  toggle: (i: number) => void;
  setLook: (i: number, look: Look) => void;
  close: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null);
  const panel = useRef<HTMLDivElement>(null);
  const layer = layers[Math.min(selected, layers.length - 1)];
  const values = layer?.values;
  const valueRange = useMemo(() => (values ? histogram(values, 2).range : null), [values]);
  if (!layer) return null;
  const i = layers.indexOf(layer);
  const look = layer.look;
  const auto = look.clim == null;
  const shownRange = valueRange ?? layer.clim;
  const range: [number, number] = [Math.min(shownRange[0], layer.clim[0]), Math.max(shownRange[1], layer.clim[1])];
  const set = (change: Partial<Look>) => setLook(i, { ...look, ...change });
  const setEnd = (end: 0 | 1, value: number) => {
    const clim: [number, number] = [...layer.clim];
    const gap = (range[1] - range[0]) / 200;
    if (end === 0) clim[0] = Math.min(value, clim[1] - gap);
    else clim[1] = Math.max(value, clim[0] + gap);
    set({ clim });
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest("button")) return;
    const el = panel.current!;
    const parent = el.parentElement!;
    const start = { x: e.clientX, y: e.clientY, left: el.offsetLeft, top: el.offsetTop };
    e.currentTarget.setPointerCapture(e.pointerId);
    const header = e.currentTarget;
    header.onpointermove = (m) =>
      setPlace({
        left: Math.max(0, Math.min(parent.clientWidth - el.offsetWidth, start.left + m.clientX - start.x)),
        top: Math.max(0, Math.min(parent.clientHeight - 40, start.top + m.clientY - start.y)),
      });
    header.onpointerup = () => (header.onpointermove = null);
  };
  return (
    <div
      ref={panel}
      className="absolute z-10 box-border max-h-[calc(100%-40px)] w-[268px] max-w-[calc(100%-16px)] overflow-auto rounded-[10px] border bg-card p-3.5 pb-4 text-[13px] shadow-[0_6px_24px_rgba(0,0,0,0.45)]"
      style={place ?? { top: 32, right: 8 }}
      onPointerDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      <div className="flex cursor-grab touch-none items-center gap-2 select-none" onPointerDown={move}>
        <Layers size={16} className="text-muted-foreground" />
        <span className="text-[12px] font-medium tracking-[1px] text-muted-foreground">{title}</span>
        <GripHorizontal size={16} className="ml-auto text-muted-foreground" />
        <button
          className="grid size-[18px] cursor-pointer place-items-center rounded-full border-0 bg-transparent p-0 text-muted-foreground hover:bg-border"
          aria-label="Close"
          title="Close (L)"
          onClick={close}
        >
          <X size={14} />
        </button>
      </div>
      <div className="mt-3 flex flex-col gap-0.5" role="listbox" aria-label="Layers">
        {layers.map((l, k) => (
          <div
            key={l.name}
            role="option"
            aria-selected={k === i}
            className={cn(
              "flex cursor-pointer items-center gap-[11px] rounded-[7px] px-2.5 py-[7px]",
              k === i ? "bg-border" : "hover:bg-accent",
            )}
            onClick={() => {
              select(k);
              setMenu(false);
            }}
          >
            <button
              className={cn(
                "grid size-4 cursor-pointer place-items-center border-0 bg-transparent p-0",
                l.visible ? "text-foreground" : "text-[#6b6b6b]",
              )}
              aria-label={`${l.visible ? "Hide" : "Show"} ${l.name}`}
              onClick={(e) => {
                e.stopPropagation();
                toggle(k);
              }}
            >
              {l.visible ? <Eye size={16} /> : <EyeOff size={16} />}
            </button>
            <Swatch color={l.look.color} />
            <span className={cn("min-w-0 flex-1 truncate", l.visible ? "text-foreground" : "text-[#6b6b6b]")}>
              {l.name}
            </span>
          </div>
        ))}
      </div>
      <div className="my-3 h-px bg-border" />
      <div className="mb-3.5 flex items-center gap-2">
        <Swatch color={look.color} className="size-[11px]" />
        <span className="font-medium">{layer.name}</span>
      </div>
      <div className="relative mb-3 flex items-center justify-between">
        <span className="text-[12px] text-muted-foreground">Colormap</span>
        <button
          className="flex w-[132px] cursor-pointer items-center gap-[7px] rounded-md border bg-panel px-2 py-1 text-[12px] text-foreground"
          aria-haspopup="listbox"
          aria-expanded={menu}
          onClick={() => setMenu(!menu)}
        >
          <Swatch color={look.color} className="size-3" />
          {look.color}
          <ChevronDown size={14} className="ml-auto text-muted-foreground" />
        </button>
        {menu && (
          <div className="absolute top-7 right-0 z-20 w-[132px] rounded border bg-panel py-0.5" role="listbox">
            {COLORS.map((c) => (
              <button
                key={c}
                role="option"
                aria-selected={c === look.color}
                className="flex w-full cursor-pointer items-center gap-[7px] border-0 bg-transparent px-2 py-1 text-left text-[12px] text-foreground hover:bg-border aria-selected:bg-border"
                onClick={() => {
                  set({ color: c });
                  setMenu(false);
                }}
              >
                <Swatch color={c} className="size-3" />
                {c}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="mb-3 flex flex-col gap-3">
        <Slider
          label="Opacity"
          value={Math.round(look.opacity * 100)}
          shown={`${Math.round(look.opacity * 100)}%`}
          min={0}
          max={100}
          step={1}
          set={(v) => set({ opacity: v / 100 })}
        />
        <Slider
          label="Gamma"
          value={look.gamma}
          shown={look.gamma.toFixed(2)}
          min={0.1}
          max={3}
          step={0.01}
          set={(v) => set({ gamma: v })}
        />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-muted-foreground">Contrast</span>
        {auto && layer.held && <span className="ml-auto text-[11px] text-muted-foreground">from projection</span>}
        <button
          aria-pressed={auto}
          title="Auto contrast: the 1st to 99th percentile"
          className={cn(
            "cursor-pointer rounded border px-2.5 py-0.5 text-[11px]",
            !layer.held || !auto ? "ml-auto" : "",
            auto
              ? "border-[var(--fibsem-primary-accent)] bg-primary/15 text-link"
              : "bg-transparent text-muted-foreground",
          )}
          onClick={() => set({ clim: auto ? layer.clim : null })}
        >
          Auto
        </button>
      </div>
      <Histogram layer={layer} range={range} />
      <RangeSlider range={range} clim={layer.clim} auto={auto} set={setEnd} />
      <div className="mt-2 flex items-center justify-between">
        {([0, 1] as const).map((end) => (
          <input
            key={`${end}:${ROUND(layer.clim[end])}`}
            type="number"
            aria-label={end ? "Contrast maximum value" : "Contrast minimum value"}
            defaultValue={ROUND(layer.clim[end])}
            className={cn(
              "w-16 rounded border bg-panel px-1.5 py-0.5 text-[11px] text-foreground tabular-nums",
              end && "text-right",
            )}
            onKeyDown={(e) => e.stopPropagation()}
            onBlur={(e) => e.target.value !== "" && setEnd(end, Number(e.target.value))}
          />
        ))}
      </div>
      <button
        className="mt-3.5 w-full cursor-pointer rounded-md border bg-transparent p-2 text-[12px] text-foreground hover:bg-border"
        onClick={() => setLook(i, { color: layer.color, opacity: 1, gamma: 1, clim: null })}
      >
        Reset adjustments
      </button>
    </div>
  );
}
