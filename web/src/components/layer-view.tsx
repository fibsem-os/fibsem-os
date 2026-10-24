import { useEffect, useLayoutEffect, useRef, useState, type MutableRefObject, type ReactNode } from "react";
import { type DrawLayer, LayerRenderer, type Placement } from "@/lib/gl-layers";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";

/** Where the view looks: the image pixel at its centre, and how far in from fitting. */
export interface View {
  zoom: number;
  cx: number;
  cy: number;
}

export const fitView = (width: number, height: number): View => ({ zoom: 1, cx: width / 2, cy: height / 2 });

/** The drawn canvas and where the image is on it, in canvas pixels. */
export interface Snapshot {
  canvas: HTMLCanvasElement;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Tools {
  ruler: boolean;
  crosshair: boolean;
  scalebar: boolean;
}

type Point = [number, number];

// The ruler's and crosshair's colour: the canvas's current-position yellow.
const MARK = "#ffee58";

// fibsem.imaging.drawing._NICE_NUMBERS
const NICE = [1, 2, 5];

/** The round length nearest *metres*, as the exported images' scalebar picks it
 * (fibsem.imaging.drawing._pick_scalebar_length_m). */
export function roundLength(metres: number): number {
  const base = 10 ** Math.floor(Math.log10(metres));
  return NICE.map((n) => n * base).reduce((best, v) => (Math.abs(metres - v) < Math.abs(metres - best) ? v : best));
}

/** The image's place on a *width* x *height* box for *view*: CSS pixels per image pixel. */
export function placement(view: View, full: [number, number], width: number, height: number): Placement {
  const scale = Math.min(width / full[1], height / full[0]) * view.zoom;
  return { x: width / 2 - view.cx * scale, y: height / 2 - view.cy * scale, scale };
}

/** Zoomed by *by* about the box point (*px*, *py*), the centre when not given. */
export function zoomed(view: View, by: number, at?: Placement, px?: number, py?: number): View {
  const zoom = Math.max(0.25, Math.min(64, view.zoom * by));
  if (!at || px == null || py == null) return { ...view, zoom };
  const ix = (px - at.x) / at.scale;
  const iy = (py - at.y) / at.scale;
  const k = view.zoom / zoom;
  return { zoom, cx: ix + (view.cx - ix) * k, cy: iy + (view.cy - iy) * k };
}

/**
 * The image drawn from its layers on the GPU, zoomed and panned in the browser.
 * Wheel zooms where it points, drag pans (or measures with the ruler); *children*
 * sit over the image itself, at its place and size, so marks follow the view.
 */
export function LayerView({
  layers,
  factor,
  full,
  pixelSize,
  tools,
  view,
  setView,
  pannable,
  dim,
  onCursor,
  onRenderer,
  snapshot,
  children,
}: {
  layers: DrawLayer[];
  /** How many image pixels each value of *layers* stands for. */
  factor: number;
  /** The full image's (height, width). */
  full: [number, number];
  pixelSize: number | null | undefined;
  tools: Tools;
  view: View;
  setView: (v: View) => void;
  pannable: boolean;
  dim?: boolean;
  onCursor: (at: Point | null) => void;
  onRenderer?: (renderer: LayerRenderer | null) => void;
  snapshot?: MutableRefObject<(() => Snapshot) | null>;
  children?: ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<LayerRenderer | null>(null);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const [line, setLine] = useState<[Point, Point] | null>(null);
  const drag = useRef<{ x: number; y: number; view: View; start: Point } | null>(null);
  const at = placement(view, full, size[0], size[1]);
  if (snapshot)
    snapshot.current = () => {
      const dpr = devicePixelRatio || 1;
      const s = at.scale * dpr;
      return { canvas: canvas.current!, x: at.x * dpr, y: at.y * dpr, width: full[1] * s, height: full[0] * s };
    };

  useLayoutEffect(() => {
    renderer.current = canvas.current ? LayerRenderer.create(canvas.current) : null;
    onRenderer?.(renderer.current);
    const observer = new ResizeObserver(([entry]) => setSize([entry.contentRect.width, entry.contentRect.height]));
    if (box.current) observer.observe(box.current);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // the ruler goes with the ruler tool
  if (!tools.ruler && line) setLine(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !renderer.current || !size[0]) return;
    const dpr = devicePixelRatio || 1;
    const w = Math.round(size[0] * dpr);
    const h = Math.round(size[1] * dpr);
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const scaled = { x: at.x * dpr, y: at.y * dpr, scale: at.scale * dpr * factor };
    renderer.current.draw(layers, scaled, w, h, at.scale * dpr > 2);
  });

  useEffect(() => {
    const c = overlay.current;
    if (!c || !size[0]) return;
    const dpr = devicePixelRatio || 1;
    c.width = Math.round(size[0] * dpr);
    c.height = Math.round(size[1] * dpr);
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size[0], size[1]);
    const toBox = ([x, y]: Point): Point => [at.x + x * at.scale, at.y + y * at.scale];
    g.strokeStyle = MARK;
    if (tools.crosshair) {
      const [mx, my] = toBox([full[1] / 2, full[0] / 2]);
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(mx - 14, my);
      g.lineTo(mx + 14, my);
      g.moveTo(mx, my - 14);
      g.lineTo(mx, my + 14);
      g.stroke();
    }
    if (line) {
      const a = toBox(line[0]);
      const b = toBox(line[1]);
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(...a);
      g.lineTo(...b);
      g.stroke();
      const pixels = Math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]);
      const label = pixelSize ? fmt.length(pixels * pixelSize) : `${pixels.toFixed(0)} px`;
      g.font = "12px system-ui, sans-serif";
      const tw = g.measureText(label).width;
      g.fillStyle = "rgba(0, 0, 0, 0.7)";
      g.fillRect(b[0] + 6, b[1] - 16, tw + 8, 18);
      g.fillStyle = MARK;
      g.fillText(label, b[0] + 10, b[1] - 3);
    }
  });

  // a wheel zooms where it points; React's wheel listener can't stop the page scrolling
  const latest = useRef({ view, at, setView });
  latest.current = { view, at, setView };
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const { view, at, setView } = latest.current;
      setView(zoomed(view, e.deltaY < 0 ? 1.2 : 1 / 1.2, at, e.clientX - r.left, e.clientY - r.top));
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, []);

  const imagePoint = (e: React.PointerEvent): Point => {
    const r = box.current!.getBoundingClientRect();
    return [(e.clientX - r.left - at.x) / at.scale, (e.clientY - r.top - at.y) / at.scale];
  };

  // a bar about a fifth of the image's width on screen, or of the box when zoomed in
  const metresPerPx = pixelSize ? pixelSize / at.scale : null;
  const across = Math.min(size[0], full[1] * at.scale);
  const bar = metresPerPx && across ? roundLength((metresPerPx * across) / 5) : null;

  return (
    <div
      ref={box}
      className={cn("relative h-full w-full touch-none overflow-hidden bg-black", tools.ruler && "cursor-crosshair")}
      onPointerDown={(e) => {
        if (e.button !== 0 || (!pannable && !tools.ruler)) return;
        const p = imagePoint(e);
        drag.current = { x: e.clientX, y: e.clientY, view, start: p };
        if (tools.ruler) setLine([p, p]);
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        const p = imagePoint(e);
        onCursor(p[0] >= 0 && p[1] >= 0 && p[0] < full[1] && p[1] < full[0] ? p : null);
        const d = drag.current;
        if (!d) return;
        if (tools.ruler) setLine([d.start, p]);
        else
          setView({
            ...d.view,
            cx: d.view.cx - (e.clientX - d.x) / at.scale,
            cy: d.view.cy - (e.clientY - d.y) / at.scale,
          });
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
      onPointerLeave={() => onCursor(null)}
    >
      <canvas
        ref={canvas}
        className={cn("absolute inset-0 h-full w-full", pannable && !tools.ruler && "cursor-grab", dim && "opacity-50")}
      />
      <canvas ref={overlay} className="pointer-events-none absolute inset-0 h-full w-full" />
      {children && (
        <div
          className="absolute"
          style={{ left: at.x, top: at.y, width: full[1] * at.scale, height: full[0] * at.scale }}
        >
          {children}
        </div>
      )}
      {tools.scalebar && bar && metresPerPx && (
        <div className="pointer-events-none absolute right-2 bottom-2 flex flex-col items-center gap-1 rounded-sm bg-black/60 px-1.5 py-1 text-[11px] leading-none text-white">
          <span>{fmt.length(bar)}</span>
          <span className="block h-1 bg-white" style={{ width: bar / metresPerPx }} />
        </div>
      )}
      <span className="pointer-events-none absolute bottom-2 left-2 text-[11px] text-white tabular-nums [text-shadow:0_0_3px_#000]">
        {Math.round(view.zoom * 100)}%
      </span>
    </div>
  );
}
