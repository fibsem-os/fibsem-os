import { Crosshair, Download, Layers, Maximize, Ruler, Scaling, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode } from "react";
import { ImageInfoBar } from "@/components/image-info";
import { LayersPanel, type PanelLayer } from "@/components/layers-panel";
import { LayerView, type Snapshot, type Tools, type View, fitView, roundLength, zoomed } from "@/components/layer-view";
import { PROJECTIONS, type StackShown, ZRow } from "@/components/z-row";
import { type DrawLayer, LayerRenderer } from "@/lib/gl-layers";
import * as fmt from "@/lib/format";
import { type Look, autoClim, cssColor, storeLook, storedLook, tint } from "@/lib/layers";
import { type Planes, type PlanesAsk, type Values, planeOf, project, readPlanes, usePlanes } from "@/lib/planes";
import type { FileRead, ImageInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

// A plane is read sharp once the slider rests on it this long.
const SETTLE_MS = 150;

/** The copies read: a stack's every plane small, for stepping without a wait, and
 * the plane or projection on screen sharp, a 1/4 copy of a large image. */
function factors(full: [number, number], planes: number) {
  const long = Math.max(...full);
  const sharp = long / 4 >= 1024 ? 4 : 1;
  const quick = planes > 1 ? (long / 8 >= 128 ? 8 : sharp) : null;
  return { sharp, quick };
}

const autoClims = new WeakMap<Values, [number, number]>();
function autoOf(values: Values, width: number) {
  let clim = autoClims.get(values);
  if (!clim) autoClims.set(values, (clim = autoClim(values, width)));
  return clim;
}

const projections = new WeakMap<Planes, Map<string, Float32Array>>();
function projectionOf(planes: Planes, c: number, shown: StackShown) {
  let kept = projections.get(planes);
  if (!kept) projections.set(planes, (kept = new Map()));
  const key = `${c}:${shown.projection}:${shown.first}:${shown.last}`;
  let values = kept.get(key);
  if (!values) kept.set(key, (values = project(planes, c, shown.projection!, shown.first, shown.last)));
  return values;
}

function askFor(shown: StackShown, factor: number): PlanesAsk {
  return shown.projection
    ? { factor, z: shown.projection, first: shown.first, last: shown.last }
    : { factor, z: shown.z };
}

const sameAsk = (a: PlanesAsk | null, b: PlanesAsk | null) => JSON.stringify(a) === JSON.stringify(b);

/** A toolbar button over the image, as the Qt canvas's. */
function Tool({
  label,
  keyName,
  pressed,
  onClick,
  children,
}: {
  label: string;
  keyName: string;
  pressed?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      className={cn(
        "grid size-[22px] cursor-pointer place-items-center rounded-[3px] border border-[#555] bg-[rgba(40,41,48,0.7)] p-0 text-[#aaa] hover:bg-[rgba(74,74,74,0.8)]",
        pressed && "border-white bg-[rgba(90,92,100,0.8)] text-white",
      )}
      title={`${label} (${keyName})`}
      aria-label={label}
      aria-pressed={pressed}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/** The image as a PNG: the part in view as drawn, or the whole image at full size,
 * with the scalebar and the bar's text when asked. */
async function exportPng({
  name,
  snapshot,
  full,
  draw,
  pixelSize,
  info,
  values,
  scalebar,
  bar,
}: {
  name: string;
  snapshot: Snapshot | null;
  full: (() => Promise<HTMLCanvasElement>) | null;
  draw: [number, number];
  pixelSize: number | null | undefined;
  info: ImageInfo | null | undefined;
  values: Record<string, string>;
  scalebar: boolean;
  bar: boolean;
}) {
  let source: CanvasImageSource;
  let sx = 0;
  let sy = 0;
  let w: number;
  let h: number;
  let metresPerPx = pixelSize ?? null;
  if (full) {
    source = await full();
    [w, h] = [source.width, source.height];
  } else {
    const s = snapshot!;
    source = s.canvas;
    sx = Math.max(0, s.x);
    sy = Math.max(0, s.y);
    w = Math.round(Math.min(s.canvas.width, s.x + s.width) - sx);
    h = Math.round(Math.min(s.canvas.height, s.y + s.height) - sy);
    if (metresPerPx) metresPerPx = (metresPerPx * draw[1]) / s.width;
  }
  const barHeight = bar && info ? Math.max(18, Math.round(h * 0.04)) : 0;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h + barHeight;
  const g = out.getContext("2d")!;
  g.imageSmoothingEnabled = false;
  g.drawImage(source, sx, sy, w, h, 0, 0, w, h);
  if (scalebar && metresPerPx) {
    const length = roundLength((metresPerPx * w) / 5);
    const px = length / metresPerPx;
    const size = Math.max(11, Math.round(h * 0.025));
    g.fillStyle = "#fff";
    g.fillRect(w - px - size, h - size * 2.2, px, Math.max(3, size / 4));
    g.font = `${size}px sans-serif`;
    g.textAlign = "right";
    g.fillText(fmt.length(length), w - size, h - size * 0.6);
  }
  if (barHeight && info) {
    const size = Math.round(barHeight * 0.5);
    g.fillStyle = "#1e2124";
    g.fillRect(0, h, w, barHeight);
    g.fillStyle = "#d6d6d6";
    g.font = `${size}px sans-serif`;
    g.textAlign = "left";
    const text = [
      info.kind,
      ...info.header.map((f) => f.value),
      ...info.fields.map((f) => `${f.label} ${values[f.key] ?? f.value}`.trim()),
      info.time ?? "",
    ];
    g.fillText(text.filter(Boolean).join("   "), size, h + barHeight * 0.68);
  }
  const blob = await new Promise<Blob | null>((done) => out.toBlob(done, "image/png"));
  if (!blob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${name.replace(/\.ome\.tiff?$|\.tiff?$|\.png$|\.jpe?g$/i, "")}_view.png`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function ExportMenu({
  save,
  close,
}: {
  save: (o: { scalebar: boolean; bar: boolean; full: boolean }) => Promise<void>;
  close: () => void;
}) {
  const [scalebar, setScalebar] = useState(true);
  const [bar, setBar] = useState(true);
  const [full, setFull] = useState(false);
  const [saving, setSaving] = useState(false);
  return (
    <div
      className="absolute top-8 right-2 z-20 flex w-[240px] flex-col gap-[7px] rounded-lg border bg-card px-3 pt-2.5 pb-3 text-[12px] shadow-[0_6px_24px_rgba(0,0,0,0.45)]"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <span className="text-[11px] font-medium tracking-[1px] text-muted-foreground">EXPORT VIEW</span>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          className="m-0 accent-[var(--primary)]"
          checked={scalebar}
          onChange={(e) => setScalebar(e.target.checked)}
        />
        Scalebar
      </label>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          className="m-0 accent-[var(--primary)]"
          checked={bar}
          onChange={(e) => setBar(e.target.checked)}
        />
        Metadata bar
      </label>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="radio"
          name="export-size"
          className="m-0 accent-[var(--primary)]"
          checked={!full}
          onChange={() => setFull(false)}
        />
        As shown on screen
      </label>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="radio"
          name="export-size"
          className="m-0 accent-[var(--primary)]"
          checked={full}
          onChange={() => setFull(true)}
        />
        Full resolution, whole image
      </label>
      <button
        className="mt-1 cursor-pointer rounded-md border-0 bg-primary p-1.5 text-[12px] text-white hover:bg-primary-hover disabled:opacity-50"
        disabled={saving}
        onClick={async () => {
          setSaving(true);
          try {
            await save({ scalebar, bar, full });
            close();
          } finally {
            setSaving(false);
          }
        }}
      >
        {saving ? "Saving…" : "Save PNG"}
      </button>
    </div>
  );
}

/** Where the viewer's keys go: true when the image used the key. */
export type ImageKeys = MutableRefObject<((e: KeyboardEvent) => boolean) | null>;

/**
 * An image drawn in the browser from its raw planes: each channel (or the detector)
 * a layer with its own colour, opacity, gamma and contrast, the stack's planes and
 * projections, zoom and pan, the ruler, crosshair and scalebar, the value under the
 * cursor, and export. *marks* is drawn over the image while marking, which keeps the
 * whole stack's max projection on screen.
 */
export function LayerImage({
  path,
  read,
  marking,
  marks,
  keys,
  onNoWebGL,
  onVisible,
}: {
  path: string;
  read: FileRead;
  marking: boolean;
  marks?: ReactNode;
  keys: ImageKeys;
  onNoWebGL: () => void;
  /** The names of the channels on screen, when they change. */
  onVisible?: (names: string[]) => void;
}) {
  const summary = read.summary ?? {};
  const fm = summary.fm;
  const shape = (summary.shape as number[] | undefined) ?? [1, 1];
  const full: [number, number] = [shape[0], shape[1]];
  const pixelSize = summary.pixel_size as number | null | undefined;
  const info = read.info;
  const count = fm?.z_planes ?? 1;
  const stack = count > 1;
  const { sharp, quick } = factors(full, count);
  const detector = info?.header.find((f) => f.key === "detector")?.value;
  const channels = fm
    ? fm.channels.map((c) => ({ name: c.name, color: c.color, key: c.name }))
    : [{ name: detector ?? info?.kind ?? "Image", color: "gray", key: `${info?.kind ?? "Image"}:${detector ?? ""}` }];

  const sharpest = fm?.channels.find((c) => c.sharpest_plane != null)?.sharpest_plane ?? 0;
  const [shown, setShown] = useState<StackShown>({
    z: sharpest,
    projection: stack ? "max" : null,
    first: 0,
    last: count - 1,
  });
  const drawn: StackShown = marking && stack ? { ...shown, projection: "max", first: 0, last: count - 1 } : shown;
  const [settled, setSettled] = useState(drawn);
  useEffect(() => {
    const t = setTimeout(() => setSettled(drawn), SETTLE_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawn.z, drawn.projection, drawn.first, drawn.last]);

  const all = usePlanes(path, quick != null ? { factor: quick, z: "all" } : null);
  const wanted = askFor(drawn, sharp);
  const sharpAsk = askFor(settled, sharp);
  const fine = usePlanes(path, sharpAsk);
  const last = useRef<{ planes: Values[]; factor: number } | null>(null);

  // What to draw: the sharp read when it is of what is shown, else the small copy
  // of it at once, else the last drawn, dimmed, while it is read.
  let current: { planes: Values[]; factor: number; width: number } | null = null;
  let dim = false;
  const width = (p: Planes) => p.shape[3];
  if (fine.data && sameAsk(sharpAsk, wanted) && !fine.isFetching) {
    current = { planes: channels.map((_, c) => planeOf(fine.data, c)), factor: sharp, width: width(fine.data) };
  } else if (all.data) {
    const p = all.data;
    current = {
      planes: channels.map((_, c) => (drawn.projection ? projectionOf(p, c, drawn) : planeOf(p, c, drawn.z))),
      factor: quick!,
      width: width(p),
    };
  } else if (last.current) {
    dim = true;
  }
  if (current) last.current = current;
  const showing = current ?? (last.current as typeof current);

  // Auto contrast while stepping planes holds each channel's projection contrast, as Qt does.
  const held = useMemo(() => {
    const p = all.data;
    if (!p) return null;
    return channels.map((_, c) =>
      autoOf(projectionOf(p, c, { z: 0, projection: "max", first: 0, last: count - 1 }), p.shape[3]),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all.data]);

  const [looks, setLooks] = useState<Look[]>(() => channels.map((c) => storedLook(c.key, c.color)));
  const [visible, setVisible] = useState<boolean[]>(() => channels.map(() => true));
  const [selected, setSelected] = useState(0);
  const [panel, setPanel] = useState(stack);
  const [tools, setTools] = useState<Tools>({ ruler: false, crosshair: false, scalebar: true });
  const [view, setView] = useState<View>(() => fitView(full[1], full[0]));
  const [exporting, setExporting] = useState(false);
  const [projMenu, setProjMenu] = useState(false);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const snapshot = useRef<(() => Snapshot) | null>(null);

  const climOf = (c: number): [number, number] | null => {
    if (looks[c].clim) return looks[c].clim;
    if (stack && !drawn.projection && held) return held[c];
    return showing ? autoOf(showing.planes[c], showing.width) : null;
  };
  const setLook = (c: number, look: Look) => {
    setLooks(looks.map((l, k) => (k === c ? look : l)));
    storeLook(channels[c].key, look, channels[c].color);
  };
  const drawLayers = (planes: Values[], w: number): DrawLayer[] =>
    channels.flatMap((_, c) => {
      const clim = climOf(c);
      if (!visible[c] || !clim) return [];
      const look = looks[c];
      return [
        {
          values: planes[c],
          width: w,
          height: planes[c].length / w,
          clim,
          gamma: look.gamma,
          tint: tint(look.color),
          opacity: look.opacity,
        },
      ];
    });
  const layers = showing ? drawLayers(showing.planes, showing.width) : [];

  // Z as the server's bar says it (fibsem.imaging.export.z_value), for what is drawn
  const step = fm?.z_step ? ` × ${fmt.length(fm.z_step)}` : "";
  const whole = drawn.first === 0 && drawn.last === count - 1;
  const zValue = !stack
    ? undefined
    : !drawn.projection
      ? `${drawn.z + 1} of ${count}${step}`
      : drawn.projection === "max" && whole
        ? `MIP · ${count}${step}`
        : `${PROJECTIONS.find((p) => p[0] === drawn.projection)![1]} · ${
            whole ? count : `${drawn.first + 1}–${drawn.last + 1} of ${count}`
          }${step}`;
  const values: Record<string, string> = zValue ? { z: zValue } : {};

  const readout =
    cursor && showing ? (
      <>
        {pixelSize && (
          <span>
            {((cursor[0] * pixelSize) / 1e-6).toFixed(2)}, {((cursor[1] * pixelSize) / 1e-6).toFixed(2)} µm
          </span>
        )}
        {channels.map((ch, c) => {
          if (!visible[c]) return null;
          const f = showing.factor;
          const v = showing.planes[c][Math.floor(cursor[1] / f) * showing.width + Math.floor(cursor[0] / f)];
          return (
            <span key={ch.name} className="inline-flex items-center gap-1" title={ch.name}>
              <span className="inline-block size-2 rounded-[2px]" style={{ background: cssColor(looks[c].color) }} />
              {v == null ? "–" : Number.isInteger(v) ? v : v.toFixed(1)}
            </span>
          );
        })}
      </>
    ) : null;

  const save = async (o: { scalebar: boolean; bar: boolean; full: boolean }) => {
    const whole = async () => {
      const planes = await readPlanes(path, askFor(drawn, 1));
      const canvas = document.createElement("canvas");
      canvas.width = planes.shape[3];
      canvas.height = planes.shape[2];
      const renderer = LayerRenderer.create(canvas);
      if (!renderer) throw new Error("WebGL2 is not available");
      const w = planes.shape[3];
      renderer.draw(
        drawLayers(
          channels.map((_, c) => planeOf(planes, c)),
          w,
        ),
        { x: 0, y: 0, scale: 1 },
        canvas.width,
        canvas.height,
        true,
      );
      return canvas;
    };
    await exportPng({
      name: path.split("/").pop() ?? "image",
      snapshot: snapshot.current?.() ?? null,
      full: o.full ? whole : null,
      draw: full,
      pixelSize,
      info,
      values,
      scalebar: o.scalebar,
      bar: o.bar,
    });
  };

  const toggleTool = (key: keyof Tools) => setTools({ ...tools, [key]: !tools[key] });
  const stepPlane = (by: number) =>
    setShown({
      ...shown,
      z: Math.max(0, Math.min(count - 1, shown.projection ? shown.z : shown.z + by)),
      projection: null,
    });
  keys.current = (e) => {
    const key = e.key.toLowerCase();
    if (key === "+" || key === "=") setView(zoomed(view, 1.5));
    else if (key === "-") setView(zoomed(view, 1 / 1.5));
    else if (key === "0") setView(fitView(full[1], full[0]));
    else if (key === "l") setPanel(!panel);
    else if (key === "r") toggleTool("ruler");
    else if (key === "c") toggleTool("crosshair");
    else if (key === "s") toggleTool("scalebar");
    else if (key === "e") setExporting(!exporting);
    else if (key === "p" && stack && !marking) setProjMenu(!projMenu);
    else if (e.key === "ArrowUp" && stack && !marking) stepPlane(1);
    else if (e.key === "ArrowDown" && stack && !marking) stepPlane(-1);
    else return false;
    return true;
  };

  const panelLayers: PanelLayer[] = channels.map((ch, c) => ({
    name: ch.name,
    color: ch.color,
    visible: visible[c],
    look: looks[c],
    clim: climOf(c) ?? [0, 1],
    held: stack && !drawn.projection && !!held,
    values: showing?.planes[c] ?? null,
  }));
  const error = fine.error ?? all.error;

  return (
    <div className="flex min-w-0 flex-col">
      <div className="relative h-[min(calc(84vh-190px),640px)] min-h-[300px]">
        <LayerView
          layers={layers}
          factor={showing?.factor ?? 1}
          full={full}
          pixelSize={pixelSize}
          tools={marking ? { ...tools, ruler: false } : tools}
          view={view}
          setView={setView}
          pannable={!marking}
          dim={dim}
          onCursor={setCursor}
          onRenderer={(r) => r || onNoWebGL()}
          snapshot={snapshot}
        >
          {marking ? marks : null}
        </LayerView>
        {(!showing || dim) && (
          <span className="absolute top-2 left-2 z-10 bg-black/70 px-2 py-0.5 text-[12px] text-white">
            {error ? error.message : "Reading…"}
          </span>
        )}
        <div className="absolute top-1 right-1 z-10 flex gap-0.5">
          <Tool label="Zoom out" keyName="−" onClick={() => setView(zoomed(view, 1 / 1.5))}>
            <ZoomOut size={14} />
          </Tool>
          <Tool label="Zoom in" keyName="+" onClick={() => setView(zoomed(view, 1.5))}>
            <ZoomIn size={14} />
          </Tool>
          <Tool label={fm ? "FM channels" : "Layers"} keyName="L" pressed={panel} onClick={() => setPanel(!panel)}>
            <Layers size={14} />
          </Tool>
          <Tool label="Measure" keyName="R" pressed={tools.ruler} onClick={() => toggleTool("ruler")}>
            <Ruler size={14} />
          </Tool>
          <Tool label="Crosshair" keyName="C" pressed={tools.crosshair} onClick={() => toggleTool("crosshair")}>
            <Crosshair size={14} />
          </Tool>
          <Tool label="Scalebar" keyName="S" pressed={tools.scalebar} onClick={() => toggleTool("scalebar")}>
            <Scaling size={14} />
          </Tool>
          <Tool label="Export view" keyName="E" pressed={exporting} onClick={() => setExporting(!exporting)}>
            <Download size={14} />
          </Tool>
          <Tool label="Reset view" keyName="0" onClick={() => setView(fitView(full[1], full[0]))}>
            <Maximize size={14} />
          </Tool>
        </div>
        {panel && (
          <LayersPanel
            title={fm ? "FM CHANNELS" : "LAYERS"}
            layers={panelLayers}
            selected={selected}
            select={setSelected}
            toggle={(c) => {
              const next = visible.map((v, k) => (k === c ? !v : v));
              setVisible(next);
              onVisible?.(channels.filter((_, k) => next[k]).map((ch) => ch.name));
            }}
            setLook={setLook}
            close={() => setPanel(false)}
          />
        )}
        {exporting && <ExportMenu save={save} close={() => setExporting(false)} />}
      </div>
      <ImageInfoBar info={info} values={values} readout={readout} />
      {stack && !marking && (
        <ZRow
          count={count}
          shown={shown}
          set={setShown}
          sharpness={fm?.channels[selected]?.sharpness ?? fm?.channels.find((c) => c.sharpness)?.sharpness ?? null}
          menu={projMenu}
          setMenu={setProjMenu}
        />
      )}
    </div>
  );
}
