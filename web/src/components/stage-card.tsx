import { Minus, Plus } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ImageInfoBar } from "@/components/image-info";
import type { ImageField, ImageInfo, MapPlace, StageDrawing, StageMapPlaces } from "@/lib/types";

// The Qt chamber view's palette (fibsem/theme/tokens.py, via theme.css).
const C = {
  bg: "var(--fibsem-canvas-bg)",
  sem: "var(--fibsem-sem-beam-muted-colour)",
  fib: "var(--fibsem-fib-beam-muted-colour)",
  plate: "var(--fibsem-neutral-650)",
  shuttle: "var(--fibsem-primary-accent)",
  light: "var(--fibsem-neutral-300)",
  barrel: "#0d0f12",
  edge: "var(--fibsem-neutral-500)",
  muted: "var(--fibsem-text-muted-color)",
  holder: "var(--fibsem-neutral-550)",
  slot: "var(--fibsem-slot-colour)",
  limits: "var(--fibsem-stage-limits-colour)",
  boundary: "var(--fibsem-grid-boundary-colour)",
  saved: "var(--fibsem-saved-position-colour)",
  here: "var(--fibsem-selected-position-colour)",
  stage: "var(--fibsem-current-position-colour)",
};

const ZOOMS = ["Travel", "Holder", "Grid"] as const;
type Zoom = 0 | 1 | 2;

/** The bar under the drawing, as the Qt stage bar reads it: `SEM · GRID-01`, the axes
 * in millimetres to the micrometre and degrees to a tenth, and the milling angle. */
function stageInfo(position: Record<string, unknown> | null, stage: StageDrawing | null): ImageInfo {
  const header: ImageField[] = [];
  if (stage?.orientation)
    header.push({ key: "orientation", label: "", title: "Stage orientation", value: stage.orientation });
  if (stage?.grid) header.push({ key: "grid", label: "", title: "Grid", value: stage.grid });
  const fields: ImageField[] = [];
  for (const [axis, scale, unit, digits] of [
    ["x", 1e3, " mm", 3],
    ["y", 1e3, " mm", 3],
    ["z", 1e3, " mm", 3],
    ["r", 180 / Math.PI, "°", 1],
    ["t", 180 / Math.PI, "°", 1],
  ] as const) {
    const v = position?.[axis];
    if (typeof v !== "number") continue;
    const label = axis.toUpperCase();
    fields.push({ key: axis, label, title: `Stage ${label}`, value: `${(v * scale).toFixed(digits)}${unit}` });
  }
  if (stage?.milling_angle != null)
    fields.push({ key: "ma", label: "MA", title: "Milling angle", value: `${stage.milling_angle.toFixed(1)}°` });
  return { kind: "Stage", header, fields, time: null, scalebar: null };
}

const rad = (deg: number) => (deg * Math.PI) / 180;

/** The stage, shuttle and both columns from the side (the Qt `StageDiagram`, as the
 * chamber view draws it): the stage at its tilt, the shuttle's pre-tilt as a wedge,
 * mirrored at a half turn, and an FM objective under the stage where there is one. */
function SideView({ stage, w, h }: { stage: StageDrawing; w: number; h: number }) {
  const cx = w * 0.56;
  const cy = stage.objective_below ? Math.min(h * 0.56, h - 110) : h * 0.66;
  const tilt = stage.stage_tilt ?? stage.pre_tilt;
  const ramp = stage.mirrored ? -stage.pre_tilt : stage.pre_tilt;
  const r = rad(ramp);
  const half = 26;
  const low = [-half * Math.cos(r), half * Math.sin(r)];
  const high = [half * Math.cos(r), -half * Math.sin(r)];
  const front = low[0] - 15;
  const back = high[0] + 15;
  const plate = 8 + half * Math.abs(Math.sin(r));
  const body = [[front, plate], [front, low[1]], low, high, [back, high[1]], [back, plate]];

  const beam = (deg: number, colour: string, name: string) => {
    const reach = Math.min(118, Math.max(24, cy - 24));
    const x2 = cx + Math.sin(rad(deg)) * reach;
    const y2 = cy - Math.cos(rad(deg)) * reach;
    return (
      <g key={name}>
        <line x1={x2} y1={y2} x2={cx} y2={cy} stroke={colour} strokeWidth={2} />
        <text x={x2} y={y2 - 6} fill={colour} fontSize={11} textAnchor="middle">
          {name}
        </text>
      </g>
    );
  };

  // The wedge, at the foot of the ramp, with its label kept upright outside the
  // stage's rotation.
  let wedge = null;
  if (stage.pre_tilt) {
    const radius = 34;
    const foot = ramp >= 0 ? low : high;
    const start = ramp >= 0 ? 0 : 180;
    const reach = ramp >= 0 ? radius + 12 : -(radius + 12);
    const at = (deg: number, d: number) => [foot[0] + Math.cos(rad(deg)) * d, foot[1] - Math.sin(rad(deg)) * d];
    const [ax, ay] = at(start, radius);
    const [bx, by] = at(start + ramp, radius);
    const [mx, my] = at(start + ramp / 2, radius + 14);
    const t = rad(tilt);
    wedge = {
      lines: (
        <g stroke={C.light} strokeWidth={1} strokeDasharray="1 2" fill="none">
          <line x1={foot[0]} y1={foot[1]} x2={foot[0] + reach} y2={foot[1]} />
          <path d={`M ${ax} ${ay} A ${radius} ${radius} 0 0 ${ramp >= 0 ? 0 : 1} ${bx} ${by}`} />
        </g>
      ),
      label: [cx + mx * Math.cos(t) - my * Math.sin(t), cy + mx * Math.sin(t) + my * Math.cos(t)],
    };
  }

  let objective = null;
  if (stage.objective_below) {
    const gap = stage.objective_inserted ? 18 : 46;
    const length = 30;
    objective = (
      <g opacity={stage.objective_inserted ? 1 : 0.47}>
        <g transform={`translate(${cx} ${cy}) rotate(180)`}>
          <polygon
            points={`-7,${-gap} 7,${-gap} 15,${-gap - length} -15,${-gap - length}`}
            fill={C.barrel}
            stroke={C.edge}
          />
          <rect x={-7} y={-gap - 3} width={14} height={3} fill="#dfe5ec" />
          <line x1={0} y1={0} x2={0} y2={-gap} stroke={C.edge} strokeDasharray="1 2" />
        </g>
        <text x={cx} y={cy + gap + length + 24} fill={C.muted} fontSize={11} textAnchor="middle">
          FM
        </text>
      </g>
    );
  }

  return (
    <g>
      {beam(0, C.sem, "SEM")}
      {beam(stage.column_tilt, C.fib, `FIB ${stage.column_tilt}°`)}
      {objective}
      <g transform={`translate(${cx} ${cy}) rotate(${tilt})`}>
        <line x1={-95} y1={plate} x2={95} y2={plate} stroke={C.plate} strokeWidth={2} />
        <line x1={-95} y1={plate + 8} x2={95} y2={plate + 8} stroke={C.plate} strokeDasharray="4 3" />
        <polygon points={body.map((p) => p.join(",")).join(" ")} fill={C.shuttle} />
        <rect x={-13} y={-4} width={26} height={4} rx={1} fill={C.light} transform={`rotate(${-ramp})`} />
        {wedge?.lines}
      </g>
      {wedge && (
        <text
          x={wedge.label[0]}
          y={wedge.label[1]}
          fill={C.light}
          fontSize={11}
          textAnchor="middle"
          dominantBaseline="middle"
        >
          {stage.pre_tilt}°
        </text>
      )}
    </g>
  );
}

// The Qt map's zoom steps (fibsem/ui/widgets/canvas/stage_map.py), in metres.
const MARGIN = 0.12;
const HOLDER_MIN_SPAN = 8e-3;
const HOLDER_MARGIN = 0.3;
const TRAVEL_MAX_SPAN = 100e-3;
const TRAVEL_MIN_SPAN = 20e-3;

/** The centre and pixels per metre a zoom step shows a *w* by *h* view at. */
function frame(places: StageMapPlaces, zoom: Zoom, w: number, h: number): [number, number, number] {
  const stage = places.stage ?? [0, 0];
  const radius = places.boundary[0] ?? 1e-3;
  if (zoom === 2) return [stage[0], stage[1], Math.min(w, h) / (2.6 * radius)];
  if (zoom === 1) {
    if (!places.holder) return [stage[0], stage[1], Math.min(w, h) / HOLDER_MIN_SPAN];
    const [hx, hy, side] = places.holder;
    const span = Math.max(HOLDER_MIN_SPAN, side * (1 + HOLDER_MARGIN));
    const on = Math.abs(stage[0] - hx) <= side / 2 && Math.abs(stage[1] - hy) <= side / 2;
    return [on ? hx : stage[0], on ? hy : stage[1], Math.min(w, h) / span];
  }
  // Travel: every grid, every station and the stage, within the span limits, slid
  // towards the stage when capped.
  const points: number[][] = [stage];
  for (const s of places.slots) points.push([s.x - radius, s.y - radius], [s.x + radius, s.y + radius]);
  for (const s of places.stations) points.push([s.x, s.y]);
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const span = (lo: number, hi: number) =>
    Math.min(Math.max((hi - lo) * (1 + MARGIN), TRAVEL_MIN_SPAN), TRAVEL_MAX_SPAN);
  let scale = Math.min(w / span(Math.min(...xs), Math.max(...xs)), h / span(Math.min(...ys), Math.max(...ys)));
  scale = Math.max(scale, Math.max(w, h) / TRAVEL_MAX_SPAN);
  const slide = (lo: number, hi: number, at: number, shown: number) => {
    const reach = (shown / 2) * (1 - MARGIN);
    return Math.min(Math.max((lo + hi) / 2, at - reach), at + reach);
  };
  return [
    slide(Math.min(...xs), Math.max(...xs), stage[0], w / scale),
    slide(Math.min(...ys), Math.max(...ys), stage[1], h / scale),
    scale,
  ];
}

/** The stage from above (the Qt `StageMap`): the travel limits, the holder and its
 * slots with each grid's boundary, other devices' stations, the lamellae at grid zoom,
 * and the stage. *detailed* is the large map: names and crosshairs. */
function MapView({
  places,
  zoom,
  w,
  h,
  detailed,
}: {
  places: StageMapPlaces;
  zoom: Zoom;
  w: number;
  h: number;
  detailed: boolean;
}) {
  const [mx, my, s] = frame(places, zoom, w, h);
  const X = (x: number) => w / 2 + (x - mx) * s;
  const Y = (y: number) => h / 2 + (y - my) * s;
  const [rx, ry] = places.boundary.length ? places.boundary : [1e-3, 1e-3];
  const cross = (p: MapPlace | number[], arm: number, colour: string, width: number, key?: string) => {
    const [x, y] = Array.isArray(p) ? [X(p[0]), Y(p[1])] : [X(p.x), Y(p.y)];
    return (
      <g key={key} stroke={colour} strokeWidth={width}>
        <line x1={x - arm} y1={y} x2={x + arm} y2={y} />
        <line x1={x} y1={y - arm} x2={x} y2={y + arm} />
      </g>
    );
  };
  const station = detailed ? 6 : 3;
  return (
    <g>
      {places.limits && (
        <rect
          x={X(places.limits[0])}
          y={Y(places.limits[1])}
          width={(places.limits[2] - places.limits[0]) * s}
          height={(places.limits[3] - places.limits[1]) * s}
          fill="none"
          stroke={C.limits}
          strokeOpacity={0.5}
          strokeDasharray="4 3"
        />
      )}
      {places.stations.map((p) => (
        <g key={`station-${p.name}`}>
          <polygon
            points={`${X(p.x)},${Y(p.y) - station} ${X(p.x) + station},${Y(p.y)} ${X(p.x)},${Y(p.y) + station} ${X(p.x) - station},${Y(p.y)}`}
            fill="none"
            stroke={C.slot}
          />
          {detailed && zoom !== 2 && (
            <text x={X(p.x) + station + 4} y={Y(p.y) + 4} fill={C.holder} fontSize={11}>
              {p.name}
            </text>
          )}
        </g>
      ))}
      {zoom !== 2 && places.holder && (
        <rect
          x={X(places.holder[0] - places.holder[2] / 2)}
          y={Y(places.holder[1] - places.holder[2] / 2)}
          width={places.holder[2] * s}
          height={places.holder[2] * s}
          rx={places.holder[2] * s * 0.08}
          fill={C.holder}
          fillOpacity={0.08}
          stroke={C.holder}
          strokeOpacity={0.7}
        />
      )}
      {places.slots.map((p) => (
        <g key={`slot-${p.name}`}>
          <ellipse
            cx={X(p.x)}
            cy={Y(p.y)}
            rx={Math.max(rx * s, 1.5)}
            ry={Math.max(ry * s, 1.5)}
            fill="none"
            stroke={C.boundary}
            strokeOpacity={0.55}
            strokeDasharray="4 3"
          />
          {cross(p, detailed ? 4 : 2, C.slot, 1)}
          {detailed && zoom === 1 && p.name && (
            <text x={X(p.x) - rx * s} y={Y(p.y) - ry * s - 6} fill={C.holder} fontSize={11}>
              {p.name}
            </text>
          )}
        </g>
      ))}
      {zoom === 2 &&
        places.positions.map((p, i) =>
          detailed ? (
            <g key={`pos-${i}`} opacity={0.65}>
              {cross(p, 3.5, C.saved, 1)}
            </g>
          ) : (
            <circle key={`pos-${i}`} cx={X(p.x)} cy={Y(p.y)} r={1} fill={C.saved} />
          ),
        )}
      {places.stage && cross(places.stage, detailed ? 7 : 4, C.stage, detailed ? 1.5 : 1)}
      {zoom === 2 &&
        places.positions
          .filter((p) => places.here.includes(p.name))
          .map((p, i) => (
            <circle
              key={`here-${i}`}
              cx={X(p.x)}
              cy={Y(p.y)}
              r={detailed ? 6 : 3}
              fill="none"
              stroke={C.here}
              strokeOpacity={0.9}
              strokeWidth={1.2}
            />
          ))}
    </g>
  );
}

/** The size of the element *ref* holds, kept current. */
function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const watch = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    watch.observe(el);
    return () => watch.disconnect();
  }, []);
  return [ref, size] as const;
}

// The inset's share of the width, its height for that width, and its margin; the
// smallest the side view is drawn at before it is shrunk into the inset.
const INSET_WIDTH = 0.3;
const INSET_ASPECT = 0.72;
const INSET_MARGIN = 8;
const SIDE_MIN = [300, 216];

/** The stage at one moment, as the Qt quad view's fourth cell draws it: the side view
 * and the map from above, one large with the other inset (click the inset to swap),
 * and the stage bar under them. */
export function StageCard({
  position,
  stage,
}: {
  position: Record<string, unknown> | null;
  stage: StageDrawing | null;
}) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const [main, setMain] = useState<"side" | "map">("side");
  const [zoom, setZoom] = useState<Zoom>(0);
  const places = stage?.map ?? null;
  const showMap = main === "map" && places != null;
  const iw = Math.max(Math.round(w * INSET_WIDTH), 60);
  const ih = Math.round(iw * INSET_ASPECT);
  const side = (sw: number, sh: number) => stage && <SideView stage={stage} w={sw} h={sh} />;
  return (
    <div className="flex min-w-0 flex-col overflow-hidden rounded-sm">
      <div ref={ref} className="relative aspect-[3/2] w-full" style={{ background: C.bg }}>
        {!stage ? (
          <div className="flex h-full items-center justify-center text-[13px] text-muted-foreground">
            No stage position yet
          </div>
        ) : (
          w > 0 && (
            <>
              <svg
                width={w}
                height={h}
                className="absolute inset-0 block"
                aria-label={showMap ? "Stage map" : "Side view"}
              >
                {showMap ? (
                  <MapView places={places} zoom={zoom} w={w} h={h} detailed />
                ) : (
                  <svg viewBox={`0 0 ${Math.max(w, SIDE_MIN[0])} ${Math.max(h, SIDE_MIN[1])}`} width={w} height={h}>
                    {side(Math.max(w, SIDE_MIN[0]), Math.max(h, SIDE_MIN[1]))}
                  </svg>
                )}
              </svg>
              {places && (
                <button
                  onClick={() => setMain(main === "side" ? "map" : "side")}
                  title={main === "side" ? "Show the map from above" : "Show the side view"}
                  className="absolute cursor-pointer overflow-hidden rounded-sm border border-border p-0"
                  style={{ left: INSET_MARGIN, top: INSET_MARGIN, width: iw, height: ih, background: C.bg }}
                >
                  <svg width={iw} height={ih} className="block">
                    {showMap ? (
                      <svg viewBox={`0 0 ${SIDE_MIN[0]} ${SIDE_MIN[1]}`} width={iw} height={ih}>
                        {side(SIDE_MIN[0], SIDE_MIN[1])}
                      </svg>
                    ) : (
                      <MapView places={places} zoom={zoom} w={iw} h={ih} detailed={false} />
                    )}
                  </svg>
                </button>
              )}
              {showMap && (
                <div className="absolute flex gap-1" style={{ right: INSET_MARGIN, bottom: INSET_MARGIN }}>
                  {(
                    [
                      [Minus, -1, "out"],
                      [Plus, 1, "in"],
                    ] as const
                  ).map(([Icon, step, way]) => {
                    const next = zoom + step;
                    const ok = next >= 0 && next <= 2;
                    return (
                      <button
                        key={way}
                        disabled={!ok}
                        onClick={() => ok && setZoom(next as Zoom)}
                        title={ok ? `Zoom ${way} to ${ZOOMS[next].toLowerCase()}` : `Zoomed ${way}`}
                        className="flex size-[26px] cursor-pointer items-center justify-center rounded border border-border bg-panel p-0 text-foreground disabled:cursor-default disabled:opacity-40"
                      >
                        <Icon size={13} />
                      </button>
                    );
                  })}
                </div>
              )}
            </>
          )
        )}
      </div>
      <ImageInfoBar info={stageInfo(position, stage)} />
    </div>
  );
}
