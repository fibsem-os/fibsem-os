import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { type Mark, lineLength } from "@/lib/marks";

// The label box: dark and see-through, a thin border in the mark's colour, no outline on the text.
const LABEL_FILL = "rgba(17, 19, 24, 0.62)";

/** Image pixels per screen pixel of *svg*, so marks keep one size on screen at any image size. */
function useScale(width: number) {
  const ref = useRef<SVGSVGElement>(null);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const svg = ref.current;
    if (!svg) return;
    const observer = new ResizeObserver(() => setShown(svg.getBoundingClientRect().width));
    observer.observe(svg);
    return () => observer.disconnect();
  }, []);
  return { ref, k: shown > 0 ? width / shown : 1 };
}

function Label({
  x,
  y,
  text,
  color,
  k,
  width,
}: {
  x: number;
  y: number;
  text: string;
  color: string;
  k: number;
  width: number;
}) {
  const size = 12 * k;
  const pad = 4 * k;
  const w = text.length * size * 0.6 + pad * 2;
  const h = size + pad * 1.6;
  // kept inside the picture
  const left = Math.max(0, Math.min(x, width - w));
  const top = Math.max(0, y - h);
  return (
    <g pointerEvents="none">
      <rect
        x={left}
        y={top}
        width={w}
        height={h}
        rx={2 * k}
        fill={LABEL_FILL}
        stroke={color}
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <text x={left + pad} y={top + h / 2} dominantBaseline="central" fill={color} fontSize={size} fontWeight={500}>
        {text}
      </text>
    </g>
  );
}

/** Where a mark's label sits: beside a point, over a box or circle, at a line's middle. */
function labelAt(m: Mark, k: number): [number, number] {
  if (m.kind === "point") return [m.x + 12 * k, m.y - 6 * k];
  if (m.kind === "area") return [m.x, m.y - 4 * k];
  if (m.kind === "circle") return [m.x + m.radius * 0.7, m.y - m.radius * 0.7];
  return [(m.x1 + m.x2) / 2 + 6 * k, (m.y1 + m.y2) / 2 - 6 * k];
}

function Shape({ m, color, k, thick }: { m: Mark; color: string; k: number; thick: boolean }) {
  const line = {
    stroke: color,
    strokeWidth: thick ? 2.5 : 1.5,
    fill: "none",
    vectorEffect: "non-scaling-stroke" as const,
  };
  if (m.kind === "point") {
    const r = 9 * k;
    return (
      <g>
        <path d={`M${m.x - r} ${m.y}H${m.x + r}M${m.x} ${m.y - r}V${m.y + r}`} {...line} />
        {m.shape === "target" && <circle cx={m.x} cy={m.y} r={r * 0.6} {...line} />}
      </g>
    );
  }
  if (m.kind === "area") return <rect x={m.x} y={m.y} width={m.width} height={m.height} {...line} />;
  if (m.kind === "circle") return <circle cx={m.x} cy={m.y} r={m.radius} {...line} />;
  const head = 10 * k;
  const angle = Math.atan2(m.y2 - m.y1, m.x2 - m.x1);
  const wing = (side: number) =>
    `${m.x2 - head * Math.cos(angle + side * 0.45)} ${m.y2 - head * Math.sin(angle + side * 0.45)}`;
  const tick = (x: number, y: number) => {
    const t = 6 * k;
    const nx = -Math.sin(angle) * t;
    const ny = Math.cos(angle) * t;
    return `M${x - nx} ${y - ny}L${x + nx} ${y + ny}`;
  };
  return (
    <g>
      <path d={`M${m.x1} ${m.y1}L${m.x2} ${m.y2}`} {...line} />
      {m.arrow && <path d={`M${wing(1)}L${m.x2} ${m.y2}L${wing(-1)}`} {...line} />}
      {m.measure && <path d={tick(m.x1, m.y1) + tick(m.x2, m.y2)} {...line} />}
    </g>
  );
}

/** Generous invisible target over a mark, so it can be picked up and moved. */
function HitArea({ m, k }: { m: Mark; k: number }) {
  const hit = { stroke: "transparent", strokeWidth: 14 * k, fill: "none" };
  if (m.kind === "point") return <circle cx={m.x} cy={m.y} r={10 * k} fill="transparent" />;
  if (m.kind === "area") return <rect x={m.x} y={m.y} width={m.width} height={m.height} {...hit} />;
  if (m.kind === "circle") return <circle cx={m.x} cy={m.y} r={m.radius} {...hit} />;
  return <path d={`M${m.x1} ${m.y1}L${m.x2} ${m.y2}`} {...hit} />;
}

export function markLabel(m: Mark, pixelSize?: number | null): string | undefined {
  if (m.kind === "line" && m.measure) return m.label || lineLength(m, pixelSize);
  return m.label || undefined;
}

/** *label* with the plane *z* a mark is on, counted from 1 as the viewer counts. */
export function planeLabel(label: string | undefined, z: number | null | undefined): string | undefined {
  if (z == null) return label;
  return label ? `${label} · plane ${z + 1}` : `plane ${z + 1}`;
}

/** Marks drawn over an image *width* x *height* source pixels, the SVG laid over it
 * exactly. With *onMarkDown*, each mark can be picked up; *hiddenLabel* is the one
 * whose label is being typed. *plane* is the stack's plane on screen: marks on
 * other planes are left out; null, the projection, shows each mark's plane. */
export function MarkLayer({
  marks,
  width,
  height,
  color,
  pixelSize,
  selected = null,
  hiddenLabel = null,
  plane,
  onMarkDown,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  className,
}: {
  marks: Mark[];
  width: number;
  height: number;
  color: string;
  pixelSize?: number | null;
  selected?: number | null;
  hiddenLabel?: number | null;
  plane?: number | null;
  onMarkDown?: (index: number, e: ReactPointerEvent<SVGGElement>) => void;
  onPointerDown?: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove?: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerUp?: (e: ReactPointerEvent<SVGSVGElement>) => void;
  className?: string;
}) {
  const { ref, k } = useScale(width);
  return (
    <svg
      ref={ref}
      className={className ?? "pointer-events-none absolute inset-0 h-full w-full"}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {marks.map((m, i) => {
        if (plane != null && m.z != null && m.z !== plane) return null;
        const text = i === hiddenLabel ? undefined : planeLabel(markLabel(m, pixelSize), plane === null ? m.z : null);
        const [lx, ly] = labelAt(m, k);
        return (
          <g
            key={i}
            className={onMarkDown ? "cursor-move" : undefined}
            onPointerDown={onMarkDown && ((e) => onMarkDown(i, e))}
          >
            <Shape m={m} color={color} k={k} thick={i === selected} />
            {onMarkDown && <HitArea m={m} k={k} />}
            {text && <Label x={lx} y={ly} text={text} color={color} k={k} width={width} />}
          </g>
        );
      })}
    </svg>
  );
}

/** Where on the image (source pixels) a pointer event is, from the layer's box. */
export function imagePoint(
  e: { clientX: number; clientY: number; currentTarget: Element },
  width: number,
  height: number,
) {
  const box = (e.currentTarget.closest("svg") ?? e.currentTarget).getBoundingClientRect();
  const x = ((e.clientX - box.left) / box.width) * width;
  const y = ((e.clientY - box.top) / box.height) * height;
  return { x: Math.max(0, Math.min(width, x)), y: Math.max(0, Math.min(height, y)) };
}

/** A mark's label anchor as a fraction of the image, for an HTML field over it. */
export function labelFraction(m: Mark, width: number, height: number): [number, number] {
  const [x, y] = labelAt(m, width / 800);
  return [x / width, y / height];
}
