import { ChevronDown } from "lucide-react";
import { useEffect, useRef } from "react";
import type { Projection } from "@/lib/planes";
import { cn } from "@/lib/utils";

/** What of a stack is drawn: one plane, or a projection over a range of planes. */
export interface StackShown {
  z: number;
  projection: Projection | null;
  first: number;
  last: number;
}

export const PROJECTIONS: [Projection, string, string][] = [
  ["max", "Max", "Max intensity"],
  ["mean", "Mean", "Mean intensity"],
  ["min", "Min", "Min intensity"],
];

/**
 * The row under a stack, as Qt's: ‹ the planes slider › and where it is, each
 * plane's sharpness drawn above the slider (click one to show it), and PROJ, which
 * opens the projection: one plane, or Max, Mean or Min over all planes or a range.
 * Choosing a plane leaves the projection.
 */
export function ZRow({
  count,
  shown,
  set,
  sharpness,
  menu,
  setMenu,
}: {
  count: number;
  shown: StackShown;
  set: (s: StackShown) => void;
  sharpness: number[] | null;
  menu: boolean;
  setMenu: (open: boolean) => void;
}) {
  const projecting = shown.projection != null;
  const go = (z: number) => set({ ...shown, z: Math.max(0, Math.min(count - 1, z)), projection: null });
  const range = (first: number, last: number) => {
    const a = Math.max(0, Math.min(count - 1, first));
    set({ ...shown, first: a, last: Math.max(a, Math.min(count - 1, last)) });
  };
  const pop = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const click = (e: MouseEvent) => pop.current?.contains(e.target as Node) || setMenu(false);
    addEventListener("mousedown", click);
    return () => removeEventListener("mousedown", click);
  }, [menu, setMenu]);
  const button = "h-[22px] cursor-pointer rounded-[3px] border bg-card text-foreground hover:bg-border";
  return (
    <div className="relative flex items-end gap-1.5 bg-card px-1.5 pt-0.5 pb-1 text-[12px] tabular-nums">
      <span className="leading-[22px]">z</span>
      <button
        className={cn(button, "w-6 pb-0.5 text-[16px] leading-none")}
        aria-label="Previous plane"
        title="Previous plane (↓)"
        onClick={() => go(shown.z - 1)}
      >
        ‹
      </button>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        {sharpness && (
          <div className="flex h-3.5 items-end gap-px px-1.5" title="How sharp each plane is">
            {sharpness.map((v, i) => (
              <button
                key={i}
                aria-label={`Plane ${i + 1}, sharpness ${v.toFixed(2)}`}
                onClick={() => go(i)}
                className={cn(
                  "min-h-0.5 min-w-0 flex-1 cursor-pointer border-0 p-0",
                  !projecting && i === shown.z
                    ? "bg-link"
                    : projecting && i >= shown.first && i <= shown.last
                      ? "bg-[var(--fibsem-primary-accent)]"
                      : "bg-[#414851] hover:bg-[var(--fibsem-gray-highlight-color)]",
                )}
                style={{ height: `${Math.max(12, v * 100)}%` }}
              />
            ))}
          </div>
        )}
        <input
          type="range"
          aria-label="Z plane"
          className={cn("h-3.5 w-full min-w-0 accent-[var(--primary)]", projecting && "opacity-45")}
          min={0}
          max={count - 1}
          value={shown.z}
          onChange={(e) => go(Number(e.target.value))}
        />
      </div>
      <button
        className={cn(button, "w-6 pb-0.5 text-[16px] leading-none")}
        aria-label="Next plane"
        title="Next plane (↑)"
        onClick={() => go(shown.z + 1)}
      >
        ›
      </button>
      <span className={cn("min-w-11 text-right leading-[22px]", projecting && "text-muted-foreground")}>
        {projecting ? PROJECTIONS.find((p) => p[0] === shown.projection)![1] : `${shown.z + 1}/${count}`}
      </span>
      <div ref={pop} className="relative">
        <button
          aria-pressed={projecting}
          aria-expanded={menu}
          title="Projection (P)"
          onClick={() => setMenu(!menu)}
          className={cn(
            "flex h-[22px] cursor-pointer items-center gap-0.5 rounded-[3px] border px-[7px] text-[10px] font-semibold tracking-[.04em]",
            projecting ? "border-primary bg-primary text-white" : "bg-card text-muted-foreground",
          )}
        >
          PROJ
          <ChevronDown size={12} />
        </button>
        {menu && (
          <div
            role="dialog"
            aria-label="Projection"
            className="absolute right-0 bottom-[calc(100%+4px)] z-20 flex w-[220px] flex-col gap-1.5 rounded-lg border bg-card px-3 pt-2.5 pb-3 shadow-[0_6px_24px_rgba(0,0,0,0.45)]"
          >
            <span className="text-[11px] font-medium tracking-[1px] text-muted-foreground">PROJECTION</span>
            {(
              [[null, "One plane"], ...PROJECTIONS.map(([p, , label]) => [p, label])] as [Projection | null, string][]
            ).map(([p, label]) => (
              <label key={label} className="flex cursor-pointer items-center gap-2">
                <input
                  type="radio"
                  name="projection"
                  className="m-0 accent-[var(--primary)]"
                  checked={shown.projection === p}
                  onChange={() => set({ ...shown, projection: p })}
                />
                {label}
              </label>
            ))}
            <div
              className={cn(
                "flex items-center gap-1.5 text-muted-foreground",
                !projecting && "pointer-events-none opacity-45",
              )}
            >
              Planes
              {(["first", "last"] as const).map((end) => (
                <input
                  key={`${end}${shown[end]}`}
                  type="number"
                  aria-label={end === "first" ? "First plane" : "Last plane"}
                  min={1}
                  max={count}
                  defaultValue={shown[end] + 1}
                  className="w-[46px] rounded border bg-panel px-1.5 py-0.5 text-[11px] text-foreground"
                  onKeyDown={(e) => e.stopPropagation()}
                  onBlur={(e) =>
                    end === "first"
                      ? range(Number(e.target.value) - 1, shown.last)
                      : range(shown.first, Number(e.target.value) - 1)
                  }
                />
              ))}
              <button
                className="ml-auto cursor-pointer border-0 bg-transparent p-0 text-[11px] text-link"
                onClick={() => range(0, count - 1)}
              >
                All
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
