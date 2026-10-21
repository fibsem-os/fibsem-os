import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Card, CardTitle } from "@/components/ui/card";
import { api, imageUrl } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import type { Go } from "@/lib/route";
import type { GridDetail, GridMarkers, GridSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

// The quality verdict's mark, as the Qt grid card draws it.
const QUALITY: Record<string, [string, string]> = {
  GOOD: ["Good", "bg-[var(--fibsem-ok-color)]"],
  REWORK: ["Rework", "bg-[var(--fibsem-defect-orange-color)]"],
  FAILED: ["Failed", "bg-[var(--fibsem-defect-red-color)]"],
};

const RUN_COLOUR: Record<string, string> = {
  Completed: "text-[var(--fibsem-ok-color)]",
  Failed: "text-destructive",
  Cancelled: "text-[var(--fibsem-semantic-warning-color)]",
  InProgress: "text-link",
};

const gridPath = (grid: string) => `/app/experiment/grids/${encodeURIComponent(grid)}`;

// The overview roles a grid records, in the order the tabs show them.
const ROLES: [string, string][] = [
  ["overview_sem", "SEM"],
  ["overview_fib", "FIB"],
  ["overview_fm", "FM"],
];

/** The grid's latest overview of each kind it has, SEM before FIB before FM. */
function overviewsOf(grid: GridSummary): [string, string][] {
  const o = grid.overviews;
  const known = ROLES.filter(([role]) => o[role]).map(([role, label]): [string, string] => [label, o[role]]);
  const others = Object.entries(o).filter(([role]) => !ROLES.some(([r]) => r === role));
  return [...known, ...others];
}

function useGridImage(grid: string, filename: string | null) {
  const url = useQuery({
    queryKey: ["grid image", grid, filename],
    queryFn: () => imageUrl(`${gridPath(grid)}/outputs/${encodeURIComponent(filename!)}`),
    enabled: !!filename,
    staleTime: Infinity,
  });
  useEffect(() => () => void (url.data && URL.revokeObjectURL(url.data)), [url.data]);
  return { src: url.data, loading: url.isLoading };
}

function GridCard({ grid, picked, onPick }: { grid: GridSummary; picked: boolean; onPick: () => void }) {
  const { src, loading } = useGridImage(grid.name, overviewsOf(grid)[0]?.[1] ?? null);
  const quality = QUALITY[grid.quality];
  const status =
    grid.status === "InProgress" && grid.current_task ? (
      <span className="text-link">{grid.current_task} · running</span>
    ) : grid.is_failure ? (
      <span className="text-destructive">Failed{grid.current_task ? ` at ${grid.current_task}` : ""}</span>
    ) : (
      <span className="text-muted-foreground">{grid.last_completed_task ?? "Not screened yet"}</span>
    );
  return (
    <button
      onClick={onPick}
      aria-pressed={picked}
      className={cn(
        "flex w-60 shrink-0 cursor-pointer flex-col gap-2 rounded-md border bg-card p-2.5 text-left text-foreground hover:border-input lg:w-full",
        picked ? "border-primary" : "border-border",
      )}
    >
      <div className="aspect-[3/2] w-full overflow-hidden rounded bg-black">
        {src ? (
          <img className="block size-full object-contain" alt="" src={src} />
        ) : (
          <div className="flex size-full items-center justify-center text-[12px] text-muted-foreground">
            {loading ? "Loading overview…" : "No overview yet"}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[14px] font-semibold">{grid.name}</span>
        {quality && (
          <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <span className={cn("inline-block size-2.5 rounded-full", quality[1])} />
            {quality[0]}
          </span>
        )}
      </div>
      <div className="text-[12px]">
        {status}
        {grid.num_items > 0 && <span className="text-muted-foreground"> · {grid.num_items} lamellae</span>}
      </div>
    </button>
  );
}

/** The overview with the grid's lamellae where they sit: green, red when failed. */
function Overview({ grid, filename }: { grid: string; filename: string }) {
  const { src } = useGridImage(grid, filename);
  const markers = useQuery({
    queryKey: ["grid markers", grid, filename],
    queryFn: () => api<GridMarkers>(`${gridPath(grid)}/outputs/${encodeURIComponent(filename)}/markers`),
  });
  const m = markers.data;
  const w = m?.image?.width ?? 1;
  const h = m?.image?.height ?? 1;
  const r = Math.max(w / 80, 4);
  const size = Math.max(w / 40, 10);
  return (
    <div className="flex flex-col gap-2">
      <div className="relative w-full max-w-[880px] overflow-hidden rounded bg-black">
        {src ? (
          <img className="block w-full" alt={filename} src={src} />
        ) : (
          <div className="flex aspect-[3/2] w-full items-center justify-center text-[12px] text-muted-foreground">
            Loading overview…
          </div>
        )}
        {src && m?.image && (
          <svg className="absolute inset-0 size-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
            {m.markers
              .filter((p) => p.inside)
              .map((p) => {
                const colour = p.is_failure ? "#e05252" : "#4caf50";
                return (
                  <g key={p.item_name}>
                    <circle cx={p.x} cy={p.y} r={r} fill="none" stroke="#000" strokeOpacity={0.8} strokeWidth={r * 0.7} />
                    <circle cx={p.x} cy={p.y} r={r} fill="none" stroke={colour} strokeWidth={r * 0.35} />
                    <text
                      x={p.x + r * 1.6}
                      y={p.y + size / 3}
                      fill={colour}
                      fontSize={size}
                      fontWeight="bold"
                      stroke="#000"
                      strokeOpacity={0.8}
                      strokeWidth={size / 8}
                      paintOrder="stroke"
                    >
                      {p.item_name}
                    </text>
                  </g>
                );
              })}
          </svg>
        )}
      </div>
      <div className="text-[12px] text-muted-foreground">
        {filename}
        {m?.reason && ` · ${m.reason}`}
        {m && m.unplaced.length > 0 && ` · not placed: ${m.unplaced.map((u) => u.item_name).join(", ")}`}
      </div>
    </div>
  );
}

function GridPanel({ name, go }: { name: string; go: Go }) {
  const detail = useQuery({ queryKey: ["app/experiment/grids", name], queryFn: () => api<GridDetail>(gridPath(name)) });
  const d = detail.data;
  const overviews = d ? overviewsOf(d) : [];
  const [kind, setKind] = useState<string | null>(null);
  const shown = overviews.find(([label]) => label === kind) ?? overviews[0];
  const overview = shown?.[1] ?? null;
  return (
    <Card className="flex min-w-0 flex-col gap-4">
      <div>
        <div className="text-[15px] font-semibold">{name}</div>
        {d?.description && <div className="text-[12px] text-muted-foreground">{d.description}</div>}
      </div>
      {detail.error && <div className="text-[13px] text-destructive">{detail.error.message}</div>}
      {overviews.length > 1 && (
        <div role="tablist" aria-label="Overview" className="flex gap-1">
          {overviews.map(([label]) => (
            <button
              key={label}
              role="tab"
              aria-selected={label === shown?.[0]}
              className={cn(
                "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[12px]",
                label === shown?.[0]
                  ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                  : "text-muted-foreground hover:text-foreground",
              )}
              onClick={() => setKind(label)}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {overview ? (
        <Overview key={overview} grid={name} filename={overview} />
      ) : (
        d && <div className="text-[13px] text-muted-foreground">No overview yet.</div>
      )}
      {d && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <CardTitle>Runs</CardTitle>
            {d.tasks.length ? (
              d.tasks.map((t, i) => (
                <div key={i} className="flex items-center gap-2 text-[13px]">
                  <span className="min-w-0 flex-1 truncate">{t.name}</span>
                  <span className={cn("text-[12px]", RUN_COLOUR[t.status] ?? "text-muted-foreground")}>{t.status === "Cancelled" ? "Stopped" : t.status}</span>
                </div>
              ))
            ) : (
              <div className="text-[13px] text-muted-foreground">No runs yet.</div>
            )}
          </div>
          <div className="flex flex-col gap-1.5">
            <CardTitle>Lamellae</CardTitle>
            {d.items.length ? (
              d.items.map((item) => (
                <button
                  key={item}
                  className="cursor-pointer border-0 bg-transparent p-0 text-left text-[13px] text-link"
                  onClick={() => go("lamellae", item)}
                >
                  {item}
                </button>
              ))
            ) : (
              <div className="text-[13px] text-muted-foreground">None on this grid.</div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}

export function GridsPage({ go }: { go: Go }) {
  const { status, open } = useOpenExperiment();
  const grids = useQuery({
    queryKey: ["app/experiment/grids"],
    queryFn: () => api<{ items: GridSummary[] }>("/app/experiment/grids"),
    enabled: !!open,
  });
  const [picked, setPicked] = useState<string | null>(null);
  if (status.data && !open) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const list = grids.data?.items ?? [];
  const name = picked && list.some((g) => g.name === picked) ? picked : (list[0]?.name ?? null);
  return (
    <div className="grid items-start gap-4 p-4 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="flex gap-3 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {grids.data && !list.length && <div className="text-muted-foreground">No grids recorded.</div>}
        {list.map((grid) => (
          <GridCard key={grid.id} grid={grid} picked={grid.name === name} onPick={() => setPicked(grid.name)} />
        ))}
      </div>
      {name && <GridPanel name={name} go={go} />}
    </div>
  );
}
