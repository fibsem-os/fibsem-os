import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { api, ApiError, imageUrl } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import { itemOf } from "@/lib/route";
import type { HistoryFilter, ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { clock, History, NO_FILTER, RUN_STATUS } from "./lamellae/History";
import { Scrubber, useStack } from "./positions/Scrubber";

/** A row of /app/experiment/task_history: one run of a task on an item. */
interface TaskRun {
  item_name: string;
  task_name: string;
  task_status: string;
  start_timestamp: string | null;
}

const itemPath = (name: string) => `/app/experiment/items/${encodeURIComponent(name)}`;

function slicesLine(item: ItemSummary) {
  const cut = item.slices_cut ?? 0;
  return item.iterations ? `${cut} of ${item.iterations} slices` : `${cut} ${cut === 1 ? "slice" : "slices"}`;
}

function PositionCard({
  item,
  running,
  last,
  picked,
  onPick,
}: {
  item: ItemSummary;
  running: string | null;
  last: TaskRun | undefined;
  picked: boolean;
  onPick: () => void;
}) {
  const name = item.item_name;
  const newest = useStack(name, "slice").data?.slices.slice(-1)[0];
  const image = useQuery({
    queryKey: ["slice image", name, newest?.filename, 640],
    queryFn: () => imageUrl(`${itemPath(name)}/outputs/${encodeURIComponent(newest!.filename)}?width=640`),
    enabled: !!newest,
    staleTime: Infinity,
  });
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (picked) ref.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [picked]);
  const lastWord =
    last && last.task_status !== "Completed"
      ? (RUN_STATUS[last.task_status] ?? [last.task_status, ""])
      : item.is_failure
        ? RUN_STATUS.Failed
        : null;
  return (
    <button
      ref={ref}
      onClick={onPick}
      aria-pressed={picked}
      className={cn(
        "flex w-60 shrink-0 cursor-pointer flex-col gap-2 rounded-md border bg-card p-2.5 text-left text-foreground hover:border-input lg:w-full",
        picked ? "border-primary" : "border-border",
      )}
    >
      <div className="aspect-[3/2] w-full overflow-hidden rounded bg-black">
        {image.data ? (
          <img className="block size-full object-contain" alt="" src={image.data} />
        ) : (
          <div className="flex size-full items-center justify-center text-[12px] text-muted-foreground">
            {newest ? "Loading slice…" : "No slices yet"}
          </div>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="min-w-0 flex-1 truncate text-[14px] font-semibold">{name}</span>
        <span className="shrink-0 text-[12px] text-muted-foreground tabular-nums">{slicesLine(item)}</span>
      </div>
      <div className="text-[12px]">
        {running ? (
          <span className="text-link">{running} · running</span>
        ) : last ? (
          <span className="text-muted-foreground">
            {last.task_name} · {clock(last.start_timestamp ? Date.parse(last.start_timestamp) / 1000 : null)}
            {lastWord && <span className={lastWord[1]}> · {lastWord[0]}</span>}
          </span>
        ) : (
          <span className="text-muted-foreground">No tasks run yet</span>
        )}
      </div>
    </button>
  );
}

const TABS = [
  ["slices", "Slices"],
  ["history", "History"],
] as const;

/** Whether this app serves an item's History; until it does, the tab stays hidden. */
function useHistoryServed(name: string | null) {
  return useQuery({
    queryKey: ["history served"],
    queryFn: () =>
      api(`${itemPath(name!)}/history`).then(
        () => true,
        (e) => !(e instanceof ApiError && e.status === 404),
      ),
    enabled: !!name,
    staleTime: Infinity,
  }).data;
}

export function PositionsPage() {
  const { status, open } = useOpenExperiment();
  const items = useQuery({
    queryKey: ["app/experiment"],
    queryFn: () => api<{ items: ItemSummary[] }>("/app/experiment"),
    enabled: !!open,
  });
  const runs = useQuery({
    queryKey: ["app/experiment/task_history"],
    queryFn: () => api<{ items: TaskRun[] }>("/app/experiment/task_history"),
    enabled: !!open,
  });
  const lastRun = new Map<string, TaskRun>();
  for (const run of runs.data?.items ?? []) {
    const seen = lastRun.get(run.item_name);
    if (!seen || (run.start_timestamp ?? "") >= (seen.start_timestamp ?? "")) lastRun.set(run.item_name, run);
  }
  const [picked, setPicked] = useState<string | null>(itemOf);
  const [tab, setTab] = useState<"slices" | "history">("slices");
  const [filter, setFilter] = useState<HistoryFilter>(NO_FILTER);
  const list = items.data?.items ?? [];
  const name = picked && list.some((i) => i.item_name === picked) ? picked : (list[0]?.item_name ?? null);
  const historyServed = useHistoryServed(name);
  if (status.data && !open) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const workflow = status.data?.workflow;
  const runningOn = (item: string) =>
    workflow?.running && workflow.current_item === item ? workflow.current_task : null;
  const item = list.find((i) => i.item_name === name);
  const shown = historyServed ? tab : "slices";
  return (
    <div className="grid items-start gap-4 p-4 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="flex gap-3 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {items.data && !list.length && <div className="text-muted-foreground">No positions yet.</div>}
        {list.map((i) => (
          <PositionCard
            key={i.item_id}
            item={i}
            running={runningOn(i.item_name)}
            last={lastRun.get(i.item_name)}
            picked={i.item_name === name}
            onPick={() => setPicked(i.item_name)}
          />
        ))}
      </div>
      {name && (
        <Card className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="min-w-0 truncate text-[15px] font-semibold">{name}</span>
            {historyServed && (
              <div role="tablist" aria-label="Position" className="flex gap-1">
                {TABS.map(([key, label]) => (
                  <button
                    key={key}
                    role="tab"
                    aria-selected={shown === key}
                    onClick={() => setTab(key)}
                    className={cn(
                      "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[13px]",
                      shown === key
                        ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>
          {shown === "slices" ? (
            <Scrubber key={name} name={name} planned={item?.iterations} running={!!runningOn(name)} />
          ) : (
            <History name={name} filter={filter} setFilter={setFilter} />
          )}
        </Card>
      )}
    </div>
  );
}
