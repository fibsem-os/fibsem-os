import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import { useFileRead } from "@/lib/files";
import { itemOf } from "@/lib/route";
import type { HistoryFilter, ItemDetail, ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { clock, History, NO_FILTER, RUN_STATUS } from "./lamellae/History";

/** A row of /app/experiment/task_history: one run of a task on an item. */
interface TaskRun {
  item_name: string;
  task_name: string;
  task_status: string;
  start_timestamp: string | null;
}

// The defect verdict's mark, as the Qt card draws it.
const VERDICT: Record<string, [string, string]> = {
  GOOD: ["Good", "bg-[var(--fibsem-ok-color)]"],
  REWORK: ["Rework", "bg-[var(--fibsem-defect-orange-color)]"],
  FAILED: ["Failed", "bg-[var(--fibsem-defect-red-color)]"],
};

function LamellaCard({
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
  const thumb = useFileRead(`${name}/thumbnail.png`, 640);
  const detail = useQuery({
    queryKey: ["app/experiment/items", name],
    queryFn: () => api<ItemDetail>(`/app/experiment/items/${encodeURIComponent(name)}`),
  });
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (picked) ref.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [picked]);
  const p = thumb.data?.preview;
  // A stopped or failed last run says so; so does a lamella marked failed.
  const lastWord =
    last && last.task_status !== "Completed"
      ? (RUN_STATUS[last.task_status] ?? [last.task_status, ""])
      : item.is_failure
        ? RUN_STATUS.Failed
        : null;
  const defect = detail.data?.defect;
  const verdict = defect ? VERDICT[defect.verdict] : undefined;
  return (
    <button
      ref={ref}
      onClick={onPick}
      aria-pressed={picked}
      className={cn(
        "flex w-60 shrink-0 cursor-pointer flex-col gap-2 lg:w-full rounded-md border bg-card p-2.5 text-left text-foreground hover:border-input",
        picked ? "border-primary" : "border-border",
      )}
    >
      <div className="aspect-[3/2] w-full overflow-hidden rounded bg-black">
        {p && <img className="block size-full object-contain" alt="" src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />}
        {thumb.isError && <div className="flex size-full items-center justify-center text-[12px] text-muted-foreground">No thumbnail yet</div>}
      </div>
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[14px] font-semibold">{name}</span>
        {verdict && (
          <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground" title={defect?.reason || undefined}>
            <span className={cn("inline-block size-2.5 rounded-full", verdict[1])} />
            {verdict[0]}
          </span>
        )}
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
      {verdict && defect?.reason && <div className="text-[12px] text-muted-foreground">{defect.reason}</div>}
    </button>
  );
}

export function LamellaePage() {
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
  const [filter, setFilter] = useState<HistoryFilter>(NO_FILTER);
  if (status.data && !open) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const list = items.data?.items ?? [];
  const name = picked && list.some((i) => i.item_name === picked) ? picked : (list[0]?.item_name ?? null);
  const workflow = status.data?.workflow;
  return (
    <div className="grid items-start gap-4 p-4 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="flex gap-3 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {items.data && !list.length && <div className="text-muted-foreground">No lamellae yet.</div>}
        {list.map((item) => (
          <LamellaCard
            key={item.item_id}
            item={item}
            running={workflow?.running && workflow.current_item === item.item_name ? workflow.current_task : null}
            last={lastRun.get(item.item_name)}
            picked={item.item_name === name}
            onPick={() => setPicked(item.item_name)}
          />
        ))}
      </div>
      {name && (
        <Card>
          <History name={name} filter={filter} setFilter={setFilter} />
        </Card>
      )}
    </div>
  );
}
