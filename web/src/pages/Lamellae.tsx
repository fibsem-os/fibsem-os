import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import { useFileRead } from "@/lib/files";
import { itemOf } from "@/lib/route";
import type { HistoryFilter, ItemDetail, ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { History, NO_FILTER } from "./lamellae/History";

// The defect verdict's mark, as the Qt card draws it.
const VERDICT: Record<string, [string, string]> = {
  GOOD: ["Good", "bg-[var(--fibsem-ok-color)]"],
  REWORK: ["Rework", "bg-[var(--fibsem-defect-orange-color)]"],
  FAILED: ["Failed", "bg-[var(--fibsem-defect-red-color)]"],
};

function LamellaCard({
  item,
  running,
  picked,
  onPick,
}: {
  item: ItemSummary;
  running: string | null;
  picked: boolean;
  onPick: () => void;
}) {
  const name = item.lamella_name;
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
        ) : item.is_failure ? (
          <span className="text-destructive">Failed{item.last_completed_task ? ` after ${item.last_completed_task}` : ""}</span>
        ) : item.last_completed_task ? (
          <span className="text-muted-foreground">
            {item.last_completed_task} · {item.last_completed_at}
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
  const [picked, setPicked] = useState<string | null>(itemOf);
  const [filter, setFilter] = useState<HistoryFilter>(NO_FILTER);
  if (status.data && !open) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const list = items.data?.items ?? [];
  const name = picked && list.some((i) => i.lamella_name === picked) ? picked : (list[0]?.lamella_name ?? null);
  const workflow = status.data?.workflow;
  return (
    <div className="grid items-start gap-4 p-4 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="flex gap-3 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {items.data && !list.length && <div className="text-muted-foreground">No lamellae yet.</div>}
        {list.map((item) => (
          <LamellaCard
            key={item.lamella_id}
            item={item}
            running={workflow?.running && workflow.current_item === item.lamella_name ? workflow.current_task : null}
            picked={item.lamella_name === name}
            onPick={() => setPicked(item.lamella_name)}
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
