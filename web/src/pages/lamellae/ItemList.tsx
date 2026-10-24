import { useQuery } from "@tanstack/react-query";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import * as fmt from "@/lib/format";
import { itemOf } from "@/lib/route";
import type { ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { RUN_STATUS } from "./History";

/** A row of /app/experiment/task_history: one run of a task on an item. */
export interface TaskRun {
  item_name: string;
  task_name: string;
  task_status: string;
  start_timestamp: string | null;
}

/** The experiment's items, each one's last run, and the one picked (from the
 * route, else the first): what the Lamellae and Positions pages list. */
export function useItemList() {
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
  const list = items.data?.items ?? [];
  const name = picked && list.some((i) => i.item_name === picked) ? picked : (list[0]?.item_name ?? null);
  const workflow = status.data?.workflow;
  const runningOn = (item: string) =>
    workflow?.running && workflow.current_item === item ? workflow.current_task : null;
  return {
    closed: !!status.data && !open,
    loaded: !!items.data,
    list,
    lastRun,
    name,
    setPicked,
    runningOn,
  };
}

/** One item in the list: its picture, its name and *badge*, and what it last
 * ran; *children* go underneath. */
export function ItemCard({
  item,
  running,
  last,
  picked,
  onPick,
  picture,
  badge,
  children,
}: {
  item: ItemSummary;
  running: string | null;
  last: TaskRun | undefined;
  picked: boolean;
  onPick: () => void;
  picture: ReactNode;
  badge?: ReactNode;
  children?: ReactNode;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (picked) ref.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [picked]);
  // A stopped or failed last run says so; so does an item marked failed.
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
      <div className="aspect-[3/2] w-full overflow-hidden rounded bg-black">{picture}</div>
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[14px] font-semibold">{item.item_name}</span>
        {badge}
      </div>
      <div className="text-[12px]">
        {running ? (
          <span className="text-link">{running} · running</span>
        ) : last ? (
          <span className="text-muted-foreground">
            {last.task_name} · {fmt.clock(last.start_timestamp)}
            {lastWord && <span className={lastWord[1]}> · {lastWord[0]}</span>}
          </span>
        ) : (
          <span className="text-muted-foreground">No tasks run yet</span>
        )}
      </div>
      {children}
    </button>
  );
}

/** The page: the cards (a row on a narrow screen, a column beside the picked
 * item's *detail* on a wide one). */
export function ItemListPage({ cards, empty, detail }: { cards: ReactNode; empty: ReactNode; detail: ReactNode }) {
  return (
    <div className="grid items-start gap-4 p-4 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="flex gap-3 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
        {empty}
        {cards}
      </div>
      {detail}
    </div>
  );
}
