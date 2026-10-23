import { useQuery } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronRight, Circle, CircleSlash, ListFilter, Loader, X } from "lucide-react";
import { useState } from "react";
import { ImageViewer, Thumb } from "@/components/images";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { HistoryFilter, HistoryOperation, HistoryRun, ItemHistory } from "@/lib/types";
import { cn } from "@/lib/utils";

export const NO_FILTER: HistoryFilter = { show: "all", status: "all", task: null, decided_by: "anyone", order: "oldest" };

// The run's status as the Qt tab says it, and its colour.
export const RUN_STATUS: Record<string, [string, string]> = {
  Completed: ["Completed", "text-[var(--fibsem-ok-color)]"],
  Failed: ["Failed", "text-destructive"],
  Cancelled: ["Stopped", "text-[var(--fibsem-semantic-warning-color)]"],
  Skipped: ["Skipped", "text-muted-foreground"],
  InProgress: ["Running", "text-link"],
};

const OP_ICON: Record<string, [typeof Check, string]> = {
  completed: [Check, "text-[var(--fibsem-ok-color)]"],
  failed: [X, "text-destructive"],
  cancelled: [CircleSlash, "text-[var(--fibsem-semantic-warning-color)]"],
  skipped: [Circle, "text-muted-foreground"],
};
const SEVERITY: Record<string, number> = { failed: 0, cancelled: 1, skipped: 2 };

function duration(seconds: number | null) {
  if (seconds == null) return "";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

function OpIcon({ status }: { status: string }) {
  const [Icon, colour] = OP_ICON[status] ?? [Circle, "text-muted-foreground"];
  return <Icon size={13} className={cn("shrink-0", colour)} />;
}

function Figure({ path }: { path: string }) {
  const read = useFileRead(path, 640);
  const p = read.data?.preview;
  return p ? <img className="block w-full max-w-[420px] rounded bg-white" alt="" src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} /> : null;
}

/** One operation: its line, and opened, its facts, stages and figure. */
function Operation({ op, nested }: { op: HistoryOperation; nested?: boolean }) {
  const [open, setOpen] = useState(false);
  const more = op.facts.length > 0 || op.stages.length > 0 || !!op.figure;
  return (
    <div className={cn("flex flex-col gap-1", nested && "pl-5")}>
      <button
        className="flex min-w-0 cursor-pointer items-center gap-2 border-0 bg-transparent p-0 text-left text-[12px] text-foreground disabled:cursor-default"
        onClick={() => setOpen(!open)}
        disabled={!more}
        aria-expanded={more ? open : undefined}
      >
        {more ? (
          open ? <ChevronDown size={12} className="shrink-0 text-muted-foreground" /> : <ChevronRight size={12} className="shrink-0 text-muted-foreground" />
        ) : (
          <span className="w-3 shrink-0" />
        )}
        <OpIcon status={op.status} />
        <span className="shrink-0 font-semibold">{op.label}</span>
        <span className="min-w-0 truncate text-muted-foreground">{op.detail}</span>
        <span className="ml-auto shrink-0 pl-2 text-muted-foreground tabular-nums">
          {duration(op.started_at != null && op.ended_at != null ? op.ended_at - op.started_at : null)}
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-1.5 pb-1 pl-10 text-[12px]">
          {op.facts.length > 0 && (
            <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
              {op.facts.map(([k, v], i) => (
                <div key={i} className="contents">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="m-0 break-words">{v}</dd>
                </div>
              ))}
            </dl>
          )}
          {op.stages.map((stage, i) => (
            <div key={i} className="flex flex-col gap-1">
              <div>
                <span className="font-semibold">{stage.name}</span> <span className="text-muted-foreground">{stage.text}</span>
              </div>
              {stage.strategy && (
                <div className="pl-3">
                  <span className="text-foreground">{stage.strategy}</span>{" "}
                  <span className="text-muted-foreground">{stage.said}</span>
                </div>
              )}
              {stage.figure && (
                <div className="pl-3">
                  <Figure path={stage.figure} />
                </div>
              )}
              {stage.monitors.map((m, j) => (
                <div key={j} className="flex flex-col gap-1 pl-3">
                  <div>
                    <span className="text-foreground">{m.name}</span> <span className="text-muted-foreground">{m.said}</span>
                  </div>
                  {m.figure && <Figure path={m.figure} />}
                </div>
              ))}
              {stage.drift.map((d, j) => (
                <Operation key={j} op={d} nested />
              ))}
            </div>
          ))}
          {op.figure && <Figure path={op.figure} />}
        </div>
      )}
    </div>
  );
}

/** "› ✓ 3 operations · all completed", opening to the operations. */
function Operations({ run, show }: { run: HistoryRun; show: HistoryFilter["show"] }) {
  const [open, setOpen] = useState(run.operations_open);
  if (!run.operations.length) return null;
  if (run.operations.length === 1 || show === "operations")
    return (
      <div className="flex flex-col gap-1">
        {run.operations.map((op, i) => (
          <Operation key={i} op={op} />
        ))}
      </div>
    );
  const worst = run.operations.map((op) => op.status).sort((a, b) => (SEVERITY[a] ?? 3) - (SEVERITY[b] ?? 3))[0];
  const total = run.operations.reduce((sum, op) => sum + (op.started_at != null && op.ended_at != null ? op.ended_at - op.started_at : 0), 0);
  return (
    <div className="flex flex-col gap-1">
      <button
        className="flex cursor-pointer items-center gap-2 border-0 bg-transparent p-0 text-left text-[12px] text-muted-foreground"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        title="Show or hide the operations"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <OpIcon status={worst} />
        <span className="min-w-0 flex-1 truncate">{run.operations_summary}</span>
        {total > 0 && <span className="shrink-0 tabular-nums">{duration(total)} in operations</span>}
      </button>
      {open && (
        <div className="flex flex-col gap-1 pl-5">
          {run.operations.map((op, i) => (
            <Operation key={i} op={op} />
          ))}
        </div>
      )}
    </div>
  );
}

function Run({
  run,
  runs,
  show,
  openImage,
}: {
  run: HistoryRun;
  runs: HistoryRun[];
  show: HistoryFilter["show"];
  openImage: (path: string) => void;
}) {
  const [word, colour] = RUN_STATUS[run.status] ?? [run.status, "text-muted-foreground"];
  const later = run.images_replaced_by ? runs.find((r) => r.task_id === run.images_replaced_by) : undefined;
  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-semibold">{run.task_name}</span>
        <span className="text-[12px] text-muted-foreground">
          {[fmt.clock(run.started_at), duration(run.duration)].filter(Boolean).join(" · ")}
        </span>
        {!run.outside_task && (
          <span className={cn("ml-auto flex items-center gap-1 text-[12px]", colour)}>
            {run.status === "InProgress" && <Loader size={12} className="animate-spin" />}
            {word}
          </span>
        )}
      </div>
      {run.status_message && (run.status === "Failed" || run.status === "Cancelled") && (
        <div className={cn("text-[12px]", colour)}>{run.status_message}</div>
      )}
      {show !== "images" && <Operations run={run} show={show} />}
      {show !== "operations" &&
        (run.images.length ? (
          <div className="grid max-w-[880px] grid-cols-2 gap-2">
            {run.images.map((path) => (
              <Thumb key={path} path={path} onOpen={() => openImage(path)} />
            ))}
          </div>
        ) : run.outside_task ? null : (
          <div className="text-[12px] text-muted-foreground">
            {run.images_replaced_by ? `Images replaced by the ${later ? `${fmt.clock(later.started_at)} ` : "later "}run.` : "No images recorded."}
          </div>
        ))}
    </div>
  );
}

type Choice<K extends keyof HistoryFilter> = [HistoryFilter[K], string];

function Section<K extends keyof HistoryFilter>({
  title,
  field,
  choices,
  value,
  set,
}: {
  title: string;
  field: K;
  choices: Choice<K>[];
  value: HistoryFilter;
  set: (next: HistoryFilter) => void;
}) {
  return (
    <div className="flex flex-col py-1">
      <div className="px-2 py-1 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">{title}</div>
      {choices.map(([choice, label]) => (
        <button
          key={String(choice)}
          className="flex cursor-pointer items-center gap-2 rounded border-0 bg-transparent px-2 py-1 text-left text-[13px] text-foreground hover:bg-accent"
          onClick={() => set({ ...value, [field]: choice })}
        >
          <Check size={13} className={cn(value[field] === choice ? "opacity-100" : "opacity-0")} />
          {label}
        </button>
      ))}
    </div>
  );
}

function filterCount(f: HistoryFilter) {
  return [f.show !== "all", f.status !== "all", f.task != null, f.decided_by !== "anyone"].filter(Boolean).length;
}

function FilterMenu({ value, set, tasks }: { value: HistoryFilter; set: (f: HistoryFilter) => void; tasks: string[] }) {
  const n = filterCount(value);
  const label = `${value.order === "newest" ? "Newest first" : "Oldest first"}${n ? ` · ${n} filter${n > 1 ? "s" : ""}` : ""}`;
  return (
    <Popup
      label="History filter"
      align="right"
      side="below"
      className="w-56"
      trigger={(open, toggle) => (
        <Button variant="ghost" onClick={toggle} aria-expanded={open} className={cn(n > 0 && "text-link")}>
          <ListFilter size={14} />
          {label}
        </Button>
      )}
    >
      {() => (
        <div className="flex flex-col divide-y divide-border">
          <Section title="Order" field="order" value={value} set={set} choices={[["newest", "Newest first"], ["oldest", "Oldest first"]]} />
          <Section title="Show" field="show" value={value} set={set} choices={[["all", "Everything"], ["images", "Images"], ["operations", "Operations"]]} />
          <Section title="Runs" field="status" value={value} set={set} choices={[["all", "All"], ["failed", "Failed"], ["cancelled", "Stopped"]]} />
          <Section
            title="Task"
            field="task"
            value={value}
            set={set}
            choices={[[null, "Every task"], ...tasks.map((t): Choice<"task"> => [t, t])]}
          />
          <Section
            title="Decided by"
            field="decided_by"
            value={value}
            set={set}
            choices={[["anyone", "Anyone"], ["agent", "The agent"], ["operator", "The operator"]]}
          />
          <div className="p-1">
            <Button variant="ghost" className="w-full justify-start" onClick={() => set(NO_FILTER)} disabled={!n && value.order === "oldest"}>
              Reset
            </Button>
          </div>
        </div>
      )}
    </Popup>
  );
}

/** A lamella's History: its runs through the filter, as the Qt tab shows them. */
export function History({ name, filter, setFilter }: { name: string; filter: HistoryFilter; setFilter: (f: HistoryFilter) => void }) {
  const query = new URLSearchParams({ show: filter.show, status: filter.status, decided_by: filter.decided_by, order: filter.order });
  if (filter.task) query.set("task", filter.task);
  const history = useQuery({
    queryKey: ["app/experiment/items", name, "history", query.toString()],
    queryFn: () => api<ItemHistory>(`/app/experiment/items/${encodeURIComponent(name)}/history?${query}`),
    placeholderData: (previous) => (previous?.item_name === name ? previous : undefined),
  });
  const [viewing, setViewing] = useState<number | null>(null);
  const runs = history.data?.runs ?? [];
  const images = filter.show === "operations" ? [] : runs.flatMap((r) => r.images);
  const shown = runs.length;
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-semibold">{name}</div>
          <div className="text-[12px] text-muted-foreground">
            {history.data
              ? filterCount(filter)
                ? `${shown} of ${history.data.total} runs`
                : `${history.data.total} ${history.data.total === 1 ? "run" : "runs"}`
              : " "}
          </div>
        </div>
        <FilterMenu value={filter} set={setFilter} tasks={history.data?.tasks ?? []} />
      </div>
      {history.error && <div className="text-[13px] text-destructive">{history.error.message}</div>}
      {history.data && !history.data.total && <div className="text-[13px] text-muted-foreground">No task runs yet.</div>}
      {history.data && history.data.total > 0 && !shown && (
        <div className="text-[13px] text-muted-foreground">
          No runs match the filter.{" "}
          <button className="cursor-pointer border-0 bg-transparent p-0 text-link" onClick={() => setFilter(NO_FILTER)}>
            Reset
          </button>
        </div>
      )}
      {runs.map((run) => (
        <Run key={run.task_id} run={run} runs={runs} show={filter.show} openImage={(path) => setViewing(images.indexOf(path))} />
      ))}
      <ImageViewer paths={images} at={viewing} onClose={() => setViewing(null)} />
    </div>
  );
}
