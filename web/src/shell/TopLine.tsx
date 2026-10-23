import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { Facts } from "@/components/ui/card";
import { HoverCard } from "@/components/ui/hover-card";
import { api } from "@/lib/api";
import type { Go } from "@/lib/route";
import type { AgentState, Capabilities, ProgressLine, WorkflowStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { AgentChip } from "./AgentChip";

const DISMISSED = "fibsem-dismissed-failure";

/** "2:14" since *iso*, counted each second while *iso* is set. */
function useElapsed(iso: string | null | undefined): string {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!iso) return;
    const clock = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(clock);
  }, [iso]);
  if (!iso) return "";
  const s = Math.max(0, Math.floor((now - new Date(iso).getTime()) / 1000));
  const [h, m] = [Math.floor(s / 3600), Math.floor((s % 3600) / 60)];
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

const progressText = (p: ProgressLine) => [p.label, p.step, p.numbers].filter(Boolean).join(" · ");

function readDismissed(): string | null {
  try {
    return localStorage.getItem(DISMISSED);
  } catch {
    return null;
  }
}

/** The run on the left, read first; the agent and the open experiment on the right.
 * The line stays short; the card under it says the rest. */
export function TopLine({ go }: { go: Go }) {
  const status = useQuery({
    queryKey: ["app/workflow/status"],
    queryFn: () => api<WorkflowStatus>("/app/workflow/status"),
    // A mill's countdown is not on the record: read it again while something runs.
    refetchInterval: (query) => (query.state.data?.workflow.running || query.state.data?.progress ? 2000 : false),
  });
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const agent = useQuery({ queryKey: ["app/agent"], queryFn: () => api<AgentState>("/app/agent") });
  const [dismissed, setDismissed] = useState(readDismissed);
  const data = status.data;
  const workflow = data?.workflow;
  const experiment = data?.experiment;
  const running = workflow?.running;
  const waited = useElapsed(running ? workflow?.waiting_since : null);
  const progress = data?.progress;

  const place =
    workflow?.queue_position && workflow.queue_total ? `${workflow.queue_position} of ${workflow.queue_total}` : null;
  const what = running
    ? [
        `Running: ${workflow?.current_task ?? "a task"}${workflow?.current_item ? ` on ${workflow.current_item}` : ""}`,
        place,
      ]
        .filter(Boolean)
        .join(" · ")
    : progress && !progress.outcome
      ? progressText(progress)
      : data
        ? "Idle"
        : "";
  const failed = workflow?.failed;
  const failureKey = failed ? `${failed.at ?? ""}|${failed.message}` : null;
  const failure =
    failed && failureKey !== dismissed
      ? `${failed.count > 1 ? `${failed.count} failed · ` : ""}${failed.what}: ${failed.message}`
      : null;
  const dismiss = () => {
    setDismissed(failureKey);
    try {
      if (failureKey) localStorage.setItem(DISMISSED, failureKey);
    } catch {
      // storage off: for this page load only
    }
  };

  const rows: [string, ReactNode][] = [];
  if (running) {
    if (workflow?.current_task) rows.push(["Task", workflow.current_task]);
    if (workflow?.current_item) rows.push(["On", workflow.current_item]);
    if (place) rows.push(["Queue", place]);
    if (workflow?.iteration != null) rows.push(["Iteration", String(workflow.iteration)]);
    if (workflow?.step) rows.push(["Step", workflow.step]);
    if (workflow?.waiting_on) rows.push(["Waiting", `${workflow.waiting_on} · ${waited}`]);
  }
  if (progress) rows.push(["Progress", progressText(progress)]);
  if (agent.data?.now) rows.push(["Agent", agent.data.now]);
  if (data?.in_control) rows.push(["In control", `${data.in_control.by}: ${data.in_control.action}`]);
  if (failed)
    rows.push([failed.count > 1 ? `Failed (${failed.count})` : "Failed", `${failed.what}: ${failed.message}`]);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-b border-border px-5 py-2.5 text-[13px]">
      <span className="flex min-w-0 items-center gap-2">
        <span
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: running ? "var(--fibsem-primary-color)" : "var(--fibsem-gray-primary-color)" }}
        />
        {data && (
          <HoverCard
            align="start"
            trigger={
              <span
                tabIndex={0}
                className={cn(
                  "cursor-default truncate border-b border-dotted border-input",
                  // A short line keeps its words; a run's line gives way to an ellipsis.
                  running ? "min-w-0 shrink" : "shrink-0",
                )}
              >
                {what}
              </span>
            }
          >
            {rows.length ? (
              <Facts rows={rows} />
            ) : (
              <div className="text-[13px] text-muted-foreground">Nothing running.</div>
            )}
          </HoverCard>
        )}
        {running && workflow?.waiting_on && (
          <span className="min-w-0 shrink-[4] truncate text-muted-foreground">
            Waiting: {workflow.waiting_on} · {waited}
          </span>
        )}
        {failure && (
          <>
            <span className="min-w-0 shrink-[4] truncate text-destructive" title={failure}>
              {failure}
            </span>
            <button
              className="shrink-0 cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted-foreground hover:text-foreground"
              onClick={dismiss}
            >
              Dismiss
            </button>
          </>
        )}
      </span>
      <div className="flex-1" />
      <AgentChip go={go} />
      {experiment ? (
        <HoverCard
          trigger={
            <span tabIndex={0} className="cursor-default border-b border-dotted border-input text-muted-foreground">
              {experiment.name}
            </span>
          }
        >
          <Facts
            rows={[
              ["App", caps.data?.app_name ?? "–"],
              ["Items", String(experiment.num_items)],
              ["Folder", <span className="break-all">{experiment.path}</span>],
            ]}
          />
        </HoverCard>
      ) : (
        data && <span className="text-muted-foreground">No experiment open</span>
      )}
    </div>
  );
}
