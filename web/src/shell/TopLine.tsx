import { useQuery } from "@tanstack/react-query";
import { Facts } from "@/components/ui/card";
import { HoverCard } from "@/components/ui/hover-card";
import { api } from "@/lib/api";
import type { Capabilities, WorkflowStatus } from "@/lib/types";

/** The run on the left, read first; the open experiment on the right. */
export function TopLine() {
  const status = useQuery({ queryKey: ["app/workflow/status"], queryFn: () => api<WorkflowStatus>("/app/workflow/status") });
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const workflow = status.data?.workflow;
  const experiment = status.data?.experiment;
  const running = workflow?.running;
  const what = running
    ? `Running: ${workflow?.current_task ?? "a task"}${workflow?.current_item ? ` on ${workflow.current_item}` : ""}`
    : status.data
      ? "Idle"
      : "";
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-b border-border px-5 py-2.5 text-[13px]">
      <span className="flex min-w-0 items-center gap-2">
        <span
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: running ? "var(--fibsem-primary-color)" : "var(--fibsem-gray-primary-color)" }}
        />
        <span className="truncate">{what}</span>
      </span>
      <div className="flex-1" />
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
        status.data && <span className="text-muted-foreground">No experiment open</span>
      )}
    </div>
  );
}
