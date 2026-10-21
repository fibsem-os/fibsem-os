import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { ExperimentEntry, WorkflowStatus } from "./types";

const folderOf = (path: string) => path.replace(/[\\/][^\\/]*\.ya?ml$/i, "");

/** The open experiment, and its entry in the recent list (its metadata). */
export function useOpenExperiment() {
  const status = useQuery({ queryKey: ["app/workflow/status"], queryFn: () => api<WorkflowStatus>("/app/workflow/status") });
  const recent = useQuery({
    queryKey: ["app/experiment/recent"],
    queryFn: () => api<{ items: ExperimentEntry[] }>("/app/experiment/recent"),
    enabled: !!status.data?.experiment,
  });
  const open = status.data?.experiment ?? null;
  const entry = open ? recent.data?.items.find((e) => folderOf(e.path) === open.path) : undefined;
  return { status, open, entry, loading: status.isPending || (!!open && recent.isPending) };
}
