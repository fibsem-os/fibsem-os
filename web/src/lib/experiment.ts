import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Capabilities, ExperimentEntry, WorkflowStatus } from "./types";

const folderOf = (path: string) => path.replace(/[\\/][^\\/]*\.ya?ml$/i, "");

/** The open experiment, and its entry in the recent list (its metadata). */
export function useOpenExperiment() {
  const status = useQuery({
    queryKey: ["app/workflow/status"],
    queryFn: () => api<WorkflowStatus>("/app/workflow/status"),
  });
  const recent = useQuery({
    queryKey: ["app/experiment/recent"],
    queryFn: () => api<{ items: ExperimentEntry[] }>("/app/experiment/recent"),
    enabled: !!status.data?.experiment,
  });
  const open = status.data?.experiment ?? null;
  const entry = open ? recent.data?.items.find((e) => folderOf(e.path) === open.path) : undefined;
  return { status, open, entry, loading: status.isPending || (!!open && recent.isPending) };
}

/** Whether the server runs the volume app, whose items are positions, not lamellae on grids;
 * undefined until the server says. */
export function useVolume(): boolean | undefined {
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  return caps.data ? caps.data.app_name === "Volume" : undefined;
}
