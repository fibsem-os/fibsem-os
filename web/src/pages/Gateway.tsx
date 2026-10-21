import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { Connections } from "@/shell/Settings";
import { cn } from "@/lib/utils";

type Mode = "off" | "window" | "headless";

interface GatewayState {
  mode: Mode;
  version: string;
  app: string | null;
  experiment: string | null;
  pid: number | null;
  since: number | null;
  who: string | null;
  via: string | null;
  switching: boolean;
  holder: { pid: number; url: string } | null;
  last_exit: { mode: Mode; app: string | null; code: number | null; at: number; killed: boolean; log: string | null } | null;
  apps: { key: string; name: string; headless: boolean }[];
}

interface Switch {
  at: number;
  to_mode: Mode;
  from_mode: Mode;
  who: string;
  via: string;
  result: string;
  app: string | null;
  experiment: string | null;
}

const MODES: { mode: Mode | "server"; name: string; what: string }[] = [
  { mode: "off", name: "Off", what: "Nothing holds the microscope." },
  { mode: "window", name: "Window", what: "The app's window at this PC, as a desktop launch opens it." },
  { mode: "headless", name: "Headless app", what: "The app with no window, for a hub or a browser." },
  { mode: "server", name: "Microscope server", what: "The microscope only, for an app on another computer. Later." },
];

const when = (seconds: number) =>
  new Date(seconds * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

function ModeCards({ state }: { state: GatewayState }) {
  const client = useQueryClient();
  const [app, setApp] = useState(state.apps[0]?.key ?? "");
  const [experiment, setExperiment] = useState("");
  const act = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) =>
      api<GatewayState>(path, { method: "POST", body: JSON.stringify(body) }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway"], answer);
      client.invalidateQueries({ queryKey: ["gateway/history"] });
    },
  });
  const busy = state.switching || act.isPending;
  const chosen = state.apps.find((a) => a.key === app);
  const start = (mode: Mode) =>
    act.mutate({
      path: "/gateway/mode",
      body: { mode, app, ...(experiment.trim() ? { experiment: experiment.trim() } : { last: true }) },
    });

  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Mode</CardTitle>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        {MODES.map(({ mode, name, what }) => {
          const on = state.mode === mode;
          return (
            <div
              key={mode}
              className={cn(
                "flex flex-col gap-1 rounded-md border border-border bg-background px-3.5 py-3 text-[13px]",
                on && "border-primary shadow-[inset_0_0_0_1px_var(--color-primary)]",
                mode === "server" && "opacity-50",
              )}
            >
              <span className="font-semibold">{name}</span>
              <span className="text-xs text-muted-foreground">{what}</span>
              {on && mode !== "off" && (
                <span className="text-xs">
                  {[state.app, state.experiment, state.who && `by ${state.who}`, state.since && when(state.since)]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              )}
            </div>
          );
        })}
      </div>
      {state.holder && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          Another App Server runs here (pid {state.holder.pid}) that the gateway didn't start: close it at the PC before
          switching.
        </p>
      )}
      <div className="flex flex-wrap items-end gap-2.5">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          App
          <select
            value={app}
            onChange={(event) => setApp(event.target.value)}
            className="rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground"
          >
            {state.apps.map((a) => (
              <option key={a.key} value={a.key}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-[260px] flex-1 flex-col gap-1 text-xs text-muted-foreground">
          Experiment folder (headless)
          <input
            value={experiment}
            onChange={(event) => setExperiment(event.target.value)}
            placeholder="The last experiment when left empty"
            className="rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground"
          />
        </label>
        <Button disabled={busy || state.mode === "window"} onClick={() => start("window")}>
          Open window
        </Button>
        <Button
          variant="primary"
          disabled={busy || state.mode === "window" || !chosen?.headless}
          onClick={() => start("headless")}
        >
          Run headless
        </Button>
        <Button
          variant="danger"
          disabled={busy || state.mode !== "headless"}
          onClick={() => act.mutate({ path: "/gateway/stop", body: {} })}
        >
          Stop
        </Button>
      </div>
      {state.mode === "window" && (
        <p className="m-0 text-xs text-muted-foreground">The window is closed at the PC, never from here.</p>
      )}
      {act.error && <p className="m-0 text-[13px] text-destructive">{(act.error as Error).message}</p>}
      {state.last_exit && (
        <p className="m-0 text-xs text-muted-foreground">
          Last ended: {state.last_exit.app ?? state.last_exit.mode},{" "}
          {state.last_exit.killed ? "killed" : `exit ${state.last_exit.code ?? "–"}`}, {when(state.last_exit.at)}
          {state.last_exit.log && ` · log ${state.last_exit.log}`}
        </p>
      )}
    </Card>
  );
}

function Recent() {
  const history = useQuery({
    queryKey: ["gateway/history"],
    queryFn: () => api<{ switches: Switch[] }>("/gateway/history"),
    refetchInterval: 5000,
  });
  const switches = history.data?.switches ?? [];
  return (
    <Card className="flex flex-col gap-2">
      <CardTitle>Recent</CardTitle>
      {switches.length === 0 && <p className="m-0 text-[13px] text-muted-foreground">No switches yet.</p>}
      {switches.map((s, i) => (
        <div key={i} className="grid grid-cols-[150px_minmax(0,1fr)_auto] gap-3 border-t border-border py-2 text-[13px]">
          <span className="text-muted-foreground tabular-nums">{when(s.at)}</span>
          <span className="min-w-0 break-words">
            {s.from_mode} → {s.to_mode}
            {s.app && ` · ${s.app}`}
            {s.experiment && ` · ${s.experiment}`}
            <span className={cn("text-muted-foreground", s.result.startsWith("refused") && "text-[var(--fibsem-semantic-warning-color)]")}>
              {" "}
              · {s.result}
            </span>
          </span>
          <span className="text-muted-foreground">
            {s.who} · {s.via}
          </span>
        </div>
      ))}
    </Card>
  );
}

/** The gateway's own page: its mode, who is paired, and what switched. */
export function GatewayPage() {
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => api<GatewayState>("/gateway"),
    refetchInterval: 2000,
  });
  const state = gateway.data;
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto flex max-w-[1100px] flex-col gap-4 px-4 py-5">
        <div className="flex items-baseline gap-3">
          <h1 className="m-0 text-[17px] font-semibold">Gateway</h1>
          <span className="text-[13px] text-muted-foreground">
            {location.host}
            {state && ` · fibsemOS ${state.version}`}
          </span>
        </div>
        {gateway.error && <p className="m-0 text-[13px] text-destructive">{(gateway.error as Error).message}</p>}
        {state && <ModeCards state={state} />}
        <Card>
          <Connections roles={["operator", "hub"]} />
        </Card>
        <Recent />
      </div>
    </div>
  );
}
