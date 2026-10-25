import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { Connections } from "@/shell/Settings";
import { cn } from "@/lib/utils";
import type { Connection } from "@/lib/types";
import { AgentCard, AgentSessions } from "@/pages/gateway/AgentCard";
import { ConnectionCard } from "@/pages/Microscope";

type Mode = "off" | "running";

interface GatewayState {
  mode: Mode;
  version: string;
  app: string | null;
  window: boolean;
  experiment: string | null;
  pid: number | null;
  since: number | null;
  who: string | null;
  via: string | null;
  switching: boolean;
  holder: { pid: number; url: string } | null;
  last_exit: { window: boolean; app: string | null; code: number | null; at: number; killed: boolean; log: string | null } | null;
  apps: { key: string; name: string; headless: boolean }[];
  last_experiment: string | null;
  connection: string[];
}

interface Switch {
  at: number;
  // older lines name the modes before the app had one start: window, headless
  to_mode: string;
  from_mode: string;
  who: string;
  via: string;
  result: string;
  app: string | null;
  experiment: string | null;
  window?: boolean | null;
}

const FIELD = "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";

function Status({ state }: { state: GatewayState }) {
  const app = state.apps.find((a) => a.key === state.app)?.name ?? state.app;
  const rows: [string, ReactNode][] = [
    ["App", app ?? "–"],
    ["Window", state.window ? "On this PC" : "None"],
  ];
  if (!state.window || state.experiment) rows.push(["Experiment", state.experiment ?? "None open"]);
  if (state.since) rows.push(["Started", `${fmt.when(state.since)} by ${state.who ?? "–"}, from the ${state.via ?? "–"}`]);
  return <Facts className="[&_dd]:text-left" rows={rows} />;
}

function Starting({ state, start, busy }: { state: GatewayState; start: (body: unknown) => void; busy: boolean }) {
  const [app, setApp] = useState(state.apps[0]?.key ?? "");
  const [window, setWindow] = useState(true);
  const [experiment, setExperiment] = useState(state.last_experiment ?? "");
  const chosen = state.apps.find((a) => a.key === app);
  // Only an app with a host (a headless command) opens an experiment as it starts.
  const host = !!chosen?.headless;
  return (
    <>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-[200px_minmax(0,1fr)]">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          App
          <select value={app} onChange={(event) => setApp(event.target.value)} className={FIELD}>
            {state.apps.map((a) => (
              <option key={a.key} value={a.key}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        {host && (
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Experiment folder
            <input
              value={experiment}
              onChange={(event) => setExperiment(event.target.value)}
              placeholder="None: start with no experiment open"
              className={cn(FIELD, "font-mono text-xs")}
            />
            {experiment && experiment === state.last_experiment && <span>The last experiment opened on this PC.</span>}
          </label>
        )}
      </div>
      <label className="flex items-center gap-2 text-[13px]">
        <input type="checkbox" checked={window} onChange={(event) => setWindow(event.target.checked)} />
        Show the window on this PC
      </label>
      {!window && !host && <p className="m-0 text-xs text-muted-foreground">{chosen?.name} runs only with its window.</p>}
      <p className="m-0 text-xs text-muted-foreground">
        {state.connection.length > 0
          ? `Connects as it starts: ${state.connection.join(" ")}.`
          : "Starts with no microscope: connect one once it is up."}
        {window && " A window is closed at this PC only."}
      </p>
      <div className="flex justify-end">
        <Button
          variant="primary"
          disabled={busy || !chosen || (!window && !host)}
          onClick={() =>
            start({ app, window, ...(host && experiment.trim() ? { experiment: experiment.trim() } : {}) })
          }
        >
          Start
        </Button>
      </div>
    </>
  );
}

function AppSection({ state }: { state: GatewayState }) {
  const client = useQueryClient();
  const act = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) =>
      api<GatewayState>(path, { method: "POST", body: JSON.stringify(body) }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway"], answer);
      client.invalidateQueries({ queryKey: ["gateway/history"] });
    },
  });
  const busy = state.switching || act.isPending;
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>App</CardTitle>
      {state.mode === "off" ? (
        <Starting state={state} busy={busy} start={(body) => act.mutate({ path: "/gateway/start", body })} />
      ) : (
        <>
          <Status state={state} />
          <div className="flex justify-end">
            {state.window ? (
              <span className="text-xs text-muted-foreground">The window is closed at the PC.</span>
            ) : (
              <Button variant="danger" disabled={busy} onClick={() => act.mutate({ path: "/gateway/stop", body: {} })}>
                Stop
              </Button>
            )}
          </div>
        </>
      )}
      {state.holder && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          Another fibsem host runs here (pid {state.holder.pid}) that the gateway didn't start: close it at the PC
          before starting the app.
        </p>
      )}
      {act.error && <p className="m-0 text-[13px] text-destructive">{(act.error as Error).message}</p>}
      {state.last_exit && (
        <p className="m-0 text-xs text-muted-foreground">
          Last ended: {state.last_exit.app ?? "the app"},{" "}
          {state.last_exit.killed ? "killed" : `exit ${state.last_exit.code ?? "–"}`}, {fmt.when(state.last_exit.at)}
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
      {switches.length === 0 && <p className="m-0 text-[13px] text-muted-foreground">No starts or stops yet.</p>}
      {switches.map((s, i) => (
        <div key={i} className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)_auto] sm:gap-3">
          <span className="text-muted-foreground tabular-nums">{fmt.when(s.at)}</span>
          <span className="min-w-0 break-words">
            {s.from_mode} → {s.to_mode}
            {s.window === true && " (window)"}
            {s.window === false && " (no window)"}
            {s.app && ` · ${s.app}`}
            {s.experiment && <span title={s.experiment}> · {s.experiment.split(/[\\/]/).filter(Boolean).pop()}</span>}
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

/** The gateway's own page: the app, its microscope, the agent, who is paired,
 * and what started and stopped. */
export function GatewayPage() {
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => api<GatewayState>("/gateway"),
    refetchInterval: 2000,
  });
  const state = gateway.data;
  const running = state?.mode === "running";
  const connection = useQuery({
    queryKey: ["/gateway/microscope/connection"],
    queryFn: () => api<Connection>("/gateway/microscope/connection"),
    enabled: running,
    retry: false,
    refetchInterval: 5000,
  });
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
        {state && <AppSection state={state} />}
        {running && <ConnectionCard base="/gateway" />}
        {state && <AgentCard appRunning={running} connected={running && !!connection.data?.connected} />}
        <Card>
          <Connections roles={["operator", "hub"]} />
        </Card>
        <Recent />
        <AgentSessions />
      </div>
    </div>
  );
}
