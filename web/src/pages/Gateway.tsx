import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { Connections } from "@/shell/Settings";
import { cn } from "@/lib/utils";
import { AgentCard, AgentSessions } from "@/pages/gateway/AgentCard";

type Mode = "off" | "window" | "headless";

interface GatewayState {
  mode: Mode;
  version: string;
  app: string | null;
  experiment: string | null;
  configuration: string | null;
  pid: number | null;
  since: number | null;
  who: string | null;
  via: string | null;
  switching: boolean;
  holder: { pid: number; url: string } | null;
  last_exit: { mode: Mode; app: string | null; code: number | null; at: number; killed: boolean; log: string | null } | null;
  apps: { key: string; name: string; headless: boolean }[];
  configurations: { name: string; path: string; default: boolean }[];
  last_experiment: string | null;
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

type Shown = Mode | "server";

const MODES: { mode: Shown; name: string; what: string }[] = [
  { mode: "off", name: "Off", what: "Nothing holds the microscope." },
  { mode: "window", name: "Window", what: "The app's window at this PC, as the desktop shortcut opens it." },
  { mode: "headless", name: "Headless app", what: "The app with no window, for a hub or a browser." },
  { mode: "server", name: "Microscope server", what: "The microscope only, for an app on another computer." },
];

const FIELD = "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";

function Status({ state }: { state: GatewayState }) {
  const app = state.apps.find((a) => a.key === state.app)?.name ?? state.app;
  const rows: [string, ReactNode][] = [["App", app ?? "–"]];
  if (state.mode === "headless") {
    rows.push(["Experiment", state.experiment ?? "None open"], ["Configuration", state.configuration ?? "–"]);
  }
  if (state.since) rows.push(["Started", `${fmt.when(state.since)} by ${state.who ?? "–"}, from the ${state.via ?? "–"}`]);
  return <Facts className="[&_dd]:text-left" rows={rows} />;
}

function ModeSection({ state }: { state: GatewayState }) {
  const client = useQueryClient();
  const [shown, setShown] = useState<Shown>(state.mode);
  const [app, setApp] = useState(state.apps[0]?.key ?? "");
  const [experiment, setExperiment] = useState(state.last_experiment ?? "");
  const [configuration, setConfiguration] = useState(
    state.configurations.find((c) => c.default)?.name ?? "",
  );
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
  const running = shown === state.mode;
  const stop = () => act.mutate({ path: "/gateway/stop", body: {} });
  const mode = MODES.find((m) => m.mode === shown)!;

  const appField = (
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
  );

  let body: ReactNode = null;
  let action: ReactNode = null;
  if (shown === "server") {
    body = <p className="m-0 text-[13px] text-muted-foreground">Comes in a later step.</p>;
  } else if (running && shown !== "off") {
    body = <Status state={state} />;
    action =
      shown === "headless" ? (
        <Button variant="danger" disabled={busy} onClick={stop}>
          Stop
        </Button>
      ) : (
        <span className="text-xs text-muted-foreground">The window is closed at the PC.</span>
      );
  } else if (shown === "off") {
    action =
      state.mode === "headless" ? (
        <Button variant="danger" disabled={busy} onClick={stop}>
          Stop the headless app
        </Button>
      ) : null;
  } else if (shown === "window") {
    body = <div className="flex flex-wrap gap-2.5">{appField}</div>;
    action = (
      <Button
        variant="primary"
        disabled={busy || state.mode === "window"}
        onClick={() => act.mutate({ path: "/gateway/mode", body: { mode: "window", app } })}
      >
        Open window
      </Button>
    );
  } else {
    body = (
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {appField}
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          Microscope configuration
          <select
            value={configuration}
            onChange={(event) => setConfiguration(event.target.value)}
            title={state.configurations.find((c) => c.name === configuration)?.path}
            className={FIELD}
          >
            {state.configurations.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
                {c.default ? " (default)" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted-foreground sm:col-span-2">
          Experiment folder
          <input
            value={experiment}
            onChange={(event) => setExperiment(event.target.value)}
            placeholder="None: start with no experiment open"
            className={cn(FIELD, "font-mono text-xs")}
          />
          {experiment && experiment === state.last_experiment && <span>The last experiment opened on this PC.</span>}
        </label>
      </div>
    );
    action = (
      <Button
        variant="primary"
        disabled={busy || state.mode === "window" || !chosen?.headless}
        onClick={() =>
          act.mutate({
            path: "/gateway/mode",
            body: {
              mode: "headless",
              app,
              ...(configuration ? { configuration } : {}),
              ...(experiment.trim() ? { experiment: experiment.trim() } : {}),
            },
          })
        }
      >
        Run headless
      </Button>
    );
  }

  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Mode</CardTitle>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-[180px_minmax(0,1fr)]">
        <nav aria-label="Modes" className="flex flex-row flex-wrap gap-0.5 sm:flex-col">
          {MODES.map((m) => (
            <button
              key={m.mode}
              onClick={() => setShown(m.mode)}
              className={cn(
                "flex cursor-pointer items-baseline justify-between gap-2 rounded border-0 bg-transparent px-2.5 py-2 text-left text-[13px] text-muted-foreground hover:bg-accent",
                shown === m.mode && "bg-accent text-foreground",
              )}
            >
              {m.name}
              {state.mode === m.mode && m.mode !== "off" && (
                <span
                  title="Running"
                  aria-label="Running"
                  className="size-2 shrink-0 self-center rounded-full bg-[var(--fibsem-ok-color)]"
                />
              )}
            </button>
          ))}
        </nav>
        <div className="flex min-h-[190px] min-w-0 flex-col gap-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-[14px] font-semibold">{mode.name}</span>
            <span className="text-xs text-muted-foreground">{mode.what}</span>
          </div>
          {body}
          {state.holder && (
            <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
              Another fibsem host runs here (pid {state.holder.pid}) that the gateway didn't start: close it at the PC
              before switching.
            </p>
          )}
          {act.error && <p className="m-0 text-[13px] text-destructive">{(act.error as Error).message}</p>}
          <div className="mt-auto flex flex-wrap items-center justify-end gap-2.5">
            {!running && shown !== "off" && shown !== "server" && state.mode === "headless" && (
              <span className="text-xs text-muted-foreground">Stops the headless app first.</span>
            )}
            {action}
          </div>
        </div>
      </div>
      {state.last_exit && (
        <p className="m-0 text-xs text-muted-foreground">
          Last ended: {state.last_exit.app ?? state.last_exit.mode},{" "}
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
      {switches.length === 0 && <p className="m-0 text-[13px] text-muted-foreground">No switches yet.</p>}
      {switches.map((s, i) => (
        <div key={i} className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)_auto] sm:gap-3">
          <span className="text-muted-foreground tabular-nums">{fmt.when(s.at)}</span>
          <span className="min-w-0 break-words">
            {s.from_mode} → {s.to_mode}
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
        {state && <ModeSection state={state} />}
        {state && <AgentCard appRunning={state.mode !== "off"} />}
        <Card>
          <Connections roles={["operator", "hub"]} />
        </Card>
        <Recent />
        <AgentSessions />
      </div>
    </div>
  );
}
