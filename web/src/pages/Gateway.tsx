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
  last_exit: {
    window: boolean;
    app: string | null;
    code: number | null;
    at: number;
    killed: boolean;
    log: string | null;
  } | null;
  apps: { key: string; name: string; headless: boolean }[];
  last_experiment: string | null;
  connection: string[];
  configurations: { name: string; default: boolean }[];
  newer: {
    latest: string | null;
    behind: number | null;
    branch: string;
  } | null;
}

/** What the next start connects: the page's choice, or the gateway's own
 * flags when it was given some. */
interface MicroscopeChoice {
  connect: boolean;
  configuration: string;
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

const FIELD =
  "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";

/** What is newer than the code the gateway runs: a release, or commits on main. */
function newerLabel(
  newer: GatewayState["newer"],
): { text: string; detail: string } | null {
  if (newer?.latest)
    return {
      text: `fibsemOS ${newer.latest} available`,
      detail: "A newer release is on PyPI.",
    };
  if (newer?.behind)
    return {
      text: "Update available",
      detail: `${newer.behind} newer change${newer.behind === 1 ? "" : "s"} on GitHub (${newer.branch}).`,
    };
  return null;
}

function Status({ state }: { state: GatewayState }) {
  const app = state.apps.find((a) => a.key === state.app)?.name ?? state.app;
  const rows: [string, ReactNode][] = [
    ["App", app ?? "–"],
    ["Window", state.window ? "On this PC" : "None"],
  ];
  if (!state.window || state.experiment)
    rows.push(["Experiment", state.experiment ?? "None open"]);
  if (state.since)
    rows.push([
      "Started",
      `${fmt.when(state.since)} by ${state.who ?? "–"}, from the ${state.via ?? "–"}`,
    ]);
  return <Facts className="[&_dd]:text-left" rows={rows} />;
}

/** Before the app starts: which configuration it connects as it starts. Once
 * it runs, the Connection card takes this place. */
function MicroscopeStart({
  state,
  choice,
  setChoice,
}: {
  state: GatewayState;
  choice: MicroscopeChoice;
  setChoice: (choice: MicroscopeChoice) => void;
}) {
  const fallback =
    state.configurations.find((c) => c.default)?.name ??
    state.configurations[0]?.name ??
    "";
  const configuration = choice.configuration || fallback;
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Microscope</CardTitle>
      {state.connection.length > 0 ? (
        <p className="m-0 text-[13px] text-muted-foreground">
          The app connects as it starts, with this gateway's flags:{" "}
          {state.connection.join(" ")}.
        </p>
      ) : (
        <>
          <label className="flex max-w-[320px] flex-col gap-1 text-xs text-muted-foreground">
            Configuration
            <select
              value={configuration}
              onChange={(event) =>
                setChoice({ ...choice, configuration: event.target.value })
              }
              className={FIELD}
            >
              {state.configurations.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.default ? `${c.name} (default)` : c.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-[13px]">
            <input
              type="checkbox"
              checked={choice.connect}
              onChange={(event) =>
                setChoice({ ...choice, connect: event.target.checked })
              }
            />
            Connect when the app starts
          </label>
          {!choice.connect && (
            <p className="m-0 text-xs text-muted-foreground">
              The app starts with no microscope: connect one once it is up.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

interface Folders {
  path: string;
  parent: string | null;
  experiment: boolean;
  default: string;
  folders: { name: string; path: string; experiment: boolean }[];
}

/** The folders on this PC, from the default experiments folder: an
 * experiment is picked (open), or the folder a new one goes in (new). */
function FolderBrowser({
  start,
  pick,
  newExperiment,
  close,
}: {
  start: string;
  pick: (path: string) => void;
  newExperiment: boolean;
  close: () => void;
}) {
  const [path, setPath] = useState(start);
  const listed = useQuery({
    queryKey: ["gateway/folders", path],
    queryFn: () =>
      api<Folders>(
        `/gateway/folders${path ? `?path=${encodeURIComponent(path)}` : ""}`,
      ),
    retry: false,
  });
  const here = listed.data;
  return (
    <div className="flex flex-col gap-1.5 rounded border border-input p-2 text-[13px]">
      <div className="flex items-center gap-2">
        <Button
          disabled={!here?.parent}
          onClick={() => here?.parent && setPath(here.parent)}
        >
          Up
        </Button>
        <span
          className="min-w-0 flex-1 truncate font-mono text-xs"
          title={here?.path}
        >
          {here?.path ?? path}
        </span>
        {here && here.path !== here.default && (
          <Button onClick={() => setPath(here.default)}>Experiments</Button>
        )}
      </div>
      {listed.error && (
        <p className="m-0 text-destructive">
          {(listed.error as Error).message}
        </p>
      )}
      <ul className="m-0 max-h-56 list-none overflow-y-auto p-0">
        {here?.folders.map((f) => (
          <li key={f.path}>
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded px-1.5 py-1 text-left hover:bg-muted"
              onClick={() =>
                f.experiment && !newExperiment ? pick(f.path) : setPath(f.path)
              }
            >
              <span className="min-w-0 flex-1 truncate">{f.name}</span>
              {f.experiment && (
                <span className="text-xs text-muted-foreground">
                  experiment
                </span>
              )}
            </button>
          </li>
        ))}
        {here && here.folders.length === 0 && (
          <li className="px-1.5 py-1 text-muted-foreground">No folders</li>
        )}
      </ul>
      <div className="flex justify-end gap-2">
        <Button onClick={close}>Cancel</Button>
        {here && (newExperiment || here.experiment) && (
          <Button variant="primary" onClick={() => pick(here.path)}>
            {newExperiment ? "Use this folder" : "Open this experiment"}
          </Button>
        )}
      </div>
    </div>
  );
}

function Starting({
  state,
  start,
  busy,
  microscope,
}: {
  state: GatewayState;
  start: (body: unknown) => void;
  busy: boolean;
  microscope: Record<string, unknown>;
}) {
  const [app, setApp] = useState(state.apps[0]?.key ?? "");
  const [window, setWindow] = useState(true);
  const [creating, setCreating] = useState(false);
  const [experiment, setExperiment] = useState(state.last_experiment ?? "");
  const [name, setName] = useState("");
  const [parent, setParent] = useState("");
  const [browsing, setBrowsing] = useState(false);
  const chosen = state.apps.find((a) => a.key === app);
  // Only an app with a host (a headless command) opens an experiment as it starts.
  const host = !!chosen?.headless;
  // A new experiment goes in the default experiments folder unless another is named.
  const defaults = useQuery({
    queryKey: ["gateway/folders", ""],
    queryFn: () => api<Folders>("/gateway/folders"),
    enabled: host && creating,
    retry: false,
  });
  const into = parent.trim() || defaults.data?.default || "";
  const badName = /[\\/]/.test(name);
  const newFolder =
    into && name.trim() && !badName
      ? `${into.replace(/[\\/]+$/, "")}${into.includes("\\") ? "\\" : "/"}${name.trim()}`
      : "";
  const experimentBody = !host
    ? {}
    : creating
      ? { experiment: newFolder, create: true }
      : experiment.trim()
        ? { experiment: experiment.trim() }
        : {};
  const ready = !host || !creating || !!newFolder;
  return (
    <>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-[200px_minmax(0,1fr)]">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          App
          <select
            value={app}
            onChange={(event) => setApp(event.target.value)}
            className={FIELD}
          >
            {state.apps.map((a) => (
              <option key={a.key} value={a.key}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        {host && (
          <div className="flex flex-col gap-1 text-xs text-muted-foreground">
            <div className="flex items-center gap-3">
              Experiment
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  checked={!creating}
                  onChange={() => setCreating(false)}
                />
                Open
              </label>
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  checked={creating}
                  onChange={() => setCreating(true)}
                />
                New
              </label>
            </div>
            {creating ? (
              <div className="flex gap-2">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Name"
                  aria-label="New experiment name"
                  className={cn(FIELD, "w-40")}
                />
                <input
                  value={parent}
                  onChange={(event) => setParent(event.target.value)}
                  placeholder={defaults.data?.default ?? "Folder"}
                  aria-label="New experiment folder"
                  className={cn(FIELD, "min-w-0 flex-1 font-mono text-xs")}
                />
                <Button onClick={() => setBrowsing(true)}>Browse</Button>
              </div>
            ) : (
              <div className="flex gap-2">
                <input
                  value={experiment}
                  onChange={(event) => setExperiment(event.target.value)}
                  placeholder="None: start with no experiment open"
                  aria-label="Experiment folder"
                  className={cn(FIELD, "min-w-0 flex-1 font-mono text-xs")}
                />
                <Button onClick={() => setBrowsing(true)}>Browse</Button>
              </div>
            )}
            {!creating &&
              experiment &&
              experiment === state.last_experiment && (
                <span>The last experiment opened on this PC.</span>
              )}
            {creating && badName && (
              <span className="text-destructive">A name, not a path.</span>
            )}
            {browsing && (
              <FolderBrowser
                start={
                  creating
                    ? into
                    : experiment
                      ? experiment.replace(/[\\/][^\\/]*[\\/]?$/, "")
                      : ""
                }
                newExperiment={creating}
                close={() => setBrowsing(false)}
                pick={(path) => {
                  if (creating) setParent(path);
                  else setExperiment(path);
                  setBrowsing(false);
                }}
              />
            )}
          </div>
        )}
      </div>
      <label className="flex items-center gap-2 text-[13px]">
        <input
          type="checkbox"
          checked={window}
          onChange={(event) => setWindow(event.target.checked)}
        />
        Show the window on this PC
      </label>
      {!window && !host && (
        <p className="m-0 text-xs text-muted-foreground">
          {chosen?.name} runs only with its window.
        </p>
      )}
      <div className="flex justify-end">
        <Button
          variant="primary"
          disabled={busy || !chosen || (!window && !host) || !ready}
          onClick={() =>
            start({ app, window, ...microscope, ...experimentBody })
          }
        >
          Start
        </Button>
      </div>
    </>
  );
}

function AppSection({
  state,
  microscope,
}: {
  state: GatewayState;
  microscope: Record<string, unknown>;
}) {
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
  const [confirmClose, setConfirmClose] = useState(false);
  const stop = () => {
    setConfirmClose(false);
    act.mutate({ path: "/gateway/stop", body: {} });
  };
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>App</CardTitle>
      {state.mode === "off" ? (
        <Starting
          state={state}
          busy={busy}
          microscope={microscope}
          start={(body) => act.mutate({ path: "/gateway/start", body })}
        />
      ) : (
        <>
          <Status state={state} />
          {confirmClose ? (
            <>
              <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
                Close the app's window on this PC? Anyone at the PC sees it
                close.
              </p>
              <div className="flex justify-end gap-2.5">
                <Button onClick={() => setConfirmClose(false)}>Cancel</Button>
                <Button variant="danger" disabled={busy} onClick={stop}>
                  Close the window
                </Button>
              </div>
            </>
          ) : (
            <div className="flex justify-end">
              <Button
                variant="danger"
                disabled={busy}
                onClick={state.window ? () => setConfirmClose(true) : stop}
              >
                {act.isPending
                  ? "Stopping…"
                  : state.window
                    ? "Close the window"
                    : "Stop"}
              </Button>
            </div>
          )}
        </>
      )}
      {state.holder && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          Another fibsem host runs here (pid {state.holder.pid}) that the
          gateway didn't start: close it at the PC before starting the app.
        </p>
      )}
      {act.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(act.error as Error).message}
        </p>
      )}
      {state.last_exit && (
        <p className="m-0 text-xs text-muted-foreground">
          Last ended: {state.last_exit.app ?? "the app"},{" "}
          {state.last_exit.killed
            ? "killed"
            : `exit ${state.last_exit.code ?? "–"}`}
          , {fmt.when(state.last_exit.at)}
          {state.last_exit.log && ` · log ${state.last_exit.log}`}
        </p>
      )}
    </Card>
  );
}

/** One line of the history in words: what happened to which app. */
function happened(s: Switch, name: string): string {
  if (s.result.startsWith("refused"))
    return `Refused: ${s.result.replace(/^refused:\s*/, "")}`;
  if (s.result === "started")
    return `Started ${name}${s.window === true ? " with its window" : s.window === false ? " without a window" : ""}`;
  if (s.result === "stopped")
    return s.window ? `Closed ${name}'s window` : `Stopped ${name}`;
  if (s.result === "closed") return `${name}'s window was closed at the PC`;
  const exit = s.result.match(/^exited \((.*)\)$/);
  if (exit) return `${name} ended on its own (exit ${exit[1]})`;
  return s.result;
}

function Recent({ apps }: { apps: GatewayState["apps"] }) {
  const history = useQuery({
    queryKey: ["gateway/history"],
    queryFn: () => api<{ switches: Switch[] }>("/gateway/history"),
    refetchInterval: 5000,
  });
  const switches = history.data?.switches ?? [];
  return (
    <Card className="flex flex-col gap-2">
      <CardTitle>Recent</CardTitle>
      {switches.length === 0 && (
        <p className="m-0 text-[13px] text-muted-foreground">
          No starts or stops yet.
        </p>
      )}
      {switches.map((s, i) => (
        <div
          key={i}
          className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)_auto] sm:gap-3"
        >
          <span className="text-muted-foreground tabular-nums">
            {fmt.when(s.at)}
          </span>
          <span
            className={cn(
              "min-w-0 break-words",
              s.result.startsWith("refused") &&
                "text-[var(--fibsem-semantic-warning-color)]",
            )}
          >
            {happened(
              s,
              apps.find((a) => a.key === s.app)?.name ?? s.app ?? "the app",
            )}
            {s.experiment && (
              <span className="text-muted-foreground" title={s.experiment}>
                {" "}
                · {s.experiment.split(/[\\/]/).filter(Boolean).pop()}
              </span>
            )}
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
  const newer = state ? newerLabel(state.newer) : null;
  const [choice, setChoice] = useState<MicroscopeChoice>({
    connect: true,
    configuration: "",
  });
  // The gateway's own flags stand when it has some; else the page's choice.
  const microscope: Record<string, unknown> =
    !state || state.connection.length > 0
      ? {}
      : choice.connect
        ? {
            connect: true,
            ...(choice.configuration
              ? { configuration: choice.configuration }
              : {}),
          }
        : { connect: false };
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
          <h1 className="m-0 text-[17px] font-semibold">Agent Gateway</h1>
          <span className="text-[13px] text-muted-foreground">
            {location.host}
            {state && ` · fibsemOS ${state.version}`}
          </span>
          {newer && (
            <span
              className="ml-auto text-[13px] text-muted-foreground"
              title={newer.detail}
            >
              {newer.text}
            </span>
          )}
        </div>
        {gateway.error && (
          <p className="m-0 text-[13px] text-destructive">
            {(gateway.error as Error).message}
          </p>
        )}
        {running && <ConnectionCard base="/gateway" title="Microscope" />}
        {state && !running && (
          <MicroscopeStart
            state={state}
            choice={choice}
            setChoice={setChoice}
          />
        )}
        {state && <AppSection state={state} microscope={microscope} />}
        {state && (
          <AgentCard
            appRunning={running}
            connected={running && !!connection.data?.connected}
          />
        )}
        <Card>
          <Connections roles={["operator", "hub"]} />
        </Card>
        <Recent apps={state?.apps ?? []} />
        <AgentSessions
          appRunning={running}
          connected={running && !!connection.data?.connected}
        />
      </div>
    </div>
  );
}
