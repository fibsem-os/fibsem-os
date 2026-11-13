import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useEffect, useState } from "react";
import { Loader, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, getToken } from "@/lib/api";
import * as fmt from "@/lib/format";
import { Connections } from "@/shell/Settings";
import { cn } from "@/lib/utils";
import type { Connection } from "@/lib/types";
import {
  AgentCard,
  type AgentState,
  notReady,
} from "@/pages/gateway/AgentCard";
import { History } from "@/pages/gateway/History";
import { Instruments } from "@/pages/gateway/Instruments";
import { type Defaults, Settings } from "@/pages/gateway/Settings";
// The same icon as the gateway's window and its Desktop shortcut.
import gatewayIcon from "../../../fibsem/ui/icons/gateway.svg?raw";

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
  /** Which switch is in progress, from any page, tray or hub. */
  switch?: "starting" | "stopping" | null;
  other_host: { pid: number; url: string } | null;
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
  recent_experiments: { name: string; folder: string; app: string | null }[];
  experiments_folder: string | null;
  app_url: string | null;
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

/** The name the app's own New dialog suggests: the app and the time,
 * e.g. AutoLamella-2026-11-06-12-30. */
function defaultName(app: string): string {
  const now = new Date();
  const two = (n: number) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}-${two(now.getMonth() + 1)}-${two(now.getDate())}-${two(now.getHours())}-${two(now.getMinutes())}`;
  return `${app.replace(/\s+/g, "")}-${stamp}`;
}

function Starting({
  state,
  start,
  busy,
  microscope,
  defaults,
}: {
  state: GatewayState;
  start: (body: unknown) => void;
  busy: boolean;
  microscope: Record<string, unknown>;
  defaults?: Defaults;
}) {
  const [app, setApp] = useState(
    state.apps.find((a) => a.key === defaults?.app)?.key ??
      state.apps[0]?.key ??
      "",
  );
  const [window, setWindow] = useState(defaults?.window ?? true);
  const [creating, setCreating] = useState(false);
  const [experiment, setExperiment] = useState(state.last_experiment ?? "");
  const [name, setName] = useState("");
  const chosen = state.apps.find((a) => a.key === app);
  // Only an app with a host (a headless command) opens an experiment as it starts.
  const host = !!chosen?.headless;
  // This app's recent experiments (one that doesn't say which app wrote it too).
  const recent = state.recent_experiments.filter(
    (r) =>
      !r.app ||
      !chosen ||
      [chosen.key, chosen.name].some(
        (n) => n.toLowerCase() === r.app!.toLowerCase(),
      ),
  );
  // A new experiment goes in the experiments folder.
  const into = state.experiments_folder ?? "";
  const badName = /[\\/]/.test(name);
  const newFolder =
    into && name.trim() && !badName
      ? `${into.replace(/[\\/]+$/, "")}${into.includes("\\") ? "\\" : "/"}${name.trim()}`
      : "";
  const experimentBody = !host
    ? {}
    : creating
      ? { experiment: newFolder, create: true }
      : experiment
        ? { experiment }
        : {};
  const ready = !host || !creating || !!newFolder;
  return (
    <>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        App
        <select
          value={app}
          onChange={(event) => {
            setApp(event.target.value);
            setExperiment("");
          }}
          className={cn(FIELD, "sm:w-60")}
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
          Experiment
          <div className="flex gap-2">
            <select
              value={creating ? "new" : "load"}
              onChange={(event) => {
                const making = event.target.value === "new";
                setCreating(making);
                if (making && !name.trim())
                  setName(defaultName(chosen?.name ?? "Experiment"));
              }}
              aria-label="Load or new experiment"
              className={FIELD}
            >
              <option value="load">Load</option>
              <option value="new">New</option>
            </select>
            {creating ? (
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Name"
                aria-label="New experiment name"
                className={cn(FIELD, "min-w-0 flex-1")}
              />
            ) : (
              <select
                value={experiment}
                onChange={(event) => setExperiment(event.target.value)}
                aria-label="Recent experiment"
                className={cn(FIELD, "min-w-0 flex-1")}
              >
                <option value="">None</option>
                {recent.map((r) => (
                  <option key={r.folder} value={r.folder} title={r.folder}>
                    {r.name}
                  </option>
                ))}
              </select>
            )}
          </div>
          {creating && into && (
            <span className="truncate" title={into}>
              In {into}
            </span>
          )}
          {!creating && experiment && (
            <span className="truncate" title={experiment}>
              {experiment}
            </span>
          )}
          {creating && badName && (
            <span className="text-destructive">A name, not a path.</span>
          )}
        </div>
      )}
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
          {busy ? "Starting…" : "Start the app"}
        </Button>
      </div>
    </>
  );
}

/** While the app starts: what it is waiting for, and for how long. */
function StartingLine({ state }: { state: GatewayState }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);
  const app = state.apps.find((a) => a.key === state.app)?.name ?? "the app";
  const secs = state.since
    ? Math.max(0, Math.round(now / 1000 - state.since))
    : 0;
  return (
    <p className="m-0 flex items-center gap-2 text-[13px] text-muted-foreground">
      <Loader size={14} className="shrink-0 animate-spin" />
      {state.window
        ? `Starting ${app}: its window opens on this PC`
        : `Starting ${app}`}
      … {secs} s{secs > 90 && " (slower than usual)"}
    </p>
  );
}

const WARN = "text-[var(--fibsem-semantic-warning-color)]";

function Dot({ colour }: { colour: "ok" | "warn" | "off" }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-2 shrink-0 rounded-full",
        colour === "ok"
          ? "bg-[var(--fibsem-ok-color)]"
          : colour === "warn"
            ? "bg-[var(--fibsem-semantic-warning-color)]"
            : "bg-[var(--fibsem-pending-color)]",
      )}
    />
  );
}

function SubHeading({
  first,
  children,
}: {
  first?: boolean;
  children: string;
}) {
  return (
    <h3
      className={cn(
        "m-0 text-sm font-semibold text-foreground",
        !first && "mt-1 border-t border-border pt-3",
      )}
    >
      {children}
    </h3>
  );
}

/** Before the app starts: which configuration it connects as it starts. */
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
  if (state.connection.length > 0)
    return (
      <p className="m-0 text-[13px] text-muted-foreground">
        The app connects as it starts, with this gateway's flags:{" "}
        {state.connection.join(" ")}.
      </p>
    );
  return (
    <>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
        <label className="flex w-full max-w-[320px] flex-col gap-1 text-xs text-muted-foreground">
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
        <label className="flex items-center gap-2 pb-1.5 text-[13px]">
          <input
            type="checkbox"
            checked={choice.connect}
            onChange={(event) =>
              setChoice({ ...choice, connect: event.target.checked })
            }
          />
          Connect when the app starts
        </label>
      </div>
      {!choice.connect && (
        <p className="m-0 text-xs text-muted-foreground">
          The app starts with no microscope: connect one once it is up.
        </p>
      )}
    </>
  );
}

/** How long the app has been up: "up 2 h 14 min". */
function uptime(since: number | null, now: number): string | null {
  if (!since) return null;
  const minutes = Math.max(0, Math.floor((now / 1000 - since) / 60));
  if (minutes < 60) return `up ${minutes} min`;
  return `up ${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** The running app's microscope in one row: connected or not, and Connect
 * or Disconnect through the gateway. */
function MicroscopeRow({
  connection,
  starting,
}: {
  connection: Connection | undefined;
  starting: boolean;
}) {
  const client = useQueryClient();
  const [chosen, setChosen] = useState<string | null>(null);
  const act = useMutation({
    mutationFn: (path: string) =>
      api<Connection>(`/gateway/microscope/${path}`, {
        method: "POST",
        body: JSON.stringify(
          path === "connect" ? { configuration: chosen ?? undefined } : {},
        ),
      }),
    onSettled: () =>
      client.invalidateQueries({
        queryKey: ["/gateway/microscope/connection"],
      }),
  });
  const c = connection?.configuration;
  const busy =
    act.isPending ||
    !!connection?.connecting ||
    connection?.run?.state === "running";
  const fallback =
    connection?.configurations.find((o) => o.default)?.name ??
    connection?.configurations[0]?.name ??
    "";
  const failed =
    (act.error as Error | null)?.message ??
    (connection && !connection.connected && !busy ? connection.error : null);
  let line: ReactNode;
  let colour: "ok" | "warn" | "off" = "off";
  let action: ReactNode = null;
  if (!connection) {
    line = (
      <span className="text-muted-foreground">
        {starting ? "Waiting for the app…" : "The app doesn't answer"}
      </span>
    );
  } else if (connection.connected && c) {
    colour = "ok";
    const name = [c.manufacturer, c.model].filter(Boolean).join(" ");
    line = (
      <>
        {name || "Microscope"} connected
        <span className="text-muted-foreground">
          {[c.name ?? c.path, c.address].filter(Boolean).map((t) => ` · ${t}`)}
        </span>
      </>
    );
    action = (
      <Button disabled={busy} onClick={() => act.mutate("disconnect")}>
        Disconnect
      </Button>
    );
  } else if (busy) {
    colour = "warn";
    line = <>{connection.connecting ?? "Connecting…"}</>;
  } else {
    line = <span className="text-muted-foreground">Not connected</span>;
    action = (
      <span className="flex items-center gap-2">
        <select
          value={chosen ?? fallback}
          onChange={(event) => setChosen(event.target.value)}
          aria-label="Configuration"
          className={FIELD}
        >
          {connection.configurations.map((o) => (
            <option key={o.name} value={o.name}>
              {o.name}
            </option>
          ))}
        </select>
        <Button
          variant="primary"
          disabled={!(chosen ?? fallback)}
          onClick={() => act.mutate("connect")}
        >
          Connect
        </Button>
      </span>
    );
  }
  return (
    <div className="flex flex-col gap-1.5 py-2.5">
      <div className="grid grid-cols-1 items-center gap-2 text-[13px] sm:grid-cols-[96px_minmax(0,1fr)_auto] sm:gap-3">
        <span className="text-muted-foreground">Microscope</span>
        <span className="flex min-w-0 items-center gap-2">
          <Dot colour={colour} />
          <span className="min-w-0 break-words">{line}</span>
        </span>
        {action}
      </div>
      {failed && <p className="m-0 text-[13px] text-destructive">{failed}</p>}
    </div>
  );
}

/** The running app in one row: which app, its window and experiment, how
 * long it has been up; Open the app, and Stop (or Close the window). */
function AppRow({
  state,
  starting,
  up,
  stop,
  busy,
}: {
  state: GatewayState;
  starting: boolean;
  up: boolean;
  stop: () => void;
  busy: boolean;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(tick);
  }, []);
  const [confirm, setConfirm] = useState(false);
  const app = state.apps.find((a) => a.key === state.app)?.name ?? state.app;
  const stopping = state.switch === "stopping";
  const facts = [
    state.experiment && state.experiment.split(/[\\/]/).filter(Boolean).pop(),
    !starting && uptime(state.since, now),
  ].filter(Boolean);
  return (
    <div className="flex flex-col gap-2 border-t border-border py-2.5">
      <div className="grid grid-cols-1 items-center gap-2 text-[13px] sm:grid-cols-[96px_minmax(0,1fr)_auto] sm:gap-3">
        <span className="text-muted-foreground">App</span>
        <span className="flex min-w-0 items-center gap-2">
          <Dot colour={starting || stopping ? "warn" : "ok"} />
          <span className="min-w-0 break-words">
            {app ?? "The app"}, {state.window ? "with its window" : "no window"}
            <span
              className="text-muted-foreground"
              title={state.experiment ?? ""}
            >
              {facts.map((t) => ` · ${t}`)}
            </span>
          </span>
        </span>
        <span className="flex gap-2.5">
          {up && state.app_url && (
            <Button
              onClick={() =>
                window.open(
                  `${state.app_url}/ui/#token=${getToken() ?? ""}`,
                  "_blank",
                  "noopener",
                )
              }
            >
              Open the app
            </Button>
          )}
          <Button
            variant="danger"
            disabled={busy}
            onClick={() => setConfirm(true)}
          >
            {busy && stopping
              ? "Stopping…"
              : state.window
                ? "Close the window"
                : "Stop"}
          </Button>
        </span>
      </div>
      {starting && <StartingLine state={state} />}
      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent
          title={
            state.window
              ? `Close ${app ?? "the app"}'s window?`
              : `Stop ${app ?? "the app"}?`
          }
          className="w-[min(460px,calc(100vw-32px))]"
        >
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">
              {state.window
                ? "This closes the app's window on this PC, and anything it is running stops with it."
                : "This stops the app on this PC, and anything it is running stops with it."}
            </p>
            {state.window && (
              <p className={cn("m-0", WARN)}>
                Anyone at the PC sees the window close.
              </p>
            )}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setConfirm(false)}>Cancel</Button>
              <Button
                variant="danger"
                disabled={busy}
                onClick={() => {
                  setConfirm(false);
                  stop();
                }}
              >
                {state.window ? "Close the window" : "Stop the app"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/** The microscope and the app: the start form while nothing runs, else one
 * row each. */
function AppCard({
  state,
  defaults,
  connection,
  starting,
  up,
}: {
  state: GatewayState;
  defaults: Defaults | undefined;
  connection: Connection | undefined;
  starting: boolean;
  up: boolean;
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
  const [choice, setChoice] = useState<MicroscopeChoice>({
    connect: defaults?.connect ?? true,
    configuration:
      state.configurations.find((c) => c.name === defaults?.configuration)
        ?.name ?? "",
  });
  // The gateway's own flags stand when it has some; else the page's choice.
  const microscope: Record<string, unknown> =
    state.connection.length > 0
      ? {}
      : choice.connect
        ? {
            connect: true,
            ...(choice.configuration
              ? { configuration: choice.configuration }
              : {}),
          }
        : { connect: false };
  const off = state.mode === "off";
  return (
    <Card className={cn("flex flex-col", off ? "gap-3" : "gap-0 py-1")}>
      {off ? (
        <>
          <div className="flex items-center gap-2">
            <Dot colour="off" />
            <CardTitle>Nothing is running</CardTitle>
          </div>
          <SubHeading first>Microscope</SubHeading>
          <MicroscopeStart
            state={state}
            choice={choice}
            setChoice={setChoice}
          />
          <SubHeading>App</SubHeading>
          <Starting
            state={state}
            busy={busy}
            microscope={microscope}
            defaults={defaults}
            start={(body) => act.mutate({ path: "/gateway/start", body })}
          />
        </>
      ) : (
        <>
          <MicroscopeRow connection={connection} starting={starting} />
          <AppRow
            state={state}
            starting={starting}
            up={up}
            busy={busy}
            stop={() => act.mutate({ path: "/gateway/stop", body: {} })}
          />
        </>
      )}
      {state.other_host && (
        <p className={cn("m-0 pb-2 text-[13px]", WARN)}>
          Another fibsem host runs here (pid {state.other_host.pid}) that the
          gateway didn't start: close it at the PC before starting the app.
        </p>
      )}
      {act.error && (
        <p className="m-0 pb-2 text-[13px] text-destructive">
          {(act.error as Error).message}
        </p>
      )}
      {off && state.last_exit && (
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

/** After Stop: the page still shows, but the gateway is gone. */
function Stopped({ at }: { at: number }) {
  return (
    <div className="flex h-full items-center justify-center px-4">
      <div className="flex max-w-[460px] flex-col items-center gap-2 text-center">
        <Square size={28} className="mb-2 text-muted-foreground" />
        <h1 className="m-0 text-[17px] font-semibold">
          The gateway has stopped
        </h1>
        <p className="m-0 text-[13px] text-muted-foreground">
          Stopped at {fmt.clock(at)} from this page. The app and any agent run
          stopped with it.
        </p>
        <p className="m-0 mt-2 text-[13px]">
          Start it again at the computer: the{" "}
          <b className="text-foreground">fibsemOS Gateway</b> shortcut, or{" "}
          <span className="font-mono text-xs">fibsem-gateway open</span>.
        </p>
      </div>
    </div>
  );
}

const RESTART_WAIT_MS = 120_000;

/** After Restart: wait for the gateway to answer again, then reload. */
function Restarting() {
  const [late, setLate] = useState(false);
  useEffect(() => {
    let done = false;
    const began = Date.now();
    const poll = async () => {
      // Give the gateway a moment to go down before asking whether it is back.
      await new Promise((r) => setTimeout(r, 2000));
      while (!done) {
        try {
          const r = await fetch("/health", { cache: "no-store" });
          if (r.ok) {
            location.reload();
            return;
          }
        } catch {
          // still down
        }
        if (Date.now() - began > RESTART_WAIT_MS) setLate(true);
        await new Promise((r) => setTimeout(r, 1000));
      }
    };
    void poll();
    return () => {
      done = true;
    };
  }, []);
  return (
    <div className="flex h-full items-center justify-center px-4">
      <div className="flex max-w-[460px] flex-col items-center gap-2 text-center">
        <Loader size={24} className="mb-2 animate-spin text-muted-foreground" />
        <h1 className="m-0 text-[17px] font-semibold">
          Restarting the gateway…
        </h1>
        <p className="m-0 text-[13px] text-muted-foreground">
          {late
            ? "It hasn't come back yet. If it doesn't, start it again at the computer."
            : "This page reloads when it is back."}
        </p>
      </div>
    </div>
  );
}

type Tab = "overview" | "instruments" | "history" | "pairing" | "settings";
const TABS: [Tab, string][] = [
  ["overview", "Overview"],
  ["instruments", "Instruments"],
  ["history", "History"],
  ["pairing", "Pairing"],
  ["settings", "Settings"],
];

/** The gateway's own page: a launcher for the app and the agent on this PC,
 * what happened, who is paired, and its settings. What a running agent is
 * doing is on the app's Agent page. */
/** The tab says which page this is: the gateway's name and icon, not the
 * app's. */
function useGatewayTab() {
  useEffect(() => {
    document.title = "Agent Gateway";
    let icon = document.querySelector<HTMLLinkElement>("link[rel='icon']");
    if (!icon) {
      icon = document.createElement("link");
      icon.rel = "icon";
      document.head.appendChild(icon);
    }
    icon.type = "image/svg+xml";
    icon.href = `data:image/svg+xml,${encodeURIComponent(gatewayIcon)}`;
  }, []);
}

export function GatewayPage() {
  useGatewayTab();
  const [tab, setTab] = useState<Tab>("overview");
  const [quit, setQuit] = useState<{ restart: boolean; at: number } | null>(
    null,
  );
  if (quit)
    return quit.restart ? <Restarting /> : <Stopped at={quit.at / 1000} />;
  return (
    <Gateway
      tab={tab}
      setTab={setTab}
      onQuit={(restart) => setQuit({ restart, at: Date.now() })}
    />
  );
}

function Gateway({
  tab,
  setTab,
  onQuit,
}: {
  tab: Tab;
  setTab: (tab: Tab) => void;
  onQuit: (restart: boolean) => void;
}) {
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => api<GatewayState>("/gateway"),
    refetchInterval: 2000,
  });
  const defaults = useQuery({
    queryKey: ["gateway/defaults"],
    queryFn: () => api<Defaults>("/gateway/defaults"),
    retry: false,
  });
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => api<AgentState>("/gateway/agent"),
    refetchInterval: 2000,
  });
  const state = gateway.data;
  const running = state?.mode === "running";
  const newer = state ? newerLabel(state.newer) : null;
  const connection = useQuery({
    queryKey: ["/gateway/microscope/connection"],
    queryFn: () => api<Connection>("/gateway/microscope/connection"),
    enabled: running,
    retry: false,
    // Often while the app starts or connects, to say so as soon as it answers.
    refetchInterval: (query) =>
      query.state.data && !query.state.data.connecting ? 5000 : 1000,
  });
  // The app is up once its server answers; until then it is starting.
  const appUp = running && !!connection.data;
  // Only an app with a host has a server to wait for.
  const hosted = !!state?.apps.find((a) => a.key === state.app)?.headless;
  const appStarting = running && hosted && !connection.data;
  const appName = running
    ? (state?.apps.find((a) => a.key === state.app)?.name ?? "the app")
    : null;
  const agentOn = !!agent.data?.running;
  const settled = !defaults.isPending;
  return (
    <div className="flex h-full flex-col overflow-hidden sm:flex-row">
      <nav className="flex shrink-0 flex-col gap-0.5 border-b border-border px-3 py-3 sm:w-52 sm:border-r sm:border-b-0 sm:py-5">
        <div className="px-3 pb-3">
          <div className="text-[15px] font-semibold">Agent Gateway</div>
          <div className="text-xs text-muted-foreground">{location.host}</div>
        </div>
        <div className="flex gap-0.5 overflow-x-auto sm:flex-col">
          {TABS.map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              aria-current={tab === key ? "page" : undefined}
              className={cn(
                "flex cursor-pointer items-center justify-between gap-3 rounded border-0 px-3 py-1.5 text-left text-[13px]",
                tab === key
                  ? "bg-accent text-foreground"
                  : "bg-transparent text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {label}
              {key === "overview" && (
                <Dot colour={running || agentOn ? "ok" : "off"} />
              )}
            </button>
          ))}
        </div>
        <div className="mt-auto hidden px-3 pt-4 text-xs text-muted-foreground sm:block">
          {state && <div>fibsemOS {state.version}</div>}
          {newer && (
            <div className="text-foreground" title={newer.detail}>
              {newer.text}
            </div>
          )}
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-auto">
        <div className="flex max-w-[860px] flex-col gap-3 px-4 py-5 sm:px-6">
          <h1 className="m-0 text-[17px] font-semibold">
            {TABS.find(([key]) => key === tab)?.[1]}
          </h1>
          {gateway.error && (
            <p className="m-0 text-[13px] text-destructive">
              {(gateway.error as Error).message}
            </p>
          )}
          {tab === "overview" && state && settled && (
            <>
              <AppCard
                key={state.mode}
                state={state}
                defaults={defaults.data}
                connection={connection.data}
                starting={appStarting}
                up={appUp}
              />
              <AgentCard
                appRunning={appUp}
                appStarting={appStarting}
                defaults={defaults.data}
              />
            </>
          )}
          {tab === "instruments" && <Instruments />}
          {tab === "history" && (
            <History
              apps={state?.apps ?? []}
              ready={notReady(agent.data, appUp, appStarting)}
            />
          )}
          {tab === "pairing" && (
            <Card>
              <Connections roles={["operator", "hub"]} />
            </Card>
          )}
          {tab === "settings" && state && (
            <Settings
              apps={state.apps}
              configurations={state.configurations}
              version={state.version}
              newer={newer}
              running={appName}
              onQuit={onQuit}
            />
          )}
        </div>
      </main>
    </div>
  );
}
