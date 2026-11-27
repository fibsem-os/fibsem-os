import {
  keepPreviousData,
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { Check, Info, Minus, Plus, Square, Trash2, X } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, ApiError, ReachedProvider, useApi } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  AgentCard,
  dollars,
  Field,
  FIELD,
  notReady,
  StartForm,
  type AgentState,
  type Session,
  title,
} from "@/pages/gateway/AgentCard";
import { AgentRow, History, stopped } from "@/pages/gateway/History";
import { SetupAt } from "@/pages/agent/Connectors";
import { AppCard, type GatewayState } from "@/pages/Gateway";
import { agentPagePath, sessionPagePath } from "@/pages/hub/InstrumentAgent";
import { Customise } from "@/pages/gateway/Customise";
import { KeyRow, type Secret } from "@/pages/gateway/Settings";
import { Connections } from "@/shell/Settings";

interface Piece {
  status: string;
  detail?: string;
  since?: number;
}

/** An instrument as the hub last read it (`fibsem.hub.hub.InstrumentState`). */
interface Instrument {
  name: string;
  url: string | null;
  ssh: string | null;
  paired: boolean;
  reachable: boolean | null;
  since: number | null;
  why: string | null;
  gateway: {
    mode?: "off" | "running";
    version?: string;
    app?: string;
    experiment?: string;
    since?: number;
  } | null;
  agent: {
    session_id?: string;
    state?: "running" | "waiting";
    playbook?: string;
    goal?: string;
    level?: string;
  } | null;
  /** Its tunnel, when the Hub opens one: the port on this PC that reaches
   * its gateway. */
  tunnel: { open?: boolean; gateway_port?: number } | null;
  health: { app?: Piece; microscope?: Piece } | null;
  api_version: string | null;
  // False when its gateway is a version this Hub was not written for.
  same_version: boolean | null;
}

interface HubState {
  instruments: Instrument[];
  /** This hub PC's simulated instrument: whether the Hub runs it now, and
   * whether it is in the list. */
  simulator: { name: string; running: boolean; listed: boolean } | null;
}

/** One step of a test of how an instrument is reached: ok, failed (with what
 * to do), or not reached (null). */
interface TestStep {
  key: string;
  name: string;
  ok: boolean | null;
  detail: string | null;
}

interface TestResult {
  ok: boolean;
  steps: TestStep[];
}

const WARN = "text-[var(--fibsem-semantic-warning-color)]";

/** The statuses that need someone's eye, and how each reads. */
const UNWELL: Record<string, string> = {
  slow: "Slow",
  not_answering: "Not answering",
  halted: "Halted",
  fault: "Fault",
  unknown: "Can't tell",
};
const MICROSCOPE: Record<string, string> = {
  ok: "Connected",
  not_connected: "Not connected",
  connecting: "Connecting…",
  stopped: "Not connected",
  starting: "Connecting…",
};

type Colour = "ok" | "warn" | "off";

function Dot({ colour }: { colour: Colour }) {
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

const unwell = (i: Instrument) =>
  [i.health?.app, i.health?.microscope].some((p) => p && p.status in UNWELL);

/** An instrument in one dot: grey while it can't be read, amber when it needs
 * someone, green otherwise. */
function colour(i: Instrument): Colour {
  if (!i.reachable) return "off";
  if (!i.paired || unwell(i) || i.agent?.state === "waiting") return "warn";
  return "ok";
}

/** An instrument in one line, for the list. */
function line(i: Instrument): string {
  if (i.reachable === null) return "Reading…";
  if (!i.reachable) return "Not reachable";
  if (!i.paired) return "Not paired";
  const app = i.gateway?.mode === "running" ? (i.gateway.app ?? "App") : "Nothing running";
  return i.agent ? `${app} · agent` : app;
}

function Hint({ children }: { children: string }) {
  return (
    <span title={children} className="inline-flex text-muted-foreground">
      <Info size={13} aria-label={children} />
    </span>
  );
}

function PieceLine({ piece }: { piece?: Piece }) {
  if (!piece?.detail || !(piece.status in UNWELL)) return null;
  return (
    <p className={cn("m-0 text-[13px]", WARN)} role="status">
      {piece.detail}
    </p>
  );
}

/** The instrument's microscope as its gateway says, in the gateway's own row:
 * read only, the Hub never connects one. */
function MicroscopeLine({ instrument }: { instrument: Instrument }) {
  const microscope = instrument.health?.microscope;
  return (
    <div className="flex flex-col gap-1.5 py-2.5">
      <div className="grid grid-cols-1 items-center gap-2 text-[13px] sm:grid-cols-[96px_minmax(0,1fr)_auto] sm:gap-3">
        <span className="text-muted-foreground">Microscope</span>
        <span className="flex min-w-0 items-center gap-2">
          <Dot
            colour={
              !microscope
                ? "off"
                : microscope.status in UNWELL
                  ? "warn"
                  : microscope.status === "ok"
                    ? "ok"
                    : "off"
            }
          />
          <span>
            {microscope
              ? (MICROSCOPE[microscope.status] ?? UNWELL[microscope.status] ?? title(microscope.status))
              : "No reading yet"}
          </span>
        </span>
      </div>
      <PieceLine piece={microscope} />
    </div>
  );
}

/** The instrument's app: the gateway's own card, read and asked through the
 * Hub. Start it while nothing runs, stop one that runs with no window. */
function InstrumentApp({ instrument }: { instrument: Instrument }) {
  const reach = useApi();
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => reach<GatewayState>("/gateway"),
    refetchInterval: 2000,
  });
  const state = gateway.data;
  if (!state)
    return gateway.error ? (
      <p className="m-0 text-[13px] text-destructive">{(gateway.error as Error).message}</p>
    ) : (
      <p className="m-0 text-[13px] text-muted-foreground">Reading…</p>
    );
  const running = state.mode === "running";
  const starting =
    running && (state.switch === "starting" || state.health?.app.status === "starting");
  return (
    <AppCard
      key={state.mode}
      state={state}
      defaults={undefined}
      connection={undefined}
      starting={starting}
      up={running && !starting}
      microscope={<MicroscopeLine instrument={instrument} />}
    />
  );
}

/** Try reaching an instrument now, and say which step failed: one in the list
 * (*name*), or an address before it is added. */
function useReachTest(target: { name: string } | { address: string }) {
  return useMutation({
    mutationFn: () => {
      if ("name" in target)
        return api<TestResult>(`/hub/instruments/${encodeURIComponent(target.name)}/test`, {
          method: "POST",
          body: "{}",
        });
      const where = target.address.trim();
      return api<TestResult>("/hub/test", {
        method: "POST",
        body: JSON.stringify(/^https?:\/\//i.test(where) ? { url: where } : { ssh: where }),
      });
    },
  });
}

function TestSteps({ result }: { result: TestResult }) {
  return (
    <ul className="m-0 flex list-none flex-col gap-1.5 p-0 text-[13px]" aria-label="Test result">
      {result.steps.map((step) => (
        <li key={step.key} className="flex gap-2">
          <span className="mt-0.5 shrink-0">
            {step.ok === true ? (
              <Check size={14} className="text-[var(--fibsem-ok-color)]" aria-label="Passed" />
            ) : step.ok === false ? (
              <X size={14} className="text-destructive" aria-label="Failed" />
            ) : (
              <Minus size={14} className="text-muted-foreground" aria-label="Not reached" />
            )}
          </span>
          <span className="flex min-w-0 flex-col">
            <span className={step.ok === null ? "text-muted-foreground" : undefined}>
              {step.name}
            </span>
            {step.detail && (
              <span
                className={cn(
                  "break-words text-xs",
                  step.ok === false ? "text-destructive" : "text-muted-foreground",
                )}
              >
                {step.detail}
              </span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Pair this hub with the instrument's gateway, with the code it shows. */
function PairCard({ instrument, simulated }: { instrument: Instrument; simulated: boolean }) {
  const client = useQueryClient();
  const [code, setCode] = useState("");
  const pair = useMutation({
    mutationFn: () =>
      api<HubState>(`/hub/instruments/${encodeURIComponent(instrument.name)}/pair`, {
        method: "POST",
        body: JSON.stringify({ code: code.trim() }),
      }),
    onSuccess: (answer) => client.setQueryData(["hub"], answer),
  });
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Dot colour="warn" />
        <CardTitle>Not paired</CardTitle>
        <Hint>
          {simulated
            ? "The Hub pairs with its own simulator by itself. If that didn't work, its gateway makes a code and you type it here."
            : "On the instrument's PC, open the gateway's page, go to Pairing and make a code for a Hub. Type that code here."}
        </Hint>
      </div>
      {simulated && (
        <p className="m-0 text-[13px]">
          In a terminal on this PC, run{" "}
          <span className="font-mono text-xs">fibsem-gateway pair --hub --may-switch</span> and
          type the code it shows.
        </p>
      )}
      <form
        className="flex flex-wrap items-end gap-2.5"
        onSubmit={(event) => {
          event.preventDefault();
          if (code.trim()) pair.mutate();
        }}
      >
        <Field label="Pairing code">
          <input
            value={code}
            onChange={(event) => setCode(event.target.value)}
            autoComplete="off"
            className={cn(FIELD, "w-40 font-mono")}
          />
        </Field>
        <Button variant="primary" disabled={pair.isPending || !code.trim()}>
          {pair.isPending ? "Pairing…" : "Pair"}
        </Button>
      </form>
      {pair.error && (
        <p className="m-0 text-[13px] text-destructive">{(pair.error as Error).message}</p>
      )}
    </Card>
  );
}

function Remove({ instrument, onDone }: { instrument: Instrument; onDone: () => void }) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const remove = useMutation({
    mutationFn: () =>
      api<HubState>(`/hub/instruments/${encodeURIComponent(instrument.name)}`, {
        method: "DELETE",
      }),
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      setOpen(false);
      onDone();
    },
  });
  return (
    <>
      <Button
        variant="ghost"
        className="px-2"
        title="Remove this instrument"
        aria-label="Remove this instrument"
        onClick={() => setOpen(true)}
      >
        <Trash2 size={14} />
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          title={`Remove ${instrument.name}?`}
          className="w-[min(460px,calc(100vw-32px))]"
        >
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">
              The Hub forgets it and its pairing. Nothing changes on the instrument: its app and
              agent keep running.
            </p>
            {remove.error && (
              <p className="m-0 text-destructive">{(remove.error as Error).message}</p>
            )}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setOpen(false)}>Cancel</Button>
              <Button variant="danger" disabled={remove.isPending} onClick={() => remove.mutate()}>
                Remove
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** The instrument's app and agent: the gateway's own cards, their calls sent
 * through the hub to that instrument's gateway. Their own query cache, so one
 * instrument's answers never show under another's name. */
/** What is inside reaches one instrument's gateway through the Hub, with a
 * cache of its own. With *hubOnly*, an agent started inside runs on this hub
 * PC; else where the form says. */
function InstrumentScope({
  instrument,
  simulated,
  hubOnly,
  children,
}: {
  instrument: Instrument;
  simulated: boolean;
  hubOnly?: boolean;
  children: ReactNode;
}) {
  const client = useMemo(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: (failures, error) =>
              failures < 1 && !(error instanceof ApiError && error.status >= 400),
            refetchOnWindowFocus: false,
            staleTime: 2000,
          },
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [instrument.name],
  );
  const reached = useMemo(
    () => ({
      prefix: `/hub/instruments/${encodeURIComponent(instrument.name)}`,
      pc: simulated ? "this hub PC" : "the microscope PC",
      places: [
        { key: "pc", label: simulated ? "The simulator's gateway" : "The microscope PC" },
        {
          key: "hub",
          label: "This hub PC",
          start: `/hub/instruments/${encodeURIComponent(instrument.name)}/hub-agent/start`,
        },
      ].slice(hubOnly ? 1 : 0),
    }),
    [instrument.name, simulated, hubOnly],
  );
  return (
    <QueryClientProvider client={client}>
      <ReachedProvider value={reached}>{children}</ReachedProvider>
    </QueryClientProvider>
  );
}

function InstrumentCards({ instrument, simulated }: { instrument: Instrument; simulated: boolean }) {
  return (
    <InstrumentScope instrument={instrument} simulated={simulated}>
      <InstrumentParts instrument={instrument} />
    </InstrumentScope>
  );
}

type Part = "now" | "history" | "about";
const PARTS: [Part, string][] = [
  ["now", "Now"],
  ["history", "History"],
  ["about", "About"],
];

/** An instrument as its gateway shows itself: what runs now, its history,
 * and what its gateway says of itself. All read through the Hub. */
function InstrumentParts({ instrument }: { instrument: Instrument }) {
  const reach = useApi();
  const [part, setPart] = useState<Part>("now");
  const appRunning = instrument.gateway?.mode === "running";
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => reach<GatewayState>("/gateway"),
    refetchInterval: 2000,
  });
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => reach<AgentState>("/gateway/agent"),
    refetchInterval: 5000,
    enabled: part === "history",
  });
  return (
    <>
      <div className="flex gap-1.5" role="tablist" aria-label="Instrument">
        {PARTS.map(([key, label]) => (
          <Button
            key={key}
            role="tab"
            aria-selected={part === key}
            variant={part === key ? "default" : "ghost"}
            onClick={() => setPart(key)}
          >
            {label}
          </Button>
        ))}
      </div>
      {part === "now" && (
        <>
          <InstrumentApp instrument={instrument} />
          <AgentCard appRunning={appRunning} />
        </>
      )}
      {part === "history" && (
        <History
          apps={gateway.data?.apps ?? []}
          ready={notReady(agent.data, appRunning, gateway.data?.switch === "starting")}
        />
      )}
      {part === "about" && <InstrumentAbout instrument={instrument} state={gateway.data} />}
    </>
  );
}

/** What the instrument's gateway says of itself, read only: it is set at
 * the microscope PC. */
function InstrumentAbout({
  instrument,
  state,
}: {
  instrument: Instrument;
  state: GatewayState | undefined;
}) {
  const rows: [string, ReactNode][] = [];
  const mono = (text: string) => <span className="font-mono text-xs">{text}</span>;
  if (instrument.ssh ?? instrument.url) rows.push(["Reached at", mono((instrument.ssh ?? instrument.url)!)]);
  if (instrument.since) rows.push(["Reachable since", fmt.when(instrument.since)]);
  if (state) {
    const newer = state.newer?.latest
      ? `fibsemOS ${state.newer.latest} available`
      : state.newer?.behind
        ? `${state.newer.behind} newer change${state.newer.behind === 1 ? "" : "s"} (${state.newer.branch})`
        : null;
    rows.push([
      "Gateway",
      <>
        fibsemOS {state.version}
        {newer && <span className={WARN}> · {newer}</span>}
      </>,
    ]);
  }
  if (instrument.api_version) rows.push(["API version", instrument.api_version]);
  if (state) {
    rows.push([
      "Microscope",
      state.connection.length ? mono(state.connection.join(" ")) : "Chosen as the app starts",
    ]);
    if (state.configurations.length)
      rows.push([
        "Configurations",
        state.configurations.map((c) => (c.default ? `${c.name} (default)` : c.name)).join(", "),
      ]);
    if (state.apps.length) rows.push(["Apps", state.apps.map((a) => a.name).join(", ")]);
    if (state.experiments_folder) rows.push(["Experiments", mono(state.experiments_folder)]);
    for (const server of state.device_servers ?? [])
      rows.push([
        `Device Server: ${server.name}`,
        <>
          {title(server.status)}
          {server.devices.length > 0 && (
            <span className="text-muted-foreground">
              {" "}
              · {server.devices.map((d) => (d.ok ? d.name : `${d.name} (not well)`)).join(", ")}
            </span>
          )}
        </>,
      ]);
  }
  return (
    <Card className="flex flex-col gap-2.5">
      <CardTitle className="flex items-center gap-1.5">
        About
        <Hint>As this instrument's gateway says. It is set at the microscope PC, not here.</Hint>
      </CardTitle>
      {state ? (
        <Facts className="[&_dd]:text-left" rows={rows} />
      ) : (
        <p className="m-0 text-[13px] text-muted-foreground">Reading…</p>
      )}
    </Card>
  );
}

function AgentPageLink({ instrument }: { instrument: Instrument }) {
  return (
    <p className="m-0 text-xs text-muted-foreground">
      <a
        href={agentPagePath(instrument.name)}
        className="text-primary underline underline-offset-2"
        title="The agent's chat, plan and review list, read through this Hub. No pairing with the microscope PC is needed."
      >
        Open the Agent page
      </a>
    </p>
  );
}

/** An instrument's SSH tunnel, in a few words beside its address: open and
 * on which port of this PC, or down. */
function Tunnel({ tunnel }: { tunnel: Instrument["tunnel"] }) {
  if (!tunnel) return null;
  const words = tunnel.open ? `SSH tunnel open, port ${tunnel.gateway_port}` : "SSH tunnel down";
  return (
    <span
      className="font-sans"
      title={
        tunnel.open
          ? `The Hub holds an SSH tunnel to this PC: 127.0.0.1:${tunnel.gateway_port} here reaches its gateway. It closes when the Hub stops.`
          : "The Hub's SSH tunnel to this PC is not open. It is tried again every 15 s."
      }
    >
      {" · "}
      {words}
    </span>
  );
}

/** Start or stop this hub PC's simulator: its gateway here, with the Demo
 * microscope. */
function useSimulator(onDone?: (state: HubState) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (act: "start" | "stop") =>
      api<HubState>(`/hub/simulator/${act}`, { method: "POST", body: "{}" }),
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      onDone?.(answer);
    },
  });
}

const SIMULATOR_HINT =
  "A simulated instrument on this hub PC: a gateway here with the Demo microscope. Nothing real is connected, and it stops when the Hub stops.";

/** An instrument the Hub can't reach: why, a test that says which step
 * fails, and for the simulator, starting it. */
function NotReachable({ instrument, simulated }: { instrument: Instrument; simulated: boolean }) {
  const test = useReachTest({ name: instrument.name });
  const simulator = useSimulator();
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Dot colour="off" />
        <CardTitle>{simulated ? "The simulator is not running" : "Not reachable"}</CardTitle>
        <span className="flex-1" />
        {simulated ? (
          <Button
            variant="primary"
            disabled={simulator.isPending}
            title={SIMULATOR_HINT}
            onClick={() => simulator.mutate("start")}
          >
            {simulator.isPending ? "Starting…" : "Start the simulator"}
          </Button>
        ) : (
          <Button
            disabled={test.isPending}
            title="Try reaching it now, and say which step fails."
            onClick={() => test.mutate()}
          >
            {test.isPending ? "Testing…" : "Test"}
          </Button>
        )}
      </div>
      {!simulated && instrument.why && !test.data && (
        <p className="m-0 text-[13px] text-muted-foreground">{instrument.why}</p>
      )}
      {test.data && <TestSteps result={test.data} />}
      {(test.error ?? simulator.error) && (
        <p className="m-0 text-[13px] text-destructive">
          {((test.error ?? simulator.error) as Error).message}
        </p>
      )}
    </Card>
  );
}

function InstrumentView({
  instrument,
  simulator,
  onRemoved,
}: {
  instrument: Instrument;
  simulator: HubState["simulator"];
  onRemoved: () => void;
}) {
  const simulated = simulator?.name === instrument.name;
  const stop = useSimulator();
  return (
    <>
      <div className="flex items-center gap-2.5">
        <Dot colour={colour(instrument)} />
        <h1 className="m-0 text-[17px] font-semibold">{instrument.name}</h1>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
          {instrument.ssh ?? instrument.url}
          {instrument.ssh && <Tunnel tunnel={instrument.tunnel} />}
        </span>
        {simulated && simulator?.running && (
          <Button
            disabled={stop.isPending}
            title="Stops the simulator's gateway on this PC, and an app it runs with no window. It stays in the list."
            onClick={() => stop.mutate("stop")}
          >
            {stop.isPending ? "Stopping…" : "Stop the simulator"}
          </Button>
        )}
        <Remove instrument={instrument} onDone={onRemoved} />
      </div>
      {stop.error && (
        <p className="m-0 text-[13px] text-destructive">{(stop.error as Error).message}</p>
      )}
      {instrument.reachable === false ? (
        <NotReachable key={instrument.name} instrument={instrument} simulated={simulated} />
      ) : !instrument.paired ? (
        <PairCard key={instrument.name} instrument={instrument} simulated={simulated} />
      ) : instrument.reachable === null ? (
        <p className="m-0 text-[13px] text-muted-foreground">Reading…</p>
      ) : (
        <>
          {instrument.why && <p className={cn("m-0 text-[13px]", WARN)}>{instrument.why}</p>}
          {instrument.same_version === false && (
            <p className={cn("m-0 text-[13px]", WARN)}>
              Its gateway is a different version from this Hub ({instrument.api_version}): update
              the older one.
            </p>
          )}
          <InstrumentCards key={instrument.name} instrument={instrument} simulated={simulated} />
          <AgentPageLink instrument={instrument} />
        </>
      )}
    </>
  );
}

/** Add an instrument: its name here, and where its PC is reached. */
function NewInstrument({
  open,
  onOpenChange,
  onAdded,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAdded: (name: string) => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const test = useReachTest({ address });
  const add = useMutation({
    mutationFn: () => {
      const where = address.trim();
      return api<HubState>("/hub/instruments", {
        method: "POST",
        body: JSON.stringify({
          name: name.trim(),
          ...(/^https?:\/\//i.test(where) ? { url: where } : { ssh: where }),
        }),
      });
    },
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      onAdded(name.trim());
      setName("");
      setAddress("");
      test.reset();
      onOpenChange(false);
    },
  });
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next);
        if (!next) {
          add.reset();
          test.reset();
        }
      }}
    >
      <DialogContent title="New instrument" className="w-[min(460px,calc(100vw-32px))]">
        <form
          className="flex flex-col gap-3 px-4 py-3.5"
          onSubmit={(event) => {
            event.preventDefault();
            add.mutate();
          }}
        >
          <Field label="Name">
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Scope 1"
              autoFocus
              className={FIELD}
            />
          </Field>
          <Field
            label={
              <span className="inline-flex items-center gap-1.5">
                SSH address
                <Hint>
                  Where ssh connects to the instrument's PC, such as user@host. The Hub opens the connection itself, with the SSH key this account already has.
                </Hint>
              </span>
            }
          >
            <input
              value={address}
              onChange={(event) => {
                setAddress(event.target.value);
                test.reset();
              }}
              placeholder="user@host"
              autoComplete="off"
              className={cn(FIELD, "font-mono")}
            />
          </Field>
          {test.data && <TestSteps result={test.data} />}
          {(test.error ?? add.error) && (
            <p className="m-0 text-[13px] text-destructive">
              {((test.error ?? add.error) as Error).message}
            </p>
          )}
          <div className="flex justify-end gap-2.5 pt-1">
            <Button
              type="button"
              className="mr-auto"
              disabled={test.isPending || !address.trim()}
              title="Try reaching it now, before adding it, and say which step fails."
              onClick={() => test.mutate()}
            >
              {test.isPending ? "Testing…" : "Test"}
            </Button>
            <Button type="button" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button variant="primary" disabled={add.isPending || !name.trim() || !address.trim()}>
              Add
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/** The key the agents on this hub PC use. Write-only, as a gateway's keys. */
function HubKeys() {
  const secrets = useQuery({
    queryKey: ["hub/secrets"],
    queryFn: () => api<Secret[]>("/hub/secrets"),
  });
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">Keys</CardTitle>
      {secrets.error && (
        <p className="m-0 text-[13px] text-destructive">{(secrets.error as Error).message}</p>
      )}
      {secrets.data?.map((s) => (
        <KeyRow key={s.name} s={s} path="/hub/secrets" page="the Hub's own page" />
      ))}
    </Card>
  );
}

function HubSettings({ version, onQuit }: { version?: string; onQuit: () => void }) {
  const [confirm, setConfirm] = useState(false);
  const quit = useMutation({
    mutationFn: () => api<{ stopping: boolean }>("/hub/quit", { method: "POST", body: "{}" }),
    onSuccess: onQuit,
  });
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Hub</CardTitle>
      <Facts
        rows={[
          ["Version", version ?? ""],
          ["Address", location.host],
        ]}
      />
      <div className="flex justify-end">
        <Button variant="danger" onClick={() => setConfirm(true)}>
          Stop the Hub
        </Button>
      </div>
      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent title="Stop the Hub?" className="w-[min(460px,calc(100vw-32px))]">
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">
              Its connections close, and any agent running on this hub PC is ended. Apps and agents
              on the microscope PCs keep running. You need to start it again at the hub PC.
            </p>
            {quit.error && <p className="m-0 text-destructive">{(quit.error as Error).message}</p>}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setConfirm(false)}>Cancel</Button>
              <Button variant="danger" disabled={quit.isPending} onClick={() => quit.mutate()}>
                Stop the Hub
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function Stopped({ at }: { at: number }) {
  return (
    <div className="flex h-full items-center justify-center px-4">
      <div className="flex max-w-[460px] flex-col items-center gap-2 text-center">
        <Square size={28} className="mb-2 text-muted-foreground" />
        <h1 className="m-0 text-[17px] font-semibold">The Hub has stopped</h1>
        <p className="m-0 text-[13px] text-muted-foreground">
          Stopped at {fmt.clock(at)} from this page. Every instrument carries on as it was.
        </p>
        <p className="m-0 mt-2 text-[13px]">
          Start it again at the hub PC: <span className="font-mono text-xs">agent-hub</span>.
        </p>
      </div>
    </div>
  );
}

/** A session in the Hub's list: the gateway's own row, with the instrument
 * it ran for and the PC it ran on. */
interface HubSession extends Session {
  /** null for a session with no instrument. */
  instrument: string | null;
  runs_on: "hub" | "pc";
}

/** What a session with no instrument is listed under. */
const NO_INSTRUMENT = "No instrument";
const listedUnder = (s: { instrument: string | null }) => s.instrument ?? NO_INSTRUMENT;

interface SessionsAnswer {
  sessions: HubSession[];
  totals: {
    spent_usd: number;
    by_instrument: Record<string, number>;
    by_day: { day: string; spent_usd: number }[];
  };
  unread: { instrument: string; why: string }[];
  more: boolean;
}

const SESSIONS_PAGE = 50;

/** Every agent session the Hub knows of, on this hub PC and on each
 * instrument's PC, newest first, with what they spent. Each row is the
 * gateway's own, and Resume is asked where the session ran. */
function HubSessions({ instruments }: { instruments: Instrument[] }) {
  const [pages, setPages] = useState(1);
  const [only, setOnly] = useState("");
  const sessions = useQuery({
    queryKey: ["hub/sessions", pages],
    queryFn: () => api<SessionsAnswer>(`/hub/sessions?limit=${SESSIONS_PAGE * pages}`),
    refetchInterval: 5000,
    placeholderData: keepPreviousData,
  });
  const data = sessions.data;
  const names = [...new Set([...instruments.map((i) => i.name), ...(data?.sessions ?? []).map(listedUnder)])];
  const shown = (data?.sessions ?? []).filter((s) => !only || listedUnder(s) === only);
  // An instrument that has an agent now takes no other until that one ends.
  const busy = (name: string) => !!instruments.find((i) => i.name === name)?.agent;
  const spent = only ? (data?.totals.by_instrument[only] ?? 0) : (data?.totals.spent_usd ?? 0);
  return (
    <>
      <div className="flex items-center gap-3">
        <h1 className="m-0 flex-1 text-[17px] font-semibold">Sessions</h1>
        <select
          value={only}
          onChange={(event) => setOnly(event.target.value)}
          aria-label="Instrument"
          className={FIELD}
        >
          <option value="">All instruments</option>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      {sessions.error && (
        <p className="m-0 text-[13px] text-destructive">{(sessions.error as Error).message}</p>
      )}
      {data?.unread.map((u) => (
        <p key={u.instrument} className={cn("m-0 text-[13px]", WARN)}>
          {u.instrument}'s sessions could not be read: {u.why}
        </p>
      ))}
      {data && (
        <Card className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <CardTitle>Spent</CardTitle>
            <Hint>
              What the sessions listed below spent, as each one's record says. Show older to count more.
            </Hint>
            <span className="flex-1" />
            <span className="text-[15px] font-semibold tabular-nums">{dollars(spent)}</span>
          </div>
          {!only && Object.keys(data.totals.by_instrument).length > 1 && (
            <Facts
              rows={Object.entries(data.totals.by_instrument).map(([name, usd]) => [name, dollars(usd)])}
            />
          )}
          {!only && data.totals.by_day.length > 0 && (
            <p className="m-0 text-xs text-muted-foreground tabular-nums">
              {data.totals.by_day
                .slice(0, 7)
                .map((d) => `${d.day.slice(5)} ${dollars(d.spent_usd)}`)
                .join(" · ")}
            </p>
          )}
        </Card>
      )}
      <Card className="flex flex-col gap-2">
        {data && shown.length === 0 && (
          <p className="m-0 text-[13px] text-muted-foreground">No sessions yet.</p>
        )}
        {shown.map((s) => (
          <ReachedProvider
            key={`${s.instrument}-${s.runs_on}-${s.session_id}`}
            value={{
              // A session with no instrument is the Hub's own to resume.
              prefix:
                s.instrument === null
                  ? "/hub/free-agents"
                  : `/hub/instruments/${encodeURIComponent(s.instrument)}`,
              pc: s.runs_on === "hub" ? "this hub PC" : "the microscope PC",
            }}
          >
            <AgentRow
              session={s}
              running={s.status === "running" || s.status === "waiting"}
              canResume={
                stopped(s) &&
                !!s.session_id &&
                !s.cannot_resume &&
                (s.instrument === null || !busy(s.instrument))
              }
              ready={null}
              where={[listedUnder(s), s.runs_on === "hub" ? "hub PC" : "microscope PC", s.who]
                .filter(Boolean)
                .join(" · ")}
              resumePath={
                s.instrument === null
                  ? "/resume"
                  : s.runs_on === "hub"
                    ? "/hub-agent/resume"
                    : undefined
              }
              noLevel={s.instrument === null}
            />
          </ReachedProvider>
        ))}
        {data?.more && (
          <Button variant="ghost" className="self-start" onClick={() => setPages((n) => n + 1)}>
            Show older
          </Button>
        )}
      </Card>
    </>
  );
}

interface HubEvent {
  at: number;
  kind: "agent" | "app" | "instrument" | "setup";
  what: string;
  who: string;
  instrument: string | null;
  ok: boolean;
}

const EVENT_KINDS: Record<HubEvent["kind"], string> = {
  agent: "Agent",
  app: "App",
  instrument: "Instrument",
  setup: "Setup",
};
const HISTORY_PAGE = 100;

/** What was asked of the Hub and what its agents did, newest first. */
function HubHistory() {
  const [pages, setPages] = useState(1);
  const [shown, setShown] = useState<HubEvent["kind"] | "all">("all");
  const history = useQuery({
    queryKey: ["hub/history", pages],
    queryFn: () =>
      api<{ events: HubEvent[]; more: boolean }>(`/hub/history?limit=${HISTORY_PAGE * pages}`),
    refetchInterval: 5000,
    placeholderData: keepPreviousData,
  });
  const events = (history.data?.events ?? []).filter((e) => shown === "all" || e.kind === shown);
  return (
    <>
      <div className="flex items-center gap-2">
        <h1 className="m-0 text-[17px] font-semibold">History</h1>
        <Hint>
          What was asked of this Hub and what its agents did. What happens at a microscope PC itself is in that gateway's own History.
        </Hint>
      </div>
      <Card className="flex flex-col gap-2">
        <div className="flex flex-wrap gap-1.5 pb-1">
          {(["all", "agent", "app", "instrument", "setup"] as const).map((k) => (
            <Button key={k} variant={shown === k ? "default" : "ghost"} onClick={() => setShown(k)}>
              {k === "all" ? "All" : EVENT_KINDS[k]}
            </Button>
          ))}
        </div>
        {history.error && (
          <p className="m-0 text-[13px] text-destructive">{(history.error as Error).message}</p>
        )}
        {history.data && events.length === 0 && (
          <p className="m-0 text-[13px] text-muted-foreground">Nothing yet.</p>
        )}
        {events.map((e, i) => (
          <div
            key={`${e.at}-${i}`}
            className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[170px_84px_minmax(0,1fr)_auto] sm:gap-3"
          >
            <span className="text-muted-foreground tabular-nums">{fmt.when(e.at)}</span>
            <span className="text-muted-foreground">{EVENT_KINDS[e.kind] ?? e.kind}</span>
            <span className={cn("min-w-0 break-words", !e.ok && WARN)}>
              {e.instrument && <span className="font-semibold">{e.instrument}: </span>}
              {e.what}
            </span>
            <span className="text-muted-foreground">{e.who}</span>
          </div>
        ))}
        {history.data?.more && (
          <Button variant="ghost" className="self-start" onClick={() => setPages((n) => n + 1)}>
            Show older
          </Button>
        )}
      </Card>
    </>
  );
}

/** An agent with no instrument on this hub PC (`fibsem.hub.free_agents`). */
interface FreeAgent {
  session_id: string;
  state: "running" | "waiting" | "ended";
  why: string | null;
  started: string | null;
  goal: string | null;
  who: string | null;
  budget_usd: number | null;
  resumed?: boolean;
  /** Whether the Hub serves its Agent page now. */
  page: boolean;
}

interface FreeAgents {
  limit: number;
  agents: FreeAgent[];
  at_this_pc: boolean;
  default_budget_usd: number | null;
}

/** An instrument's agent on this hub PC (`fibsem.hub.agents`). */
interface HubPcAgent {
  session_id?: string;
  state?: "running" | "away" | "ended";
  why?: string | null;
  started?: string | null;
  playbook?: string;
  goal?: string | null;
  level?: string;
  who?: string;
}

const useFreeAgents = () =>
  useQuery({
    queryKey: ["hub/free-agents"],
    queryFn: () => api<FreeAgents>("/hub/free-agents"),
    refetchInterval: 2000,
  });

/** One agent on this hub PC in a row: what it is for, for which instrument
 * (or none), and its page and End. */
function HubAgentRow({
  instrument,
  goal,
  state,
  why,
  about,
  page,
  endPath,
}: {
  instrument: string | null;
  goal: string | null | undefined;
  state: string;
  why?: string | null;
  about: (string | null | undefined | false)[];
  page: string | null;
  /** Where its End is asked; none for one that has ended. */
  endPath?: string;
}) {
  const client = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const end = useMutation({
    mutationFn: () => api<unknown>(endPath ?? "", { method: "POST", body: "{}" }),
    onSuccess: () => {
      setConfirm(false);
      client.invalidateQueries({ queryKey: ["hub/free-agents"] });
      client.invalidateQueries({ queryKey: ["hub/agents"] });
    },
  });
  const ended = state === "ended";
  return (
    <div className="flex items-start gap-2.5 border-t border-border pt-2 first:border-t-0 first:pt-0">
      <span className="pt-1.5">
        <Dot colour={ended ? "off" : state === "running" ? "ok" : "warn"} />
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="whitespace-pre-wrap break-words text-[13px] text-foreground">
          {goal || "No goal given"}
        </span>
        <span className="text-xs text-muted-foreground">
          {[instrument ?? NO_INSTRUMENT, ...about, ended ? `ended: ${why ?? "it ended"}` : state !== "running" && (why ?? state)]
            .filter(Boolean)
            .join(" · ")}
        </span>
        {end.error && !confirm && (
          <span className="text-xs text-destructive">{(end.error as Error).message}</span>
        )}
      </div>
      {page && (
        <a href={page} className="shrink-0 pt-0.5 text-xs text-primary underline underline-offset-2">
          Open the Agent page
        </a>
      )}
      {endPath && (
        <Button variant="danger" disabled={end.isPending} onClick={() => setConfirm(true)}>
          End
        </Button>
      )}
      <Dialog
        open={confirm}
        onOpenChange={(open) => {
          setConfirm(open);
          if (!open) end.reset();
        }}
      >
        <DialogContent title="End the agent?" className="w-[min(460px,calc(100vw-32px))]">
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">The agent stops now. Its session stays in Sessions, where it can be resumed.</p>
            {end.error && <p className="m-0 text-destructive">{(end.error as Error).message}</p>}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setConfirm(false)}>Cancel</Button>
              <Button variant="danger" disabled={end.isPending} onClick={() => end.mutate()}>
                {end.isPending ? "Ending…" : "End the agent"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/** Start an agent on this hub PC: for an instrument, with that instrument's
 * own start form, or for none, with a goal and a budget. */
function NewAgent({
  open,
  onOpenChange,
  instruments,
  simulator,
  free,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  instruments: Instrument[];
  simulator: HubState["simulator"];
  free?: FreeAgents;
}) {
  const client = useQueryClient();
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [budget, setBudget] = useState("");
  const usable = instruments.filter((i) => i.paired && i.reachable);
  const instrument = usable.find((i) => i.name === name) ?? null;
  const done = () => {
    client.invalidateQueries({ queryKey: ["hub/free-agents"] });
    client.invalidateQueries({ queryKey: ["hub/agents"] });
    setGoal("");
    setBudget("");
    onOpenChange(false);
  };
  const start = useMutation({
    mutationFn: () =>
      api<FreeAgent>("/hub/free-agents/start", {
        method: "POST",
        body: JSON.stringify({
          goal: goal.trim(),
          ...(budget.trim() ? { budget_usd: budget.trim() } : {}),
        }),
      }),
    onSuccess: done,
  });
  const running = free?.agents.filter((a) => a.state !== "ended").length ?? 0;
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next);
        if (!next) start.reset();
      }}
    >
      <DialogContent title="New agent" className="w-[min(640px,calc(100vw-32px))]">
        <div className="flex flex-col gap-3 px-4 py-3.5">
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
            <Field
              label={
                <span className="flex items-center gap-1.5">
                  Instrument
                  <Hint>
                    An agent with no instrument acts on no microscope: you give it a goal and talk to it on its Agent page. It runs on this hub PC and uses its API key. An agent for an instrument takes that instrument's one agent slot.
                  </Hint>
                </span>
              }
            >
              <select value={instrument?.name ?? ""} onChange={(event) => setName(event.target.value)} className={FIELD}>
                <option value="">{NO_INSTRUMENT}</option>
                {usable.map((i) => (
                  <option key={i.name} value={i.name}>
                    {i.name}
                  </option>
                ))}
              </select>
            </Field>
            {!instrument && (
              <Field label="Budget ($)">
                <input
                  value={budget}
                  onChange={(event) => setBudget(event.target.value)}
                  inputMode="decimal"
                  placeholder={free?.default_budget_usd != null ? String(free.default_budget_usd) : "No limit"}
                  className={FIELD}
                />
              </Field>
            )}
            {!instrument && (
              <Field label="Goal" wide>
                <textarea
                  value={goal}
                  onChange={(event) => setGoal(event.target.value)}
                  placeholder="What the agent should do this session"
                  rows={5}
                  className={cn(FIELD, "min-h-24 resize-y leading-snug")}
                />
              </Field>
            )}
          </div>
          {instrument ? (
            <InstrumentScope
              key={instrument.name}
              instrument={instrument}
              simulated={simulator?.name === instrument.name}
              hubOnly
            >
              <StartForm appRunning={instrument.gateway?.mode === "running"} onStarted={done} />
            </InstrumentScope>
          ) : (
            <>
              {start.error && <p className="m-0 text-[13px] text-destructive">{(start.error as Error).message}</p>}
              <div className="flex flex-wrap items-center justify-end gap-2.5">
                <span className="text-xs text-muted-foreground">
                  {free ? `${running} of ${free.limit} running. ` : ""}Uses ANTHROPIC_API_KEY on this hub PC.
                </span>
                <Button variant="primary" disabled={start.isPending} onClick={() => start.mutate()}>
                  {start.isPending ? "Starting…" : "Start the agent"}
                </Button>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** The agents this hub PC runs: each instrument's, and those with no
 * instrument. An agent on a microscope PC is on its instrument's page. */
function HubAgents({ instruments, simulator }: { instruments: Instrument[]; simulator: HubState["simulator"] }) {
  const [adding, setAdding] = useState(false);
  const free = useFreeAgents();
  const forInstruments = useQuery({
    queryKey: ["hub/agents"],
    queryFn: () => api<Record<string, HubPcAgent>>("/hub/agents"),
    refetchInterval: 2000,
  });
  const since = (iso: string | null | undefined, resumed?: boolean) =>
    iso ? `${resumed ? "resumed" : "since"} ${fmt.when(iso)}` : null;
  const rows = [
    ...Object.entries(forInstruments.data ?? {}).map(([name, a]) => ({
      key: `i-${name}`,
      ended: a.state === "ended",
      started: a.started ?? "",
      row: (
        <HubAgentRow
          key={`i-${name}`}
          instrument={name}
          goal={a.goal}
          state={a.state ?? "running"}
          why={a.why}
          about={[a.playbook, a.level && title(a.level), a.who, since(a.started)]}
          page={agentPagePath(name)}
          endPath={
            a.state === "ended" ? undefined : `/hub/instruments/${encodeURIComponent(name)}/hub-agent/end`
          }
        />
      ),
    })),
    ...(free.data?.agents ?? []).map((a) => ({
      key: `s-${a.session_id}`,
      ended: a.state === "ended",
      started: a.started ?? "",
      row: (
        <HubAgentRow
          key={`s-${a.session_id}`}
          instrument={null}
          goal={a.goal}
          state={a.state}
          why={a.why}
          about={[
            a.budget_usd != null && `budget ${dollars(a.budget_usd)}`,
            a.who,
            since(a.started, a.resumed),
          ]}
          page={a.page ? sessionPagePath(a.session_id) : null}
          endPath={a.state === "ended" ? undefined : `/hub/free-agents/${a.session_id}/end`}
        />
      ),
    })),
  ].sort((a, b) => b.started.localeCompare(a.started));
  const running = rows.filter((r) => !r.ended);
  const ended = rows.filter((r) => r.ended);
  const error = free.error ?? forInstruments.error;
  return (
    <>
      <div className="flex items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold">Agents</h1>
        <Hint>
          The agents running on this hub PC: for an instrument, or with no instrument. An agent started on a microscope PC is on its instrument's page.
        </Hint>
        <span className="flex-1" />
        <Button variant="primary" onClick={() => setAdding(true)}>
          <Plus size={14} />
          New agent
        </Button>
      </div>
      {error && <p className="m-0 text-[13px] text-destructive">{(error as Error).message}</p>}
      <Card className="flex flex-col gap-2">
        <CardTitle>Running on this hub PC</CardTitle>
        {running.length === 0 && (
          <p className="m-0 text-[13px] text-muted-foreground">No agent is running on this hub PC.</p>
        )}
        {running.map((r) => r.row)}
      </Card>
      {ended.length > 0 && (
        <Card className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <CardTitle>Ended</CardTitle>
            <Hint>Ended since the Hub last started. Every session, and Resume, is in Sessions.</Hint>
          </div>
          {ended.map((r) => r.row)}
        </Card>
      )}
      <NewAgent
        open={adding}
        onOpenChange={setAdding}
        instruments={instruments}
        simulator={simulator}
        free={free.data}
      />
    </>
  );
}

/** How many agents with no instrument run on this hub PC at once. */
function FreeAgentsLimit() {
  const client = useQueryClient();
  const free = useFreeAgents();
  const [typed, setTyped] = useState<string | null>(null);
  const save = useMutation({
    mutationFn: (limit: number) =>
      api<{ limit: number }>("/hub/free-agents/limit", { method: "PUT", body: JSON.stringify({ limit }) }),
    onSuccess: () => {
      setTyped(null);
      client.invalidateQueries({ queryKey: ["hub/free-agents"] });
    },
  });
  if (!free.data) return null;
  const here = free.data.at_this_pc;
  const value = typed ?? String(free.data.limit);
  const wanted = /^\d+$/.test(value.trim()) ? Number(value.trim()) : null;
  return (
    <Card className="flex flex-col gap-2">
      <CardTitle>Agents</CardTitle>
      <div className="flex flex-wrap items-center gap-2.5 text-[13px]">
        <span className="flex items-center gap-1.5">
          Agents with no instrument at once
          <Hint>
            How many agents with no instrument this hub PC runs at the same time. An instrument's agent is not counted. Changed only at this PC, on the Hub's own page.
          </Hint>
        </span>
        <span className="flex-1" />
        <input
          value={value}
          onChange={(event) => setTyped(event.target.value)}
          inputMode="numeric"
          aria-label="Agents with no instrument at once"
          disabled={!here}
          className={cn(FIELD, "w-20")}
        />
        <Button
          disabled={!here || save.isPending || wanted === null || wanted === free.data.limit}
          onClick={() => wanted !== null && save.mutate(wanted)}
        >
          Save
        </Button>
      </div>
      {save.error && <p className="m-0 text-[13px] text-destructive">{(save.error as Error).message}</p>}
    </Card>
  );
}

type Tab = "instruments" | "agents" | "sessions" | "history" | "pairing" | "customise" | "settings";

function NavRow({
  current,
  onClick,
  children,
}: {
  current: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-current={current ? "page" : undefined}
      className={cn(
        "flex w-full cursor-pointer items-center gap-2.5 rounded border-0 px-3 py-1.5 text-left text-[13px]",
        current
          ? "bg-accent text-foreground"
          : "bg-transparent text-muted-foreground hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

/** The Hub's page: the lab's instruments down the side, and the one picked
 * beside them as its own gateway shows it: microscope, app, agent. */
export function HubPage() {
  useEffect(() => {
    document.title = "Agent Hub";
  }, []);
  const [tab, setTab] = useState<Tab>("instruments");
  const [picked, setPicked] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [stopped, setStopped] = useState<number | null>(null);
  const hub = useQuery({
    queryKey: ["hub"],
    queryFn: () => api<HubState>("/hub"),
    refetchInterval: 2000,
    enabled: stopped === null,
  });
  const health = useQuery({
    queryKey: ["hub/health"],
    queryFn: () => fetch("/health").then((r) => r.json() as Promise<{ version: string }>),
    staleTime: Infinity,
  });
  if (stopped !== null) return <Stopped at={stopped / 1000} />;
  const simulator = useSimulator((answer) => {
    if (answer.simulator) setPicked(answer.simulator.name);
    setTab("instruments");
  });
  const instruments = hub.data?.instruments ?? [];
  const instrument = instruments.find((i) => i.name === picked) ?? instruments[0] ?? null;
  return (
    <div className="flex h-full flex-col overflow-hidden sm:flex-row">
      <nav className="flex shrink-0 flex-col gap-0.5 border-b border-border px-3 py-3 sm:w-60 sm:border-r sm:border-b-0 sm:py-5">
        <div className="px-3 pb-3">
          <div className="text-[15px] font-semibold">Agent Hub</div>
          <div className="text-xs text-muted-foreground">{location.host}</div>
        </div>
        <div className="flex items-center justify-between px-3 pt-1 pb-1">
          <span className="text-xs font-semibold text-muted-foreground">Instruments</span>
          <Button
            variant="ghost"
            className="px-1.5 py-1"
            title="New instrument"
            aria-label="New instrument"
            onClick={() => setAdding(true)}
          >
            <Plus size={14} />
          </Button>
        </div>
        <div className="flex gap-0.5 overflow-x-auto sm:flex-col">
          {instruments.map((i) => (
            <NavRow
              key={i.name}
              current={tab === "instruments" && instrument?.name === i.name}
              onClick={() => {
                setPicked(i.name);
                setTab("instruments");
              }}
            >
              <Dot colour={colour(i)} />
              <span className="flex min-w-0 flex-col">
                <span className="truncate text-foreground">{i.name}</span>
                <span className="truncate text-xs text-muted-foreground">{line(i)}</span>
              </span>
            </NavRow>
          ))}
          {hub.data && instruments.length === 0 && (
            <span className="px-3 py-1.5 text-xs text-muted-foreground">None yet</span>
          )}
          {hub.data?.simulator && !hub.data.simulator.listed && instruments.length > 0 && (
            <NavRow current={false} onClick={() => simulator.mutate("start")}>
              <Plus size={12} className="shrink-0" />
              <span title={SIMULATOR_HINT}>
                {simulator.isPending ? "Starting the simulator…" : "Start a simulator"}
              </span>
            </NavRow>
          )}
        </div>
        <div className="mt-3 flex gap-0.5 border-t border-border pt-3 sm:flex-col">
          <NavRow current={tab === "agents"} onClick={() => setTab("agents")}>
            Agents
          </NavRow>
          <NavRow current={tab === "sessions"} onClick={() => setTab("sessions")}>
            Sessions
          </NavRow>
          <NavRow current={tab === "history"} onClick={() => setTab("history")}>
            History
          </NavRow>
          <NavRow current={tab === "pairing"} onClick={() => setTab("pairing")}>
            Browsers
          </NavRow>
          <NavRow current={tab === "customise"} onClick={() => setTab("customise")}>
            Customise
          </NavRow>
          <NavRow current={tab === "settings"} onClick={() => setTab("settings")}>
            Settings
          </NavRow>
        </div>
        <div className="mt-auto hidden px-3 pt-4 text-xs text-muted-foreground sm:block">
          {health.data && <div>fibsemOS {health.data.version}</div>}
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-auto">
        <div className="flex max-w-[860px] flex-col gap-3 px-4 py-5 sm:px-6">
          {(hub.error ?? simulator.error) && (
            <p className="m-0 text-[13px] text-destructive">
              {((hub.error ?? simulator.error) as Error).message}
            </p>
          )}
          {tab === "instruments" &&
            (instrument ? (
              <InstrumentView
                instrument={instrument}
                simulator={hub.data?.simulator ?? null}
                onRemoved={() => setPicked(null)}
              />
            ) : (
              hub.data && (
                <div className="flex flex-col items-start gap-3 pt-6">
                  <h1 className="m-0 text-[17px] font-semibold">No instruments yet</h1>
                  <div className="flex flex-wrap gap-2.5">
                    <Button variant="primary" onClick={() => setAdding(true)}>
                      <Plus size={14} />
                      New instrument
                    </Button>
                    {hub.data.simulator && (
                      <Button
                        disabled={simulator.isPending}
                        title={SIMULATOR_HINT}
                        onClick={() => simulator.mutate("start")}
                      >
                        {simulator.isPending ? "Starting the simulator…" : "Start a simulator"}
                      </Button>
                    )}
                  </div>
                </div>
              )
            ))}
          {tab === "agents" && (
            <HubAgents instruments={instruments} simulator={hub.data?.simulator ?? null} />
          )}
          {tab === "sessions" && <HubSessions instruments={instruments} />}
          {tab === "history" && <HubHistory />}
          {tab === "pairing" && (
            <>
              <h1 className="m-0 text-[17px] font-semibold">Browsers</h1>
              <p className="m-0 text-[13px] text-muted-foreground">
                The browsers that may use this Hub. The browser on this PC is signed in by{" "}
                <span className="font-mono text-xs">agent-hub open</span>. To let another one in,
                make a code here and type it in that browser. This is separate from pairing an
                instrument.
              </p>
              <Card>
                <Connections roles={["operator"]} />
              </Card>
            </>
          )}
          {tab === "customise" && (
            <>
              <h1 className="m-0 text-[17px] font-semibold">Customise</h1>
              <p className="m-0 text-[13px] text-muted-foreground">
                What an agent started on this hub PC runs with. An agent started on a microscope
                PC uses that PC's own, set on its gateway page. Changes are made at this PC only.
              </p>
              <SetupAt.Provider value={{ at: "/hub", page: "the Hub's own page" }}>
                <Customise only={["playbooks", "skills", "agents", "connectors"]} />
              </SetupAt.Provider>
            </>
          )}
          {tab === "settings" && (
            <>
              <h1 className="m-0 text-[17px] font-semibold">Settings</h1>
              <HubKeys />
              <FreeAgentsLimit />
              <HubSettings version={health.data?.version} onQuit={() => setStopped(Date.now())} />
            </>
          )}
        </div>
      </main>
      <NewInstrument open={adding} onOpenChange={setAdding} onAdded={setPicked} />
    </div>
  );
}
