import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { Info, Plus, Square, Trash2 } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, ApiError, ReachedProvider } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";
import { AgentCard, Field, FIELD, title } from "@/pages/gateway/AgentCard";
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

function SubHeading({ first, children }: { first?: boolean; children: string }) {
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

function Hint({ children }: { children: string }) {
  return (
    <span title={children} className="inline-flex text-muted-foreground">
      <Info size={13} aria-label={children} />
    </span>
  );
}

function uptime(since: number | undefined): string | null {
  if (!since) return null;
  const minutes = Math.max(0, Math.floor((Date.now() / 1000 - since) / 60));
  if (minutes < 60) return `up ${minutes} min`;
  return `up ${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function PieceLine({ piece }: { piece?: Piece }) {
  if (!piece?.detail || !(piece.status in UNWELL)) return null;
  return (
    <p className={cn("m-0 text-[13px]", WARN)} role="status">
      {piece.detail}
    </p>
  );
}

/** The instrument's microscope and app, as its gateway says: read only. */
function StatusCard({ instrument }: { instrument: Instrument }) {
  const gateway = instrument.gateway;
  const running = gateway?.mode === "running";
  const microscope = instrument.health?.microscope;
  const app = instrument.health?.app;
  const up = uptime(gateway?.since);
  return (
    <Card className="flex flex-col gap-3">
      <SubHeading first>Microscope</SubHeading>
      <div className="flex items-center gap-2 text-[13px]">
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
            : running
              ? "No reading yet"
              : "Not connected"}
        </span>
      </div>
      <PieceLine piece={microscope} />
      <SubHeading>App</SubHeading>
      <div className="flex items-center gap-2 text-[13px]">
        <Dot colour={!running ? "off" : app && app.status in UNWELL ? "warn" : "ok"} />
        <span>{running ? (gateway?.app ?? "An app") : "Nothing is running"}</span>
        {running && (
          <span className="text-muted-foreground">
            {[gateway?.experiment, up].filter(Boolean).join(" · ")}
          </span>
        )}
      </div>
      <PieceLine piece={app} />
    </Card>
  );
}

/** Pair this hub with the instrument's gateway, with the code it shows. */
function PairCard({ instrument }: { instrument: Instrument }) {
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
          On the instrument's PC, open the gateway's page, go to Pairing and make a code for a Hub. Type that code here.
        </Hint>
      </div>
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

/** The instrument's agent: the gateway's own card, its calls sent through the
 * hub to that instrument's gateway. Its own query cache, so one instrument's
 * answers never show under another's name. */
function InstrumentAgent({ instrument }: { instrument: Instrument }) {
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
      pc: "the microscope PC",
      places: [
        { key: "pc", label: "The microscope PC" },
        {
          key: "hub",
          label: "This hub PC",
          start: `/hub/instruments/${encodeURIComponent(instrument.name)}/hub-agent/start`,
        },
      ],
    }),
    [instrument.name],
  );
  return (
    <QueryClientProvider client={client}>
      <ReachedProvider value={reached}>
        <AgentCard appRunning={instrument.gateway?.mode === "running"} />
      </ReachedProvider>
    </QueryClientProvider>
  );
}

/** Where this PC's browser reaches the instrument's own pages: through the
 * Hub's tunnel, or at its address. That gateway signs the browser in itself,
 * with a code from its Pairing page the first time. */
function gatewayAddress(instrument: Instrument): string | null {
  if (instrument.ssh)
    return instrument.tunnel?.open && instrument.tunnel.gateway_port
      ? `http://127.0.0.1:${instrument.tunnel.gateway_port}`
      : null;
  return instrument.url;
}

function AgentPageLink({ instrument }: { instrument: Instrument }) {
  const address = gatewayAddress(instrument);
  return (
    <p className="m-0 text-xs text-muted-foreground">
      The agent's chat and plan are on the instrument's Agent page.{" "}
      {address && (
        <a
          href={`${address}/ui/agent`}
          target="_blank"
          rel="noreferrer"
          className="text-primary underline underline-offset-2"
          title="Opens the instrument's own page. The first time, it asks for a code from the Pairing page at the microscope PC."
        >
          Open the Agent page
        </a>
      )}
    </p>
  );
}

function InstrumentView({ instrument, onRemoved }: { instrument: Instrument; onRemoved: () => void }) {
  return (
    <>
      <div className="flex items-center gap-2.5">
        <Dot colour={colour(instrument)} />
        <h1 className="m-0 text-[17px] font-semibold">{instrument.name}</h1>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
          {instrument.ssh ?? instrument.url}
        </span>
        <Remove instrument={instrument} onDone={onRemoved} />
      </div>
      {instrument.reachable === false ? (
        <Card className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Dot colour="off" />
            <CardTitle>Not reachable</CardTitle>
          </div>
          {instrument.why && <p className="m-0 text-[13px] text-muted-foreground">{instrument.why}</p>}
        </Card>
      ) : !instrument.paired ? (
        <PairCard key={instrument.name} instrument={instrument} />
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
          <StatusCard instrument={instrument} />
          <InstrumentAgent key={instrument.name} instrument={instrument} />
          {instrument.agent && <AgentPageLink instrument={instrument} />}
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
      onOpenChange(false);
    },
  });
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next);
        if (!next) add.reset();
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
              onChange={(event) => setAddress(event.target.value)}
              placeholder="user@host"
              autoComplete="off"
              className={cn(FIELD, "font-mono")}
            />
          </Field>
          {add.error && (
            <p className="m-0 text-[13px] text-destructive">{(add.error as Error).message}</p>
          )}
          <div className="flex justify-end gap-2.5 pt-1">
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

type Tab = "instruments" | "pairing" | "customise" | "settings";

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
        </div>
        <div className="mt-3 flex gap-0.5 border-t border-border pt-3 sm:flex-col">
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
          {hub.error && (
            <p className="m-0 text-[13px] text-destructive">{(hub.error as Error).message}</p>
          )}
          {tab === "instruments" &&
            (instrument ? (
              <InstrumentView instrument={instrument} onRemoved={() => setPicked(null)} />
            ) : (
              hub.data && (
                <div className="flex flex-col items-start gap-3 pt-6">
                  <h1 className="m-0 text-[17px] font-semibold">No instruments yet</h1>
                  <Button variant="primary" onClick={() => setAdding(true)}>
                    <Plus size={14} />
                    New instrument
                  </Button>
                </div>
              )
            ))}
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
              <Customise only={["playbooks", "skills", "agents"]} />
            </>
          )}
          {tab === "settings" && (
            <>
              <h1 className="m-0 text-[17px] font-semibold">Settings</h1>
              <HubKeys />
              <HubSettings version={health.data?.version} onQuit={() => setStopped(Date.now())} />
            </>
          )}
        </div>
      </main>
      <NewInstrument open={adding} onOpenChange={setAdding} onAdded={setPicked} />
    </div>
  );
}
