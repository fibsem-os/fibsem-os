import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";

interface HubSession {
  session_id: string;
  state: string;
  started?: string;
  playbook?: string;
  goal?: string;
  level?: string;
  who?: string;
  budget_usd?: number;
}

interface Instrument {
  name: string;
  url: string | null;
  ssh: string | null;
  tunnel: { open: boolean; gateway_port: number; app_port: number } | null;
  paired: boolean;
  reachable: boolean | null;
  since: number | null;
  checked: number | null;
  why: string | null;
  gateway: {
    mode?: string;
    version?: string;
    app?: string;
    experiment?: string;
    switching?: boolean;
  } | null;
  agent: { state?: string; playbook?: string; level?: string } | null;
  session: HubSession | null;
}

interface HubState {
  hub: boolean;
  instruments: Instrument[];
}

interface Playbook {
  name: string;
  error?: string | null;
}

const FIELD =
  "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
const LEVELS = ["watch", "assist", "drive"];
const WARNING = "text-[var(--fibsem-semantic-warning-color)]";
const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

function Failed({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p className="m-0 text-[13px] text-destructive">
      {(error as Error).message}
    </p>
  );
}

/** What the instrument's PC is doing, in a few words; the rest is the
 * line's hover text. */
function status(i: Instrument): { text: string; warn: boolean; more?: string } {
  if (i.reachable === null) return { text: "Checking", warn: false };
  if (!i.reachable)
    return {
      text: "Not reachable",
      warn: true,
      more: [i.since && `Since ${fmt.when(i.since)}`, i.why]
        .filter(Boolean)
        .join(". "),
    };
  if (!i.paired) return { text: "Not paired", warn: true };
  if (!i.gateway) return { text: "Reachable", warn: !!i.why, more: i.why ?? undefined };
  if (i.gateway.switching) return { text: "Starting or stopping", warn: false };
  if (i.gateway.mode !== "running") return { text: "No app running", warn: false };
  const experiment = i.gateway.experiment
    ?.split(/[\\/]/)
    .filter(Boolean)
    .pop();
  return {
    text: `${i.gateway.app ?? "App"}${experiment ? ` · ${experiment}` : ""}`,
    warn: false,
  };
}

function useHubAction<T>(act: (value: T) => Promise<unknown>) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: act,
    onSuccess: () => client.invalidateQueries({ queryKey: ["hub"] }),
  });
}

const at = (name: string) => `/hub/instruments/${encodeURIComponent(name)}`;

function Pair({ name, onDone }: { name: string; onDone: () => void }) {
  const [code, setCode] = useState("");
  const client = useQueryClient();
  const pair = useMutation({
    mutationFn: () =>
      api<HubState>(`${at(name)}/pair`, {
        method: "POST",
        body: JSON.stringify({ code: code.trim() }),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      onDone();
    },
  });
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <input
          value={code}
          onChange={(event) => setCode(event.target.value)}
          inputMode="numeric"
          placeholder="Code"
          title={`The code "fibsem-gateway pair --hub" shows on ${name}`}
          className={cn(FIELD, "w-28")}
        />
        <Button
          variant="primary"
          disabled={!code.trim() || pair.isPending}
          onClick={() => pair.mutate()}
        >
          Pair
        </Button>
        <Button onClick={onDone}>Cancel</Button>
      </div>
      <Failed error={pair.error} />
    </div>
  );
}

function Start({ name, onDone }: { name: string; onDone: () => void }) {
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
  });
  const usable = (playbooks.data ?? []).filter((p) => !p.error);
  const [playbook, setPlaybook] = useState("");
  const [goal, setGoal] = useState("");
  const [level, setLevel] = useState("watch");
  const [budget, setBudget] = useState("");
  const chosen =
    playbook ||
    usable.find((p) => p.name === "fibsem-agent")?.name ||
    usable[0]?.name ||
    "";
  const client = useQueryClient();
  const start = useMutation({
    mutationFn: () =>
      api<HubState>(`${at(name)}/agent/start`, {
        method: "POST",
        body: JSON.stringify({
          playbook: chosen,
          goal: goal.trim(),
          level,
          ...(budget.trim() ? { budget_usd: budget.trim() } : {}),
        }),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      onDone();
    },
  });
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <select
          aria-label="Playbook"
          value={chosen}
          onChange={(event) => setPlaybook(event.target.value)}
          className={FIELD}
        >
          {usable.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        <select
          aria-label="Level"
          value={level}
          onChange={(event) => setLevel(event.target.value)}
          className={FIELD}
        >
          {LEVELS.map((l) => (
            <option key={l} value={l}>
              {title(l)}
            </option>
          ))}
        </select>
        <input
          aria-label="Budget in dollars"
          value={budget}
          onChange={(event) => setBudget(event.target.value)}
          inputMode="decimal"
          placeholder="Budget ($)"
          className={cn(FIELD, "w-28")}
        />
        <input
          aria-label="Goal"
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          placeholder="Goal"
          className={cn(FIELD, "min-w-40 flex-1")}
        />
        <Button
          variant="primary"
          disabled={!chosen || start.isPending}
          onClick={() => start.mutate()}
        >
          Start
        </Button>
        <Button onClick={onDone}>Cancel</Button>
      </div>
      <Failed error={start.error} />
    </div>
  );
}

function Agent({ name, session }: { name: string; session: HubSession }) {
  const level = useHubAction((value: string) =>
    api(`${at(name)}/level`, {
      method: "PUT",
      body: JSON.stringify({ level: value }),
    }),
  );
  return (
    <span className="flex items-center gap-2">
      <span
        className="truncate"
        title={[
          session.goal,
          session.started && `Since ${fmt.when(session.started)}`,
          session.state !== "running" && "Waiting for the app",
          level.error && (level.error as Error).message,
        ]
          .filter(Boolean)
          .join("\n")}
      >
        {session.playbook ?? "Agent"}
      </span>
      <select
        aria-label="Level"
        value={session.level ?? "watch"}
        disabled={level.isPending}
        onChange={(event) => level.mutate(event.target.value)}
        className={cn(FIELD, "py-1", level.error && "border-destructive")}
      >
        {LEVELS.map((l) => (
          <option key={l} value={l}>
            {title(l)}
          </option>
        ))}
      </select>
    </span>
  );
}

const CELL = "flex min-h-11 items-center border-t border-border py-1.5 pr-6";

function Row({ instrument: i, editing }: { instrument: Instrument; editing: boolean }) {
  const [open, setOpen] = useState<"start" | "pair" | null>(null);
  const remove = useHubAction(() => api(at(i.name), { method: "DELETE" }));
  const end = useHubAction(() => api(`${at(i.name)}/agent/end`, { method: "POST" }));
  const { text, warn, more } = status(i);
  const ready = i.reachable && i.paired && i.gateway?.mode === "running";
  const failed = remove.error ?? end.error;
  return (
    <Fragment>
      <span
        className={cn(CELL, "font-semibold")}
        title={[i.ssh ?? i.url, i.gateway?.version && `fibsemOS ${i.gateway.version}`]
          .filter(Boolean)
          .join(" · ")}
      >
        {i.name}
      </span>
      <span className={cn(CELL, warn && WARNING)} title={more}>
        <span className="truncate">{text}</span>
      </span>
      <span className={cn(CELL, "min-w-0 text-muted-foreground")}>
        {i.session ? (
          <Agent name={i.name} session={i.session} />
        ) : i.agent ? (
          <span title={`Started at ${i.name}, not from this hub.`}>
            {i.agent.playbook ?? "Agent"} · {title(i.agent.level ?? "watch")}
          </span>
        ) : more ? (
          <span className="truncate" title={more}>
            {more}
          </span>
        ) : null}
      </span>
      <span className={cn(CELL, "justify-end pr-0")}>
        {editing ? (
          <Button
            variant="danger"
            disabled={!!i.session || remove.isPending}
            title={i.session ? "End its agent first." : undefined}
            onClick={() => remove.mutate(undefined)}
          >
            Remove
          </Button>
        ) : i.session ? (
          <Button disabled={end.isPending} onClick={() => end.mutate(undefined)}>
            End
          </Button>
        ) : i.reachable && !i.paired ? (
          <Button onClick={() => setOpen(open ? null : "pair")}>Pair</Button>
        ) : ready && !i.agent ? (
          <Button onClick={() => setOpen(open ? null : "start")}>Start</Button>
        ) : null}
      </span>
      {(open || failed) && (
        <div className="col-span-4 flex flex-col gap-1.5 pb-3">
          {open === "start" && <Start name={i.name} onDone={() => setOpen(null)} />}
          {open === "pair" && <Pair name={i.name} onDone={() => setOpen(null)} />}
          <Failed error={failed} />
        </div>
      )}
    </Fragment>
  );
}

function Add({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [ssh, setSsh] = useState("");
  const client = useQueryClient();
  const add = useMutation({
    mutationFn: () =>
      api<HubState>("/hub/instruments", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), ssh: ssh.trim() }),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["hub"], answer);
      onDone();
    },
  });
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <input
          aria-label="Name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Name"
          className={cn(FIELD, "w-40")}
        />
        <input
          aria-label="Where ssh connects to its PC"
          value={ssh}
          onChange={(event) => setSsh(event.target.value)}
          placeholder="user@host"
          title="Where ssh connects to its PC. This PC's ssh key must be set up there first."
          className={cn(FIELD, "w-56")}
        />
        <Button
          variant="primary"
          disabled={!name.trim() || !ssh.trim() || add.isPending}
          onClick={() => add.mutate()}
        >
          Add
        </Button>
        <Button onClick={onDone}>Cancel</Button>
      </div>
      <Failed error={add.error} />
    </div>
  );
}

/** The instruments this gateway is the hub for, one line each: what its PC
 * runs, the agent this PC runs for it, and the one thing to do next. */
export function Instruments() {
  const hub = useQuery({
    queryKey: ["hub"],
    queryFn: () => api<HubState>("/hub"),
    refetchInterval: 3000,
  });
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState(false);
  const state = hub.data;
  if (!state) return null;
  const any = state.instruments.length > 0;
  return (
    <Card className="flex flex-col gap-3">
      {any ? (
        <div className="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)_auto] items-stretch text-[13px]">
          {["Instrument", "Running", "Agent", ""].map((heading) => (
            <span key={heading} className="pb-1.5 pr-6 text-xs text-muted-foreground">
              {heading}
            </span>
          ))}
          {state.instruments.map((instrument) => (
            <Row key={instrument.name} instrument={instrument} editing={editing} />
          ))}
        </div>
      ) : (
        <p className="m-0 text-[13px] text-muted-foreground">No instruments yet.</p>
      )}
      {adding ? (
        <Add onDone={() => setAdding(false)} />
      ) : (
        <div className="flex gap-2">
          <Button onClick={() => setAdding(true)}>Add instrument</Button>
          {any && (
            <Button variant="ghost" onClick={() => setEditing(!editing)}>
              {editing ? "Done" : "Edit list"}
            </Button>
          )}
        </div>
      )}
    </Card>
  );
}
