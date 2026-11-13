import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
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

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted-foreground">
      {label}
      {children}
    </label>
  );
}

function Failed({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p className="m-0 text-[13px] text-destructive">
      {(error as Error).message}
    </p>
  );
}

/** What the instrument's PC is doing, in a line. */
function status(i: Instrument): { text: string; warn: boolean } {
  if (i.reachable === null) return { text: "Not read yet", warn: false };
  if (!i.reachable)
    return {
      text: `Not reachable${i.since ? ` since ${fmt.when(i.since)}` : ""}`,
      warn: true,
    };
  if (!i.paired) return { text: "Reachable, not paired", warn: true };
  if (!i.gateway) return { text: "Reachable", warn: !!i.why };
  if (i.gateway.switching) return { text: "Its app is starting or stopping", warn: false };
  if (i.gateway.mode !== "running") return { text: "No app running", warn: false };
  const experiment = i.gateway.experiment
    ?.split(/[\\/]/)
    .filter(Boolean)
    .pop();
  return {
    text: `Running ${i.gateway.app ?? "an app"}${experiment ? ` · ${experiment}` : ""}`,
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

function Pair({ name }: { name: string }) {
  const [code, setCode] = useState("");
  const pair = useHubAction((value: string) =>
    api(`${at(name)}/pair`, {
      method: "POST",
      body: JSON.stringify({ code: value }),
    }),
  );
  return (
    <div className="flex flex-col gap-1.5">
      <p className="m-0 text-[13px] text-muted-foreground">
        On that PC, run <code>fibsem-gateway pair --hub</code> and type the code
        it shows.
      </p>
      <div className="flex items-end gap-2">
        <Field label="Code">
          <input
            value={code}
            onChange={(event) => setCode(event.target.value)}
            inputMode="numeric"
            className={cn(FIELD, "w-28")}
          />
        </Field>
        <Button variant="primary"
          disabled={!code.trim() || pair.isPending}
          onClick={() => pair.mutate(code.trim())}
        >
          Pair
        </Button>
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
    <div className="flex flex-col gap-2.5">
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <Field label="Playbook">
          <select
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
        </Field>
        <Field label="Level">
          <select
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
        </Field>
        <Field label="Budget ($)">
          <input
            value={budget}
            onChange={(event) => setBudget(event.target.value)}
            inputMode="decimal"
            placeholder="The playbook's"
            className={FIELD}
          />
        </Field>
      </div>
      <Field label="Goal">
        <textarea
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          rows={2}
          className={FIELD}
        />
      </Field>
      <p className="m-0 text-[13px] text-muted-foreground">
        The agent runs on this PC and works through {name}'s app. Stop and the
        level at {name} still decide what it may do.
      </p>
      <Failed error={start.error} />
      <div className="flex justify-end gap-2">
        <Button onClick={onDone}>
          Cancel
        </Button>
        <Button variant="primary"
          disabled={!chosen || start.isPending}
          onClick={() => start.mutate()}
        >
          Start agent
        </Button>
      </div>
    </div>
  );
}

function Session({ name, session }: { name: string; session: HubSession }) {
  const end = useHubAction(() =>
    api(`${at(name)}/agent/end`, { method: "POST" }),
  );
  const level = useHubAction((value: string) =>
    api(`${at(name)}/level`, {
      method: "PUT",
      body: JSON.stringify({ level: value }),
    }),
  );
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[13px]">
        <span>
          Agent {session.state === "running" ? "running" : "waiting for the app"}
          {session.playbook && (
            <span className="text-muted-foreground"> · {session.playbook}</span>
          )}
          {session.started && (
            <span className="text-muted-foreground">
              {" "}
              · since{" "}
              {fmt.clock(session.started)}
            </span>
          )}
        </span>
        <label className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
          Level
          <select
            value={session.level ?? "watch"}
            disabled={level.isPending}
            onChange={(event) => level.mutate(event.target.value)}
            className={FIELD}
          >
            {LEVELS.map((l) => (
              <option key={l} value={l}>
                {title(l)}
              </option>
            ))}
          </select>
        </label>
        <Button
         
          disabled={end.isPending}
          onClick={() => end.mutate(undefined)}
        >
          End agent
        </Button>
      </div>
      {session.goal && (
        <p className="m-0 break-words text-[13px] text-muted-foreground">
          {session.goal}
        </p>
      )}
      <Failed error={level.error ?? end.error} />
    </div>
  );
}

function Row({ instrument: i }: { instrument: Instrument }) {
  const [starting, setStarting] = useState(false);
  const [removing, setRemoving] = useState(false);
  const remove = useHubAction(() => api(at(i.name), { method: "DELETE" }));
  const { text, warn } = status(i);
  const ready = i.reachable && i.paired && i.gateway?.mode === "running";
  return (
    <div className="flex flex-col gap-2 border-t border-border py-3 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[13px]">
        <span className="font-semibold">{i.name}</span>
        <span className={cn(warn && WARNING)}>{text}</span>
        <span className="ml-auto text-muted-foreground">
          {i.ssh ?? i.url}
          {i.gateway?.version && ` · fibsemOS ${i.gateway.version}`}
        </span>
      </div>
      {i.why && (
        <p className="m-0 break-words text-[13px] text-muted-foreground">
          {i.why}
        </p>
      )}
      {i.reachable && !i.paired && <Pair name={i.name} />}
      {i.session && <Session name={i.name} session={i.session} />}
      {!i.session && i.agent && (
        <p className="m-0 text-[13px] text-muted-foreground">
          An agent started at {i.name} is running there
          {i.agent.playbook && ` (${i.agent.playbook})`}.
        </p>
      )}
      {starting && <Start name={i.name} onDone={() => setStarting(false)} />}
      {!starting && (
        <div className="flex items-center justify-end gap-2">
          {removing ? (
            <>
              <span className="text-[13px] text-muted-foreground">
                Remove {i.name} from this hub?
              </span>
              <Button onClick={() => setRemoving(false)}>
                Keep
              </Button>
              <Button
               
                disabled={remove.isPending}
                onClick={() => remove.mutate(undefined)}
              >
                Remove
              </Button>
            </>
          ) : (
            <>
              {!i.session && (
                <Button onClick={() => setRemoving(true)}>
                  Remove
                </Button>
              )}
              {!i.session && !i.agent && (
                <Button variant="primary"
                  disabled={!ready}
                  title={
                    ready
                      ? undefined
                      : `Start the app at ${i.name} first: the agent works through it.`
                  }
                  onClick={() => setStarting(true)}
                >
                  Start agent
                </Button>
              )}
            </>
          )}
        </div>
      )}
      <Failed error={remove.error} />
    </div>
  );
}

function Add({ onDone }: { onDone?: () => void }) {
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
      setName("");
      setSsh("");
      onDone?.();
    },
  });
  return (
    <div className="flex flex-col gap-1.5 border-t border-border pt-3">
      <div className="flex flex-wrap items-end gap-2">
        <Field label="Name">
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            className={cn(FIELD, "w-44")}
          />
        </Field>
        <Field label="Where ssh connects to its PC">
          <input
            value={ssh}
            onChange={(event) => setSsh(event.target.value)}
            placeholder="user@host"
            className={cn(FIELD, "w-60")}
          />
        </Field>
        <Button variant="primary"
          disabled={!name.trim() || !ssh.trim() || add.isPending}
          onClick={() => add.mutate()}
        >
          Add instrument
        </Button>
      </div>
      <p className="m-0 text-[13px] text-muted-foreground">
        This PC needs its ssh key on that PC, and one connection made by hand
        first. The gateway there keeps listening on that PC only.
      </p>
      <Failed error={add.error} />
    </div>
  );
}

/** The instruments this gateway is the hub for: what each PC runs, pairing,
 * and the agent this PC runs for it. A gateway with none says so in a line. */
export function Instruments() {
  const hub = useQuery({
    queryKey: ["hub"],
    queryFn: () => api<HubState>("/hub"),
    refetchInterval: 3000,
  });
  const [adding, setAdding] = useState(false);
  const state = hub.data;
  if (!state) return null;
  if (!state.hub && !adding)
    return (
      <p className="m-0 px-1 text-[13px] text-muted-foreground">
        This gateway looks after this PC only.{" "}
        <button
          type="button"
          className="cursor-pointer border-0 bg-transparent p-0 text-[13px] text-foreground underline"
          onClick={() => setAdding(true)}
        >
          Add another microscope PC
        </button>{" "}
        to start and watch agents for it from here.
      </p>
    );
  return (
    <Card className="flex flex-col gap-2">
      {state.instruments.map((instrument) => (
        <Row key={instrument.name} instrument={instrument} />
      ))}
      <Add onDone={() => setAdding(false)} />
    </Card>
  );
}
