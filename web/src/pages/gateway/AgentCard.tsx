import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Session {
  session_id: string | null;
  started: string | null;
  playbook: string | null;
  playbook_hash: string | null;
  goal: string | null;
  spent_usd: number | null;
  ended: string | null;
  resumed?: boolean;
  // Why the session can't be resumed here; null when it can.
  cannot_resume?: string | null;
}

interface Running extends Session {
  state: "running" | "waiting";
  level?: string;
  who?: string;
  via?: string;
  budget_usd?: number | null;
  restarts: { at: number; kind: "app" | "crash" }[];
}

interface AgentState {
  running: Running | null;
  sessions: Session[];
  api_key: boolean;
  microscope: string;
}

interface Playbook {
  name: string;
  hash?: string;
  error?: string;
}

interface Configuration {
  playbook: {
    name: string;
    path: string;
    hash: string;
    extends: string[];
    reminders: { when: string; label: string; say: string }[];
    rules: string[];
  };
  level: string | null;
  description: string | null;
  model: string | null;
  effort: string | null;
  max_budget_usd: number | null;
  max_turns: number | null;
  mode: "none" | "files" | "code";
  folder: string | null;
  skills: string[];
  system_prompt: string;
}

const FIELD =
  "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
const LEVELS = ["watch", "assist", "drive"];

const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
const dollars = (value: number) =>
  Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`;
const clock = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "–";
const day = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "–";

function Field({
  label,
  wide,
  children,
}: {
  label: string;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <label
      className={cn(
        "flex flex-col gap-1 text-xs text-muted-foreground",
        wide && "sm:col-span-3",
      )}
    >
      {label}
      {children}
    </label>
  );
}

function ConfigurationFacts({ name }: { name: string }) {
  const config = useQuery({
    queryKey: ["gateway/agent/playbooks", name],
    queryFn: () =>
      api<Configuration>(
        `/gateway/agent/playbooks/${encodeURIComponent(name)}`,
      ),
    staleTime: 30_000,
  });
  if (config.error)
    return (
      <p className="m-0 text-[13px] text-destructive">
        {(config.error as Error).message}
      </p>
    );
  const c = config.data;
  if (!c)
    return (
      <p className="m-0 text-[13px] text-muted-foreground">
        Reading the playbook…
      </p>
    );
  const rows: [string, ReactNode][] = [];
  const p = c.playbook;
  rows.push([
    "File",
    <span className="font-mono text-xs">{`${p.path} · ${p.hash}`}</span>,
  ]);
  if (p.extends.length) rows.push(["Builds on", p.extends.join(", ")]);
  rows.push([
    "Model",
    [c.model ?? "the default model", c.effort && `effort ${c.effort}`]
      .filter(Boolean)
      .join(", "),
  ]);
  rows.push([
    "Budget and turns",
    [
      c.max_budget_usd != null && dollars(c.max_budget_usd),
      c.max_turns != null && `${c.max_turns} turns`,
    ]
      .filter(Boolean)
      .join(", ") || "–",
  ]);
  rows.push([
    "Files and code",
    c.mode === "none"
      ? "Off"
      : `${c.mode === "code" ? "Writes and runs code" : "Reads and writes files"} in ${c.folder}`,
  ]);
  rows.push(["Skills", c.skills.join(", ") || "–"]);
  rows.push([
    "Reminders",
    p.reminders.length === 0 ? (
      "None"
    ) : (
      <span className="flex flex-col gap-1">
        {p.reminders.map((hook, i) => (
          <span key={i}>
            <span className="text-muted-foreground">{hook.label}:</span>{" "}
            {hook.say}
          </span>
        ))}
      </span>
    ),
  ]);
  if (p.rules.length)
    rows.push([
      "Rules",
      <span className="flex flex-col gap-1">
        {p.rules.map((rule, i) => (
          <span key={i}>{rule}</span>
        ))}
      </span>,
    ]);
  return (
    <Facts
      className="border-t border-border pt-2.5 [&_dd]:text-left"
      rows={rows}
    />
  );
}

function Start({
  state,
  appRunning,
  connected,
}: {
  state: AgentState;
  appRunning: boolean;
  connected: boolean;
}) {
  const client = useQueryClient();
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
  });
  const usable = (playbooks.data ?? []).filter((p) => !p.error);
  const [playbook, setPlaybook] = useState("");
  // Picked here, or else the playbook's level, or Watch.
  const [picked, setPicked] = useState<string | null>(null);
  // Typed here, or else the playbook's own budget.
  const [budget, setBudget] = useState<string | null>(null);
  const [goal, setGoal] = useState("");
  const [shown, setShown] = useState(false);
  const chosen =
    playbook ||
    usable.find((p) => p.name === "fibsem-agent")?.name ||
    usable[0]?.name ||
    "";
  const config = useQuery({
    queryKey: ["gateway/agent/playbooks", chosen],
    queryFn: () =>
      api<Configuration>(
        `/gateway/agent/playbooks/${encodeURIComponent(chosen)}`,
      ),
    enabled: !!chosen,
    staleTime: 30_000,
  });
  const fromPlaybook = config.data?.level ?? null;
  const level = picked ?? fromPlaybook ?? "watch";
  const playbookBudget = config.data?.max_budget_usd ?? null;
  const shownBudget =
    budget ?? (playbookBudget != null ? String(playbookBudget) : "");
  const start = useMutation({
    mutationFn: (replace: boolean) =>
      api<AgentState>("/gateway/agent/start", {
        method: "POST",
        body: JSON.stringify({
          playbook: chosen,
          goal: goal.trim(),
          level,
          ...(budget?.trim() ? { budget_usd: budget.trim() } : {}),
          replace,
        }),
      }),
    onSuccess: (answer) => client.setQueryData(["gateway/agent"], answer),
  });
  const paired =
    start.error instanceof ApiError && start.error.errorType === "agent_paired";
  const last = state.sessions[0];

  return (
    <>
      <p className="m-0 text-[13px] text-muted-foreground">
        No agent is running.
        {last?.ended && ` Last session (${day(last.started)}): ${last.ended}.`}
      </p>
      {!state.api_key && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          No API key on this PC, so the agent can't start. Run fibsem-cli key
          set ANTHROPIC_API_KEY on this PC.
        </p>
      )}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <Field label="Playbook">
          <select
            value={chosen}
            onChange={(event) => {
              setPlaybook(event.target.value);
              setPicked(null);
              setBudget(null);
            }}
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
            onChange={(event) => setPicked(event.target.value)}
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
            value={shownBudget}
            onChange={(event) => setBudget(event.target.value)}
            // Left empty, it goes back to the playbook's budget.
            onBlur={() => !budget?.trim() && setBudget(null)}
            inputMode="decimal"
            placeholder="No limit"
            className={FIELD}
          />
        </Field>
        <Field label="Goal" wide>
          <input
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="What the agent should do this session"
            className={FIELD}
          />
        </Field>
      </div>
      {playbooks.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(playbooks.error as Error).message}
        </p>
      )}
      {chosen && (
        <div className="flex flex-col gap-2.5">
          {config.data?.description && (
            <p className="m-0 text-[13px] text-muted-foreground">
              {config.data.description}
            </p>
          )}
          <button
            onClick={() => setShown(!shown)}
            className="w-fit cursor-pointer border-0 bg-transparent p-0 text-xs text-[var(--primary)] hover:underline"
          >
            {shown ? "Hide configuration" : "Show configuration"}
          </button>
          {shown && <ConfigurationFacts name={chosen} />}
        </div>
      )}
      {paired ? (
        <>
          <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
            {(start.error as Error).message}
          </p>
          <div className="flex flex-wrap items-center justify-end gap-2.5">
            <Button onClick={() => start.reset()}>Cancel</Button>
            <Button
              variant="primary"
              disabled={start.isPending}
              onClick={() => start.mutate(true)}
            >
              Disconnect and start
            </Button>
          </div>
        </>
      ) : (
        <>
          {start.error && (
            <p className="m-0 text-[13px] text-destructive">
              {(start.error as Error).message}
            </p>
          )}
          <div className="flex flex-wrap items-center justify-end gap-2.5">
            {!appRunning ? (
              <span className="text-xs text-muted-foreground">
                Start the app first: the agent acts through it.
              </span>
            ) : !connected ? (
              <span className="text-xs text-muted-foreground">
                Connect the microscope first: the agent works on it.
              </span>
            ) : !state.api_key ? null : (
              <span className="text-xs text-muted-foreground">
                Uses ANTHROPIC_API_KEY on this PC.
              </span>
            )}
            <Button
              variant="primary"
              disabled={
                start.isPending ||
                !appRunning ||
                !connected ||
                !state.api_key ||
                !chosen
              }
              onClick={() => start.mutate(false)}
            >
              Start
            </Button>
          </div>
        </>
      )}
    </>
  );
}

function Run({ running }: { running: Running }) {
  const client = useQueryClient();
  // The app's own Agent page, signed in as the operator; absent while it is away.
  const page = useQuery({
    queryKey: ["gateway/app/agent-page"],
    queryFn: () => api<{ url: string }>("/gateway/app/agent-page"),
    enabled: running.state === "running",
    retry: false,
  });
  const end = useMutation({
    mutationFn: () =>
      api<AgentState>("/gateway/agent/end", { method: "POST", body: "{}" }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway/agent"], answer);
      client.invalidateQueries({ queryKey: ["gateway"] });
    },
  });
  const rows: [string, ReactNode][] = [];
  if (running.state === "waiting")
    rows.push(["State", "Waiting for the app to come back, then it resumes"]);
  rows.push(["Goal", running.goal ?? "None given"]);
  rows.push([
    "Playbook",
    <>
      {running.playbook ?? "–"}
      {running.playbook_hash && (
        <span className="text-muted-foreground">
          {" "}
          · {running.playbook_hash}
        </span>
      )}
    </>,
  ]);
  if (running.level) {
    rows.push([
      "Level",
      <>
        {title(running.level)}
        {running.level !== "watch" && (
          <span className="text-muted-foreground">
            {" "}
            · set again after each app restart
          </span>
        )}
      </>,
    ]);
  }
  const spent = running.spent_usd != null ? dollars(running.spent_usd) : "$0";
  rows.push([
    "Spent",
    running.budget_usd != null
      ? `${spent} of ${dollars(running.budget_usd)}`
      : spent,
  ]);
  if (running.restarts.length > 0) {
    const lastRestart = running.restarts[running.restarts.length - 1];
    rows.push([
      "Restarts",
      `${running.restarts.length}, the last ${lastRestart.kind === "app" ? "after an app restart" : "after a crash"} at ${new Date(
        lastRestart.at * 1000,
      ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
    ]);
  }
  rows.push([
    running.resumed ? "Resumed" : "Started",
    `${clock(running.started)}${running.who ? ` by ${running.who}, from the ${running.via}` : ""}`,
  ]);

  return (
    <>
      <Facts className="[&_dd]:text-left" rows={rows} />
      {end.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(end.error as Error).message}
        </p>
      )}
      <div className="flex flex-wrap items-center justify-end gap-2.5">
        {page.data ? (
          <a
            href={page.data.url}
            target="_blank"
            rel="noopener"
            className="mr-auto text-xs text-[var(--primary)] hover:underline"
          >
            Open the Agent page
          </a>
        ) : (
          <span className="mr-auto" />
        )}
        <Button
          variant="danger"
          disabled={end.isPending}
          onClick={() => end.mutate()}
        >
          {end.isPending ? "Ending…" : "End"}
        </Button>
      </div>
    </>
  );
}

/** This PC's agent: start one with a playbook, or the one running and its End. */
export function AgentCard({
  appRunning,
  connected,
}: {
  appRunning: boolean;
  connected: boolean;
}) {
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => api<AgentState>("/gateway/agent"),
    refetchInterval: 2000,
  });
  const state = agent.data;
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <CardTitle>Agent</CardTitle>
        {state?.running && (
          <span
            title={
              state.running.state === "running"
                ? "Running"
                : "Waiting for the app"
            }
            aria-label={
              state.running.state === "running"
                ? "Running"
                : "Waiting for the app"
            }
            className={cn(
              "size-2 rounded-full",
              state.running.state === "running"
                ? "bg-[var(--fibsem-ok-color)]"
                : "bg-[var(--fibsem-semantic-warning-color)]",
            )}
          />
        )}
      </div>
      {agent.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(agent.error as Error).message}
        </p>
      )}
      {state &&
        (state.running ? (
          <Run running={state.running} />
        ) : (
          <Start state={state} appRunning={appRunning} connected={connected} />
        ))}
    </Card>
  );
}

/** Level and Budget for carrying an ended session on, then Resume. */
function ResumeForm({
  session,
  ready,
  onDone,
}: {
  session: Session;
  ready: string | null;
  onDone: () => void;
}) {
  const client = useQueryClient();
  const [level, setLevel] = useState("watch");
  // Typed here, or else the playbook's own budget.
  const [budget, setBudget] = useState<string | null>(null);
  const config = useQuery({
    queryKey: ["gateway/agent/playbooks", session.playbook],
    queryFn: () =>
      api<Configuration>(
        `/gateway/agent/playbooks/${encodeURIComponent(session.playbook ?? "")}`,
      ),
    enabled: !!session.playbook,
    staleTime: 30_000,
  });
  const playbookBudget = config.data?.max_budget_usd ?? null;
  const shownBudget =
    budget ?? (playbookBudget != null ? String(playbookBudget) : "");
  const resume = useMutation({
    mutationFn: (replace: boolean) =>
      api<AgentState>("/gateway/agent/resume", {
        method: "POST",
        body: JSON.stringify({
          session_id: session.session_id,
          level,
          ...(budget?.trim() ? { budget_usd: budget.trim() } : {}),
          replace,
        }),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway/agent"], answer);
      onDone();
    },
  });
  const paired =
    resume.error instanceof ApiError &&
    resume.error.errorType === "agent_paired";
  return (
    <div className="flex flex-col gap-2.5 pt-2 sm:col-span-3">
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
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
            value={shownBudget}
            onChange={(event) => setBudget(event.target.value)}
            // Left empty, it goes back to the playbook's budget.
            onBlur={() => !budget?.trim() && setBudget(null)}
            inputMode="decimal"
            placeholder="No limit"
            className={FIELD}
          />
        </Field>
      </div>
      <p className="m-0 text-xs text-muted-foreground">
        Carries on with the same playbook, goal and folder it had.
      </p>
      {resume.error && (
        <p
          className={cn(
            "m-0 text-[13px]",
            paired
              ? "text-[var(--fibsem-semantic-warning-color)]"
              : "text-destructive",
          )}
        >
          {(resume.error as Error).message}
        </p>
      )}
      <div className="flex flex-wrap items-center justify-end gap-2.5">
        {ready && (
          <span className="text-xs text-muted-foreground">{ready}</span>
        )}
        <Button onClick={onDone}>Cancel</Button>
        <Button
          variant="primary"
          disabled={resume.isPending || !!ready}
          onClick={() => resume.mutate(paired)}
        >
          {paired
            ? "Disconnect and resume"
            : resume.isPending
              ? "Resuming…"
              : "Resume"}
        </Button>
      </div>
    </div>
  );
}

/** The latest agent sessions on this PC, run by the gateway or by hand; an
 * ended one can be resumed while no agent runs. */
export function AgentSessions({
  appRunning,
  connected,
}: {
  appRunning: boolean;
  connected: boolean;
}) {
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => api<AgentState>("/gateway/agent"),
  });
  const [open, setOpen] = useState<string | null>(null);
  const state = agent.data;
  const sessions = state?.sessions ?? [];
  if (sessions.length === 0) return null;
  const ready = !appRunning
    ? "Start the app first: the agent acts through it."
    : !connected
      ? "Connect the microscope first: the agent works on it."
      : !state?.api_key
        ? "No API key on this PC."
        : null;
  return (
    <Card className="flex flex-col gap-2">
      <CardTitle>Recent agent sessions</CardTitle>
      {sessions.map((s) => {
        const ended = s.ended != null && !!s.session_id;
        const canResume = ended && !s.cannot_resume && !state?.running;
        return (
          <div
            key={s.session_id ?? s.started}
            className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)_auto] sm:gap-3"
          >
            <span className="text-muted-foreground tabular-nums">
              {day(s.started)}
            </span>
            <span className="min-w-0 break-words">
              {s.playbook ?? "no playbook"}
              {s.goal && ` · ${s.goal}`}
            </span>
            <span className="flex items-center gap-3">
              <span
                className={cn(
                  "text-muted-foreground",
                  s.ended === "restarts used up" &&
                    "text-[var(--fibsem-semantic-warning-color)]",
                )}
              >
                {s.ended ?? "running"}
                {s.spent_usd != null && ` · ${dollars(s.spent_usd)}`}
              </span>
              {canResume && open !== s.session_id && (
                <button
                  onClick={() => setOpen(s.session_id)}
                  className="cursor-pointer border-0 bg-transparent p-0 text-xs text-[var(--primary)] hover:underline"
                >
                  Resume
                </button>
              )}
            </span>
            {ended && s.cannot_resume && (
              <span className="text-xs text-muted-foreground sm:col-start-2 sm:col-span-2">
                Can't resume:{" "}
                {s.cannot_resume.charAt(0).toLowerCase() +
                  s.cannot_resume.slice(1)}
              </span>
            )}
            {canResume && open === s.session_id && (
              <ResumeForm
                session={s}
                ready={ready}
                onDone={() => setOpen(null)}
              />
            )}
          </div>
        );
      })}
    </Card>
  );
}
