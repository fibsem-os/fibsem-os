import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface Session {
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
  // What the agent last printed, when it ended before its log said why.
  output?: string | null;
  // Started by the gateway, not by hand.
  gateway?: boolean;
}

export interface Running extends Session {
  state: "running" | "waiting";
  level?: string;
  who?: string;
  via?: string;
  budget_usd?: number | null;
  restarts: { at: number; kind: "app" | "crash" }[];
}

// An agent the gateway didn't start, connected to the app.
interface Connected {
  name: string;
  last_heard_s: number | null;
  // What to check when the app hasn't heard from it in a while.
  quiet: string | null;
}

export interface AgentState {
  running: Running | null;
  connected: Connected | null;
  sessions: Session[];
  api_key: boolean;
  microscope: string;
}

export interface Playbook {
  name: string;
  hash?: string;
  error?: string;
}

export interface Configuration {
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

export const FIELD =
  "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
export const LEVELS = ["watch", "assist", "drive"];

export const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
export const dollars = (value: number) =>
  Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`;
export const clock = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "–";
export const day = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "–";

export function Field({
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

export function ConfigurationFacts({ name }: { name: string }) {
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

/** What the gateway's Settings fill in for the agent; unset falls back to the
 * playbook's own. */
export interface AgentDefaults {
  playbook: string | null;
  level: string | null;
  budget_usd: number | null;
}

/** Why an agent can't start or resume yet; null when it can. */
export function notReady(
  state: AgentState | undefined,
  appRunning: boolean,
  appStarting?: boolean,
): string | null {
  if (appStarting) return "Waiting for the app to start…";
  if (!appRunning) return "Start the app first: the agent acts through it.";
  if (!state?.api_key) return "No API key on this PC.";
  return null;
}

function Start({
  state,
  appRunning,
  appStarting,
  defaults,
}: {
  state: AgentState;
  appRunning: boolean;
  appStarting?: boolean;
  defaults?: AgentDefaults;
}) {
  const client = useQueryClient();
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
  });
  const usable = (playbooks.data ?? []).filter((p) => !p.error);
  const [playbook, setPlaybook] = useState("");
  // Picked here, or else the default, the playbook's level, or Watch.
  const [picked, setPicked] = useState<string | null>(null);
  // Typed here, or else the default, or the playbook's own budget.
  const [budget, setBudget] = useState<string | null>(null);
  const [goal, setGoal] = useState("");
  const [shown, setShown] = useState(false);
  const chosen =
    playbook ||
    usable.find((p) => p.name === defaults?.playbook)?.name ||
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
  const level = picked ?? defaults?.level ?? fromPlaybook ?? "watch";
  const playbookBudget = config.data?.max_budget_usd ?? null;
  const savedBudget =
    defaults?.budget_usd != null ? String(defaults.budget_usd) : null;
  const shownBudget =
    budget ??
    savedBudget ??
    (playbookBudget != null ? String(playbookBudget) : "");
  const sentBudget = budget?.trim() || savedBudget;
  const start = useMutation({
    mutationFn: (replace: boolean) =>
      api<AgentState>("/gateway/agent/start", {
        method: "POST",
        body: JSON.stringify({
          playbook: chosen,
          goal: goal.trim(),
          level,
          ...(sentBudget ? { budget_usd: sentBudget } : {}),
          replace,
        }),
      }),
    onSuccess: (answer) => client.setQueryData(["gateway/agent"], answer),
  });
  const paired =
    start.error instanceof ApiError && start.error.errorType === "agent_paired";

  return (
    <>
      {state.connected && <ConnectedAgent connected={state.connected} />}
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
            {appStarting ? (
              <span className="text-xs text-muted-foreground">
                Waiting for the app to start…
              </span>
            ) : !appRunning ? (
              <span className="text-xs text-muted-foreground">
                Start the app first: the agent acts through it.
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
                !state.api_key ||
                !chosen
              }
              onClick={() => start.mutate(false)}
            >
              {start.isPending ? (
                <span className="flex items-center gap-1.5">
                  <Loader size={13} className="animate-spin" />
                  Starting…
                </span>
              ) : (
                "Start the agent"
              )}
            </Button>
          </div>
        </>
      )}
    </>
  );
}

/** The running agent in one line, with the app's Agent page for the rest:
 * what it is doing, and its questions, live there. */
function Run({ running }: { running: Running }) {
  const client = useQueryClient();
  // The app's own Agent page, signed in as the operator; absent while it is away.
  const page = useQuery({
    queryKey: ["gateway/app/agent-page"],
    queryFn: () => api<{ url: string }>("/gateway/app/agent-page"),
    enabled: running.state === "running",
    retry: false,
  });
  const [confirm, setConfirm] = useState(false);
  const end = useMutation({
    mutationFn: () =>
      api<AgentState>("/gateway/agent/end", { method: "POST", body: "{}" }),
    onSuccess: (answer) => {
      setConfirm(false);
      client.setQueryData(["gateway/agent"], answer);
      client.invalidateQueries({ queryKey: ["gateway"] });
    },
  });
  const spent = running.spent_usd != null ? dollars(running.spent_usd) : "$0";
  const about = [
    running.playbook ?? "no playbook",
    running.level && title(running.level),
    running.budget_usd != null
      ? `${spent} of ${dollars(running.budget_usd)}`
      : spent,
    `${running.resumed ? "resumed" : "since"} ${clock(running.started)}`,
  ].filter(Boolean);
  const lastRestart = running.restarts[running.restarts.length - 1];
  return (
    <>
      <div className="flex flex-col gap-1">
        <p className="m-0 text-[15px] text-foreground">
          {running.goal ?? "No goal given"}
        </p>
        <p className="m-0 text-[13px] text-muted-foreground">
          {about.join(" · ")}
        </p>
      </div>
      {running.state === "waiting" && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          Waiting for the app to come back, then it resumes.
        </p>
      )}
      {lastRestart && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          Restarted {running.restarts.length === 1 ? "once" : `${running.restarts.length} times`}, the last{" "}
          {lastRestart.kind === "app" ? "after an app restart" : "after a crash"} at{" "}
          {new Date(lastRestart.at * 1000).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
          .
        </p>
      )}
      {end.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(end.error as Error).message}
        </p>
      )}
      <div className="flex flex-wrap items-center justify-end gap-2.5">
        {page.data && (
          <Button
            onClick={() => window.open(page.data.url, "_blank", "noopener")}
          >
            Open the Agent page
          </Button>
        )}
        <Button
          variant="danger"
          disabled={end.isPending}
          onClick={() => setConfirm(true)}
        >
          {end.isPending ? "Ending…" : "End"}
        </Button>
      </div>
      <Dialog
        open={confirm}
        onOpenChange={(open) => {
          setConfirm(open);
          if (!open) end.reset();
        }}
      >
        <DialogContent
          title="End the agent?"
          className="w-[min(460px,calc(100vw-32px))]"
        >
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">
              The agent stops now. The app keeps running, and the session
              stays in History.
            </p>
            {end.error && (
              <p className="m-0 text-destructive">
                {(end.error as Error).message}
              </p>
            )}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setConfirm(false)}>Cancel</Button>
              <Button
                variant="danger"
                disabled={end.isPending}
                onClick={() => end.mutate()}
              >
                {end.isPending ? "Ending…" : "End the agent"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** An agent connected to the app from elsewhere, and when it was last heard. */
function ConnectedAgent({ connected }: { connected: Connected }) {
  const seconds = connected.last_heard_s;
  const heard =
    seconds == null
      ? ""
      : seconds < 60
        ? ` · last heard ${Math.round(seconds)} s ago`
        : ` · last heard ${Math.floor(seconds / 60)} min ago`;
  return (
    <div className="flex flex-col gap-1">
      <p className="m-0 text-[13px]">
        {connected.name} is connected{heard}
      </p>
      {connected.quiet && (
        <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
          {connected.quiet}
        </p>
      )}
    </div>
  );
}

/** The last session, and Resume when it can carry on. */
function LastSession({
  session,
  ready,
}: {
  session: Session;
  ready: string | null;
}) {
  const [open, setOpen] = useState(false);
  const canResume = !!session.session_id && !session.cannot_resume;
  return (
    <div className="flex flex-col gap-1 border-t border-border pt-2.5 text-[13px] text-muted-foreground">
      <span>
        Last session {day(session.started)}: {session.playbook ?? "no playbook"}
        , {session.ended}
        {session.spent_usd != null && `, ${dollars(session.spent_usd)}`}
        {canResume && !open && (
          <>
            {" · "}
            <button
              onClick={() => setOpen(true)}
              className="cursor-pointer border-0 bg-transparent p-0 text-[13px] text-[var(--primary)] hover:underline"
            >
              Resume
            </button>
          </>
        )}
      </span>
      {open && (
        <ResumeForm
          session={session}
          ready={ready}
          onDone={() => setOpen(false)}
        />
      )}
    </div>
  );
}

/** This PC's agent: start one with a playbook, or the one running and its End. */
export function AgentCard({
  appRunning,
  appStarting,
  defaults,
}: {
  appRunning: boolean;
  appStarting?: boolean;
  defaults?: AgentDefaults;
}) {
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => api<AgentState>("/gateway/agent"),
    refetchInterval: 2000,
  });
  const state = agent.data;
  const last = state?.sessions.find((s) => s.ended != null);
  const waiting = state?.running?.state === "waiting";
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className={cn(
            "size-2 rounded-full",
            !state?.running
              ? "bg-[var(--fibsem-pending-color)]"
              : waiting
                ? "bg-[var(--fibsem-semantic-warning-color)]"
                : "bg-[var(--fibsem-ok-color)]",
          )}
        />
        <CardTitle>
          {state?.running
            ? waiting
              ? "Agent, waiting for the app"
              : "Agent"
            : "No agent is running"}
        </CardTitle>
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
          <>
            <Start
              state={state}
              appRunning={appRunning}
              appStarting={appStarting}
              defaults={defaults}
            />
            {last && (
              <LastSession
                session={last}
                ready={notReady(state, appRunning, appStarting)}
              />
            )}
          </>
        ))}
    </Card>
  );
}

/** Level and Budget for carrying an ended session on, then Resume. */
export function ResumeForm({
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
