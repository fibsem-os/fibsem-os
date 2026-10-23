import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Session {
  session_id: string | null;
  started: string | null;
  profile: string | null;
  profile_hash: string | null;
  goal: string | null;
  spent_usd: number | null;
  ended: string | null;
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

interface Profile {
  name: string;
  hash?: string;
  error?: string;
}

interface Configuration {
  profile: { name: string; path: string; hash: string } | null;
  model: string | null;
  effort: string | null;
  max_budget_usd: number | null;
  max_turns: number | null;
  mode: "none" | "files" | "code";
  folder: string | null;
  skills: string[];
  hooks: { when: string; label: string; say: string }[];
  system_prompt: string;
}

const FIELD = "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
const LEVELS = ["watch", "assist", "drive"];

const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
const dollars = (value: number) => (Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`);
const clock = (iso: string | null) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "–";
const day = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "–";

function Field({ label, wide, children }: { label: string; wide?: boolean; children: ReactNode }) {
  return (
    <label className={cn("flex flex-col gap-1 text-xs text-muted-foreground", wide && "sm:col-span-3")}>
      {label}
      {children}
    </label>
  );
}

function ConfigurationFacts({ name }: { name: string }) {
  const config = useQuery({
    queryKey: ["gateway/agent/profiles", name],
    queryFn: () => api<Configuration>(`/gateway/agent/profiles/${encodeURIComponent(name)}`),
    staleTime: 30_000,
  });
  if (config.error) return <p className="m-0 text-[13px] text-destructive">{(config.error as Error).message}</p>;
  const c = config.data;
  if (!c) return <p className="m-0 text-[13px] text-muted-foreground">Reading the profile…</p>;
  const rows: [string, ReactNode][] = [];
  if (c.profile) rows.push(["File", <span className="font-mono text-xs">{`${c.profile.path} · ${c.profile.hash}`}</span>]);
  rows.push(["Model", [c.model ?? "the default model", c.effort && `effort ${c.effort}`].filter(Boolean).join(", ")]);
  rows.push([
    "Budget and turns",
    [c.max_budget_usd != null && dollars(c.max_budget_usd), c.max_turns != null && `${c.max_turns} turns`]
      .filter(Boolean)
      .join(", ") || "–",
  ]);
  rows.push([
    "Files and code",
    c.mode === "none" ? "Off" : `${c.mode === "code" ? "Writes and runs code" : "Reads and writes files"} in ${c.folder}`,
  ]);
  rows.push(["Skills", c.skills.join(", ") || "–"]);
  rows.push([
    "Reminders",
    c.hooks.length === 0 ? (
      "None"
    ) : (
      <span className="flex flex-col gap-1">
        {c.hooks.map((hook, i) => (
          <span key={i}>
            <span className="text-muted-foreground">{hook.label}:</span> {hook.say}
          </span>
        ))}
      </span>
    ),
  ]);
  return <Facts className="border-t border-border pt-2.5 [&_dd]:text-left" rows={rows} />;
}

function Start({ state, appRunning }: { state: AgentState; appRunning: boolean }) {
  const client = useQueryClient();
  const profiles = useQuery({
    queryKey: ["gateway/agent/profiles"],
    queryFn: () => api<Profile[]>("/gateway/agent/profiles"),
    staleTime: 30_000,
  });
  const usable = (profiles.data ?? []).filter((p) => !p.error);
  const [profile, setProfile] = useState("");
  const [level, setLevel] = useState("watch");
  const [budget, setBudget] = useState("");
  const [goal, setGoal] = useState("");
  const [shown, setShown] = useState(false);
  const chosen = profile || usable.find((p) => p.name === "default")?.name || usable[0]?.name || "";
  const start = useMutation({
    mutationFn: (replace: boolean) =>
      api<AgentState>("/gateway/agent/start", {
        method: "POST",
        body: JSON.stringify({
          profile: chosen,
          goal: goal.trim(),
          level,
          ...(budget.trim() ? { budget_usd: budget.trim() } : {}),
          replace,
        }),
      }),
    onSuccess: (answer) => client.setQueryData(["gateway/agent"], answer),
  });
  const paired = start.error instanceof ApiError && start.error.errorType === "agent_paired";
  const last = state.sessions[0];

  return (
    <>
      <p className="m-0 text-[13px] text-muted-foreground">
        No agent is running.
        {last?.ended && ` Last session (${day(last.started)}): ${last.ended}.`}
      </p>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <Field label="Profile">
          <select value={chosen} onChange={(event) => setProfile(event.target.value)} className={FIELD}>
            {usable.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Level">
          <select value={level} onChange={(event) => setLevel(event.target.value)} className={FIELD}>
            {LEVELS.map((l) => (
              <option key={l} value={l}>
                {title(l)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Budget">
          <input
            value={budget}
            onChange={(event) => setBudget(event.target.value)}
            inputMode="decimal"
            placeholder="From the profile"
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
      {profiles.error && <p className="m-0 text-[13px] text-destructive">{(profiles.error as Error).message}</p>}
      {chosen && (
        <div className="flex flex-col gap-2.5">
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
          <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">{(start.error as Error).message}</p>
          <div className="flex flex-wrap items-center justify-end gap-2.5">
            <Button onClick={() => start.reset()}>Cancel</Button>
            <Button variant="primary" disabled={start.isPending} onClick={() => start.mutate(true)}>
              Disconnect and start
            </Button>
          </div>
        </>
      ) : (
        <>
          {start.error && <p className="m-0 text-[13px] text-destructive">{(start.error as Error).message}</p>}
          <div className="flex flex-wrap items-center justify-end gap-2.5">
            {!appRunning ? (
              <span className="text-xs text-muted-foreground">Run the app first: the agent acts through it.</span>
            ) : !state.api_key ? (
              <span className="text-xs text-muted-foreground">
                Set ANTHROPIC_API_KEY for this PC's user, then restart the gateway.
              </span>
            ) : (
              <span className="text-xs text-muted-foreground">Uses ANTHROPIC_API_KEY on this PC.</span>
            )}
            <Button
              variant="primary"
              disabled={start.isPending || !appRunning || !state.api_key || !chosen}
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
  const [confirm, setConfirm] = useState(false);
  const end = useMutation({
    mutationFn: (stopApp: boolean) =>
      api<AgentState>("/gateway/agent/end", { method: "POST", body: JSON.stringify({ stop_app: stopApp }) }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway/agent"], answer);
      client.invalidateQueries({ queryKey: ["gateway"] });
    },
  });
  const rows: [string, ReactNode][] = [];
  if (running.state === "waiting") rows.push(["State", "Waiting for the app to come back, then it resumes"]);
  rows.push(["Goal", running.goal ?? "None given"]);
  rows.push([
    "Profile",
    <>
      {running.profile ?? "–"}
      {running.profile_hash && <span className="text-muted-foreground"> · {running.profile_hash}</span>}
    </>,
  ]);
  if (running.level) {
    rows.push([
      "Level",
      <>
        {title(running.level)}
        {running.level !== "watch" && <span className="text-muted-foreground"> · set again after each app restart</span>}
      </>,
    ]);
  }
  const spent = running.spent_usd != null ? dollars(running.spent_usd) : "$0";
  rows.push(["Spent", running.budget_usd != null ? `${spent} of ${dollars(running.budget_usd)}` : spent]);
  if (running.restarts.length > 0) {
    const lastRestart = running.restarts[running.restarts.length - 1];
    rows.push([
      "Restarts",
      `${running.restarts.length}, the last ${lastRestart.kind === "app" ? "after an app restart" : "after a crash"} at ${new Date(
        lastRestart.at * 1000,
      ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
    ]);
  }
  rows.push(["Started", `${clock(running.started)}${running.who ? ` by ${running.who}, from the ${running.via}` : ""}`]);

  return (
    <>
      <Facts className="[&_dd]:text-left" rows={rows} />
      {end.error && <p className="m-0 text-[13px] text-destructive">{(end.error as Error).message}</p>}
      {confirm ? (
        <>
          <p className="m-0 text-[13px] text-[var(--fibsem-semantic-warning-color)]">
            End the agent and stop the app? Nothing holds the microscope afterwards.
          </p>
          <div className="flex flex-wrap items-center justify-end gap-2.5">
            <Button onClick={() => setConfirm(false)}>Cancel</Button>
            <Button variant="danger" disabled={end.isPending} onClick={() => end.mutate(true)}>
              End and stop the app
            </Button>
          </div>
        </>
      ) : (
        <div className="flex flex-wrap items-center justify-end gap-2.5">
          <a href="/ui/agent" className="mr-auto text-xs text-[var(--primary)] hover:underline">
            Open the Agent page
          </a>
          <Button disabled={end.isPending} onClick={() => setConfirm(true)}>
            End and stop the app
          </Button>
          <Button variant="danger" disabled={end.isPending} onClick={() => end.mutate(false)}>
            {end.isPending ? "Ending…" : "End"}
          </Button>
        </div>
      )}
    </>
  );
}

/** This PC's agent: start one with a profile, or the one running and its End. */
export function AgentCard({ appRunning }: { appRunning: boolean }) {
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
            title={state.running.state === "running" ? "Running" : "Waiting for the app"}
            aria-label={state.running.state === "running" ? "Running" : "Waiting for the app"}
            className={cn(
              "size-2 rounded-full",
              state.running.state === "running"
                ? "bg-[var(--fibsem-ok-color)]"
                : "bg-[var(--fibsem-semantic-warning-color)]",
            )}
          />
        )}
      </div>
      {agent.error && <p className="m-0 text-[13px] text-destructive">{(agent.error as Error).message}</p>}
      {state && (state.running ? <Run running={state.running} /> : <Start state={state} appRunning={appRunning} />)}
    </Card>
  );
}

/** The latest agent sessions on this PC, run by the gateway or by hand. */
export function AgentSessions() {
  const agent = useQuery({ queryKey: ["gateway/agent"], queryFn: () => api<AgentState>("/gateway/agent") });
  const sessions = agent.data?.sessions ?? [];
  if (sessions.length === 0) return null;
  return (
    <Card className="flex flex-col gap-2">
      <CardTitle>Recent agent sessions</CardTitle>
      {sessions.map((s) => (
        <div
          key={s.session_id ?? s.started}
          className="grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)_auto] sm:gap-3"
        >
          <span className="text-muted-foreground tabular-nums">{day(s.started)}</span>
          <span className="min-w-0 break-words">
            {s.profile ?? "no profile"}
            {s.goal && ` · ${s.goal}`}
          </span>
          <span
            className={cn(
              "text-muted-foreground",
              s.ended === "restarts used up" && "text-[var(--fibsem-semantic-warning-color)]",
            )}
          >
            {s.ended ?? "running"}
            {s.spent_usd != null && ` · ${dollars(s.spent_usd)}`}
          </span>
        </div>
      ))}
    </Card>
  );
}
