import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, Facts } from "@/components/ui/card";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  type AgentState,
  dollars,
  ResumeForm,
  type Session,
} from "@/pages/gateway/AgentCard";

/** One start or stop of the app, as the gateway's history file has it. */
export interface Switch {
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

type Kind = "app" | "agent";
type Entry =
  | { kind: "app"; at: number; switch: Switch }
  | { kind: "agent"; at: number; session: Session; running: boolean };

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

const lastPart = (folder: string) =>
  folder.split(/[\\/]/).filter(Boolean).pop();

const ROW =
  "grid grid-cols-1 gap-0.5 border-t border-border py-2 text-[13px] sm:grid-cols-[170px_52px_minmax(0,1fr)_auto] sm:gap-3";

function Tag({ kind }: { kind: Kind }) {
  return (
    <span
      className={
        kind === "agent"
          ? "text-[var(--fibsem-agent-mark-color)]"
          : "text-muted-foreground"
      }
    >
      {kind === "agent" ? "Agent" : "App"}
    </span>
  );
}

function AppRow({ s, name }: { s: Switch; name: string }) {
  return (
    <div className={ROW}>
      <span className="text-muted-foreground tabular-nums">
        {fmt.when(s.at)}
      </span>
      <Tag kind="app" />
      <span
        className={cn(
          "min-w-0 break-words",
          s.result.startsWith("refused") &&
            "text-[var(--fibsem-semantic-warning-color)]",
        )}
      >
        {happened(s, name)}
        {s.experiment && (
          <span className="text-muted-foreground" title={s.experiment}>
            {" "}
            · {lastPart(s.experiment)}
          </span>
        )}
      </span>
      <span className="text-muted-foreground">
        {s.who} · {s.via}
      </span>
    </div>
  );
}

function AgentRow({
  session,
  running,
  canResume,
  ready,
}: {
  session: Session;
  running: boolean;
  canResume: boolean;
  ready: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [resuming, setResuming] = useState(false);
  const status = running ? "running" : (session.ended ?? "running");
  const rows: [string, ReactNode][] = [
    [
      "Playbook",
      <>
        {session.playbook ?? "–"}
        {session.playbook_hash && (
          <span className="text-muted-foreground">
            {" "}
            · {session.playbook_hash}
          </span>
        )}
      </>,
    ],
    ["Goal", session.goal ?? "None given"],
    ["Spent", session.spent_usd != null ? dollars(session.spent_usd) : "–"],
    ["Ended", running ? "Still running" : (session.ended ?? "–")],
  ];
  if (session.output)
    rows.push([
      "Last said",
      <span className="font-mono text-xs">{session.output}</span>,
    ]);
  if (session.cannot_resume) rows.push(["Can't resume", session.cannot_resume]);
  if (session.session_id)
    rows.push([
      "Session",
      <span className="font-mono text-xs">{session.session_id}</span>,
    ]);
  const Chevron = open ? ChevronDown : ChevronRight;
  return (
    <>
      <div className={ROW}>
        <span className="text-muted-foreground tabular-nums">
          {fmt.when(session.started)}
        </span>
        <Tag kind="agent" />
        <button
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex min-w-0 cursor-pointer items-start gap-1 border-0 bg-transparent p-0 text-left text-[13px] text-foreground"
        >
          <Chevron size={14} className="mt-0.5 shrink-0 text-muted-foreground" />
          <span className="min-w-0 break-words">
            {session.playbook ?? "no playbook"}
            {session.goal && ` · ${session.goal}`}
            <span
              className={cn(
                "text-muted-foreground",
                status === "restarts used up" &&
                  "text-[var(--fibsem-semantic-warning-color)]",
              )}
            >
              {" "}
              · {status}
              {session.spent_usd != null && ` · ${dollars(session.spent_usd)}`}
            </span>
          </span>
        </button>
        <span className="text-muted-foreground">
          {session.gateway ? "gateway" : "by hand"}
        </span>
      </div>
      {open && (
        <div className="flex flex-col gap-2.5 bg-background/40 px-3 py-2.5 sm:ml-[234px]">
          <Facts className="[&_dd]:text-left" rows={rows} />
          {canResume && !resuming && (
            <div className="flex justify-end">
              <Button onClick={() => setResuming(true)}>Resume…</Button>
            </div>
          )}
          {canResume && resuming && (
            <ResumeForm
              session={session}
              ready={ready}
              onDone={() => setResuming(false)}
            />
          )}
        </div>
      )}
    </>
  );
}

/** Starts and stops of the app and this PC's agent sessions, newest first. */
export function History({
  apps,
  ready,
}: {
  apps: { key: string; name: string }[];
  /** Why a session can't be resumed yet; null when it can. */
  ready: string | null;
}) {
  const [shown, setShown] = useState<Kind | "all">("all");
  const history = useQuery({
    queryKey: ["gateway/history"],
    queryFn: () => api<{ switches: Switch[] }>("/gateway/history"),
    refetchInterval: 5000,
  });
  const agent = useQuery({
    queryKey: ["gateway/agent"],
    queryFn: () => api<AgentState>("/gateway/agent"),
    refetchInterval: 5000,
  });
  const state = agent.data;
  const entries: Entry[] = [
    ...(history.data?.switches ?? []).map(
      (s): Entry => ({ kind: "app", at: s.at * 1000, switch: s }),
    ),
    ...[...(state?.running ? [state.running] : []), ...(state?.sessions ?? [])]
      .filter((s) => s.started)
      .map(
        (s): Entry => ({
          kind: "agent",
          at: Date.parse(s.started!),
          session: s,
          running: s === state?.running,
        }),
      ),
  ]
    .filter((e) => shown === "all" || e.kind === shown)
    .sort((a, b) => b.at - a.at);
  const error = (history.error ?? agent.error) as Error | null;
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex gap-1.5 pb-1">
        {(["all", "app", "agent"] as const).map((k) => (
          <Button
            key={k}
            variant={shown === k ? "default" : "ghost"}
            onClick={() => setShown(k)}
          >
            {k === "all" ? "All" : k === "app" ? "App" : "Agent"}
          </Button>
        ))}
      </div>
      {error && (
        <p className="m-0 text-[13px] text-destructive">{error.message}</p>
      )}
      {entries.length === 0 && !error && (
        <p className="m-0 text-[13px] text-muted-foreground">
          Nothing yet.
        </p>
      )}
      {entries.map((e, i) =>
        e.kind === "app" ? (
          <AppRow
            key={`app-${i}`}
            s={e.switch}
            name={
              apps.find((a) => a.key === e.switch.app)?.name ??
              e.switch.app ??
              "the app"
            }
          />
        ) : (
          <AgentRow
            key={e.session.session_id ?? `agent-${i}`}
            session={e.session}
            running={e.running}
            canResume={
              !e.running &&
              e.session.ended != null &&
              !!e.session.session_id &&
              !e.session.cannot_resume &&
              !state?.running
            }
            ready={ready}
          />
        ),
      )}
    </Card>
  );
}
