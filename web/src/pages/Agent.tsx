import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, BookOpen, ChevronDown, ChevronRight, ListChecks, ListFilter, Loader } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import { onEvents } from "@/lib/events";
import { Markdown } from "@/lib/markdown";
import type { AgentPlan, AgentState, RecentRow, ReportedSession, TranscriptLine } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ConnectCard, RequestCard, SuggestionCard } from "./agent/Cards";
import { carriesFiles, splitAttached, useAttachments } from "./agent/attach";
import { SentFiles } from "./agent/Attachments";
import { Composer } from "./agent/Composer";
import { KnowledgeDialog } from "./agent/Knowledge";
import { EvidenceImage } from "./agent/Pictures";
import { PlanDialog } from "./agent/PlanDialog";
import { hasPanel, SidePanel } from "./agent/SidePanel";

// The agent's news (lib/events.ts) reads it, its session and its plan again at once.
// Last heard, spend and the transcript move with no news, so they are also read on one
// clock: every 10 s while the agent works, every minute while none is heard.
const WORKING_EVERY_MS = 10000;
const IDLE_EVERY_MS = 60000;
// Heard this recently: working (fibsem.ai.agent.levels.AGENT_LISTENING_S).
const WORKING_S = 120;
// While it works, the app's events also read the transcript, at most this often;
// fibsem-agent's own agent_transcript event reads it at once.
const TRANSCRIPT_GAP_MS = 3000;

const working = (agent: AgentState | undefined) => agent?.last_heard_s != null && agent.last_heard_s < WORKING_S;

const SHOWS = [
  ["all", "Everything"],
  ["conversation", "Conversation"],
  ["actions", "Actions"],
  ["transcript", "Transcript"],
] as const;
type Show = (typeof SHOWS)[number][0];

const TONE: Record<string, string> = {
  ok: "text-[var(--fibsem-ok-color)]",
  error: "text-destructive",
  waiting: "text-[var(--fibsem-semantic-warning-color)]",
  running: "text-link",
  muted: "text-muted-foreground",
};

const capital = (text: string) => text.replace(/^./, (c) => c.toUpperCase());

function Time({ when }: { when: string }) {
  return <div className="w-12 shrink-0 pt-0.5 text-[12px] text-muted-foreground tabular-nums">{when}</div>;
}

/** Why a message isn't read yet, and when it will be. */
function unreadWhy(agent: AgentState | undefined): [string, string] {
  if (agent?.last_heard_s == null) return ["No agent connected", "An agent reads it when one connects."];
  if (agent.level === "off") return ["The agent is off", "The agent reads it when its level is back on."];
  return ["The agent is working...", "The agent reads it when its current turn ends."];
}

function Message({ row, unread }: { row: RecentRow; unread?: [string, string] }) {
  const sent = splitAttached(row.text);
  if (row.kind === "you")
    return (
      <div className={cn("my-1 flex gap-2 rounded-md bg-card py-2 pr-2.5 pl-2.5", unread && "opacity-75")}>
        <Time when={row.when} />
        <div className="min-w-0 flex-1">
          <div className="flex gap-2 text-[12px] font-semibold text-muted-foreground">
            You
            {unread && (
              <span className="ml-auto cursor-help font-normal" title={unread[1]}>
                Not read yet. {unread[0]}
              </span>
            )}
          </div>
          <div className="whitespace-pre-wrap break-words">{sent.text}</div>
          <SentFiles files={sent.files} />
          {row.evidence && <EvidenceImage id={row.evidence.evidence_id} />}
        </div>
      </div>
    );
  return (
    <div className="my-1 flex gap-2 py-2 pl-2.5">
      <Time when={row.when} />
      <div className="min-w-0 flex-1 break-words">
        <div className="text-[12px] font-semibold text-[#c4b5fd]">{row.kind === "note" ? "Agent · note" : "Agent"}</div>
        <Markdown text={row.text} />
        {row.evidence && <EvidenceImage id={row.evidence.evidence_id} />}
      </div>
    </div>
  );
}

function Action({ row }: { row: RecentRow }) {
  const [evidence, setEvidence] = useState(false);
  const at = row.level ? `at ${capital(row.level)}` : "";
  const tip = [row.why ? `${row.outcome}: ${row.why}` : row.outcome, [at, row.when].filter(Boolean).join(" · ")].join("\n");
  return (
    <>
      <span className={cn("text-[13px]", TONE[row.tone] ?? "")} title={tip}>
        {capital(row.outcome)}
      </span>
      <span className="min-w-0 text-[13px] break-words">
        {row.text}
        {row.why && row.tone === "error" && <span className="text-muted-foreground"> · {row.why}</span>}
        {row.evidence && (
          <button
            className="ml-1.5 cursor-pointer border-0 bg-transparent p-0 text-[12px] text-link"
            onClick={() => setEvidence(!evidence)}
          >
            evidence
          </button>
        )}
        {evidence && row.evidence && (
          <div className="mt-1.5">
            <EvidenceImage id={row.evidence.evidence_id} />
          </div>
        )}
      </span>
    </>
  );
}

/** A run of actions in one line, "4 actions, 1 refused". A run seen running stays
 * open when it finishes, so the list doesn't jump; the reader folds it. */
function Actions({ rows, folded }: { rows: RecentRow[]; folded: boolean }) {
  const live = rows.some((r) => r.tone === "running" || r.tone === "waiting");
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  const shown = !folded || open;
  const counts = new Map<string, number>();
  for (const r of rows) if (r.tone !== "ok") counts.set(r.outcome, (counts.get(r.outcome) ?? 0) + 1);
  const summary = [
    `${rows.length} ${rows.length === 1 ? "action" : "actions"}`,
    ...[...counts].map(([outcome, n]) => `${n} ${outcome}`),
  ].join(", ");
  return (
    <div className="flex gap-2 py-1.5 pl-2.5">
      <Time when={rows[0].when} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {folded && (
          <button
            className="flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-left text-[13px] text-muted-foreground hover:text-foreground"
            onClick={() => setOpen(!shown)}
            aria-expanded={shown}
          >
            {shown ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {summary}
          </button>
        )}
        {shown && (
          <div className={cn("grid grid-cols-[72px_minmax(0,1fr)] gap-x-2.5 gap-y-1", folded && "pl-5")}>
            {rows.map((r, i) => (
              <Action key={i} row={r} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** How long a step took, when it took long enough to say: "12 s", "3 min 5 s". */
function took(s: number | undefined): string | null {
  if (s == null || s < 2) return null;
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  const rest = Math.round(s - m * 60);
  return rest ? `${m} min ${rest} s` : `${m} min`;
}

/** A step's answer, done or failed, by its session and number. */
type Answers = Map<string, TranscriptLine>;
const stepKey = (session: string, seq: number | undefined) => `${session}|${seq}`;
/** A line that answers a step: shown on the step, never on its own. */
const answers = (l: TranscriptLine) => (l.kind === "done" || l.kind === "failed") && l.step != null;

/** What fibsem-agent printed between two rows, in one line, "Working · 6 steps".
 * Open while the agent still adds to it, and it stays open when the agent moves
 * on, as a run of actions does; the reader folds it. *answered* holds every
 * step's answer, which may come after the next row; a done step says how long
 * it took. *running* says whether a step still runs (`Record`). */
function Working({
  lines,
  folded,
  live,
  answered,
  running,
}: {
  lines: TranscriptLine[];
  folded: boolean;
  live: boolean;
  answered: Answers;
  running: (step: TranscriptLine) => boolean;
}) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  const shown = !folded || open;
  const answerOf = (l: TranscriptLine) => (l.kind === "step" ? answered.get(stepKey(l.session_id, l.seq)) : undefined);
  const steps = lines.filter((l) => l.kind === "step").length;
  const failed = lines.filter((l) => l.kind === "failed" || answerOf(l)?.kind === "failed").length;
  // A block of the agent's words alone is "Working", not "Working · 0 steps".
  const count = steps ? [`${steps} ${steps === 1 ? "step" : "steps"}`] : [];
  const summary = ["Working", ...count, ...(failed ? [`${failed} failed`] : [])].join(" · ");
  return (
    <div className="flex gap-2 py-1.5 pl-2.5">
      <Time when={lines[0].t.split("T")[1]?.slice(0, 5) ?? ""} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {folded && (
          <button
            className="flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-left text-[13px] text-muted-foreground hover:text-foreground"
            onClick={() => setOpen(!shown)}
            aria-expanded={shown}
          >
            {shown ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {summary}
          </button>
        )}
        {shown && (
          <div className={cn("grid grid-cols-[72px_minmax(0,1fr)] gap-x-2.5 gap-y-1 text-[13px]", folded && "pl-5")}>
            {lines.map((l, i) => {
              if (l.kind === "said")
                return (
                  <div key={i} className="col-span-2 whitespace-pre-wrap break-words text-muted-foreground italic">
                    {l.text}
                  </div>
                );
              if (l.kind === "context") {
                const [head, ...rest] = l.text.split(": ");
                return (
                  <div key={i} className="col-span-2 break-words">
                    <span className="font-medium">{rest.length ? head : l.text}</span>
                    {rest.length > 0 && <span className="text-muted-foreground"> · {rest.join(": ")}</span>}
                  </div>
                );
              }
              const answer = answerOf(l);
              const reason = l.kind === "failed" ? l.text : answer?.kind === "failed" ? answer.text : undefined;
              const runs = l.kind === "step" && !answer && running(l);
              const duration = took(answer?.took_s);
              return (
                <div key={i} className="contents">
                  <span
                    className={cn(
                      "flex items-center gap-1",
                      reason != null ? "text-destructive" : "text-muted-foreground",
                    )}
                  >
                    {l.kind === "failed" ? "Failed" : l.verb}
                    {runs && <Loader size={11} className="shrink-0 animate-spin" aria-label="running" />}
                  </span>
                  <span className="min-w-0 break-words">
                    {l.kind === "step" && <span className="font-mono text-[12px]">{l.text}</span>}
                    {duration && reason == null && <span className="text-muted-foreground"> · {duration}</span>}
                    {reason != null && (
                      <span className="text-muted-foreground">
                        {l.kind === "step" ? " · failed: " : ""}
                        {reason}
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

const time = (t: string) => Date.parse(t);
const LIVE_FOR_MS = 60_000;

/** *heard*: the agent was heard from lately, so a step it started may still run
 * with no lines coming (a long milling call). A step runs until its answer
 * comes, while the agent is heard from and no later step of its session has
 * been answered (a call cut short never answers). An agent that never says a
 * step is done shows only its newest step as running. */
function Record({
  rows,
  lines,
  show,
  heard,
}: {
  rows: RecentRow[];
  lines: TranscriptLine[];
  show: Show;
  heard: boolean;
}) {
  const out: JSX.Element[] = [];
  let run: RecentRow[] = [];
  let working: TranscriptLine[] = [];
  const answered: Answers = new Map();
  const lastAnswered = new Map<string, number>();
  for (const l of lines) {
    if (!answers(l)) continue;
    answered.set(stepKey(l.session_id, l.step), l);
    lastAnswered.set(l.session_id, Math.max(lastAnswered.get(l.session_id) ?? 0, l.step ?? 0));
  }
  const saysDone = lines.some((l) => l.kind === "done");
  const lastStep = [...lines].reverse().find((l) => l.kind === "step");
  const running = (step: TranscriptLine) =>
    heard && (saysDone ? (step.seq ?? 0) > (lastAnswered.get(step.session_id) ?? 0) : step === lastStep);
  // The answers show on their steps, not as lines of their own.
  lines = lines.filter((l) => !answers(l));
  // Keyed by what a row says, not where it sits: the oldest rows drop off the
  // window, and a run keyed by position would lose whether the reader opened it.
  const seen = new Map<string, number>();
  const keyOf = (row: RecentRow) => {
    const base = `${row.kind}|${row.when}|${row.text}`;
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    return `${base}|${n}`;
  };
  const end = () => {
    if (run.length) out.push(<Actions key={keyOf(run[0])} rows={run} folded={show === "all"} />);
    run = [];
  };
  const endWorking = (live = false) => {
    if (working.length) {
      end();
      out.push(
        <Working
          key={`w|${working[0].session_id}|${working[0].seq ?? working[0].t}`}
          lines={working}
          folded={show === "all"}
          live={live}
          answered={answered}
          running={running}
        />,
      );
    }
    working = [];
  };
  const transcript = show === "all" || show === "transcript";
  let next = 0;
  for (const row of rows) {
    // What the agent printed before this row, between it and the one before.
    while (next < lines.length && row.at && time(lines[next].t) < time(row.at)) working.push(lines[next++]);
    if (transcript) endWorking();
    else working = [];
    if (row.kind === "action") {
      if (show !== "conversation") run.push(row);
    } else if (show !== "actions") {
      end();
      out.push(<Message key={keyOf(row)} row={row} />);
    }
  }
  end();
  if (transcript) {
    working.push(...lines.slice(next));
    // The newest block is live, and open, while its lines still arrive or a
    // step in it still runs.
    const last = working[working.length - 1];
    const runs = working.some((l) => l.kind === "step" && !answered.has(stepKey(l.session_id, l.seq)) && running(l));
    endWorking(!!last && (runs || Date.now() - time(last.t) < LIVE_FOR_MS));
  }
  return <>{out}</>;
}

export function AgentPage() {
  const client = useQueryClient();
  const agent = useQuery({
    queryKey: ["app/agent"],
    queryFn: () => api<AgentState>("/app/agent"),
    refetchInterval: (query) => (working(query.state.data) ? WORKING_EVERY_MS : IDLE_EVERY_MS),
  });
  const isWorking = working(agent.data);
  const recent = useQuery({
    queryKey: ["app/agent/recent"],
    queryFn: () => api<{ rows: RecentRow[] }>("/app/agent/recent"),
  });
  // Under "app/agent", so news about the agent reads it again.
  const reported = useQuery({
    queryKey: ["app/agent", "session"],
    queryFn: () => api<{ session: ReportedSession | null }>("/app/agent/session"),
  });
  // plan.md: the side panel's steps and the plan dialog. Every edit is news.
  const plan = useQuery({
    queryKey: ["app/agent", "plan"],
    queryFn: () => api<AgentPlan>("/app/agent/plan"),
  });
  const [planOpen, setPlanOpen] = useState(false);
  const rowsNow = recent.data?.rows ?? [];
  const since = rowsNow.find((r) => r.at)?.at;
  const transcript = useQuery({
    queryKey: ["app/agent/transcript", since],
    queryFn: () =>
      api<{ entries: TranscriptLine[] }>(`/app/agent/transcript${since ? `?since=${encodeURIComponent(since)}` : ""}`),
    // A new oldest row starts a new read: keep the lines shown until it answers,
    // so the blocks don't vanish, fold and jump.
    placeholderData: keepPreviousData,
  });
  // Each read of the agent while it works reads its spend and transcript too.
  useEffect(() => {
    if (!isWorking) return;
    void client.invalidateQueries({ queryKey: ["app/agent", "session"], exact: true }, { cancelRefetch: false });
    void client.invalidateQueries({ queryKey: ["app/agent/transcript"] }, { cancelRefetch: false });
  }, [agent.dataUpdatedAt, isWorking, client]);
  // What it does shows in the app's events before its transcript is read again.
  useEffect(() => {
    if (!isWorking) return;
    return onEvents((events) => {
      const read = client.getQueryState(["app/agent/transcript", since])?.dataUpdatedAt ?? 0;
      const sent = events.some((e) => e.kind === "agent_transcript");
      if (sent || Date.now() - read > TRANSCRIPT_GAP_MS) {
        void client.invalidateQueries({ queryKey: ["app/agent/transcript"] }, { cancelRefetch: false });
      }
    });
  }, [isWorking, since, client]);
  const [show, setShow] = useState<Show>("all");
  const [knowledge, setKnowledge] = useState(false);
  const list = useRef<HTMLDivElement>(null);
  // Kept at the newest row while the reader hasn't scrolled up from it.
  const pinned = useRef(true);
  // The operator's messages the agent hasn't read wait below the chat, under
  // whatever it does meanwhile; read, each sits where the agent read it.
  const all = recent.data?.rows ?? [];
  const rows = all.filter((r) => !(r.kind === "you" && r.message_id && !r.read));
  const unread = all.filter((r) => r.kind === "you" && r.message_id && !r.read);
  const a = agent.data;
  // A row that arrives while the reader is scrolled up shows the way down to it.
  const [newer, setNewer] = useState(false);
  const attach = useAttachments();
  // Files dragged over the chat: enter and leave fire for each child, so count them.
  const drags = useRef(0);
  const [dragging, setDragging] = useState(false);
  const newest = rows.length ? rows[rows.length - 1].at : "";
  const seen = useRef(newest);
  useEffect(() => {
    if (newest === seen.current) return;
    seen.current = newest;
    if (!pinned.current) setNewer(true);
  }, [newest]);
  const toNewest = () => {
    const el = list.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };
  useLayoutEffect(() => {
    if (pinned.current && list.current) list.current.scrollTop = list.current.scrollHeight;
  });
  // Pictures land after the rows are drawn, and a card or the composer growing
  // below shrinks the record: keep the newest in view.
  useEffect(() => {
    const el = list.current;
    if (!el) return;
    const keep = () => pinned.current && (el.scrollTop = el.scrollHeight);
    const watcher = new ResizeObserver(keep);
    watcher.observe(el);
    for (const child of el.children) watcher.observe(child);
    return () => watcher.disconnect();
  });

  // A failed read keeps the last answer on screen; the top line says why.
  if (!a) return agent.error ? <div className="p-5 text-destructive">{agent.error.message}</div> : null;
  const request = a.request && ["pending", "applying"].includes(a.request.status) ? a.request : null;
  const suggestion = a.suggestion && a.request?.status !== "pending" ? a.suggestion : null;
  const label = SHOWS.find(([key]) => key === show)![1];
  const session = reported.data?.session;
  const panel = hasPanel(session) ? (
    <SidePanel session={session} plan={plan.data} budgetLow={a.budget_low} onOpenPlan={() => setPlanOpen(true)} />
  ) : null;
  return (
    <div className="flex h-full">
      <div
        className="mx-auto flex h-full w-full max-w-[880px] min-w-0 flex-col px-4 md:px-5"
        onDragEnter={(e) => {
          if (!carriesFiles(e)) return;
          drags.current += 1;
          setDragging(true);
        }}
        onDragLeave={(e) => {
          if (carriesFiles(e) && --drags.current <= 0) setDragging(false);
        }}
        onDragOver={(e) => carriesFiles(e) && e.preventDefault()}
        onDrop={(e) => {
          if (!carriesFiles(e)) return;
          e.preventDefault();
          drags.current = 0;
          setDragging(false);
          if (a.experiment) attach.add(e.dataTransfer.files);
        }}
      >
        <div className="flex items-center justify-end gap-3 pt-2.5">
          <button
            className="flex cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={() => setKnowledge(true)}
          >
            <BookOpen size={14} />
            Knowledge
          </button>
          <KnowledgeDialog open={knowledge} onOpenChange={setKnowledge} />
          <Popup
            label="Show"
            align="right"
            className="top-[calc(100%+4px)] bottom-auto w-44"
            trigger={(open, toggle) => (
              <button
                className="flex cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-haspopup="menu"
                aria-expanded={open}
                onClick={toggle}
              >
                <ListFilter size={14} />
                {label}
              </button>
            )}
          >
            {(close) => (
              <div role="menu" className="flex flex-col">
                {SHOWS.map(([key, word]) => (
                  <button
                    key={key}
                    role="menuitemradio"
                    aria-checked={key === show}
                    className={cn(
                      "cursor-pointer rounded border-0 bg-transparent px-2 py-1.5 text-left text-[13px] text-foreground hover:bg-accent",
                      key === show && "bg-accent",
                    )}
                    onClick={() => {
                      setShow(key);
                      close();
                    }}
                  >
                    {word}
                  </button>
                ))}
              </div>
            )}
          </Popup>
          {panel && (
            <div className="lg:hidden">
              <Popup
                label="Goal and to do"
                align="right"
                className="top-[calc(100%+4px)] bottom-auto w-[min(18rem,calc(100vw-2rem))] p-3"
                trigger={(open, toggle) => (
                  <button
                    className="flex cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground"
                    aria-expanded={open}
                    onClick={toggle}
                  >
                    <ListChecks size={14} />
                    Goal and to do
                  </button>
                )}
              >
                {() => panel}
              </Popup>
            </div>
          )}
        </div>
        <div className="relative flex min-h-0 flex-1 flex-col">
          <section
            aria-label="Record"
            ref={list}
            className="min-h-0 flex-1 overflow-y-auto py-1"
            onScroll={(e) => {
              const el = e.currentTarget;
              pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
              if (pinned.current) setNewer(false);
            }}
          >
            <div>
              {a.session && <div className="py-1.5 pl-2.5 text-[13px] text-muted-foreground">{a.session}</div>}
              {all.length ? (
                <Record
                  rows={rows}
                  lines={transcript.data?.entries ?? []}
                  show={show}
                  heard={isWorking && !agent.data?.stopped}
                />
              ) : (
                <div className="py-8 text-center text-[13px] text-muted-foreground">
                  No messages yet. What you write here waits for the agent: it reads it on its next turn, or when one
                  connects.
                </div>
              )}
            </div>
          </section>
          {dragging && (
            <div className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed border-primary bg-primary/5 text-center">
              {a.experiment ? (
                <>
                  <div className="text-[15px]">Drop to attach</div>
                  <div className="text-[13px] text-muted-foreground">
                    Saved in the experiment folder, under notes/, where the agent reads them
                  </div>
                </>
              ) : (
                <div className="text-[15px]">Open an experiment to attach files</div>
              )}
            </div>
          )}
          {newer && (
            <button
              className="absolute bottom-3 left-1/2 flex -translate-x-1/2 cursor-pointer items-center gap-1.5 rounded-md border border-input bg-popover px-3 py-1.5 text-[12px] text-foreground shadow-[0_4px_14px_rgba(0,0,0,0.4)] hover:bg-accent"
              onClick={toNewest}
            >
              <ArrowDown size={14} />
              Newer messages
            </button>
          )}
        </div>
        {unread.length > 0 && (
          <section
            aria-label="Not read yet"
            className="max-h-[30vh] overflow-y-auto border-t border-dashed border-border pt-1"
          >
            {unread.map((row) => (
              <Message key={row.message_id} row={row} unread={unreadWhy(a)} />
            ))}
          </section>
        )}
        <div className="flex flex-col gap-2.5 border-t border-border pt-3 pb-4">
          {/* Kept while a session runs, so the composer doesn't move as commands start and end. */}
          {(a.session || a.now) && <div className="min-h-5 text-[13px] text-muted-foreground">{a.now}</div>}
          {a.waiting && (
            <div className="text-[13px]">
              <b>Waiting on a decision:</b> {a.waiting.proposal}{" "}
              <span className="text-muted-foreground">· {a.waiting.who}</span>
            </div>
          )}
          {a.asking && <ConnectCard key={a.asking.id} ask={a.asking} />}
          {request && <RequestCard key={request.request_id} request={request} agent={a} />}
          {!request && suggestion && <SuggestionCard key={suggestion.proposal_id} suggestion={suggestion} agent={a} />}
          <div
            className={cn(
              "rounded-md",
              dragging && a.experiment && "outline-2 outline-offset-2 outline-primary outline-dashed",
            )}
          >
            <Composer agent={a} attach={attach} onSent={() => (pinned.current = true)} />
          </div>
        </div>
      </div>
      {panel && (
        <aside
          aria-label="Goal and to do"
          className="hidden w-[clamp(232px,22vw,400px)] shrink-0 overflow-y-auto border-l border-border px-6 pt-[38px] pb-4 lg:block"
        >
          {panel}
        </aside>
      )}
      <PlanDialog plan={plan.data} open={planOpen} onOpenChange={setPlanOpen} />
    </div>
  );
}
