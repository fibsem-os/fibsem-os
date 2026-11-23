import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  BookOpen,
  Bot,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Compass,
  FileText,
  Layers,
  ListChecks,
  ListFilter,
  MessageCircle,
  Loader,
  Microscope,
  PanelRight,
  Search,
  Settings as SettingsIcon,
  SlidersHorizontal,
  SquareSlash,
} from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Popup } from "@/components/ui/popup";
import { api, imageUrl } from "@/lib/api";
import { ago } from "@/lib/format";
import { onEvents } from "@/lib/events";
import { Markdown } from "@/lib/markdown";
import type {
  ItemSummary,
  ReviewItem,
  AgentPlan,
  AgentState,
  Capabilities,
  RecentRow,
  ReportedSession,
  TranscriptLine,
  WorkflowStatus,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { ConnectCard, RequestCard, SuggestionCard, useGatewayAsk } from "./agent/Cards";
import { carriesFiles, splitAttached, useAttachments } from "./agent/attach";
import { SentFiles } from "./agent/Attachments";
import { useAct } from "./agent/act";
import { COMMANDS, compose, Composer } from "./agent/Composer";
import { KnowledgeDialog } from "./agent/Knowledge";
import { MicroscopeCustomise, MicroscopeTab, useMicroscopeShown } from "./agent/MicroscopeTab";
import { EvidenceImage } from "./agent/Pictures";
import { Palette, PALETTE_KEY, type PaletteEntry, usePaletteKey } from "./agent/Palette";
import { PlanDialog } from "./agent/PlanDialog";
import { say, since } from "./agent/steps";
import { asking, SideChat } from "./agent/SideChat";
import { ReviewTab, useReview } from "./agent/ReviewTab";
import { type ColumnTab, SafetyRow, SideColumn, useSafetyActions } from "./agent/SideColumn";
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
// Rows the chat reads at first, and each Show earlier adds (recent.SIZE).
const CHAT_PAGE = 50;

const working = (agent: AgentState | undefined) => agent?.last_heard_s != null && agent.last_heard_s < WORKING_S;

const SHOWS = [
  ["all", "Everything"],
  ["conversation", "Conversation"],
  ["actions", "Actions"],
  ["transcript", "Transcript"],
] as const;
type Show = (typeof SHOWS)[number][0];

// Whether steps show their tool calls rather than words, kept in this browser.
const CALLS_KEY = "fibsem.agent.toolCalls";
function readCalls(): boolean {
  try {
    return localStorage.getItem(CALLS_KEY) === "1";
  } catch {
    return false;
  }
}
function keepCalls(on: boolean) {
  try {
    localStorage.setItem(CALLS_KEY, on ? "1" : "0");
  } catch {
    // A browser that keeps nothing still shows the choice until reload.
  }
}

const TONE: Record<string, string> = {
  ok: "text-[var(--fibsem-ok-color)]",
  error: "text-destructive",
  waiting: "text-[var(--fibsem-semantic-warning-color)]",
  running: "text-link",
  muted: "text-muted-foreground",
};

/** Another page of the app, as the rail opens it. */
function openPage(path: string) {
  history.pushState(null, "", path);
  dispatchEvent(new PopStateEvent("popstate"));
}

const capital = (text: string) => text.replace(/^./, (c) => c.toUpperCase());

function Time({ when }: { when: string }) {
  return <div className="w-12 shrink-0 pt-0.5 text-[12px] text-muted-foreground tabular-nums">{when}</div>;
}

/** Why a message isn't read yet, and when it will be; the last says whether the
 * agent is mid turn, so Interrupt now can reach it. */
function unreadWhy(agent: AgentState | undefined): [string, string, boolean] {
  if (agent?.last_heard_s == null) return ["No agent connected", "An agent reads it when one connects.", false];
  if (agent.level === "off") return ["The agent is off", "The agent reads it when its level is back on.", false];
  return [
    "The agent is working...",
    "The agent reads it when its current turn ends. Interrupt now, or start a message with /now, to have it read it at once.",
    true,
  ];
}

/** Cut the agent's turn short so it reads this waiting message now. */
function InterruptNow({ id }: { id: string }) {
  const { act, busy, error } = useAct();
  const [asked, setAsked] = useState(false);
  if (asked) return <span className="font-normal">Interrupting...</span>;
  return (
    <button
      type="button"
      disabled={busy}
      title={error ?? "Stop the agent's turn so it reads this message now. The run carries on."}
      className="cursor-pointer rounded border border-input px-1.5 font-normal text-foreground hover:bg-accent disabled:opacity-50"
      onClick={async () => {
        if ((await act(`/app/agent/messages/${encodeURIComponent(id)}/interrupt`)) !== undefined) setAsked(true);
      }}
    >
      {error ? "Retry interrupt" : "Interrupt now"}
    </button>
  );
}

function Message({ row, unread }: { row: RecentRow; unread?: [string, string, boolean] }) {
  const sent = splitAttached(row.text);
  if (row.kind === "you")
    return (
      <div
        data-row={row.at || undefined}
        className={cn("my-1 flex gap-2 rounded-md bg-card py-2 pr-2.5 pl-2.5", unread && "opacity-75")}
      >
        <Time when={row.when} />
        <div className="min-w-0 flex-1">
          <div className="flex gap-2 text-[12px] font-semibold text-muted-foreground">
            You
            {unread && (
              <span className="ml-auto cursor-help font-normal" title={unread[1]}>
                Not read yet. {unread[0]}
              </span>
            )}
            {unread?.[2] && row.message_id && <InterruptNow id={row.message_id} />}
          </div>
          <div className="whitespace-pre-wrap break-words">{sent.text}</div>
          <SentFiles files={sent.files} />
          {row.evidence && <EvidenceImage id={row.evidence.evidence_id} />}
        </div>
      </div>
    );
  return (
    <div data-row={row.at || undefined} className="my-2 flex gap-2 rounded-md py-1 pl-2.5">
      <Time when={row.when} />
      <div className="min-w-0 flex-1 break-words text-[15px]">
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
/** Now, ticking each second while *on*: the running step's clock. */
function useClock(on: boolean): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!on) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [on]);
  return now;
}

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
/** A step's pictures, by its session and number. */
type Pictures = Map<string, TranscriptLine[]>;
/** A subagent run's lines, by the session and number of the Agent call. */
type Runs = Map<string, TranscriptLine[]>;

/** A picture a lab's tool answered a step with; a click opens it full size. */
function StepPicture({ line }: { line: TranscriptLine }) {
  const name = line.path?.split("/").pop() ?? "";
  const { data: url } = useQuery({
    queryKey: ["app/agent/transcript/pictures", name],
    queryFn: () => imageUrl(`/app/agent/transcript/pictures/${encodeURIComponent(name)}`),
    enabled: !!name,
    staleTime: Infinity,
  });
  if (!url) return null;
  return (
    <a href={url} target="_blank" rel="noreferrer" title="Open the picture" className="block w-fit">
      <img src={url} alt={line.text || "The step's picture"} className="max-h-48 max-w-full rounded border object-contain" />
    </a>
  );
}

/** What fibsem-agent printed between two rows, in one line, "Working · 6 steps".
 * Open while the agent still adds to it, and it stays open when the agent moves
 * on, as a run of actions does; the reader folds it. *answered* holds every
 * step's answer, which may come after the next row; a done step says how long
 * it took. *running* says whether a step still runs (`Record`). */
function Working({
  lines,
  folded,
  live,
  ...rest
}: {
  lines: TranscriptLine[];
  folded: boolean;
  live: boolean;
} & Shared) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  const shown = !folded || open;
  const { answered } = rest;
  const answerOf = (l: TranscriptLine) => (l.kind === "step" ? answered.get(stepKey(l.session_id, l.seq)) : undefined);
  const steps = lines.filter((l) => l.kind === "step").length;
  const failed = lines.filter((l) => l.kind === "failed" || answerOf(l)?.kind === "failed").length;
  // A block of the agent's words alone is "Working", not "Working · 0 steps".
  const count = steps ? [`${steps} ${steps === 1 ? "step" : "steps"}`] : [];
  const summary = ["Working", ...count, ...(failed ? [`${failed} failed`] : [])].join(" · ");
  return (
    <div className="flex gap-2 py-0.5 pl-2.5 opacity-80">
      <Time when="" />
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
        {shown && <Lines lines={lines} live={live} className={cn(folded && "pl-5")} {...rest} />}
      </div>
    </div>
  );
}

/** What a block of lines needs from the whole transcript (`Record`). *runs*
 * holds each subagent run's lines, by the Agent call that started it. */
type Shared = {
  answered: Answers;
  pictures: Pictures;
  runs: Runs;
  running: (step: TranscriptLine) => boolean;
  now?: string | null;
  calls: boolean;
  /** Shows a run in the side column's Subagents tab. */
  onOpen?: (call: TranscriptLine) => void;
};

/** The lines of a Working block, or of a subagent's run: its words, and a row
 * per step. An Agent call shows as its run (`RunCard`). */
function Lines({
  lines,
  live,
  className,
  ...shared
}: { lines: TranscriptLine[]; live: boolean; className?: string } & Shared) {
  const { answered, pictures, running, now, calls } = shared;
  const [shownCall, setShownCall] = useState<number | null>(null);
  const clock = useClock(live);
  const answerOf = (l: TranscriptLine) => (l.kind === "step" ? answered.get(stepKey(l.session_id, l.seq)) : undefined);
  return (
    <div className={cn("grid grid-cols-[72px_minmax(0,1fr)] gap-x-2.5 gap-y-0.5 text-[12px]", className)}>
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
        if (l.kind === "step" && l.subagent != null)
          return <RunCard key={i} call={l} answer={answer} live={live} {...shared} />;
        const reason = l.kind === "failed" ? l.text : answer?.kind === "failed" ? answer.text : undefined;
        const runs = l.kind === "step" && !answer && running(l);
        const duration = took(answer?.took_s);
        const said = l.kind === "step" ? say(l.verb ?? "Called", l.text) : null;
        return (
          <div key={i} className="contents">
            <span
              className={cn("flex items-center gap-1", reason != null ? "text-destructive" : "text-muted-foreground")}
            >
              {l.kind === "failed" ? "Failed" : calls ? l.verb : runs ? "Running" : said?.verb ?? l.verb}
              {runs && <Loader size={11} className="shrink-0 animate-spin" aria-label="running" />}
            </span>
            <span className="min-w-0 break-words">
              {said && calls && <span className="font-mono text-[12px]">{l.text}</span>}
              {said && !calls && (
                <button
                  className="cursor-pointer border-0 bg-transparent p-0 text-left text-foreground"
                  title={said.call ? "Show the call" : undefined}
                  onClick={() => setShownCall(shownCall === i ? null : i)}
                >
                  {runs && now ? now : said.words}
                </button>
              )}
              {runs && (
                <span className="text-muted-foreground">
                  {" · "}
                  {calls && now ? `${now} · ` : ""}
                  {since(l.t, clock)}
                </span>
              )}
              {duration && reason == null && <span className="text-muted-foreground"> · {duration}</span>}
              {said?.call && !calls && shownCall === i && (
                <span className="block font-mono text-[12px] text-muted-foreground">{said.call}</span>
              )}
              {reason != null && (
                <span className="text-muted-foreground">
                  {l.kind === "step" ? " · failed: " : ""}
                  {reason}
                </span>
              )}
            </span>
            {l.kind === "step" &&
              pictures.get(stepKey(l.session_id, l.seq))?.map((p) => (
                <span key={p.seq} className="col-start-2">
                  <StepPicture line={p} />
                </span>
              ))}
          </div>
        );
      })}
    </div>
  );
}

/** "verifier" as "Verifier", "autolamella-verifier" as "Verifier",
 * "autolamella-site-finder" as "Site finder": the app's prefix goes. */
function agentName(agent: string): string {
  const parts = agent.split(/[-_:]/).filter(Boolean);
  const name = (parts.length > 1 ? parts.slice(1) : parts).join(" ") || "Subagent";
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/** What fibsem-agent says of a step its run ended before answering
 * (harness.STEP_ENDED): the run stopped, not failed. */
const STEP_ENDED = "Ended without an answer";

/** When a run started: "2 h ago" today, else "Mon 16 Nov". */
function when(t: string, now: number): string {
  const at = new Date(t);
  if (now - at.getTime() < 20 * 3600 * 1000) return ago(t, now);
  return at.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

/** A run's state: running, done or failed, and its badge when the subagent
 * gave a verdict or asked for attention. */
function runState(call: TranscriptLine, answer: TranscriptLine | undefined, runs: boolean, clock: number) {
  const stopped = answer?.kind === "failed" && answer.text.startsWith(STEP_ENDED);
  const failed = answer?.kind === "failed" && !stopped;
  const verdict = answer?.verdict;
  const badge = verdict
    ? { text: verdict === "agree" ? "Agrees" : "Concern", warn: verdict === "concern" }
    : answer?.attention
      ? { text: "Needs attention", warn: true }
      : null;
  const text = stopped
    ? ["Stopped", took(answer?.took_s)].filter(Boolean).join(" after ")
    : failed
      ? "Failed"
      : answer
        ? [badge ? "" : "Done", took(answer.took_s)].filter(Boolean).join(" · ")
        : runs
          ? `Running · ${since(call.t, clock)}`
          : "Stopped";
  return { failed, badge, text };
}

function RunBadge({ badge }: { badge: { text: string; warn: boolean } | null }) {
  if (!badge) return null;
  return (
    <span
      className={cn(
        "rounded-full bg-muted px-2 py-px font-semibold",
        badge.warn ? "text-[var(--fibsem-warn-color)]" : "text-[var(--fibsem-ok-color)]",
      )}
    >
      {badge.text}
    </span>
  );
}

/** A subagent's run, where the agent called it: one card with what it was
 * asked, whether it still runs, and its answer once it gives one. Its own steps
 * and words fold under it; Open shows the run in the side column's Subagents tab.
 * *tab*: the run as that tab shows it, its lines always open. */
function RunCard({
  call,
  answer,
  live,
  tab = false,
  ...shared
}: { call: TranscriptLine; answer?: TranscriptLine; live: boolean; tab?: boolean } & Shared) {
  const [open, setOpen] = useState(false);
  const clock = useClock(live);
  const lines = shared.runs.get(stepKey(call.session_id, call.seq)) ?? [];
  const steps = lines.filter((l) => l.kind === "step").length;
  const runs = !answer && shared.running(call);
  const state = runState(call, answer, runs, clock);
  const shown = tab || open;
  return (
    <div
      className={cn(
        "flex flex-col gap-1 text-[13px]",
        tab ? "" : "col-span-2 my-1 max-w-[640px] rounded-md border bg-card px-3 py-2",
      )}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 break-words">
          <span className="font-semibold text-foreground">{agentName(call.subagent ?? "")}</span>
          {call.text && <span className="text-foreground"> · {call.text}</span>}
        </span>
        <span
          className={cn(
            "flex shrink-0 items-center gap-1 text-[12px]",
            state.failed ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {runs && <Loader size={11} className="animate-spin" aria-label="running" />}
          <RunBadge badge={state.badge} />
          {state.text}
        </span>
      </div>
      {answer?.text && (
        <div className={cn("whitespace-pre-wrap break-words", state.failed ? "text-muted-foreground" : "text-foreground")}>
          {answer.text}
        </div>
      )}
      {!tab && (lines.length > 0 || shared.onOpen) && (
        <div className="flex items-center justify-between gap-3">
          {lines.length > 0 ? (
            <button
              className="flex w-fit cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-left text-[12px] text-muted-foreground hover:text-foreground"
              onClick={() => setOpen(!open)}
              aria-expanded={open}
            >
              {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
              {steps ? `${steps} ${steps === 1 ? "step" : "steps"}` : "What it said"}
            </button>
          ) : (
            <span />
          )}
          {shared.onOpen && (
            <button
              className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-primary underline underline-offset-2"
              onClick={() => shared.onOpen?.(call)}
              title="Show this run in the side column"
            >
              Open
            </button>
          )}
        </div>
      )}
      {tab && lines.length === 0 && <p className="m-0 text-[12px] text-muted-foreground">Nothing from it yet.</p>}
      {shown && lines.length > 0 && (
        <Lines lines={lines} live={live && runs} className={cn(tab ? "mt-2" : "pl-5")} {...shared} />
      )}
    </div>
  );
}

/** The subagent runs, newest first, those still running above the finished
 * ones: each opens in the Subagents tab. */
function SubagentList({
  calls,
  answered,
  running,
  onOpen,
}: {
  calls: TranscriptLine[];
  answered: Answers;
  running: (step: TranscriptLine) => boolean;
  onOpen: (call: TranscriptLine) => void;
}) {
  const runs = (c: TranscriptLine) => !answered.has(stepKey(c.session_id, c.seq)) && running(c);
  const clock = useClock(calls.some(runs));
  if (!calls.length) return null;
  const newest = [...calls].reverse();
  const groups: [string, TranscriptLine[]][] = [
    ["Running", newest.filter(runs)],
    ["Finished", newest.filter((c) => !runs(c))],
  ];
  return (
    <div className="flex flex-col gap-3.5">
      {groups.map(([title, shown]) =>
        shown.length ? (
          <section key={title} className="border-b border-border/60 pb-3.5 last:border-b-0 last:pb-0">
            <h4 className="m-0 mb-1 text-[12px] font-semibold text-muted-foreground">
              {title} · {shown.length}
            </h4>
            {shown.map((c) => {
              const answer = answered.get(stepKey(c.session_id, c.seq));
              const state = runState(c, answer, runs(c), clock);
              return (
                <button
                  key={stepKey(c.session_id, c.seq)}
                  className="flex w-full cursor-pointer flex-col gap-0.5 border-0 bg-transparent px-0 py-1 text-left text-[13px] text-foreground hover:text-primary"
                  onClick={() => onOpen(c)}
                  title={c.text || undefined}
                >
                  <span className="min-w-0 truncate">
                    {agentName(c.subagent ?? "")}
                    {c.text && <span className="text-muted-foreground"> · {c.text}</span>}
                  </span>
                  <span
                    className={cn(
                      "flex items-center gap-1 text-[12px]",
                      state.failed ? "text-destructive" : "text-muted-foreground",
                    )}
                  >
                    {runs(c) && <Loader size={11} className="animate-spin" aria-label="running" />}
                    <RunBadge badge={state.badge} />
                    {state.text}
                    {!runs(c) && <span className="text-muted-foreground/70"> · {when(c.t, clock)}</span>}
                  </span>
                </button>
              );
            })}
          </section>
        ) : null,
      )}
    </div>
  );
}

const time = (t: string) => Date.parse(t);
const LIVE_FOR_MS = 60_000;

/** The transcript's lines sorted for showing: each step's answer and pictures
 * by the step, each subagent run's lines by its Agent call, and the agent's own
 * lines (*top*). *heard*: the agent was heard from lately (`Record`);
 * *live*: the session it reports, whose steps alone may still run. */
function indexLines(lines: TranscriptLine[], heard: boolean, live?: string | null) {
  const answered: Answers = new Map();
  const lastAnswered = new Map<string, number>();
  const pictures: Pictures = new Map();
  const runs: Runs = new Map();
  // A subagent's steps are answered beside the agent's, so each run, and the
  // agent's own lines, count their answers apart.
  const flow = (l: TranscriptLine) => `${l.session_id}|${l.run ?? ""}`;
  for (const l of lines) {
    if (l.kind === "picture" && l.step != null && l.path) {
      const key = stepKey(l.session_id, l.step);
      pictures.set(key, [...(pictures.get(key) ?? []), l]);
    }
    if (!answers(l)) continue;
    answered.set(stepKey(l.session_id, l.step), l);
    lastAnswered.set(flow(l), Math.max(lastAnswered.get(flow(l)) ?? 0, l.step ?? 0));
  }
  const saysDone = lines.some((l) => l.kind === "done");
  const lastStep = [...lines].reverse().find((l) => l.kind === "step");
  // Only the live session runs (else the newest): an older one's open steps,
  // and a step its run ended before answering (`ended`), stopped with it.
  const newest = live ?? (lines.length ? lines[lines.length - 1].session_id : null);
  const running = (step: TranscriptLine) =>
    heard &&
    !step.ended &&
    step.session_id === newest &&
    (step.subagent != null
      ? !answered.has(stepKey(step.session_id, step.seq))
      : saysDone
        ? (step.seq ?? 0) > (lastAnswered.get(flow(step)) ?? 0)
        : step === lastStep);
  // The answers and pictures show on their steps, not as lines of their own,
  // and a subagent's lines under its run.
  const shown = lines.filter((l) => !answers(l) && l.kind !== "picture");
  for (const l of shown) {
    if (l.run == null) continue;
    const key = stepKey(l.session_id, l.run);
    runs.set(key, [...(runs.get(key) ?? []), l]);
  }
  const calls = shown.filter((l) => l.kind === "step" && l.subagent != null);
  return { answered, pictures, runs, running, calls, top: shown.filter((l) => l.run == null) };
}

/** Every subagent run in the experiment, however old, not only those the chat
 * has loaded: read again when a loaded run starts or ends. *shown*, a run the
 * chat hasn't loaded: its own lines, read when the Subagents tab opens it. */
function useAllRuns(lines: TranscriptLine[], shown: string | null) {
  const calls = new Set<string>();
  for (const l of lines) if (l.kind === "step" && l.subagent != null) calls.add(stepKey(l.session_id, l.seq));
  const seen = [...calls]
    .map((key) => (lines.some((l) => answers(l) && stepKey(l.session_id, l.step) === key) ? `${key}+` : key))
    .join(",");
  const all = useQuery({
    queryKey: ["app/agent/transcript-runs", seen],
    queryFn: () => api<{ entries: TranscriptLine[] }>("/app/agent/transcript/runs"),
    placeholderData: keepPreviousData,
  });
  const [session, call] = shown ? [shown.slice(0, shown.lastIndexOf("|")), shown.slice(shown.lastIndexOf("|") + 1)] : [];
  const loaded = shown != null && lines.some((l) => l.run != null && stepKey(l.session_id, l.run) === shown);
  const run = useQuery({
    queryKey: ["app/agent/transcript-runs", "run", shown],
    queryFn: () =>
      api<{ entries: TranscriptLine[] }>(
        `/app/agent/transcript/runs/${encodeURIComponent(session!)}/${encodeURIComponent(call!)}`,
      ),
    enabled: shown != null && !loaded && /^\d+$/.test(call ?? ""),
  });
  return { all: all.data?.entries ?? [], run: loaded ? [] : (run.data?.entries ?? []) };
}

/** The chat's index with every run in it: the calls and answers of *all*, and
 * the lines of *run*, a run the chat hasn't loaded. */
function withAllRuns(idx: ReturnType<typeof indexLines>, all: TranscriptLine[], run: TranscriptLine[]) {
  const answered: Answers = new Map(idx.answered);
  const pictures: Pictures = new Map(idx.pictures);
  const runs: Runs = new Map(idx.runs);
  const byKey = new Map<string, TranscriptLine>();
  for (const l of [...all, ...run]) {
    if (answers(l)) {
      if (!answered.has(stepKey(l.session_id, l.step))) answered.set(stepKey(l.session_id, l.step), l);
    } else if (l.kind === "picture" && l.step != null && l.path) {
      const key = stepKey(l.session_id, l.step);
      if (!idx.pictures.has(key)) pictures.set(key, [...(pictures.get(key) ?? []), l]);
    } else if (l.run != null) {
      const key = stepKey(l.session_id, l.run);
      if (!idx.runs.has(key)) runs.set(key, [...(runs.get(key) ?? []), l]);
    } else if (l.kind === "step" && l.subagent != null) {
      byKey.set(stepKey(l.session_id, l.seq), l);
    }
  }
  // The loaded lines win: they are the newest read.
  for (const c of idx.calls) byKey.set(stepKey(c.session_id, c.seq), c);
  const calls = [...byKey.values()].sort((a, b) => time(a.t) - time(b.t));
  return { ...idx, answered, pictures, runs, calls };
}

/** *heard*: the agent was heard from lately, so a step it started may still run
 * with no lines coming (a long milling call). A step runs until its answer
 * comes, while the agent is heard from and no later step of its session has
 * been answered (a call cut short never answers). An agent that never says a
 * step is done shows only its newest step as running. */
function Record({
  rows,
  lines: all,
  show,
  heard,
  live,
  now,
  calls,
  onOpen,
}: {
  rows: RecentRow[];
  lines: TranscriptLine[];
  show: Show;
  heard: boolean;
  live?: string | null;
  now?: string | null;
  calls: boolean;
  onOpen?: (call: TranscriptLine) => void;
}) {
  const out: JSX.Element[] = [];
  let run: RecentRow[] = [];
  let working: TranscriptLine[] = [];
  const indexed = indexLines(all, heard, live);
  const { answered, pictures, runs, running } = indexed;
  const lines = indexed.top;
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
          pictures={pictures}
          runs={runs}
          running={running}
          onOpen={onOpen}
          now={now}
          calls={calls}
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
    const stillRuns = working.some((l) => l.kind === "step" && !answered.has(stepKey(l.session_id, l.seq)) && running(l));
    endWorking(!!last && (stillRuns || Date.now() - time(last.t) < LIVE_FOR_MS));
  }
  return <>{out}</>;
}

// While something runs, its progress is read this often, as the top line reads it.
const RUNNING_EVERY_MS = 2000;

/** 75 s as "1:15", an hour and more as "1:02:05". */
function elapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const two = (n: number) => String(n).padStart(2, "0");
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${two(m)}:${two(s % 60)}` : `${m}:${two(s % 60)}`;
}

/** Above the message box: what runs on the microscope, live. A run's task or
 * the agent's mill turns with a spinner, its bar fills and its time counts, and
 * while the agent is heard it says the agent watches. Nothing running, the
 * agent's now line, as before. */
function LiveLine({ agent, heard, hide }: { agent: AgentState; heard: boolean; hide: boolean }) {
  const status = useQuery({
    queryKey: ["app/workflow/status"],
    queryFn: () => api<WorkflowStatus>("/app/workflow/status"),
    refetchInterval: (query) =>
      query.state.data?.workflow.running || query.state.data?.progress ? RUNNING_EVERY_MS : false,
  });
  const workflow = status.data?.workflow;
  const progress = status.data?.progress && !status.data.progress.outcome ? status.data.progress : null;
  const task = workflow?.running
    ? [workflow.current_task ?? "A task", workflow.current_item].filter(Boolean).join(" on ")
    : null;
  const text = [task ?? progress?.label ?? (progress ? null : agent.now), progress?.step, progress?.numbers]
    .filter(Boolean)
    .join(" · ");
  const running = !!(task || progress);
  // Counted from when this page first saw it run: the line has no start time.
  const key = task ?? progress?.label ?? null;
  const since = useRef<{ key: string | null; at: number }>({ key: null, at: 0 });
  if (since.current.key !== key) since.current = { key, at: Date.now() };
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!running) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [running]);
  if (!running)
    return (agent.session || agent.now) && !hide ? (
      <div className="min-h-5 text-[13px] text-muted-foreground">{agent.now}</div>
    ) : null;
  return (
    <div className="flex flex-col gap-1" aria-live="polite">
      <div className="flex min-h-5 items-center gap-1.5 text-[13px] text-muted-foreground">
        <Loader size={13} className="shrink-0 animate-spin" aria-hidden />
        <span className="min-w-0 truncate" title={text}>
          {text}
        </span>
        <span className="ml-auto shrink-0 tabular-nums" title="How long it has run, since this page saw it start">
          {elapsed((now - since.current.at) / 1000)}
        </span>
        {heard && (
          <span className="shrink-0" title="The agent waits on the microscope's events and wakes on the next one">
            · Agent watching
          </span>
        )}
      </div>
      {progress?.fraction != null && (
        <div className="h-[2px] rounded-sm bg-accent">
          <div
            className="h-full rounded-sm bg-[var(--fibsem-primary-color)] transition-[width] duration-1000"
            style={{ width: `${Math.min(100, 100 * progress.fraction)}%` }}
          />
        </div>
      )}
    </div>
  );
}

// How long a message found by the quick search stays lit.
const FOUND_MS = 1600;

// The app's pages, as the quick search lists them.
const PAGE_LINKS: [string, string][] = [
  ["microscope", "Microscope"],
  ["experiment", "Experiment"],
  ["lamellae", "Lamellae"],
  ["grids", "Grids"],
  ["positions", "Positions"],
  ["events", "Events"],
];
const LEVELS: [string, string][] = [
  ["off", "the agent can read and do nothing"],
  ["watch", "it reads and adds notes, acts on nothing"],
  ["assist", "each action waits for your approval"],
  ["drive", "it acts on the session itself"],
];

/** What the quick search lists on the agent page: commands, panels, pages and,
 * once something is typed, the chat, review items, plan steps, lamellae and
 * subagent runs the page has read. Stop and the safety buttons stay where they are. */
function paletteEntries(p: {
  tabs: ColumnTab[];
  rows: RecentRow[];
  review: ReviewItem[];
  steps: AgentPlan["steps"];
  lamellae: ItemSummary[];
  runs: TranscriptLine[];
  show: Show;
  calls: boolean;
  onTab: (id: string) => void;
  onPlan: () => void;
  onKnowledge: () => void;
  onShow: (show: Show) => void;
  onCalls: () => void;
  onReview: (id: string) => void;
  onRun: (call: TranscriptLine) => void;
  onMessage: (at: string) => void;
}): PaletteEntry[] {
  const out: PaletteEntry[] = [];
  const slash = <SquareSlash size={14} />;
  for (const c of COMMANDS) {
    if (c.name === "/level") continue;
    out.push({
      id: `cmd${c.name}`,
      group: "Commands",
      label: c.name,
      detail: c.say,
      icon: slash,
      run: () => compose(`${c.name} `),
    });
  }
  for (const [level, say] of LEVELS)
    out.push({
      id: `level-${level}`,
      group: "Commands",
      label: `/level ${level}`,
      detail: `Set the level: ${say}`,
      icon: slash,
      words: "level",
      run: () => compose(`/level ${level}`),
    });
  const panel = <PanelRight size={14} />;
  out.push({
    id: "plan",
    group: "Panels",
    label: "Open the plan",
    icon: <FileText size={14} />,
    words: "plan.md steps",
    run: p.onPlan,
  });
  for (const t of p.tabs)
    out.push({
      id: `tab-${t.id}`,
      group: "Panels",
      label: t.label,
      detail: "Side column",
      icon: t.icon ?? panel,
      run: () => p.onTab(t.id),
    });
  out.push({
    id: "knowledge",
    group: "Panels",
    label: "Knowledge",
    detail: "The lab's notes and skills",
    icon: <BookOpen size={14} />,
    run: p.onKnowledge,
  });
  for (const [key, word] of SHOWS)
    if (key !== p.show)
      out.push({
        id: `show-${key}`,
        group: "Panels",
        label: `Show ${word.toLowerCase()}`,
        detail: "In the chat",
        icon: <ListFilter size={14} />,
        words: "filter",
        run: () => p.onShow(key),
      });
  out.push({
    id: "calls",
    group: "Panels",
    label: p.calls ? "Hide tool calls" : "Show tool calls",
    detail: "In the chat",
    icon: <SlidersHorizontal size={14} />,
    run: p.onCalls,
  });
  for (const [page, name] of PAGE_LINKS)
    out.push({
      id: `page-${page}`,
      group: "Go to",
      label: name,
      detail: "Page",
      icon: <Compass size={14} />,
      run: () => openPage(`/ui/${page}`),
    });
  out.push({
    id: "page-settings",
    group: "Go to",
    label: "Settings",
    icon: <SettingsIcon size={14} />,
    run: () => openPage("/ui/settings"),
  });
  // Found only once something is typed, newest first.
  for (const item of [...p.review].reverse())
    out.push({
      id: `review-${item.id}`,
      group: "Review",
      label: item.title,
      detail: item.about || item.text,
      words: `${item.text} ${item.state.replace("_", " ")}`,
      icon: <ClipboardCheck size={14} />,
      find: true,
      run: () => p.onReview(item.id),
    });
  p.steps.forEach((step, i) =>
    out.push({
      id: `step-${i}`,
      group: "Plan",
      label: step.text,
      detail: [step.group, step.done ? "done" : ""].filter(Boolean).join(" · "),
      icon: step.done ? <Check size={14} /> : <ListChecks size={14} />,
      find: true,
      run: p.onPlan,
    }),
  );
  for (const item of p.lamellae)
    out.push({
      id: `item-${item.item_name}`,
      group: "Lamellae",
      label: item.item_name,
      detail: item.last_completed_task ? `Last: ${item.last_completed_task}` : undefined,
      words: "lamella item",
      icon: <Layers size={14} />,
      find: true,
      run: () => openPage(`/ui/lamellae?item=${encodeURIComponent(item.item_name)}`),
    });
  for (const c of [...p.runs].reverse())
    out.push({
      id: `run-${stepKey(c.session_id, c.seq)}`,
      group: "Subagents",
      label: agentName(c.subagent ?? ""),
      detail: c.text,
      words: "subagent run",
      icon: <Bot size={14} />,
      find: true,
      run: () => p.onRun(c),
    });
  for (const row of [...p.rows].reverse()) {
    if (row.kind === "action" || !row.at) continue;
    const text = splitAttached(row.text).text.replace(/\s+/g, " ").trim();
    if (!text) continue;
    out.push({
      id: `row-${row.at}-${row.kind}`,
      group: "Chat",
      label: row.kind === "you" ? "You" : row.kind === "note" ? "Agent note" : "Agent",
      detail: `${row.when} · ${text}`,
      words: text,
      icon: <MessageCircle size={14} />,
      find: true,
      run: () => p.onMessage(row.at),
    });
  }
  return out;
}

export function AgentPage() {
  const client = useQueryClient();
  const agent = useQuery({
    queryKey: ["app/agent"],
    queryFn: () => api<AgentState>("/app/agent"),
    refetchInterval: (query) => (working(query.state.data) ? WORKING_EVERY_MS : IDLE_EVERY_MS),
  });
  const isWorking = working(agent.data);
  // The chat's rows, a page at a time: Show earlier reads one more page back.
  const [pages, setPages] = useState(1);
  const recent = useQuery({
    queryKey: ["app/agent/recent", pages],
    queryFn: () => api<{ rows: RecentRow[]; more: boolean }>(`/app/agent/recent?rows=${pages * CHAT_PAGE}`),
    // Keep the rows shown while the longer read answers, so the chat doesn't empty.
    placeholderData: keepPreviousData,
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
  // The review list: the agent's finished work, its checks and its waits.
  const review = useReview();
  // The side column's Microscope tab and safety row, while the microscope's routes are served.
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const instrument =
    !!caps.data?.routers.microscope && (caps.data.connected !== false || !!caps.data.instruments?.length);
  const fibsem = (caps.data?.instruments?.[0] ?? "fibsem") === "fibsem";
  const safety = useSafetyActions(instrument);
  const gatewayAsk = useGatewayAsk();
  const [shown, setShown] = useMicroscopeShown();
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
  const [calls, setCalls] = useState(readCalls);
  const [knowledge, setKnowledge] = useState(false);
  const list = useRef<HTMLDivElement>(null);
  // Kept at the newest row while the reader hasn't scrolled up from it.
  const pinned = useRef(true);
  // The operator's messages the agent hasn't read wait below the chat, under
  // whatever it does meanwhile; read, each sits where the agent read it.
  // Side questions (/btw) and their answers go to the side chat, not here.
  const all = (recent.data?.rows ?? []).filter((r) => !r.aside);
  const asides = (recent.data?.rows ?? []).filter((r) => r.aside);
  const rows = all.filter((r) => !(r.kind === "you" && r.message_id && !r.read));
  const unread = all.filter((r) => r.kind === "you" && r.message_id && !r.read);
  const a = agent.data;
  // What fibsem-agent printed: shown from its first step, before any message.
  const lines = transcript.data?.entries ?? [];
  // A row that arrives while the reader is scrolled up shows the way down to it.
  const [newer, setNewer] = useState(false);
  const attach = useAttachments();
  // Files dragged over the chat: enter and leave fire for each child, so count them.
  const drags = useRef(0);
  const [dragging, setDragging] = useState(false);
  // The subagent run the Subagents tab shows, by its call's key (null: the
  // list); *focus* brings the tab to the front, then clears so opening a run
  // again does too.
  const [shownRun, setShownRun] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  useEffect(() => {
    if (focus) setFocus(null);
  }, [focus]);
  // The quick search (Cmd/Ctrl+K), and the review item it last opened.
  const [palette, setPalette] = useState(false);
  usePaletteKey(() => setPalette((on) => !on));
  const [reviewOpen, setReviewOpen] = useState<{ id: string } | null>(null);
  // Read once by the Review tab as it opens, then cleared, so going back to the tab doesn't reopen it.
  useEffect(() => {
    if (reviewOpen) setReviewOpen(null);
  }, [reviewOpen]);
  const lamellae = useQuery({
    queryKey: ["app/experiment"],
    queryFn: () => api<{ items: ItemSummary[] }>("/app/experiment"),
    enabled: palette && !!agent.data?.experiment,
  });
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
  // Show earlier keeps the reader where they were: the rows it adds go above.
  const fromBottom = useRef<number | null>(null);
  const showEarlier = () => {
    const el = list.current;
    if (el) fromBottom.current = el.scrollHeight - el.scrollTop;
    pinned.current = false;
    setPages((n) => n + 1);
  };
  useLayoutEffect(() => {
    const el = list.current;
    if (!el) return;
    if (fromBottom.current != null && !recent.isPlaceholderData) {
      el.scrollTop = el.scrollHeight - fromBottom.current;
      fromBottom.current = null;
    } else if (pinned.current) el.scrollTop = el.scrollHeight;
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

  const allRuns = useAllRuns(transcript.data?.entries ?? [], shownRun);
  // A failed read keeps the last answer on screen; the top line says why.
  if (!a) return agent.error ? <div className="p-5 text-destructive">{agent.error.message}</div> : null;
  const request = a.request && ["pending", "applying"].includes(a.request.status) ? a.request : null;
  const suggestion = a.suggestion && a.request?.status !== "pending" ? a.suggestion : null;
  const label = SHOWS.find(([key]) => key === show)![1];
  // The newest step, unanswered, is the one the record shows running.
  const entries = transcript.data?.entries ?? [];
  const lastStep = [...entries].reverse().find((l) => l.kind === "step");
  const nowOnStep =
    (show === "all" || show === "transcript") &&
    isWorking &&
    !a.stopped &&
    lastStep != null &&
    !lastStep.ended &&
    !entries.some((l) => answers(l) && l.session_id === lastStep.session_id && l.step === lastStep.seq);
  const session = reported.data?.session;
  const heard = isWorking && !a.stopped;
  const idx = withAllRuns(indexLines(entries, heard, session?.session_id), allRuns.all, allRuns.run);
  const openRun = (call: TranscriptLine) => {
    setShownRun(stepKey(call.session_id, call.seq));
    setFocus("subagents");
  };
  const panel = hasPanel(session) ? (
    <SidePanel
      session={session}
      plan={plan.data}
      budgetLow={a.budget_low}
      onOpenPlan={() => setPlanOpen(true)}
      runs={idx.calls}
      onOpenRun={openRun}
    />
  ) : null;
  const tabs: ColumnTab[] = [
    {
      id: "session",
      label: "Session",
      icon: <ListChecks size={14} />,
      render: () =>
        panel ?? (
          <p className="m-0 text-[13px] text-muted-foreground">
            The agent's goal, plan and budget show here once it reports them.
          </p>
        ),
    },
  ];
  if (review.data?.items.length)
    tabs.push({
      id: "review",
      label: "Review",
      icon: <ClipboardCheck size={14} />,
      badge: review.data.counts.waiting ? (
        <span className="block size-1.5 rounded-full bg-[#e0a030]" aria-label="waiting on you" />
      ) : undefined,
      render: () => <ReviewTab open={reviewOpen} />,
    });
  if (instrument)
    tabs.push({
      id: "microscope",
      label: "Microscope",
      icon: <Microscope size={14} />,
      menu: <MicroscopeCustomise fibsem={fibsem} shown={shown} onChange={setShown} />,
      render: () => <MicroscopeTab fibsem={fibsem} shown={shown} onOpenPage={() => openPage("/ui/microscope")} />,
    });
  tabs.push({
    id: "side",
    label: "Side chat",
    icon: <MessageCircle size={14} />,
    badge: asking(asides) ? (
      <span className="block size-1.5 rounded-full bg-primary" aria-label="waiting for an answer" />
    ) : undefined,
    render: () => <SideChat rows={asides} agent={a} />,
  });
  if (idx.calls.length) {
    const call = idx.calls.find((c) => stepKey(c.session_id, c.seq) === shownRun);
    const answer = call && idx.answered.get(shownRun!);
    tabs.push({
      id: "subagents",
      label: "Subagents",
      icon: <Bot size={14} />,
      badge: idx.calls.some((c) => !idx.answered.has(stepKey(c.session_id, c.seq)) && idx.running(c)) ? (
        <span className="block size-1.5 rounded-full bg-primary" aria-label="one is running" />
      ) : undefined,
      render: () =>
        call ? (
          <div className="flex flex-col gap-3">
            <button
              className="flex w-fit cursor-pointer items-center gap-1 border-0 bg-transparent p-0 text-[12px] text-muted-foreground hover:text-foreground"
              onClick={() => setShownRun(null)}
            >
              <ChevronLeft size={13} />
              All runs ({idx.calls.length})
            </button>
            <RunCard
              tab
              call={call}
              answer={answer}
              live={!answer && idx.running(call)}
              answered={idx.answered}
              pictures={idx.pictures}
              runs={idx.runs}
              running={idx.running}
              now={a.now}
              calls={calls}
            />
          </div>
        ) : (
          <SubagentList calls={idx.calls} answered={idx.answered} running={idx.running} onOpen={openRun} />
        ),
    });
  }
  // A message found in the quick search: shown (Actions alone hides messages), scrolled to and lit briefly.
  const jumpTo = (at: string) => {
    if (show === "actions") setShow("all");
    pinned.current = false;
    setTimeout(() => {
      const el = list.current?.querySelector<HTMLElement>(`[data-row="${CSS.escape(at)}"]`);
      if (!el) return;
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      el.classList.add("ring-2", "ring-primary");
      setTimeout(() => el.classList.remove("ring-2", "ring-primary"), FOUND_MS);
    }, 50);
  };
  const found = paletteEntries({
    tabs,
    rows: all,
    review: review.data?.items ?? [],
    steps: plan.data?.steps ?? [],
    lamellae: lamellae.data?.items ?? [],
    runs: idx.calls,
    show,
    calls,
    onTab: setFocus,
    onPlan: () => setPlanOpen(true),
    onKnowledge: () => setKnowledge(true),
    onShow: setShow,
    onCalls: () => {
      keepCalls(!calls);
      setCalls(!calls);
    },
    onReview: (id) => {
      setReviewOpen({ id });
      setFocus("review");
    },
    onRun: openRun,
    onMessage: jumpTo,
  });
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
            className="mr-auto flex cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={() => setPalette(true)}
            aria-keyshortcuts="Meta+K Control+K"
          >
            <Search size={14} />
            Search
            <kbd className="rounded border border-border px-1 text-[11px]">{PALETTE_KEY}</kbd>
          </button>
          <Palette open={palette} onOpenChange={setPalette} entries={found} />
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
                <div className="my-1 border-t border-border" />
                <button
                  role="menuitemcheckbox"
                  aria-checked={calls}
                  className="flex cursor-pointer items-center justify-between rounded border-0 bg-transparent px-2 py-1.5 text-left text-[13px] text-foreground hover:bg-accent"
                  onClick={() => {
                    keepCalls(!calls);
                    setCalls(!calls);
                    close();
                  }}
                >
                  Show tool calls
                  {calls && <Check size={14} />}
                </button>
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
              {recent.data?.more && (
                <div className="flex justify-center py-1.5">
                  <button
                    type="button"
                    className="rounded-md border px-2.5 py-1 text-[13px] text-muted-foreground hover:bg-muted disabled:opacity-60"
                    disabled={recent.isPlaceholderData}
                    onClick={showEarlier}
                  >
                    {recent.isPlaceholderData ? "Reading…" : "Show earlier"}
                  </button>
                </div>
              )}
              {all.length || lines.length ? (
                <Record
                  rows={rows}
                  lines={lines}
                  show={show}
                  heard={heard}
                  live={reported.data?.session?.session_id}
                  now={a.now}
                  calls={calls}
                  onOpen={openRun}
                />
              ) : isWorking && !agent.data?.stopped ? (
                <div className="flex items-center justify-center gap-1.5 py-8 text-[13px] text-muted-foreground">
                  <Loader size={13} className="shrink-0 animate-spin" aria-hidden />
                  The agent is starting: it connects, then reads its skills and playbook. Its steps show here as it takes
                  them.
                </div>
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
          {/* Kept while a session runs, so the composer doesn't move as commands start and end.
              Live while something runs; else the running step says it in the record, and
              here only when no step shows it. */}
          <LiveLine agent={a} heard={isWorking && !a.stopped} hide={nowOnStep} />
          {a.waiting &&
            (a.waiting.agent_reviewing ? (
              <div className="text-[13px]">
                <b>Agent is reviewing:</b> {a.waiting.proposal}
              </div>
            ) : (
              <div className="text-[13px]">
                <b>Waiting on a decision:</b> {a.waiting.proposal}{" "}
                <span className="text-muted-foreground">· {a.waiting.who}</span>
              </div>
            ))}
          {gatewayAsk && <ConnectCard key={gatewayAsk.id} ask={gatewayAsk} />}
          {a.asking && <ConnectCard key={a.asking.id} ask={a.asking} decide="/app/agent/connections" />}
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
      <SideColumn
        tabs={tabs}
        focus={focus}
        footer={<SafetyRow actions={safety} />}
        rail={<SafetyRow actions={safety} compact />}
      />
      <PlanDialog plan={plan.data} open={planOpen} onOpenChange={setPlanOpen} />
    </div>
  );
}
