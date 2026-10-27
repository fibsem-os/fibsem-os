import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowDown, ChevronDown, ChevronRight, ListChecks, ListFilter } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import type { AgentState, RecentRow, ReportedSession, TranscriptLine } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ConnectCard, RequestCard, SuggestionCard } from "./agent/Cards";
import { Composer } from "./agent/Composer";
import { EvidenceImage } from "./agent/Pictures";
import { hasPanel, SidePanel } from "./agent/SidePanel";

// Last heard moves with no events, so the agent is read again on a clock.
const AGENT_EVERY_MS = 10000;
// fibsem-agent sends its transcript every 10 s while it works.
const TRANSCRIPT_EVERY_MS = 5000;

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

function Message({ row }: { row: RecentRow }) {
  if (row.kind === "you")
    return (
      <div className="my-1 flex gap-2 rounded-md bg-card py-2 pr-2.5 pl-2.5">
        <Time when={row.when} />
        <div className="min-w-0 flex-1">
          <div className="text-[12px] font-semibold text-muted-foreground">You</div>
          <div className="whitespace-pre-wrap break-words">{row.text}</div>
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

/** What fibsem-agent printed between two rows, in one line, "Working · 6 steps".
 * Open while the agent still adds to it, and it stays open when the agent moves
 * on, as a run of actions does; the reader folds it. */
function Working({ lines, folded, live }: { lines: TranscriptLine[]; folded: boolean; live: boolean }) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  const shown = !folded || open;
  const why = new Map<string, string>();
  for (const l of lines) if (l.kind === "failed" && l.step != null) why.set(`${l.session_id}|${l.step}`, l.text);
  const steps = lines.filter((l) => l.kind === "step").length;
  const failed = lines.filter((l) => l.kind === "failed").length;
  const summary = [`Working · ${steps} ${steps === 1 ? "step" : "steps"}`, ...(failed ? [`${failed} failed`] : [])].join(
    " · ",
  );
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
              const reason = l.kind === "failed" ? l.text : why.get(`${l.session_id}|${l.seq}`);
              if (l.kind === "failed" && l.step != null) return null;
              return (
                <div key={i} className="contents">
                  <span className={reason != null ? "text-destructive" : "text-muted-foreground"}>
                    {l.kind === "failed" ? "Failed" : l.verb}
                  </span>
                  <span className="min-w-0 break-words">
                    {l.kind === "step" && <span className="font-mono text-[12px]">{l.text}</span>}
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

function Record({ rows, lines, show }: { rows: RecentRow[]; lines: TranscriptLine[]; show: Show }) {
  const out: JSX.Element[] = [];
  let run: RecentRow[] = [];
  let working: TranscriptLine[] = [];
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
    // The newest block is live while its lines still arrive (every 10 s).
    const last = working[working.length - 1];
    endWorking(!!last && Date.now() - time(last.t) < LIVE_FOR_MS);
  }
  return <>{out}</>;
}

export function AgentPage() {
  const agent = useQuery({
    queryKey: ["app/agent"],
    queryFn: () => api<AgentState>("/app/agent"),
    refetchInterval: AGENT_EVERY_MS,
  });
  const recent = useQuery({
    queryKey: ["app/agent/recent"],
    queryFn: () => api<{ rows: RecentRow[] }>("/app/agent/recent"),
  });
  // Under "app/agent", so news about the agent reads it again; the checklist
  // changes without news, so it is also read on the agent's clock.
  const reported = useQuery({
    queryKey: ["app/agent", "session"],
    queryFn: () => api<{ session: ReportedSession | null }>("/app/agent/session"),
    refetchInterval: AGENT_EVERY_MS,
  });
  const rowsNow = recent.data?.rows ?? [];
  const since = rowsNow.find((r) => r.at)?.at;
  const transcript = useQuery({
    queryKey: ["app/agent/transcript", since],
    queryFn: () =>
      api<{ entries: TranscriptLine[] }>(
        `/app/agent/transcript${since ? `?since=${encodeURIComponent(since)}` : ""}`,
      ),
    refetchInterval: TRANSCRIPT_EVERY_MS,
    // A new oldest row starts a new read: keep the lines shown until it answers,
    // so the blocks don't vanish, fold and jump.
    placeholderData: keepPreviousData,
  });
  const [show, setShow] = useState<Show>("all");
  const list = useRef<HTMLDivElement>(null);
  // Kept at the newest row while the reader hasn't scrolled up from it.
  const pinned = useRef(true);
  const rows = recent.data?.rows ?? [];
  const a = agent.data;
  // A row that arrives while the reader is scrolled up shows the way down to it.
  const [newer, setNewer] = useState(false);
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
  const panel = hasPanel(session) ? <SidePanel session={session} budgetLow={a.budget_low} /> : null;
  return (
    <div className="flex h-full justify-center gap-8 lg:pr-6">
      <div className="flex h-full w-full max-w-[880px] min-w-0 flex-col px-4 md:px-5">
        <div className="flex items-center justify-end gap-3 pt-2.5">
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
              {rows.length ? (
                <Record rows={rows} lines={transcript.data?.entries ?? []} show={show} />
              ) : (
                <div className="py-8 text-center text-[13px] text-muted-foreground">
                  No messages yet. What you write here waits for the agent: it reads it on its next turn, or when one
                  connects.
                </div>
              )}
            </div>
          </section>
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
          <Composer agent={a} onSent={() => (pinned.current = true)} />
        </div>
      </div>
      {panel && (
        <aside
          aria-label="Goal and to do"
          className="hidden w-[232px] shrink-0 overflow-y-auto pt-[38px] pb-4 lg:block"
        >
          {panel}
        </aside>
      )}
    </div>
  );
}
