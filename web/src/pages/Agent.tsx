import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, ListFilter } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Popup } from "@/components/ui/popup";
import { api, getToken } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import type { AgentState, RecentRow, TranscriptLine } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ConnectCard, RequestCard, SuggestionCard } from "./agent/Cards";
import { Composer } from "./agent/Composer";
import { EvidenceImage } from "./agent/Pictures";

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

/** A run of actions in one line, "4 actions, 1 refused"; open while one still runs. */
function Actions({ rows, folded }: { rows: RecentRow[]; folded: boolean }) {
  const live = rows.some((r) => r.tone === "running" || r.tone === "waiting");
  const [open, setOpen] = useState<boolean | null>(null);
  const shown = !folded || (open ?? live);
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

/** What fibsem-agent printed between two rows, in one line, "Working · 6 steps". */
function Working({ lines, folded }: { lines: TranscriptLine[]; folded: boolean }) {
  const [open, setOpen] = useState(false);
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

function Record({ rows, lines, show }: { rows: RecentRow[]; lines: TranscriptLine[]; show: Show }) {
  const out: JSX.Element[] = [];
  let run: RecentRow[] = [];
  let working: TranscriptLine[] = [];
  const end = () => {
    if (run.length) out.push(<Actions key={`a${out.length}`} rows={run} folded={show === "all"} />);
    run = [];
  };
  const endWorking = () => {
    if (working.length) {
      end();
      out.push(<Working key={`w${out.length}`} lines={working} folded={show === "all"} />);
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
      out.push(<Message key={`m${out.length}`} row={row} />);
    }
  }
  end();
  if (transcript) {
    working.push(...lines.slice(next));
    endWorking();
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
  const rowsNow = recent.data?.rows ?? [];
  const since = rowsNow.find((r) => r.at)?.at;
  const transcript = useQuery({
    queryKey: ["app/agent/transcript", since],
    queryFn: () =>
      api<{ entries: TranscriptLine[] }>(
        `/app/agent/transcript${since ? `?since=${encodeURIComponent(since)}` : ""}`,
      ),
    refetchInterval: TRANSCRIPT_EVERY_MS,
  });
  const [show, setShow] = useState<Show>("all");
  const list = useRef<HTMLDivElement>(null);
  // Kept at the newest row while the reader hasn't scrolled up from it.
  const pinned = useRef(true);
  const rows = recent.data?.rows ?? [];
  const a = agent.data;
  useLayoutEffect(() => {
    if (pinned.current && list.current) list.current.scrollTop = list.current.scrollHeight;
  });
  // Pictures land after the rows are drawn: keep the newest in view.
  useEffect(() => {
    const el = list.current;
    if (!el) return;
    const keep = () => pinned.current && (el.scrollTop = el.scrollHeight);
    const watcher = new ResizeObserver(keep);
    for (const child of el.children) watcher.observe(child);
    return () => watcher.disconnect();
  });

  if (agent.error) return <div className="p-5 text-destructive">{String(agent.error.message)}</div>;
  if (!a) return null;
  const request = a.request && ["pending", "applying"].includes(a.request.status) ? a.request : null;
  const suggestion = a.suggestion && a.request?.status !== "pending" ? a.suggestion : null;
  const label = SHOWS.find(([key]) => key === show)![1];
  return (
    <div className="mx-auto flex h-full w-full max-w-[880px] flex-col px-4 md:px-5">
      <div className="flex items-center justify-end gap-3 pt-2.5">
        <a
          className="text-[12px] text-link no-underline hover:underline"
          href={`/dashboard/log#token=${encodeURIComponent(getToken() ?? "")}`}
          target="_blank"
          rel="noopener"
        >
          Event log
        </a>
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
      </div>
      <section
        aria-label="Record"
        ref={list}
        className="min-h-0 flex-1 overflow-y-auto py-1"
        onScroll={(e) => {
          const el = e.currentTarget;
          pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
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
      <div className="flex flex-col gap-2.5 border-t border-border pt-3 pb-4">
        {a.now && <div className="text-[13px] text-muted-foreground">{a.now}</div>}
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
  );
}
