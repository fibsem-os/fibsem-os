import { useQuery } from "@tanstack/react-query";
import { CircleAlert, CircleHelp, Info, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { chime, systemNotices, useNoticePrefs } from "@/lib/notify";
import type { Go, Page } from "@/lib/route";
import type { AgentState, RecentRow } from "@/lib/types";
import { cn } from "@/lib/utils";

/** What the agent needs an answer to, or an error: nothing else is a notice. */
interface Notice {
  id: string;
  kind: "ask" | "message" | "error";
  title: string;
  text: string;
  link: string;
}

const ICON = {
  ask: <CircleHelp size={16} className="text-[var(--fibsem-waiting-color)]" />,
  message: <Info size={16} className="text-link" />,
  error: <CircleAlert size={16} className="text-destructive" />,
};

const TITLE = { ask: "Agent asks", message: "Agent message", error: "Agent error" };

const plain = (markdown: string) =>
  markdown
    .replace(/\*\*|`/g, "")
    .replace(/\s+/g, " ")
    .trim();
const rowId = (row: RecentRow) => `${row.kind}|${row.at}|${row.text}`;

/** The rows that make a notice: the agent's messages, and actions that failed
 * (a refusal is the agent's to handle, not the operator's). */
function rowNotice(row: RecentRow): Notice | null {
  if (row.kind === "message")
    return { id: rowId(row), kind: "message", title: "Agent", text: plain(row.text), link: "Reply in Agent page" };
  if (row.kind === "action" && row.outcome === "failed")
    return {
      id: rowId(row),
      kind: "error",
      title: "The agent's action failed",
      text: [row.text, row.why].filter(Boolean).join(": "),
      link: "Show in Agent page",
    };
  return null;
}

function askNotices(a: AgentState | undefined): Notice[] {
  const out: Notice[] = [];
  if (a?.request?.status === "pending")
    out.push({
      id: `request|${a.request.request_id}`,
      kind: "ask",
      title: "The agent asks",
      text: a.request.actions.map((x) => x.what || x.tool).join(", "),
      link: "Open in Agent page",
    });
  if (a?.suggestion && a.request?.status !== "pending")
    out.push({
      id: `suggestion|${a.suggestion.proposal_id}`,
      kind: "ask",
      title: "The agent suggests",
      text: `${a.suggestion.decision} · ${a.suggestion.proposal}`,
      link: "Open in Agent page",
    });
  if (a?.asking)
    out.push({
      id: `connect|${a.asking.id}`,
      kind: "ask",
      title: "An agent asks to connect",
      text: `${a.asking.name} (${a.asking.origin})`,
      link: "Open in Agent page",
    });
  return out;
}

/** Toasts for what needs the operator, on every page but the Agent page, which
 * shows it all; the tab title counts them. An ask goes when it is answered,
 * a message or an error when it is closed or seen on the Agent page. */
export function Notices({ page, go }: { page: Page; go: Go }) {
  const agent = useQuery({ queryKey: ["app/agent"], queryFn: () => api<AgentState>("/app/agent") });
  const recent = useQuery({
    queryKey: ["app/agent/recent"],
    queryFn: () => api<{ rows: RecentRow[] }>("/app/agent/recent"),
  });
  const prefs = useNoticePrefs();
  // The rows there when the page opened are history, not news.
  const before = useRef<Set<string> | null>(null);
  if (!before.current && recent.data) before.current = new Set(recent.data.rows.map(rowId));
  const [closed, setClosed] = useState<Set<string>>(() => new Set());
  const close = (ids: string[]) => setClosed((was) => new Set([...was, ...ids]));

  const rows = before.current ? (recent.data?.rows ?? []).filter((r) => !before.current!.has(rowId(r))) : [];
  const news = rows
    .map(rowNotice)
    .filter((n): n is Notice => !!n && !closed.has(n.id) && (prefs.errors || n.kind !== "error"));
  const asks = askNotices(agent.data).filter((n) => !closed.has(n.id));
  const onAgentPage = page === "agent";
  useEffect(() => {
    if (onAgentPage && news.length) close(news.map((n) => n.id));
  });
  const shown = onAgentPage || !prefs.toasts ? [] : [...news, ...asks];
  const waiting = [...asks, ...(onAgentPage ? [] : news)];

  useEffect(() => {
    const base = "fibsemOS";
    document.title = waiting.length ? `(${waiting.length}) ${TITLE[waiting[0].kind]} · ${base}` : base;
  }, [waiting.length, waiting[0]?.kind]);

  // Sound and a system notification once per new ask or message, not for what was there on opening.
  const told = useRef<Set<string> | null>(null);
  const tellable = [...asks, ...news.filter((n) => n.kind === "message")];
  useEffect(() => {
    if (!agent.data || !recent.data) return;
    if (!told.current) {
      told.current = new Set(tellable.map((n) => n.id));
      return;
    }
    const fresh = tellable.filter((n) => !told.current!.has(n.id));
    if (!fresh.length) return;
    for (const n of fresh) told.current.add(n.id);
    if (prefs.sound) chime();
    if (prefs.system && document.hidden && systemNotices() && Notification.permission === "granted")
      for (const n of fresh) {
        const shown = new Notification(n.title, { body: n.text, tag: n.id });
        shown.onclick = () => {
          window.focus();
          go("agent");
          shown.close();
        };
      }
  });

  if (!shown.length) return null;
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed right-4 bottom-4 z-40 flex w-[min(420px,calc(100vw-32px))] flex-col gap-2.5 max-md:bottom-20"
    >
      {shown.map((n) => (
        <div
          key={n.id}
          role={n.kind === "ask" ? "alertdialog" : "status"}
          className={cn(
            "pointer-events-auto flex gap-2.5 rounded-md border bg-card px-3.5 py-3 text-[13px] shadow-lg",
            n.kind === "ask" ? "border-[color:var(--fibsem-waiting-color)]" : "border-border",
          )}
        >
          <span className="pt-0.5">{ICON[n.kind]}</span>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="font-semibold">{n.title}</span>
            <span className="line-clamp-4 break-words">{n.text}</span>
            <button
              className="cursor-pointer self-start border-0 bg-transparent p-0 text-link hover:underline"
              onClick={() => go("agent")}
            >
              {n.link}
            </button>
          </div>
          <button
            aria-label="Close"
            className="h-fit cursor-pointer border-0 bg-transparent p-0.5 text-muted-foreground hover:text-foreground"
            onClick={() => close([n.id])}
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
