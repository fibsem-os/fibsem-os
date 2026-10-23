import { Check, ListFilter } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import {
  describe,
  GROUPS,
  groupOf,
  itemOf,
  keyOf,
  noteAsked,
  taskOf,
  timeOf,
  WHO,
  whoOf,
  type AppEvent,
  type Asked,
  type Group,
  type Who,
} from "@/lib/eventText";
import { onEvents, type EventsAnswer } from "@/lib/events";
import { useVolume } from "@/lib/experiment";
import { cn } from "@/lib/utils";

const KEEP = 50000; // records held in the page; the server's buffer holds 1000
const SHOW = 3000; // rows drawn at once
const HISTORY_PAGE = 1000;
const FILTERS = "fibsem-event-filters";

const WHO_NAMES: Record<Who, string> = { agent: "The agent", operator: "The operator", task: "The run" };

/** A gap: records the page never read. */
interface Gap {
  gap: number;
  seq: number;
}
type Row = AppEvent | Gap;
const isGap = (row: Row): row is Gap => "gap" in row;
const rowKey = (row: Row) => (isGap(row) ? `gap:${row.seq}` : keyOf(row));

interface Filters {
  who: Who[];
  groups: Group[];
}
// Live progress is off at first: it fires many times a second.
const DEFAULT_FILTERS: Filters = { who: [...WHO], groups: GROUPS.map(([g]) => g).filter((g) => g !== "live") };

function loadFilters(): Filters {
  try {
    const saved = JSON.parse(localStorage.getItem(FILTERS) ?? "null");
    if (saved && Array.isArray(saved.who) && Array.isArray(saved.groups)) return saved;
  } catch {
    // the defaults stand
  }
  return DEFAULT_FILTERS;
}

function useFilters(): [Filters, (next: Filters) => void] {
  const [filters, setFilters] = useState(loadFilters);
  const set = (next: Filters) => {
    setFilters(next);
    try {
      localStorage.setItem(FILTERS, JSON.stringify(next));
    } catch {
      // storage off
    }
  };
  return [filters, set];
}

/** The records so far, oldest first: the live buffer, the app's long-poll after it,
 * and pages of the experiment's events file before it on request. */
function useRecords() {
  const [rows, setRows] = useState<Row[]>([]);
  const [unavailable, setUnavailable] = useState(false);
  // The events file's offset for the page before; null before the first read, 0 at its start.
  const [start, setStart] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const asked = useRef<Asked>(new Map());

  const add = useCallback((batch: AppEvent[], where: "newer" | "older", missed = 0) => {
    batch.forEach((e) => noteAsked(asked.current, e));
    setRows((rows) => {
      const seen = new Set(rows.map(rowKey));
      const fresh: Row[] = batch.filter((e) => !seen.has(keyOf(e)));
      if (!fresh.length && !missed) return rows;
      const next =
        where === "older"
          ? [...fresh, ...rows]
          : [...rows, ...(missed ? [{ gap: missed, seq: fresh[0]?.seq ?? 0 }] : []), ...fresh];
      return next.length > KEEP ? next.slice(next.length - KEEP) : next;
    });
  }, []);

  const earlier = useCallback(async () => {
    if (loading || start === 0) return;
    setLoading(true);
    try {
      const before = start === null ? "" : `&before=${start}`;
      const page = await api<{ available: boolean; events?: AppEvent[]; start?: number | null }>(
        `/app/events/history?limit=${HISTORY_PAGE}${before}`,
      );
      if (!page.available) setStart(0);
      else {
        add(page.events ?? [], "older");
        setStart(page.start ?? 0);
      }
    } catch {
      // the button stays for another try
    } finally {
      setLoading(false);
    }
  }, [add, loading, start]);

  useEffect(() => {
    const stop = onEvents((events, missed) => add(events, "newer", missed));
    let gone = false;
    (async () => {
      try {
        const seed = await api<EventsAnswer>("/app/events?since=0&timeout=0");
        if (gone) return;
        if (seed.available === false) return setUnavailable(true);
        add(seed.events ?? [], "newer");
      } catch {
        // the live loop fills in
      }
    })();
    return () => {
      gone = true;
      stop();
    };
  }, [add]);

  // Open on the run so far, not only what the live buffer still holds.
  const opened = useRef(false);
  useEffect(() => {
    if (opened.current) return;
    opened.current = true;
    void earlier();
  }, [earlier]);

  return { rows, unavailable, atStart: start === 0, loading, earlier, asked: asked.current };
}

function Choice({ on, label, count, toggle }: { on: boolean; label: string; count?: number; toggle: () => void }) {
  return (
    <button
      className="flex cursor-pointer items-center gap-2 rounded border-0 bg-transparent px-2 py-1 text-left text-[13px] text-foreground hover:bg-accent"
      onClick={toggle}
      aria-pressed={on}
    >
      <Check size={13} className={cn(on ? "opacity-100" : "opacity-0")} />
      <span className="flex-1">{label}</span>
      {count !== undefined && <span className="text-muted-foreground tabular-nums">{count}</span>}
    </button>
  );
}

const flip = <T,>(list: T[], value: T) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

function FilterMenu({
  filters,
  set,
  counts,
}: {
  filters: Filters;
  set: (f: Filters) => void;
  counts: Record<string, number>;
}) {
  const hidden = WHO.length - filters.who.length + GROUPS.length - filters.groups.length;
  return (
    <Popup
      label="Event filter"
      side="below"
      className="w-64"
      trigger={(open, toggle) => (
        <Button variant="ghost" onClick={toggle} aria-expanded={open}>
          <ListFilter size={14} />
          {hidden ? `${hidden} hidden` : "Everything"}
        </Button>
      )}
    >
      {() => (
        <div className="flex flex-col divide-y divide-border">
          <div className="flex flex-col py-1">
            <div className="px-2 py-1 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">Who</div>
            {WHO.map((w) => (
              <Choice
                key={w}
                on={filters.who.includes(w)}
                label={WHO_NAMES[w]}
                toggle={() => set({ ...filters, who: flip(filters.who, w) })}
              />
            ))}
          </div>
          <div className="flex flex-col py-1">
            <div className="px-2 py-1 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
              What
            </div>
            {GROUPS.map(([g, name]) => (
              <Choice
                key={g}
                on={filters.groups.includes(g)}
                label={name}
                count={counts[g] ?? 0}
                toggle={() => set({ ...filters, groups: flip(filters.groups, g) })}
              />
            ))}
          </div>
          <div className="p-1">
            <Button variant="ghost" className="w-full justify-start" onClick={() => set(DEFAULT_FILTERS)}>
              Reset
            </Button>
          </div>
        </div>
      )}
    </Popup>
  );
}

function EventRow({ e, asked, open, toggle }: { e: AppEvent; asked: Asked; open: boolean; toggle: () => void }) {
  const text = describe(e, asked);
  const who = whoOf(e);
  const { payload, seq, t, kind, actor, item, task, session } = e;
  return (
    <>
      <tr
        className={cn("cursor-pointer align-top hover:bg-accent", open && "bg-accent")}
        onClick={() => window.getSelection()?.toString() || toggle()}
        aria-expanded={open}
      >
        <td className="py-1 pr-3 pl-4 whitespace-nowrap text-muted-foreground tabular-nums md:pl-5">{timeOf(e)}</td>
        <td className={cn("py-1 pr-3 whitespace-nowrap", who === "task" && "text-muted-foreground")}>
          {who === "task" ? "run" : who}
        </td>
        <td className="py-1 pr-3 whitespace-nowrap">{itemOf(e)}</td>
        <td className="py-1 pr-3 whitespace-nowrap text-muted-foreground">{taskOf(e)}</td>
        <td className={cn("py-1 pr-3 whitespace-nowrap", text.failed && "text-destructive")}>{text.label}</td>
        <td className="w-full py-1 pr-4 break-words md:pr-5">
          {text.detail}
          {text.note && (
            <span className="text-muted-foreground">
              {text.detail ? " · " : ""}
              {text.note}
            </span>
          )}
          {text.error && (
            <span className="text-destructive">
              {text.detail || text.note ? " · " : ""}
              {text.error}
            </span>
          )}
        </td>
      </tr>
      {open && (
        <tr className="bg-accent">
          <td colSpan={6} className="px-4 pb-2 md:px-5">
            <pre className="m-0 max-h-[40vh] overflow-auto rounded border border-border bg-background px-2 py-1.5 font-mono text-[11px] break-words whitespace-pre-wrap">
              {JSON.stringify({ seq, t, kind, actor, item, task, session, payload }, null, 2)}
            </pre>
          </td>
        </tr>
      )}
    </>
  );
}

/** The session's event record as one table, newest first: what the run, the agent and the
 * operator did. It only reads, so the read-only link the Qt app opens shows it whole. */
export function EventsPage() {
  const { rows, unavailable, atStart, loading, earlier, asked } = useRecords();
  const [filters, setFilters] = useFilters();
  const [item, setItem] = useState("");
  const [text, setText] = useState("");
  const [paused, setPaused] = useState<Row[] | null>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const volume = useVolume();

  const all = paused ?? rows;
  const records = useMemo(() => all.filter((r): r is AppEvent => !isGap(r)), [all]);
  const counts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const e of records) counts[groupOf(e.kind)] = (counts[groupOf(e.kind)] ?? 0) + 1;
    return counts;
  }, [records]);
  const items = useMemo(() => [...new Set(records.map(itemOf).filter(Boolean))].sort(), [records]);
  const shown = useMemo(() => {
    const needle = text.trim().toLowerCase();
    const matches = (e: AppEvent) =>
      filters.who.includes(whoOf(e)) &&
      filters.groups.includes(groupOf(e.kind)) &&
      (!item || itemOf(e) === item) &&
      (!needle ||
        `${e.kind} ${itemOf(e)} ${taskOf(e)} ${JSON.stringify(e.payload ?? {})}`.toLowerCase().includes(needle));
    const out: Row[] = [];
    for (let i = all.length - 1; i >= 0 && out.length < SHOW; i--)
      if (isGap(all[i]) || matches(all[i] as AppEvent)) out.push(all[i]);
    return out;
  }, [all, filters, item, text]);
  const newer = paused ? rows.length - paused.length : 0;
  const count = shown.filter((r) => !isGap(r)).length;

  if (unavailable) return <div className="p-5 text-[13px] text-muted-foreground">This server records no events.</div>;
  const toggle = (key: string) => {
    const next = new Set(open);
    next.has(key) ? next.delete(key) : next.add(key);
    setOpen(next);
  };
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 px-4 pt-2.5 pb-2 md:px-5">
        <FilterMenu filters={filters} set={setFilters} counts={counts} />
        <select
          aria-label={volume ? "Position" : "Lamella"}
          className="rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground"
          value={item}
          onChange={(e) => setItem(e.target.value)}
        >
          <option value="">{volume ? "Every position" : "Every lamella"}</option>
          {items.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <input
          type="search"
          aria-label="Search the events"
          placeholder="Search"
          className="w-48 rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground outline-none focus:border-primary"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <Button variant="ghost" onClick={() => setPaused(paused ? null : rows)} aria-pressed={!!paused}>
          {paused ? (newer ? `Paused · ${newer} new` : "Paused") : "Pause"}
        </Button>
        <span className="ml-auto text-[12px] text-muted-foreground tabular-nums">
          {count} of {records.length} events{count >= SHOW ? `, newest ${SHOW} shown` : ""}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto border-t border-border">
        <table className="w-full border-collapse text-[12.5px]">
          <thead className="sticky top-0 bg-background text-left text-[11px] text-muted-foreground">
            <tr className="border-b border-border">
              <th className="py-1.5 pr-3 pl-4 font-normal md:pl-5">Time</th>
              <th className="py-1.5 pr-3 font-normal">Who</th>
              <th className="py-1.5 pr-3 font-normal">{volume ? "Position" : "Lamella"}</th>
              <th className="py-1.5 pr-3 font-normal">Task</th>
              <th className="py-1.5 pr-3 font-normal">Event</th>
              <th className="py-1.5 pr-4 font-normal md:pr-5">Detail</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((row) =>
              isGap(row) ? (
                <tr key={rowKey(row)} className="border-y border-border">
                  <td colSpan={6} className="px-4 py-1 text-muted-foreground md:px-5">
                    {row.gap} events went past before this page read them
                  </td>
                </tr>
              ) : (
                <EventRow
                  key={rowKey(row)}
                  e={row}
                  asked={asked}
                  open={open.has(rowKey(row))}
                  toggle={() => toggle(rowKey(row))}
                />
              ),
            )}
            {!count && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-muted-foreground md:px-5">
                  {records.length ? "Nothing matches these filters." : "No events yet."}
                </td>
              </tr>
            )}
            {shown.length < SHOW && (
              <tr>
                <td colSpan={6} className="px-4 py-3 text-center text-muted-foreground md:px-5">
                  {atStart ? (
                    "Start of this experiment's record."
                  ) : (
                    <Button variant="ghost" onClick={earlier} disabled={loading}>
                      {loading ? "Loading…" : "Load earlier events"}
                    </Button>
                  )}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
