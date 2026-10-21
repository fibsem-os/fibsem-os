import { useQueryClient, type Query } from "@tanstack/react-query";
import { useEffect } from "react";
import { api, getToken } from "./api";

interface AppEvent {
  kind: string;
  item?: { id?: string; name?: string } | null;
}

interface EventsAnswer {
  latest_seq?: number;
  events?: AppEvent[];
}

// Read again on any news: what the run, the agent and the microscope are doing now.
const LIVE = new Set([
  "app/workflow/status",
  "app/workflow/queue",
  "app/agent",
  "app/agent/recent",
  "app/experiment/images",
  "microscope",
]);

// The steps a run reports many times a second; they change only the live reads.
const PROGRESS = new Set([
  "image_acquired",
  "milling_progress",
  "beam_shifted",
  "stage_moved",
  "stage_position_changed",
  "task_step",
  "command_started",
  "command_ended",
  "tiled_acquisition",
  "fib_acquisition",
]);

// What an item's news makes stale: its record, its History and its thumbnail,
// the experiment's lists, and a grid's overviews and markers.
const RECORDS = new Set([
  "app/experiment",
  "app/experiment/task_history",
  "app/experiment/grids",
  "grid markers",
]);

function stale(query: Query, items: Set<string>, records: boolean, grids: boolean): boolean {
  const [root, second, third] = query.queryKey as unknown[];
  const key = String(root);
  if (LIVE.has(key)) return true;
  if (key === "microscope/devices") return third === "values";
  if (records && RECORDS.has(key)) return true;
  if (grids && key === "grid image") return true;
  if (key === "app/experiment/items") return records && items.has(String(second));
  if (key === "app/experiment/files") {
    // A folder listing, or an item's thumbnail; the files a run saved don't change.
    if (third === undefined) return records;
    return records && [...items].some((name) => second === `${name}/thumbnail.png`);
  }
  return false;
}

/** The app's one clock: a long-poll on /app/events. Each answer marks stale only
 * the reads its events touch, and the pages on screen read those again. */
export function useEventLoop() {
  const client = useQueryClient();
  useEffect(() => {
    let stopped = false;
    let since = 0;
    (async () => {
      while (!stopped && getToken()) {
        try {
          const answer = await api<EventsAnswer>(`/app/events?since=${since}&timeout=25`);
          const latest = answer.latest_seq ?? 0;
          if (latest === since) continue;
          since = latest;
          const events = answer.events ?? [];
          const news = events.filter((e) => !PROGRESS.has(e.kind));
          const items = new Set(news.flatMap((e) => (e.item?.name ? [e.item.name] : [])));
          // A gap in what the buffer held: treat it as news about everything.
          const records = news.length > 0 || events.length === 0;
          const grids = news.some((e) => e.kind.startsWith("grid"));
          await client.invalidateQueries({ predicate: (query) => stale(query, items, records, grids) });
        } catch {
          await new Promise((resolve) => setTimeout(resolve, 3000));
        }
      }
    })();
    return () => {
      stopped = true;
    };
  }, [client]);
}
