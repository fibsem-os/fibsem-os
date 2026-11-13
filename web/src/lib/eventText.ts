// One line per event record: who, where, and what it says, for the Events page
// and any list of recent events. What a record says, and its group, the server
// writes (`fibsem.workflows.event_text`, asked with text=true).

type Payload = Record<string, any>;

/** A record of /app/events (`fibsem.workflows.events`). */
export interface AppEvent {
  seq: number;
  kind: string;
  t?: string;
  timestamp?: number;
  actor?: string;
  session?: string;
  item?: { id?: string; name?: string } | null;
  task?: { id?: string; name?: string } | null;
  payload?: Payload | null;
  /** What it says, from the server. */
  text?: EventText;
  /** Its group on this page, from the server's registry of kinds. */
  group?: Group;
}

/** What a record says: a short label, the detail, a quieter note, and an error. */
export interface EventText {
  label: string;
  detail?: string;
  note?: string;
  error?: string;
  failed?: boolean;
}

export const keyOf = (e: AppEvent) => `${e.session ?? ""}:${e.seq}`;

export const GROUPS = [
  ["proposals", "Proposals"],
  ["notes", "Actions, notes and messages"],
  ["requests", "Requests"],
  ["workflow", "Workflow"],
  ["stage", "Stage and beam"],
  ["images", "Images"],
  ["milling", "Milling"],
  ["edits", "Edits"],
  ["live", "Live progress"],
  ["other", "Other"],
] as const;
export type Group = (typeof GROUPS)[number][0];

export const WHO = ["agent", "operator", "task"] as const;
export type Who = (typeof WHO)[number];

// The questions a run raises are the run's, whatever thread posted them.
const RUN_ASKS = new Set([
  "prompt_raised",
  "prompt_cancelled",
  "proposal_held",
  "proposal_released",
  "question_asked",
  "question_released",
]);

export function whoOf(e: AppEvent): Who {
  if (RUN_ASKS.has(e.kind)) return "task";
  return (WHO as readonly string[]).includes(e.actor ?? "") ? (e.actor as Who) : "task";
}

export const itemOf = (e: AppEvent): string => e.item?.name || e.payload?.item_name || "";
export const taskOf = (e: AppEvent): string => e.task?.name || e.payload?.task_name || "";

/** The time, with the day once a record is not from today: a night's run crosses midnight. */
export function timeOf(e: AppEvent, today = new Date().toLocaleDateString("sv")): string {
  if (typeof e.t === "string" && e.t.includes("T")) {
    const [day, time] = e.t.split("T");
    return (day === today ? "" : `${day.slice(5)} `) + time.slice(0, 8);
  }
  return typeof e.timestamp === "number" ? new Date(e.timestamp * 1000).toTimeString().slice(0, 8) : "";
}


/** The record's group; "other" when the server did not say. */
export const groupOf = (e: AppEvent): Group => e.group ?? "other";

/** What the record says; its kind when the server did not say. */
export const textOf = (e: AppEvent): EventText => e.text ?? { label: e.kind.replace(/_/g, " ") };
