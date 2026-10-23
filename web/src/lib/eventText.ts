// One line per event record: who, where, and what it says, for the Events page
// and any list of recent events.

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

const STAGE = new Set([
  "stage_moved",
  "beam_shifted",
  "alignment",
  "autofocus",
  "fm_autofocus",
  "coincidence_measured",
  "objective_state_changed",
  "grid_loaded",
  "grid_unloaded",
]);
const NOTES = new Set([
  "action",
  "note",
  "evidence",
  "agent_note",
  "agent_stop",
  "agent_refused",
  "agent_called",
  "operator_message",
  "workflow_stop",
]);

export function groupOf(kind: string): Group {
  if (
    kind.startsWith("prompt_") ||
    kind.startsWith("question_") ||
    kind.startsWith("proposal_") ||
    kind === "review_decided"
  )
    return "proposals";
  if (NOTES.has(kind)) return "notes";
  if (kind.startsWith("request_")) return "requests";
  if (
    kind.startsWith("task_") ||
    kind.startsWith("workflow_") ||
    kind === "item_completed" ||
    kind === "experiment_completed"
  )
    return kind === "workflow_changed" ? "edits" : "workflow";
  if (STAGE.has(kind)) return "stage";
  if (kind === "image_acquired" || kind === "fm_image_acquired") return "images";
  if (kind === "milling" || kind === "milling_stage_started" || kind === "spot_burn_started") return "milling";
  if (kind === "edit") return "edits";
  // Many a second while milling or imaging live.
  if (
    kind.endsWith("_progress") ||
    ["stage_position_changed", "sem_acquisition", "fib_acquisition", "tiled_acquisition"].includes(kind)
  )
    return "live";
  return "other";
}

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

const um = (v: unknown) => (typeof v === "number" ? `${(v * 1e6).toFixed(2)} µm` : "?");
const signed = (v: unknown) =>
  typeof v === "number" ? `${v >= 0 ? "+" : "−"}${Math.abs(v * 1e6).toFixed(2)} µm` : null;
const join = (...parts: unknown[]) => parts.filter((p) => p != null && p !== "").join(" · ");
const fileName = (path: unknown) =>
  String(path ?? "")
    .split(/[\\/]/)
    .pop() ?? "";
const cites = (p: Payload) => (p.evidence ? `evidence ${p.evidence}` : undefined);

function position(p: unknown): string {
  if (!p || typeof p !== "object") return "";
  const at = p as Payload;
  const parts = ["x", "y", "z"].filter((a) => typeof at[a] === "number").map((a) => `${a} ${um(at[a])}`);
  for (const a of ["r", "t"])
    if (typeof at[a] === "number") parts.push(`${a} ${((at[a] * 180) / Math.PI).toFixed(1)}°`);
  return parts.join(" · ");
}

function stageMove(p: Payload): EventText {
  const r: Payload = p.request ?? {};
  const failed = p.error ? { error: `failed: ${p.error}`, failed: true } : {};
  if (p.move === "stable_move" || p.move === "vertical_move")
    return {
      label: "moved stage",
      detail: join(p.move.replace("_", " "), r.beam_type),
      note: join(signed(r.dx), signed(r.dy)),
      ...failed,
    };
  if (p.move === "move_to_orientation")
    return { label: "moved stage", detail: `to the ${r.orientation} orientation`, ...failed };
  if (p.move === "move_to_milling_angle")
    return { label: "moved stage", detail: `to milling angle ${r.milling_angle}°`, ...failed };
  return {
    label: "moved stage",
    detail: String(p.move || "move").replace(/_/g, " "),
    note: position(r.position || r.stage_position),
    ...failed,
  };
}

/** Each question as raised, by session and id, so its answer can say what was asked and which button. */
export type Asked = Map<string, Payload>;

export function noteAsked(asked: Asked, e: AppEvent) {
  if (e.kind === "prompt_raised" && e.payload) asked.set(`${e.session ?? ""}:${e.payload.question_id}`, e.payload);
}

export function describe(e: AppEvent, asked: Asked = new Map()): EventText {
  const p: Payload = e.payload ?? {};
  switch (e.kind) {
    case "prompt_raised":
      return {
        label: "asked",
        detail: p.message || p.type,
        note: p.positive ? `${p.positive}${p.negative ? ` or ${p.negative}` : ""}` : undefined,
      };
    case "prompt_answered": {
      const q = asked.get(`${e.session ?? ""}:${p.question_id}`);
      const button = (q && (p.response ? q.positive : q.negative)) || (p.response ? "yes" : "no");
      return {
        label: "answered",
        detail: join(q?.message || p.type, button),
        note: join(p.adjusted && "adjusted", p.suggested_by && `suggested by ${p.suggested_by}`),
      };
    }
    case "prompt_suggested":
      return { label: "suggested", detail: join(p.type, p.response ? "yes" : "no"), note: p.reason };
    case "prompt_cancelled":
      return { label: "withdrawn", detail: p.type };
    case "prompt_suggestion_dismissed":
      return { label: "suggestion dismissed", detail: p.type, note: p.reason };
    case "proposal_held":
    case "question_asked":
      return { label: "proposal held", detail: "the run waits on a proposal" };
    case "proposal_released":
    case "question_released":
      return { label: "proposal released" };
    case "proposal_asked":
      return { label: "proposed", detail: p.field || p.kind || p.type, note: cites(p) };
    case "proposal_decided":
      return { label: "decided", detail: join(p.field || p.kind, p.outcome) };
    case "review_decided":
      return { label: "reviewed", detail: join(p.field, p.outcome || p.decision) };
    case "action":
      if (p.stops)
        return {
          label: "stopped",
          detail: `${p.target || p.what}${p.own ? " (its own)" : ""}`,
          note: p.reason || p.via,
        };
      if (p.refused) return { label: "refused", detail: p.what, note: `${p.error_type} at ${p.level}` };
      return {
        label: p.ok ? "did" : "failed",
        detail: p.what,
        note: join(p.request_id && `request ${p.request_id}`, cites(p)),
        error: p.ok ? undefined : p.error_type || p.status,
        failed: !p.ok,
      };
    case "note":
      return {
        label: p.to || p.reply_to || (e.actor === "operator" && p.id) ? "message" : "note",
        detail: join(p.text, p.item_name),
        note: cites(p),
      };
    case "evidence":
      return {
        label: "evidence",
        detail: `${fileName(p.image)}${p.uploaded ? " (the agent's own)" : ""}`,
        note: p.evidence_id,
      };
    case "agent_note":
      return { label: p.reply_to ? "message" : "note", detail: p.text };
    case "operator_message":
      return { label: "message", detail: p.text };
    case "agent_stop":
      return { label: "stopped", detail: `${p.what || p.action}${p.own ? " (its own)" : ""}`, note: p.reason };
    case "agent_session_budget": {
      const used = p.budget_usd ? Math.round((100 * p.spent_usd) / p.budget_usd) : 0;
      const budget = Number.isInteger(p.budget_usd) ? p.budget_usd : Number(p.budget_usd).toFixed(2);
      return { label: "budget", detail: `The agent has used ${used}% of its $${budget} budget` };
    }
    case "agent_called":
      return p.ok
        ? { label: "called", detail: p.action, note: `at ${p.level}` }
        : {
            label: "call failed",
            detail: p.action,
            note: `at ${p.level}`,
            error: p.error_type || p.status,
            failed: true,
          };
    case "agent_refused":
      return { label: "refused", detail: p.action, note: `${p.error_type} at ${p.level}` };
    case "request_raised":
      return { label: "requested", detail: (p.actions ?? []).map((a: Payload) => a.tool).join(", "), note: p.reason };
    case "request_decided":
      return {
        label: p.outcome === "rejected" ? "rejected" : "approved",
        detail: `request ${p.request_id}`,
        note: p.reason,
      };
    case "request_applied":
      return { label: "applied", detail: `request ${p.request_id}` };
    case "request_finished":
      return {
        label: "finished",
        detail: `request ${p.request_id} · ${p.status}`,
        note: `${p.applied} of ${p.of} applied`,
      };
    case "request_cancelled":
      return { label: "cancelled", detail: `request ${p.request_id}`, note: p.why };
    case "task_started":
      return { label: "started", detail: p.task_name };
    case "task_completed":
      return { label: "completed", detail: p.task_name };
    case "task_failed":
      return { label: "failed", detail: p.task_name, error: p.error, failed: true };
    case "task_cancelled":
      return { label: "stopped", detail: p.task_name };
    case "task_skipped":
      return {
        label: "skipped",
        detail: p.task_name,
        note: p.missing_prereqs ? `${p.skip_reason}: ${p.missing_prereqs.join(", ")}` : p.skip_reason,
      };
    case "task_step":
      return { label: "step", detail: p.display_message || p.step };
    case "workflow_started":
      return { label: "run started" };
    case "workflow_completed":
      return { label: "run completed" };
    case "workflow_cancelled":
      return { label: "run stopped" };
    case "workflow_stalled":
      return { label: "run stalled", error: p.error, failed: true };
    case "workflow_failed":
      return { label: "run failed", error: p.error, failed: true };
    case "item_completed":
      return { label: "item done" };
    case "experiment_completed":
      return { label: "experiment done" };
    case "workflow_changed":
      return { label: "setting", detail: `${p.field} → ${String(p.value)}` };
    case "edit":
      return { label: "edited", detail: join(p.item?.name, p.task, p.target), note: p.via };
    case "stage_moved":
      return stageMove(p);
    case "beam_shifted":
      return { label: "beam shift", detail: p.beam_type };
    case "image_acquired":
    case "fm_image_acquired":
      return {
        label: "image",
        detail: join(
          p.beam_type || (e.kind === "fm_image_acquired" ? "FM" : ""),
          typeof p.hfw === "number" && `${(p.hfw * 1e6).toFixed(0)} µm`,
        ),
        note: p.path ? fileName(p.path) : undefined,
      };
    case "milling":
      return {
        label: "milling",
        detail: join(p.milling_task_name, p.status !== "failed" && p.status),
        note: p.reason,
        error: p.status === "failed" ? "failed" : undefined,
        failed: p.status === "failed",
      };
    case "milling_stage_started":
      return { label: "milling stage", detail: p.name || p.stage_name || p.stage?.name };
    case "autofocus":
      return { label: "autofocus", detail: join(p.beam_type, p.method) };
    case "alignment":
      return { label: "alignment", detail: p.status };
  }
  // Anything else: its plain fields, briefly.
  const brief = Object.entries(p)
    .filter(([, v]) => v === null || ["string", "number", "boolean"].includes(typeof v))
    .slice(0, 4)
    .map(([key, v]) => `${key} ${v}`)
    .join(" · ");
  return { label: e.kind.replace(/_/g, " "), note: brief || undefined };
}
