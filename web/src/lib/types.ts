// The answers the pages read. Each is the server's class from schema.ts, which
// `bun run types` writes from /openapi.json; where the class says less than the
// page needs (a dict, or text that is one of a few words), the field is narrowed
// here. Routes that declare no class yet are typed by hand.

import type { Marks } from "./marks";
import type { components } from "./schema";

type S = components["schemas"];
/** *T* with the fields of *K* as *K* types them. */
type Narrow<T, K> = Omit<T, keyof K> & K;

export interface Capabilities {
  api_version: string;
  routers: { microscope: boolean; services: boolean; app: boolean };
  app_name?: string;
  level?: string;
  // whether this install serves an agent at all (fibsem.unreleased)
  agent?: boolean;
  connected?: boolean;
  // the optional /app parts the app serves (fibsem.server.parts)
  parts?: string[];
  fibsem?: { version: string; path: string };
  // the gateway on this PC; given to this PC's operator only
  gateway?: string | null;
}

/** The microscope connection, as the operator sees it (`/microscope/connection`). */
export type Connection = Narrow<
  S["ConnectionView"],
  { run: Narrow<S["RunAnswer"], { error: { message?: string } | null }> | null }
>;

/** Measurable progress on the microscope, as the Qt status bar says it. */
export type ProgressLine = S["ProgressLine"];

export type WorkflowStatus = Narrow<
  S["AppStatus"],
  {
    workflow: Narrow<
      S["WorkflowState"],
      {
        /** Where the run is. */
        phase: "idle" | "starting" | "running" | "waiting" | "held" | "stopping";
        last_run: Narrow<S["RunEnded"], { ended: "completed" | "cancelled" | "stalled" | "failed" }> | null;
      }
    >;
  }
>;

export type SystemInfo = S["SystemInfo"];

export type MicroscopeInfo = Narrow<S["MicroscopeInfo"], { system: { info: SystemInfo } }>;

export type ParameterSpec = Narrow<
  S["ParameterDescription"],
  { display: { label?: string; scale?: number; unit?: string; decimals?: number; advanced?: boolean } | null }
>;

// /microscope/devices writes its class inline in /openapi.json, so it has no name in schema.ts.
export interface DeviceSpec {
  name: string;
  class: string;
  driver: string;
  type: string;
  parameters: Record<string, ParameterSpec>;
}

export type DeviceListing = Record<string, DeviceSpec>;

export type DeviceValues = S["Values"];

export type ImagePayload = Narrow<S["ImagePayload"], { info: ImageInfo | null }>;

export type DisplayImages = Narrow<
  S["DisplayImages"],
  { sem: ImagePayload | null; fib: ImagePayload | null; fm: ImagePayload | null }
>;

export interface PairedComputer {
  id: string;
  name: string;
  paired_at: number; // Unix seconds
  role: PairingRole;
  grants: string[];
}

export type PairingRole = "operator" | "agent" | "hub";

export interface Pairing {
  open: boolean;
  role: PairingRole | null;
  grants: string[];
  code: string | null;
  expires_in_s: number;
  computers: PairedComputer[];
  /** Agents on the microscope PC that ask to connect are allowed without a click. */
  always_allow_local: boolean;
}

export type AgentLevel = "off" | "watch" | "assist" | "drive";

export type LevelChoice = Narrow<S["LevelOption"], { level: AgentLevel }>;

export interface Evidence {
  evidence_id: string;
  [key: string]: unknown;
}

export type RequestAction = Narrow<S["ActionView"], { details: { columns?: string[]; rows?: string[][] } | null }>;

export type AgentRequest = Narrow<
  S["AgentRequestView"],
  { actions: RequestAction[]; results: { ok?: boolean }[]; evidence: Evidence | null }
>;

export type Suggestion = Narrow<S["SuggestionCard"], { evidence: Evidence | null }>;

/** An agent on this PC asking to connect, on GET /app/agent. */
export type ConnectAsk = S["ConnectAskView"];

/** GET /app/agent: the agent as the chip and card show it. */
export type AgentState = Narrow<
  S["AgentView"],
  {
    level: AgentLevel | null;
    request: AgentRequest | null;
    suggestion: Suggestion | null;
    pictures: Partial<Record<"request" | "suggestion", S["PictureInfo"]>>;
    levels: LevelChoice[];
    /** What the agent said it can do, as [name, value] rows. */
    facts: [string, string][];
  }
>;

/** One of the agent's tools, as GET /app/agent/customisation lists it. */
export type ToolRow = Narrow<S["ToolRow"], { level: "watch" | "assist" | "drive" }>;

export type SkillRow = S["SkillRow"];

/** GET /app/agent/customisation: what the agent runs with. */
export type AgentCustomisation = Narrow<S["AgentCustomisation"], { tools: ToolRow[] }>;

/** GET /app/agent/session: what the agent reported of its session. */
export interface ReportedSession {
  /** The reminders on now. */
  hooks?: { when: string; label: string; say: string }[];
  /** The skills it has read, in order. */
  skills?: string[];
  /** The playbook it runs ({} for none): who chose it and its optional skills. */
  playbook?: {
    name?: string;
    hash?: string;
    chosen_by?: string;
    skills?: { skill: string; state: "not loaded" | "loaded" | "done" | string; summary?: string }[];
  };
  /** When the server first heard of it, as an ISO time with its offset. */
  started?: string;
  goal?: string;
  budget_usd?: number;
  spent_usd?: number;
  /** What its model calls used this run, in tokens. */
  tokens?: { input: number; output: number; cache_read: number; cache_write: number };
  /** How full its context is, in tokens: compacted at `compact_at`. */
  context?: { used: number; window?: number; compact_at?: number };
  /** The plan's Rules that give a time; `by` is an ISO time with its offset. */
  deadlines?: { text: string; by: string }[];
  checklist?: { text: string; state: "open" | "done" | "always" }[];
  ended?: boolean;
}

/** GET /app/agent/plan: the session's plan.md, "" before the agent writes one. */
export interface AgentPlan {
  session_id: string | null;
  markdown: string;
  sections: { name: string; text: string }[];
  sample: string | null;
  /** The Steps' boxes; group is the ### line above, "" for none. */
  steps: { group: string; text: string; done: boolean }[];
  done: number;
  updated: string | null;
  updated_by: "agent" | "operator" | null;
}

/** One row of GET /app/agent/recent. */
export type RecentRow = Narrow<
  S["RecentRow"],
  { evidence: Evidence | null; kind: "you" | "message" | "note" | "action" }
>;

/** One line of GET /app/agent/transcript: what fibsem-agent printed. */
export interface TranscriptLine {
  t: string;
  /** context: what fibsem-agent says of the session, "Playbook set: autolamella, ...". */
  kind: "said" | "step" | "failed" | "context";
  text: string;
  verb?: string;
  seq?: number;
  /** For "failed": the step it was. */
  step?: number;
  session_id: string;
}

export type EvidenceAnswer = Narrow<
  S["EvidenceView"],
  {
    preview: ImagePayload | null;
    marks: Marks | null;
    by: "operator" | "agent";
    /** The marks drawn on *source*, when it is not this image. */
    source_marks: Marks | null;
    verdict: "good" | "bad" | null;
    judged:
      | { kind: string; index: number; label?: string; verdict?: "good" | "bad"; moved_to?: { x: number; y: number } }[]
      | null;
    /** What the operator last said of this whole picture. */
    said: "good" | "bad" | null;
  }
>;

/** One experiment in /app/experiment/recent. */
export type ExperimentEntry = S["ExperimentEntry"];

export type FileEntry = Narrow<S["FileEntry"], { kind: "folder" | "text" | "image" | "other" }>;

export type FolderListing = Narrow<S["FolderListing"], { entries: FileEntry[] }>;

export interface StackFacts {
  channels: {
    name: string;
    color: string;
    excitation_wavelength: number | null;
    emission: string | null;
    exposure_time?: number | null;
    // per plane, 1 the sharpest; absent for a single plane
    sharpness?: number[];
    sharpest_plane?: number;
  }[];
  z_planes: number;
  z_step: number | null;
  z_positions?: number[] | null;
  acquired_at?: string | null;
}

/** GET /app/experiment/files/{path}: a text file's text, or an image's preview. */
export type FileRead = FileEntry & {
  text?: string;
  truncated?: boolean;
  preview?: ImagePayload;
  summary?: { fm?: StackFacts; [key: string]: unknown };
  info?: ImageInfo | null;
};

/** One value on an image's bar: `HFW 80 µm`, its *title* said in full on hover. */
export interface ImageField {
  key: string;
  label: string;
  title: string;
  value: string;
}

/** What the bar under an image says (fibsem.server.image_payloads.fields_info). */
export interface ImageInfo {
  kind: string;
  header: ImageField[];
  fields: ImageField[];
  time: string | null;
  scalebar: { length: number; label: string; fraction: number } | null;
}

/** GET /app/experiment: one row per item; Volume's positions add the slices planned and cut. */
export type ItemSummary = S["LamellaRow"] & { iterations?: number; slices_cut?: number };

/** One image of an item's stack: a slice, by the iteration that cut it. */
export type StackSlice = S["StackSlice"];

/** GET /app/experiment/items/{name}/stack?role=: the item's images of one role, in iteration order. */
export type ItemStack = S["ItemStack"];

export interface Defect {
  verdict: string;
  actor: string;
  reason: string;
  at_task: string;
}

/** GET /app/experiment/items/{name}: the facts the cards use. */
export type ItemDetail = Narrow<S["LamellaDetail"], { defect: Defect | null }>;

export interface HistoryOperation {
  kind: string;
  label: string;
  status: string;
  detail: string;
  started_at: number | null;
  ended_at: number | null;
  facts: [string, string][];
  figure: string | null;
  stages: {
    name: string;
    text: string;
    strategy: string;
    said: string;
    figure: string | null;
    monitors: { name: string; said: string; figure: string | null }[];
    drift: HistoryOperation[];
  }[];
}

export interface HistoryRun {
  task_id: string;
  task_name: string;
  status: string;
  status_message: string;
  started_at: number | null;
  ended_at: number | null;
  duration: number | null;
  images: string[];
  images_replaced_by: string | null;
  outside_task: boolean;
  decided_by: string[];
  operations_summary: string;
  operations_open: boolean;
  operations: HistoryOperation[];
}

export interface HistoryFilter {
  show: "all" | "images" | "operations";
  status: "all" | "failed" | "cancelled";
  task: string | null;
  decided_by: "anyone" | "agent" | "operator";
  order: "oldest" | "newest";
}

/** GET /app/experiment/items/{name}/history */
export type ItemHistory = Narrow<S["ItemHistory"], { filter: HistoryFilter; runs: HistoryRun[] }>;

/** GET /app/experiment/grids: one row per grid. */
export type GridSummary = S["GridSummary"];

/** GET /app/experiment/grids/{name} */
export type GridDetail = S["GridDetail"];

/** GET /app/experiment/grids/{name}/outputs/{file}/markers */
export type GridMarkers = Narrow<
  S["GridMarkers"],
  {
    image: {
      width: number;
      height: number;
      pixel_size?: number | null;
      hfw?: number | null;
      beam_type?: string;
    } | null;
  }
>;

/** One recorded action of the experiment replay. */
export type ReplayRow = S["ReplayRow"];

/** GET /app/experiment/replay: every recorded action, in time order. */
export type ReplayTimeline = S["ReplayTimeline"];

/** A replay image and what is drawn over it, in fractions of the image (+y down). */
export type ReplayImage = S["ReplayImage"];

/** GET /app/experiment/replay/{index}: the instrument at one recorded action. */
export type ReplayView = Narrow<S["ReplayView"], { stage_position: Record<string, number | string | null> | null }>;

/** A named place on a stage map, in metres on the map's plane (+y down). */
export type MapPlace = S["MapPlace"];

/** What a stage map draws (fibsem.geometry.stage_places.StageMapPlaces). */
export type StageMapPlaces = S["StageMapPlaces"];

/** The stage at a moment, as the stage card draws it; angles in degrees. */
export type StageDrawing = S["ReplayStage"];

/** This instrument's Slack notifications (`/app/notifications`); the token is never served. */
export type NotificationSettings = S["NotificationSettings"];
