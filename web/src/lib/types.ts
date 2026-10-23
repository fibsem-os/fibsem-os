// The answers the pages read, typed by hand until the routes declare response
// models and the types are generated from openapi.json.

export interface Capabilities {
  api_version: string;
  routers: { microscope: boolean; services: boolean; app: boolean };
  app_name?: string;
  level?: string;
  fibsem?: { version: string; path: string };
}

export interface WorkflowStatus {
  microscope_connected: boolean;
  experiment: { name: string; path: string; num_items: number } | null;
  workflow: {
    running: boolean;
    current_task: string | null;
    current_item: string | null;
    queue_total: number | null;
  };
}

export interface SystemInfo {
  name: string;
  ip_address: string;
  manufacturer: string;
  model: string;
  serial_number: string;
  hardware_version: string;
  software_version: string;
  fibsem_version: string;
  fibsem_revision: string | null;
  port: number | null;
}

export interface MicroscopeInfo {
  manufacturer: string;
  model: string;
  serial_number: string;
  software_version: string;
  system: { info: SystemInfo };
}

export interface ParameterSpec {
  type: string;
  unit: string | null;
  display?: { label?: string; scale?: number; unit?: string; decimals?: number; advanced?: boolean };
}

export interface DeviceSpec {
  name: string;
  class: string;
  driver: string;
  type: string;
  parameters: Record<string, ParameterSpec>;
}

export type DeviceListing = Record<string, DeviceSpec>;

export interface DeviceValues {
  values: Record<string, unknown>;
  unavailable: string[];
}

export interface ImagePayload {
  image_b64_jpeg: string;
  width: number;
  height: number;
  hfw?: number;
  beam_type?: string;
  acquired_at?: string | null;
}

export interface DisplayImages {
  available: boolean;
  sem: ImagePayload | null;
  fib: ImagePayload | null;
  fm?: ImagePayload | null;
  fm_available?: boolean;
}

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

export interface LevelChoice {
  level: AgentLevel;
  name: string;
  word: string;
  tip: string;
}

export interface Evidence {
  evidence_id: string;
  [key: string]: unknown;
}

export interface RequestAction {
  tool: string;
  what?: string;
  change?: string;
  details?: { columns?: string[]; rows?: string[][] } | null;
}

export interface AgentRequest {
  request_id: string;
  status: string;
  reason: string;
  actions: RequestAction[];
  results: { ok?: boolean }[];
  evidence?: Evidence | null;
}

export interface Suggestion {
  proposal_id: string;
  proposal: string;
  decision: string;
  reason: string;
  evidence: Evidence | null;
  no_evidence?: string;
}

/** GET /app/agent: the agent as the chip and card show it. */
/** An agent on this PC asking to connect, on GET /app/agent. */
export interface ConnectAsk {
  id: string;
  /** As it names itself. */
  client: string;
  /** "this PC", or an address over SSH. */
  origin: string;
  /** The PC user it runs as. */
  user: string;
  asked_at: number;
  /** What the allowed agents' list will call it. */
  name: string;
  /** What allowing it means at the level now. */
  if_allowed: string;
  /** The allowed agent that allowing this one unpairs: one per microscope. */
  replaces: string | null;
}

export interface AgentState {
  experiment: string | null;
  level: AgentLevel | null;
  stopped: boolean;
  last_heard_s: number | null;
  client: string | null;
  session: string | null;
  request: AgentRequest | null;
  suggestion: Suggestion | null;
  now: string | null;
  waiting: { proposal: string; who: string } | null;
  pictures: Partial<Record<"request" | "suggestion", { caption?: string; image: boolean; version: number }>>;
  levels: LevelChoice[];
  asking: ConnectAsk | null;
  /** What the agent said it can do, as [name, value] rows. */
  facts: [string, string][];
  /** The agent has spent most of its budget: the session line warns. */
  budget_low: boolean;
}

/** One row of GET /app/agent/recent. */
export interface RecentRow {
  when: string;
  text: string;
  outcome: string;
  tone: "waiting" | "running" | "ok" | "error" | "muted" | string;
  level: string;
  evidence: Evidence | null;
  why: string;
  kind: "you" | "message" | "note" | "action";
  /** When, as an ISO time with its offset. */
  at: string;
}

/** One line of GET /app/agent/transcript: what fibsem-agent printed. */
export interface TranscriptLine {
  t: string;
  kind: "said" | "step" | "failed";
  text: string;
  verb?: string;
  seq?: number;
  /** For "failed": the step it was. */
  step?: number;
  session_id: string;
}

export interface EvidenceAnswer {
  preview: (ImagePayload & { full_width?: number; full_height?: number }) | null;
  marks: {
    points?: { x: number; y: number; label?: string }[];
    areas?: { x: number; y: number; width: number; height: number; label?: string }[];
  } | null;
  path: string;
}

/** One experiment in /app/experiment/recent. */
export interface ExperimentEntry {
  name: string;
  path: string;
  app: string | null;
  created_at: string | null;
  num_items: number;
  available: boolean;
  instrument_model: string | null;
  instrument_serial: string | null;
  operator: string | null;
  software_version: string | null;
}

export interface FileEntry {
  name: string;
  path: string;
  kind: "folder" | "text" | "image" | "other";
  size: number | null;
  modified: string;
  images?: number;
}

export interface FolderListing {
  root: string;
  dir: string;
  entries: FileEntry[];
}

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

/** GET /app/experiment: one row per item. */
export interface ItemSummary {
  item_name: string;
  item_id: string;
  last_completed_task: string | null;
  last_completed_at: string | null;
  is_completed: boolean;
  is_failure: boolean;
  // Volume's positions: slices planned and cut.
  iterations?: number;
  slices_cut?: number;
}

/** One image of an item's stack: a slice, by the iteration that cut it. */
export interface StackSlice {
  iteration: number;
  filename: string;
  acquired_at: string | null;
  z: number | null;
  pixel_size: number | null;
  hfw: number | null;
}

/** GET /app/experiment/items/{name}/stack?role=: the item's images of one role, in iteration order. */
export interface ItemStack {
  item_name: string;
  item_id: string;
  role: string;
  slices: StackSlice[];
}

export interface Defect {
  verdict: string;
  actor: string;
  reason: string;
  at_task: string;
}

/** GET /app/experiment/items/{name}: the facts the cards use. */
export interface ItemDetail {
  item_name: string;
  is_failure: boolean;
  defect?: Defect;
  milling_angle?: number | null;
}

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
export interface ItemHistory {
  item_name: string;
  filter: HistoryFilter;
  tasks: string[];
  total: number;
  runs: HistoryRun[];
}

/** GET /app/experiment/grids: one row per grid. */
export interface GridSummary {
  name: string;
  id: string;
  description: string;
  quality: string;
  status: string;
  current_task: string | null;
  is_failure: boolean;
  last_completed_task: string | null;
  num_items: number;
  overviews: Record<string, string>;
}

/** GET /app/experiment/grids/{name} */
export type GridDetail = GridSummary & {
  tasks: { name: string; status: string; files: Record<string, string[]> }[];
  items: string[];
};

/** GET /app/experiment/grids/{name}/outputs/{file}/markers */
export interface GridMarkers {
  image?: { width: number; height: number; pixel_size?: number | null; hfw?: number | null; beam_type?: string };
  markers: { item_name: string; x: number; y: number; inside: boolean; is_completed: boolean; is_failure: boolean }[];
  unplaced: { item_name: string; reason: string }[];
  linked?: boolean;
  reason?: string;
  error?: string;
}
