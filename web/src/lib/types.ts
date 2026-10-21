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
}

export interface PairedComputer {
  id: string;
  name: string;
  paired_at: number; // Unix seconds
  role: "operator" | "agent";
}

export interface Pairing {
  open: boolean;
  role: "operator" | "agent" | null;
  code: string | null;
  expires_in_s: number;
  computers: PairedComputer[];
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
  kind: "you" | "reply" | "note" | "action";
}

export interface EvidenceAnswer {
  preview: (ImagePayload & { full_width?: number; full_height?: number }) | null;
  marks: {
    points?: { x: number; y: number; label?: string }[];
    areas?: { x: number; y: number; width: number; height: number; label?: string }[];
  } | null;
  path: string;
}
