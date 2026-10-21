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
  paired_at: string;
  role: "operator" | "agent";
}

export interface Pairing {
  open: boolean;
  role: "operator" | "agent" | null;
  code: string | null;
  expires_in_s: number;
  computers: PairedComputer[];
}
