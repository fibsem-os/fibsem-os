import { useQuery } from "@tanstack/react-query";
import { ApiError, forgetToken, getToken } from "./api";
import { encodePath } from "./files";

export type Values = Uint8Array | Uint16Array | Float32Array;

/** What GET /app/experiment/planes answers: (C, Z, Y, X) values as stored. */
export interface Planes {
  shape: [number, number, number, number];
  dtype: "uint8" | "uint16" | "float32";
  factor: number;
  channels: string[];
  /** The full image's (Y, X), which these planes are a copy of at *factor*. */
  full_shape: [number, number];
  values: Values;
}

export type Projection = "max" | "mean" | "min";

/** Which planes to read: plane *z*, every plane, or a projection over *first*..*last*. */
export interface PlanesAsk {
  factor: number;
  z: number | "all" | Projection;
  first?: number;
  last?: number;
}

const ARRAYS = { uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array };

export async function readPlanes(path: string, ask: PlanesAsk): Promise<Planes> {
  const query = new URLSearchParams({ factor: String(ask.factor), z: String(ask.z) });
  if (ask.first != null) query.set("first", String(ask.first));
  if (ask.last != null) query.set("last", String(ask.last));
  const response = await fetch(`/app/experiment/planes/${encodePath(path)}?${query}`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (response.status === 401) forgetToken();
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body?.detail?.message ?? response.statusText, body?.detail?.error_type);
  }
  const header = JSON.parse(response.headers.get("X-Planes") ?? "{}");
  const buffer = await response.arrayBuffer();
  return { ...header, values: new ARRAYS[header.dtype as Planes["dtype"]](buffer) };
}

/** The planes *ask* names of the image at *path*, kept while the viewer is open. */
export function usePlanes(path: string | null, ask: PlanesAsk | null) {
  return useQuery({
    queryKey: ["app/experiment/planes", path, ask],
    queryFn: () => readPlanes(path!, ask!),
    enabled: !!path && !!ask,
    staleTime: Infinity,
    gcTime: 2 * 60_000,
  });
}

/** Channel *c*, plane *z* of *planes*, as a view of its values. */
export function planeOf(planes: Planes, c: number, z = 0): Values {
  const [, nz, h, w] = planes.shape;
  const start = (c * nz + z) * h * w;
  return planes.values.subarray(start, start + h * w);
}

/** One channel's projection over planes *first* to *last*, from every plane in hand. */
export function project(planes: Planes, c: number, mode: Projection, first: number, last: number): Float32Array {
  const [, , h, w] = planes.shape;
  const out = new Float32Array(h * w);
  out.set(planeOf(planes, c, first));
  for (let z = first + 1; z <= last; z++) {
    const p = planeOf(planes, c, z);
    if (mode === "max") for (let i = 0; i < out.length; i++) out[i] = p[i] > out[i] ? p[i] : out[i];
    else if (mode === "min") for (let i = 0; i < out.length; i++) out[i] = p[i] < out[i] ? p[i] : out[i];
    else for (let i = 0; i < out.length; i++) out[i] += p[i];
  }
  if (mode === "mean") for (let i = 0; i < out.length; i++) out[i] /= last - first + 1;
  return out;
}
