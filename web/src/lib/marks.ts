import * as fmt from "./format";

/** Marks on an image, as evidence records keep them: full-resolution source
 * pixels, origin top left (plans/agent-drawing, "Marks schema"). */

/** Said right or wrong, by the operator. */
export type Verdict = "good" | "bad";

/** The plane of a fluorescence stack a mark is on, 0 the first; none for the projection;
 * and what was said of it. */
interface OnPlane {
  z?: number;
  verdict?: Verdict;
}

export interface PointMark extends OnPlane {
  x: number;
  y: number;
  shape?: "cross" | "target";
  label?: string;
}

export interface AreaMark extends OnPlane {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
}

export interface CircleMark extends OnPlane {
  x: number;
  y: number;
  radius: number;
  label?: string;
}

export interface LineMark extends OnPlane {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  arrow?: boolean;
  measure?: boolean;
  label?: string;
}

/** A closed outline: its corners, in order. */
export interface OutlineMark extends OnPlane {
  points: [number, number][];
  label?: string;
}

export interface Marks {
  points?: PointMark[];
  areas?: AreaMark[];
  circles?: CircleMark[];
  lines?: LineMark[];
  outlines?: OutlineMark[];
}

/** One mark of any kind, as the editor keeps them in the order they were made. */
export type Mark =
  | ({ kind: "point" } & PointMark)
  | ({ kind: "area" } & AreaMark)
  | ({ kind: "circle" } & CircleMark)
  | ({ kind: "line" } & LineMark)
  | ({ kind: "outline" } & OutlineMark);

/** A kind as a record lists it. */
export const KIND_LIST = {
  point: "points",
  area: "areas",
  circle: "circles",
  line: "lines",
  outline: "outlines",
} as const;

export const MAX_CORNERS = 64;
export const VERDICT_SIGN: Record<Verdict, string> = { good: "\u2713", bad: "\u2717" };
export const GOOD_MARK = "#4ade80";
export const BAD_MARK = "#f87171";

export const MAX_MARKS = 50;
export const MAX_LABEL = 32;

/** The agent's purple and the operator's blue (the app's accent). */
export const AGENT_MARK = "#a78bfa";
export const OPERATOR_MARK = "#50a6ff";

/** *marks* as one list, in kind order. */
export function listMarks(marks: Marks | null | undefined): Mark[] {
  if (!marks) return [];
  return [
    ...(marks.points ?? []).map((m) => ({ kind: "point" as const, ...m })),
    ...(marks.areas ?? []).map((m) => ({ kind: "area" as const, ...m })),
    ...(marks.circles ?? []).map((m) => ({ kind: "circle" as const, ...m })),
    ...(marks.lines ?? []).map((m) => ({ kind: "line" as const, ...m })),
    ...(marks.outlines ?? []).map((m) => ({ kind: "outline" as const, ...m })),
  ];
}

/** Each of *list*'s marks as a record names it: its kind's list and its place there. */
export function markRefs(list: Mark[]): { kind: (typeof KIND_LIST)[Mark["kind"]]; index: number }[] {
  const seen: Record<string, number> = {};
  return list.map((m) => {
    const kind = KIND_LIST[m.kind];
    seen[kind] = (seen[kind] ?? -1) + 1;
    return { kind, index: seen[kind] };
  });
}

/** At most *max* of *corners*, evenly spaced along them: a drawn outline kept small. */
export function fewerCorners(corners: [number, number][], max = MAX_CORNERS): [number, number][] {
  if (corners.length <= max) return corners;
  const step = corners.length / max;
  return Array.from({ length: max }, (_, i) => corners[Math.floor(i * step)]);
}

const px = (v: number) => Math.round(v);

/** *list* as a record keeps it: whole pixels, empty labels and kinds left out,
 * a measured line's label left to the server, each mark's plane kept. */
export function toMarks(list: Mark[]): Marks {
  const label = (m: { label?: string; z?: number; verdict?: Verdict }) => ({
    ...(m.label?.trim() ? { label: m.label.trim().slice(0, MAX_LABEL) } : {}),
    ...plane(m),
  });
  const plane = (m: { z?: number; verdict?: Verdict }) => ({
    ...(m.z != null ? { z: m.z } : {}),
    ...(m.verdict ? { verdict: m.verdict } : {}),
  });
  const out: Marks = {};
  for (const m of list) {
    if (m.kind === "point")
      (out.points ??= []).push({ x: px(m.x), y: px(m.y), ...(m.shape ? { shape: m.shape } : {}), ...label(m) });
    else if (m.kind === "area")
      (out.areas ??= []).push({ x: px(m.x), y: px(m.y), width: px(m.width), height: px(m.height), ...label(m) });
    else if (m.kind === "circle")
      (out.circles ??= []).push({ x: px(m.x), y: px(m.y), radius: px(m.radius), ...label(m) });
    else if (m.kind === "outline")
      (out.outlines ??= []).push({
        points: fewerCorners(m.points).map(([x, y]) => [px(x), px(y)] as [number, number]),
        ...label(m),
      });
    else
      (out.lines ??= []).push({
        x1: px(m.x1),
        y1: px(m.y1),
        x2: px(m.x2),
        y2: px(m.y2),
        arrow: !!m.arrow,
        measure: !!m.measure,
        ...(m.measure ? plane(m) : label(m)),
      });
  }
  return out;
}

/** A line's length, in the image's units when its pixel size is known. */
export function lineLength(m: LineMark, pixelSize?: number | null): string {
  const n = Math.hypot(m.x2 - m.x1, m.y2 - m.y1);
  return pixelSize ? fmt.length(n * pixelSize) : `${Math.round(n)} px`;
}
