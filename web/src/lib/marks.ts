import * as fmt from "./format";

/** Marks on an image, as evidence records keep them: full-resolution source
 * pixels, origin top left (plans/agent-drawing, "Marks schema"). */

export interface PointMark {
  x: number;
  y: number;
  shape?: "cross" | "target";
  label?: string;
}

export interface AreaMark {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
}

export interface CircleMark {
  x: number;
  y: number;
  radius: number;
  label?: string;
}

export interface LineMark {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  arrow?: boolean;
  measure?: boolean;
  label?: string;
}

export interface Marks {
  points?: PointMark[];
  areas?: AreaMark[];
  circles?: CircleMark[];
  lines?: LineMark[];
}

/** One mark of any kind, as the editor keeps them in the order they were made. */
export type Mark =
  | ({ kind: "point" } & PointMark)
  | ({ kind: "area" } & AreaMark)
  | ({ kind: "circle" } & CircleMark)
  | ({ kind: "line" } & LineMark);

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
  ];
}

const px = (v: number) => Math.round(v);

/** *list* as a record keeps it: whole pixels, empty labels and kinds left out,
 * a measured line's label left to the server. */
export function toMarks(list: Mark[]): Marks {
  const label = (m: { label?: string }) => (m.label?.trim() ? { label: m.label.trim().slice(0, MAX_LABEL) } : {});
  const out: Marks = {};
  for (const m of list) {
    if (m.kind === "point")
      (out.points ??= []).push({ x: px(m.x), y: px(m.y), ...(m.shape ? { shape: m.shape } : {}), ...label(m) });
    else if (m.kind === "area")
      (out.areas ??= []).push({ x: px(m.x), y: px(m.y), width: px(m.width), height: px(m.height), ...label(m) });
    else if (m.kind === "circle")
      (out.circles ??= []).push({ x: px(m.x), y: px(m.y), radius: px(m.radius), ...label(m) });
    else
      (out.lines ??= []).push({
        x1: px(m.x1),
        y1: px(m.y1),
        x2: px(m.x2),
        y2: px(m.y2),
        arrow: !!m.arrow,
        measure: !!m.measure,
        ...(m.measure ? {} : label(m)),
      });
  }
  return out;
}

/** A line's length, in the image's units when its pixel size is known. */
export function lineLength(m: LineMark, pixelSize?: number | null): string {
  const n = Math.hypot(m.x2 - m.x1, m.y2 - m.y1);
  return pixelSize ? fmt.length(n * pixelSize) : `${Math.round(n)} px`;
}
