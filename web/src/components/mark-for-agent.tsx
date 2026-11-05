import { useQuery, useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Button } from "@/components/ui/button";
import { MarkLayer, imagePoint, labelFraction, markLabel, markPlace } from "@/components/marks";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import {
  AGENT_MARK,
  BAD_MARK,
  GOOD_MARK,
  MAX_LABEL,
  MAX_MARKS,
  type Mark,
  OPERATOR_MARK,
  VERDICT_SIGN,
  type Verdict,
  lineLength,
  markRefs,
  toMarks,
} from "@/lib/marks";
import { cn } from "@/lib/utils";

export type Tool = "point" | "box" | "outline" | "line" | "arrow";

export const TOOLS: { tool: Tool; name: string; key: string; hint: string }[] = [
  { tool: "point", name: "Point", key: "p", hint: "Point: click where it is" },
  { tool: "box", name: "Box", key: "b", hint: "Box: drag from corner to corner" },
  { tool: "outline", name: "Outline", key: "o", hint: "Outline: draw round it" },
  { tool: "line", name: "Line", key: "l", hint: "Line: drag to measure" },
  { tool: "arrow", name: "Arrow", key: "a", hint: "Arrow: drag from the tail to the head" },
];

/** What a key says of a mark: g good, x bad. */
export function verdictForKey(key: string): Verdict | undefined {
  return key === "g" ? "good" : key === "x" ? "bad" : undefined;
}

/** The tool a key picks, or undefined. */
export function toolForKey(key: string): Tool | undefined {
  return TOOLS.find((t) => t.key === key)?.tool;
}

/** The strip over the image: "Mark for the agent" turns the mode on and off; while
 * it is on, the tools with their keys, and what the picked one does. */
export function MarkToolbar({
  editor,
  marking,
  mark,
}: {
  editor: MarkEditor;
  marking: boolean;
  mark: (on: boolean) => void;
}) {
  const tool = TOOLS.find((t) => t.tool === editor.tool);
  return (
    <div className="flex min-h-[36px] flex-wrap items-center gap-1.5 border-b border-border bg-background px-2 py-1">
      <Button
        aria-pressed={marking}
        className={cn("px-2.5", marking && "border-primary")}
        onClick={() => mark(!marking)}
      >
        Mark for the agent
      </Button>
      {marking && (
        <>
          <span className="mx-1 h-5 w-px bg-border" />
          {TOOLS.map((t) => (
            <Button
              key={t.tool}
              title={t.hint}
              aria-pressed={editor.tool === t.tool}
              className={cn("px-2.5", editor.tool === t.tool && "border-primary")}
              onClick={() => editor.setTool(t.tool)}
            >
              {t.name}
              <span className="text-[11px] text-muted-foreground uppercase">{t.key}</span>
            </Button>
          ))}
          <span className="mx-1 h-5 w-px bg-border" />
          {(["good", "bad"] as const).map((v) => (
            <Button
              key={v}
              title={`New marks are ${v}: press again for plain marks`}
              aria-pressed={editor.verdict === v}
              className={cn("px-2.5", editor.verdict === v && "border-primary")}
              onClick={() => editor.setVerdict(editor.verdict === v ? undefined : v)}
            >
              <span style={{ color: v === "good" ? GOOD_MARK : BAD_MARK }}>{VERDICT_SIGN[v]}</span>
              {v === "good" ? "Good" : "Bad"}
              <span className="text-[11px] text-muted-foreground uppercase">{v === "good" ? "g" : "x"}</span>
            </Button>
          ))}
          <span className="ml-auto text-[12px] text-muted-foreground">
            {editor.moving != null ? "Click where it should be" : tool?.hint} · Esc to stop
          </span>
        </>
      )}
    </div>
  );
}

type Spot = { x: number; y: number };

/** Keep the keys with the viewer: no mousedown to move focus off it, and focus on it,
 * so a tool's key works after a click and a new mark's label field keeps its focus. */
function hold(e: { preventDefault: () => void; currentTarget: Element }) {
  e.preventDefault();
  (e.currentTarget.closest("[tabindex]") as HTMLElement | null)?.focus();
}
type Drag = { index: number; from: Spot; was: Mark; creating: boolean };

function moved(m: Mark, dx: number, dy: number): Mark {
  if (m.kind === "outline") return { ...m, points: m.points.map(([x, y]) => [x + dx, y + dy] as [number, number]) };
  if (m.kind === "line") return { ...m, x1: m.x1 + dx, y1: m.y1 + dy, x2: m.x2 + dx, y2: m.y2 + dy };
  return { ...m, x: m.x + dx, y: m.y + dy };
}

function drawn(m: Mark, from: Spot, to: Spot): Mark {
  if (m.kind === "area")
    return {
      ...m,
      x: Math.min(from.x, to.x),
      y: Math.min(from.y, to.y),
      width: Math.abs(to.x - from.x),
      height: Math.abs(to.y - from.y),
    };
  if (m.kind === "line") return { ...m, x2: to.x, y2: to.y };
  if (m.kind === "outline") {
    const last = m.points[m.points.length - 1];
    return last && Math.hypot(to.x - last[0], to.y - last[1]) < 2 ? m : { ...m, points: [...m.points, [to.x, to.y]] };
  }
  return m;
}

/** Too small to have meant: a click with a drag tool. */
function tiny(m: Mark, k: number): boolean {
  if (m.kind === "area") return m.width < 4 * k || m.height < 4 * k;
  if (m.kind === "line") return Math.hypot(m.x2 - m.x1, m.y2 - m.y1) < 6 * k;
  if (m.kind === "outline") {
    const xs = m.points.map((p) => p[0]);
    const ys = m.points.map((p) => p[1]);
    return m.points.length < 3 || Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) < 8 * k;
  }
  return false;
}

/** What the operator said of one of the agent's picks: good or bad, and where it should be. */
export interface Judgement {
  verdict?: Verdict;
  moved_to?: Spot;
}

/** The marks being made on one image of *width* x *height* source pixels, and what is
 * said of the agent's picks on it. */
export function useMarkEditor(width: number, height: number) {
  const [tool, setTool] = useState<Tool>("point");
  const [marks, setMarks] = useState<Mark[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [labelling, setLabelling] = useState<number | null>(null);
  // what new marks are said to be
  const [verdict, setVerdict] = useState<Verdict | undefined>(undefined);
  // the agent's picks: what is said of each, by its place; the one asked about; the one being moved
  const [judged, setJudged] = useState<Record<number, Judgement>>({});
  const [choosing, setChoosing] = useState<number | null>(null);
  const [moving, setMoving] = useState<number | null>(null);
  // The stack's plane on screen (null, the projection; undefined, not a stack): a mark
  // made on a plane is on it. *goTo* shows a plane, set by the image.
  const [plane, setPlane] = useState<number | null | undefined>(undefined);
  const goTo = useRef<((z: number) => void) | null>(null);
  const on = { ...(plane != null ? { z: plane } : {}), ...(verdict ? { verdict } : {}) };
  const drag = useRef<Drag | null>(null);
  // the marks as last drawn, for the end of a drag that moved them since this render
  const latest = useRef(marks);
  latest.current = marks;

  const start = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    setChoosing(null);
    if (moving != null) {
      hold(e);
      const to = imagePoint(e, width, height);
      setJudged((all) => ({ ...all, [moving]: { ...all[moving], moved_to: to } }));
      setMoving(null);
      return;
    }
    if (marks.length >= MAX_MARKS) return;
    hold(e);
    const p = imagePoint(e, width, height);
    const index = marks.length;
    e.currentTarget.setPointerCapture(e.pointerId);
    setSelected(index);
    setLabelling(null);
    if (tool === "point") {
      setMarks([...marks, { kind: "point", ...p, ...on }]);
      setLabelling(index);
      return;
    }
    const mark: Mark =
      tool === "box"
        ? { kind: "area", ...p, width: 0, height: 0, ...on }
        : tool === "outline"
          ? { kind: "outline", points: [[p.x, p.y]], ...on }
          : {
              kind: "line",
              x1: p.x,
              y1: p.y,
              x2: p.x,
              y2: p.y,
              arrow: tool === "arrow",
              measure: tool === "line",
              ...on,
            };
    drag.current = { index, from: p, was: mark, creating: true };
    setMarks([...marks, mark]);
  };

  const pickUp = (index: number, e: ReactPointerEvent<SVGGElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    hold(e);
    (e.currentTarget.ownerSVGElement ?? e.currentTarget).setPointerCapture(e.pointerId);
    drag.current = { index, from: imagePoint(e, width, height), was: marks[index], creating: false };
    setSelected(index);
    setLabelling(null);
  };

  const move = (e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    const p = imagePoint(e, width, height);
    if (d.creating && d.was.kind === "outline") {
      // an outline grows from where it has got to
      setMarks((all) => all.map((m, i) => (i === d.index ? drawn(m, d.from, p) : m)));
      return;
    }
    const next = d.creating ? drawn(d.was, d.from, p) : moved(d.was, p.x - d.from.x, p.y - d.from.y);
    setMarks((all) => all.map((m, i) => (i === d.index ? next : m)));
  };

  const end = (e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    drag.current = null;
    if (!d?.creating) return;
    const k = width / Math.max(1, e.currentTarget.getBoundingClientRect().width);
    const made = latest.current[d.index];
    if (!made || tiny(made, k)) {
      setMarks((all) => all.filter((_, i) => i !== d.index));
      setSelected(null);
    } else if (!(made.kind === "line" && made.measure)) setLabelling(d.index);
  };

  const remove = (index: number) => {
    setMarks((all) => all.filter((_, i) => i !== index));
    setSelected(null);
    setLabelling(null);
  };
  const label = (index: number, text: string) =>
    setMarks((all) => all.map((m, i) => (i === index ? { ...m, label: text.slice(0, MAX_LABEL) } : m)));
  /** *v* said of the operator's mark *index*; said again, taken back. */
  const rate = (index: number, v: Verdict) =>
    setMarks((all) => all.map((m, i) => (i === index ? { ...m, verdict: m.verdict === v ? undefined : v } : m)));
  /** *v* said of the agent's pick *index*; said again, taken back. */
  const judge = (index: number, v: Verdict) => {
    setJudged((all) => ({ ...all, [index]: { ...all[index], verdict: all[index]?.verdict === v ? undefined : v } }));
    setChoosing(null);
  };
  const startMoving = (index: number | null) => {
    setMoving(index);
    setChoosing(null);
  };
  const unjudge = (index: number) => {
    setJudged((all) => {
      const { [index]: _, ...rest } = all;
      return rest;
    });
    setChoosing(null);
  };
  /** The agent's pick *index* clicked: asked what it is. */
  const choose = (index: number, e: ReactPointerEvent<SVGGElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    hold(e);
    setMoving(null);
    setLabelling(null);
    setChoosing(choosing === index ? null : index);
  };
  const clear = () => {
    setMarks([]);
    setSelected(null);
    setLabelling(null);
    setJudged({});
    setChoosing(null);
    setMoving(null);
  };
  return {
    tool,
    setTool,
    marks,
    selected,
    labelling,
    setLabelling,
    start,
    pickUp,
    move,
    end,
    verdict,
    setVerdict,
    rate,
    judged,
    judge,
    unjudge,
    choosing,
    choose,
    moving,
    startMoving,
    remove,
    label,
    clear,
    plane,
    setPlane,
    goTo,
    width,
    height,
  };
}

export type MarkEditor = ReturnType<typeof useMarkEditor>;

/** The agent's picks a moved one leaves behind: an arrow from it to where it should be. */
function movedPicks(picks: Mark[], judged: Record<number, Judgement>): Mark[] {
  return picks.flatMap((m, i) => {
    const to = judged[i]?.moved_to;
    if (!to) return [];
    const [x1, y1] = markPlace(m);
    return [{ kind: "line" as const, x1, y1, x2: to.x, y2: to.y, arrow: true, label: m.label, z: m.z }];
  });
}

/** What the agent's pick *index* is: good, bad, or somewhere else. */
function PickChoice({ editor, picks, index }: { editor: MarkEditor; picks: Mark[]; index: number }) {
  const m = picks[index];
  if (!m) return null;
  const [x, y] = markPlace(m);
  const said = editor.judged[index];
  const name = markLabel(m) ? `"${markLabel(m)}"` : `pick ${index + 1}`;
  return (
    <div
      role="dialog"
      aria-label={`The agent's ${name}`}
      className="absolute z-10 flex items-center gap-1 rounded border px-1.5 py-1 text-[12px] text-white"
      style={{
        left: `min(calc(${(x / editor.width) * 100}% + 12px), calc(100% - 15rem))`,
        top: `min(calc(${(y / editor.height) * 100}% + 12px), calc(100% - 2.5rem))`,
        borderColor: AGENT_MARK,
        background: "rgba(17, 19, 24, 0.9)",
      }}
    >
      <span className="mr-1 text-[#c4b5fd]">{name}</span>
      {(["good", "bad"] as const).map((v) => (
        <Button
          key={v}
          className={cn("h-6 px-1.5 text-[12px]", said?.verdict === v && "border-primary")}
          aria-pressed={said?.verdict === v}
          onClick={() => editor.judge(index, v)}
        >
          <span style={{ color: v === "good" ? GOOD_MARK : BAD_MARK }}>{VERDICT_SIGN[v]}</span>
          {v === "good" ? "Good" : "Bad"}
        </Button>
      ))}
      <Button className="h-6 px-1.5 text-[12px]" onClick={() => editor.startMoving(index)}>
        Move
      </Button>
      {said && (
        <Button className="h-6 px-1.5 text-[12px]" onClick={() => editor.unjudge(index)}>
          Clear
        </Button>
      )}
    </div>
  );
}

/** The marks over the image while marking, and the label field of the one just made.
 * *picks*, the agent's marks on the image: purple, fixed, each said good or bad by a click. */
export function MarkingLayer({
  editor,
  pixelSize,
  picks,
}: {
  editor: MarkEditor;
  pixelSize?: number | null;
  picks?: Mark[];
}) {
  const { marks, labelling, width, height, judged } = editor;
  const typing = labelling != null ? marks[labelling] : undefined;
  const [fx, fy] = typing ? labelFraction(typing, width, height) : [0, 0];
  return (
    <>
      <MarkLayer
        className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
        marks={marks}
        width={width}
        height={height}
        color={OPERATOR_MARK}
        pixelSize={pixelSize}
        selected={editor.selected}
        hiddenLabel={labelling}
        plane={editor.plane}
        onMarkDown={editor.pickUp}
        onPointerDown={editor.start}
        onPointerMove={editor.move}
        onPointerUp={editor.end}
      />
      {picks && picks.length > 0 && (
        <>
          <MarkLayer
            marks={movedPicks(picks, judged)}
            width={width}
            height={height}
            color={OPERATOR_MARK}
            plane={editor.plane}
          />
          <MarkLayer
            // only the picks themselves take a click; the rest goes to the marks under
            className="pointer-events-none absolute inset-0 h-full w-full [&>g]:pointer-events-auto [&>g]:cursor-pointer"
            marks={picks}
            width={width}
            height={height}
            color={AGENT_MARK}
            pixelSize={pixelSize}
            plane={editor.plane}
            selected={editor.choosing ?? editor.moving}
            verdicts={picks.map((_, i) => judged[i]?.verdict)}
            dimmed={picks.map((_, i) => !!judged[i]?.moved_to)}
            onMarkDown={editor.choose}
          />
        </>
      )}
      {picks && editor.choosing != null && <PickChoice editor={editor} picks={picks} index={editor.choosing} />}
      {typing && (
        <input
          autoFocus
          aria-label="Label"
          placeholder="Label, or Enter for none"
          maxLength={MAX_LABEL}
          className="absolute w-44 rounded-sm border px-1.5 py-0.5 text-[12px] text-[#50a6ff] outline-none"
          style={{
            left: `min(${fx * 100}%, calc(100% - 11rem))`,
            top: `max(0px, calc(${fy * 100}% - 1.6rem))`,
            borderColor: OPERATOR_MARK,
            background: "rgba(17, 19, 24, 0.8)",
          }}
          value={typing.label ?? ""}
          onChange={(e) => editor.label(labelling!, e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === "Escape") {
              e.preventDefault();
              e.stopPropagation();
              (e.currentTarget.closest("[tabindex]") as HTMLElement | null)?.focus();
              editor.setLabelling(null);
            }
          }}
          onBlur={() => editor.setLabelling(null)}
        />
      )}
    </>
  );
}

function where(m: Mark, pixelSize?: number | null): string {
  const n = (v: number) => Math.round(v);
  if (m.kind === "point") return `${n(m.x)}, ${n(m.y)} px`;
  if (m.kind === "area") {
    const size = pixelSize ? ` · ${fmt.length(m.width * pixelSize)} × ${fmt.length(m.height * pixelSize)}` : "";
    return `${n(m.x)}, ${n(m.y)} px${size}`;
  }
  if (m.kind === "circle") return `${n(m.x)}, ${n(m.y)} px, r ${n(m.radius)}`;
  if (m.kind === "outline") {
    const [x, y] = markPlace(m);
    return `${n(x)}, ${n(y)} px · ${m.points.length} corners`;
  }
  return `${n(m.x1)}, ${n(m.y1)} → ${n(m.x2)}, ${n(m.y2)} px`;
}

/** What a stack's marks were made on, for the Send line: "max projection", "plane 12", or both; before any, what is on screen. */
function madeOn(marks: Mark[], plane?: number | null): string {
  if (!marks.length) return plane != null ? `plane ${plane + 1}` : "max projection";
  const planes = [...new Set(marks.filter((m) => m.z != null).map((m) => m.z! + 1))].sort((a, b) => a - b);
  const on = planes.length ? `plane${planes.length > 1 ? "s" : ""} ${planes.join(", ")}` : "";
  const flat = marks.some((m) => m.z == null) ? "max projection" : "";
  return [flat, on].filter(Boolean).join(", ");
}

const GLYPH: Record<string, string> = { point: "+", area: "▭", circle: "○", outline: "⬠", arrow: "↗", line: "—" };

/** Good and bad, each pressed when said, for a list row. */
function Verdicts({ said, name, set }: { said?: Verdict; name: string; set: (v: Verdict) => void }) {
  return (
    <span className="flex gap-0.5">
      {(["good", "bad"] as const).map((v) => (
        <button
          key={v}
          aria-label={`${name} is ${v}`}
          aria-pressed={said === v}
          title={v === "good" ? "Good (g)" : "Bad (x)"}
          className={cn(
            "cursor-pointer rounded border bg-transparent px-1.5 py-0.5 text-[12px] leading-none",
            said === v ? "border-current" : "border-transparent text-muted-foreground hover:text-foreground",
          )}
          style={said === v ? { color: v === "good" ? GOOD_MARK : BAD_MARK } : undefined}
          onClick={() => set(v)}
        >
          {VERDICT_SIGN[v]}
        </button>
      ))}
    </span>
  );
}

/** The agent's picks on the image, each said good or bad, or moved. */
function PickList({ editor, picks }: { editor: MarkEditor; picks: Mark[] }) {
  const n = (v: number) => Math.round(v);
  return (
    <>
      <h3 className="m-0 text-[13px] font-semibold">The agent's picks</h3>
      <p className="m-0 text-[12px] text-muted-foreground">
        Rate the ones you care about: click one on the image, or here.
      </p>
      <ul className="m-0 flex max-h-[24vh] list-none flex-col gap-1.5 overflow-y-auto p-0">
        {picks.map((m, i) => {
          const said = editor.judged[i];
          const name = markLabel(m) || `Pick ${i + 1}`;
          return (
            <li key={i} className="grid grid-cols-[14px_1fr_auto] items-center gap-x-2">
              <span className="text-center text-[13px] text-[#a78bfa]">
                {GLYPH[m.kind === "line" && m.arrow ? "arrow" : m.kind]}
              </span>
              <span className="min-w-0 truncate text-[13px]">
                {name}
                {!said?.verdict && !said?.moved_to && (
                  <span className="text-[12px] text-muted-foreground"> · unrated</span>
                )}
                {said?.moved_to && (
                  <span className="text-[12px] text-muted-foreground">
                    {" "}
                    · moved to {n(said.moved_to.x)}, {n(said.moved_to.y)} px
                  </span>
                )}
              </span>
              <Verdicts said={said?.verdict} name={name} set={(v) => editor.judge(i, v)} />
            </li>
          );
        })}
      </ul>
    </>
  );
}

/** The sample the agent's plan names, or null before it names one. */
export function usePlanSample(): string | null {
  const plan = useQuery({
    queryKey: ["app/agent/plan"],
    queryFn: () => api<{ sample: string | null }>("/app/agent/plan"),
    staleTime: 30_000,
  });
  return plan.data?.sample ?? null;
}

/** Keep as example: on by default once something is said good or bad; off, and why,
 * before the plan names a sample. *keep* null follows the default. */
export function KeepTick({
  said,
  sample,
  keep,
  setKeep,
}: {
  said: boolean;
  sample: string | null;
  keep: boolean | null;
  setKeep: (on: boolean) => void;
}) {
  const can = said && !!sample;
  const why = !sample
    ? "The plan names no sample yet"
    : !said
      ? "Say good or bad of something to keep it"
      : `Kept in the knowledge base as an example of ${sample}`;
  return (
    <label
      className={cn("flex items-center gap-1.5 text-[12px]", can ? "cursor-pointer" : "text-muted-foreground")}
      title={why}
    >
      <input
        type="checkbox"
        className="m-0 accent-[var(--primary)]"
        disabled={!can}
        checked={can && (keep ?? true)}
        onChange={(e) => setKeep(e.target.checked)}
      />
      Keep as example
    </label>
  );
}

/** The side panel while marking: the marks with their labels, the message, and Send. *image* is the file's path; *channels* the FM channels shown. */
export function MarkPanel({
  editor,
  image,
  pixelSize,
  channels,
  answers,
  picks,
  onSent,
}: {
  editor: MarkEditor;
  image: string;
  pixelSize?: number | null;
  channels?: string[] | null;
  /** The evidence id of the agent's picture this answers, and its picks on *image*. */
  answers?: string;
  picks?: Mark[];
  onSent: () => void;
}) {
  const client = useQueryClient();
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [keep, setKeep] = useState<boolean | null>(null);
  const sample = usePlanSample();
  const { marks } = editor;
  const refs = markRefs(picks ?? []);
  const judged = Object.entries(editor.judged)
    .filter(([i, j]) => refs[+i] && (j.verdict || j.moved_to))
    .map(([i, j]) => ({
      ...refs[+i],
      ...(j.verdict ? { verdict: j.verdict } : {}),
      ...(j.moved_to ? { moved_to: { x: Math.round(j.moved_to.x), y: Math.round(j.moved_to.y) } } : {}),
    }));
  const said = judged.some((j) => j.verdict) || marks.some((m) => m.verdict);
  const kept = said && !!sample && (keep ?? true);
  const send = async () => {
    setError(null);
    setSending(true);
    try {
      const evidence = {
        image,
        marks: toMarks(marks),
        ...(channels ? { channels } : {}),
        ...(answers ? { answers, ...(judged.length ? { judged } : {}) } : {}),
        ...(kept ? { keep: true } : {}),
      };
      await api("/app/agent/messages", { method: "POST", body: JSON.stringify({ text: text.trim(), evidence }) });
      setText("");
      setKeep(null);
      editor.clear();
      onSent();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
      if (kept) await client.invalidateQueries({ queryKey: ["app/knowledge"] });
    }
  };
  return (
    <section className="flex min-h-0 flex-col gap-2.5">
      {picks && picks.length > 0 && <PickList editor={editor} picks={picks} />}
      <h3 className="m-0 text-[13px] font-semibold">{picks?.length ? "Your marks" : "Marks"}</h3>
      {marks.length === 0 && <p className="m-0 text-[12px] text-muted-foreground">No marks yet.</p>}
      <ul className="m-0 flex max-h-[34vh] list-none flex-col gap-2 overflow-y-auto p-0">
        {marks.map((m, i) => {
          const glyph = GLYPH[m.kind === "line" && m.arrow ? "arrow" : m.kind];
          const measured = m.kind === "line" && m.measure;
          return (
            <li
              key={i}
              className={cn(
                "grid grid-cols-[14px_1fr_auto_auto] items-center gap-x-2",
                i === editor.selected && "text-foreground",
              )}
            >
              <span className="text-center text-[13px] text-[#50a6ff]">{glyph}</span>
              {measured ? (
                <span className="text-[13px]">Measured · {lineLength(m, pixelSize)}</span>
              ) : (
                <input
                  aria-label={`Label of mark ${i + 1}`}
                  placeholder="No label"
                  maxLength={MAX_LABEL}
                  className="min-w-0 rounded border border-input bg-background px-2 py-1 text-[13px]"
                  value={m.label ?? ""}
                  onChange={(e) => editor.label(i, e.target.value)}
                />
              )}
              <Verdicts said={m.verdict} name={`Mark ${i + 1}`} set={(v) => editor.rate(i, v)} />
              <button
                className="cursor-pointer rounded border-0 bg-transparent p-1 text-muted-foreground hover:text-foreground"
                aria-label={`Remove mark ${i + 1}`}
                onClick={() => editor.remove(i)}
              >
                <X size={13} />
              </button>
              <span />
              <span className="col-span-3 text-[12px] text-muted-foreground tabular-nums">
                {where(m, pixelSize)}
                {m.z != null && (
                  <>
                    {" · "}
                    <button
                      className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                      aria-label={`Show plane ${m.z + 1}`}
                      onClick={() => editor.goTo.current?.(m.z!)}
                    >
                      plane {m.z + 1}
                    </button>
                  </>
                )}
              </span>
            </li>
          );
        })}
      </ul>
      <textarea
        rows={3}
        aria-label="Message to the agent"
        placeholder="What should the agent know?"
        className="resize-y rounded border border-input bg-background px-2 py-1.5 text-[13px]"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setError(null);
        }}
      />
      {error && <p className="m-0 text-[12px] text-destructive">{error}</p>}
      <KeepTick said={said} sample={sample} keep={keep} setKeep={setKeep} />
      <div className="flex items-center gap-2">
        <span className="flex-1 text-[12px] text-muted-foreground">
          {channels
            ? `Sent with: ${channels.join(", ")} · ${madeOn(marks, editor.plane)}`
            : "Goes to the agent like any message"}
        </span>
        <Button onClick={editor.clear} disabled={!marks.length && !judged.length}>
          Clear
        </Button>
        <Button
          variant="primary"
          onClick={send}
          disabled={sending || (!marks.length && !judged.length && !text.trim())}
        >
          Send
        </Button>
      </div>
    </section>
  );
}
