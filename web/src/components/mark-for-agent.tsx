import { useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Button } from "@/components/ui/button";
import { MarkLayer, imagePoint, labelFraction } from "@/components/marks";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { MAX_LABEL, MAX_MARKS, type Mark, OPERATOR_MARK, lineLength, toMarks } from "@/lib/marks";
import { cn } from "@/lib/utils";

export type Tool = "point" | "box" | "line" | "arrow";

export const TOOLS: { tool: Tool; name: string; key: string; hint: string }[] = [
  { tool: "point", name: "Point", key: "p", hint: "Point: click where it is" },
  { tool: "box", name: "Box", key: "b", hint: "Box: drag from corner to corner" },
  { tool: "line", name: "Line", key: "l", hint: "Line: drag to measure" },
  { tool: "arrow", name: "Arrow", key: "a", hint: "Arrow: drag from the tail to the head" },
];

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
          <span className="ml-auto text-[12px] text-muted-foreground">{tool?.hint} · Esc to stop</span>
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
  return m;
}

/** Too small to have meant: a click with a drag tool. */
function tiny(m: Mark, k: number): boolean {
  if (m.kind === "area") return m.width < 4 * k || m.height < 4 * k;
  if (m.kind === "line") return Math.hypot(m.x2 - m.x1, m.y2 - m.y1) < 6 * k;
  return false;
}

/** The marks being made on one image of *width* x *height* source pixels. */
export function useMarkEditor(width: number, height: number) {
  const [tool, setTool] = useState<Tool>("point");
  const [marks, setMarks] = useState<Mark[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [labelling, setLabelling] = useState<number | null>(null);
  // The stack's plane on screen (null, the projection; undefined, not a stack): a mark
  // made on a plane is on it. *goTo* shows a plane, set by the image.
  const [plane, setPlane] = useState<number | null | undefined>(undefined);
  const goTo = useRef<((z: number) => void) | null>(null);
  const on = plane != null ? { z: plane } : {};
  const drag = useRef<Drag | null>(null);
  // the marks as last drawn, for the end of a drag that moved them since this render
  const latest = useRef(marks);
  latest.current = marks;

  const start = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0 || marks.length >= MAX_MARKS) return;
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
  const clear = () => {
    setMarks([]);
    setSelected(null);
    setLabelling(null);
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

/** The marks over the image while marking, and the label field of the one just made. */
export function MarkingLayer({ editor, pixelSize }: { editor: MarkEditor; pixelSize?: number | null }) {
  const { marks, labelling, width, height } = editor;
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

const GLYPH: Record<string, string> = { point: "+", area: "▭", circle: "○", arrow: "↗", line: "—" };

/** The side panel while marking: the marks with their labels, the message, and Send. *image* is the file's path; *channels* the FM channels shown. */
export function MarkPanel({
  editor,
  image,
  pixelSize,
  channels,
  onSent,
}: {
  editor: MarkEditor;
  image: string;
  pixelSize?: number | null;
  channels?: string[] | null;
  onSent: () => void;
}) {
  const client = useQueryClient();
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const { marks } = editor;
  const send = async () => {
    setError(null);
    setSending(true);
    try {
      const evidence = { image, marks: toMarks(marks), ...(channels ? { channels } : {}) };
      await api("/app/agent/messages", { method: "POST", body: JSON.stringify({ text: text.trim(), evidence }) });
      setText("");
      editor.clear();
      onSent();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
    }
  };
  return (
    <section className="flex min-h-0 flex-col gap-2.5">
      <h3 className="m-0 text-[13px] font-semibold">Marks</h3>
      {marks.length === 0 && <p className="m-0 text-[12px] text-muted-foreground">No marks yet.</p>}
      <ul className="m-0 flex max-h-[34vh] list-none flex-col gap-2 overflow-y-auto p-0">
        {marks.map((m, i) => {
          const glyph = GLYPH[m.kind === "line" && m.arrow ? "arrow" : m.kind];
          const measured = m.kind === "line" && m.measure;
          return (
            <li
              key={i}
              className={cn(
                "grid grid-cols-[14px_1fr_auto] items-center gap-x-2",
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
              <button
                className="cursor-pointer rounded border-0 bg-transparent p-1 text-muted-foreground hover:text-foreground"
                aria-label={`Remove mark ${i + 1}`}
                onClick={() => editor.remove(i)}
              >
                <X size={13} />
              </button>
              <span />
              <span className="text-[12px] text-muted-foreground tabular-nums">
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
      <div className="flex items-center gap-2">
        <span className="flex-1 text-[12px] text-muted-foreground">
          {channels
            ? `Sent with: ${channels.join(", ")} · ${madeOn(marks, editor.plane)}`
            : "Goes to the agent like any message"}
        </span>
        <Button onClick={editor.clear} disabled={!marks.length}>
          Clear
        </Button>
        <Button variant="primary" onClick={send} disabled={sending || (!marks.length && !text.trim())}>
          Send
        </Button>
      </div>
    </section>
  );
}
