import { ChevronLeft, ChevronRight } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { MarkPanel, MarkingLayer, TOOLS, useMarkEditor } from "@/components/mark-for-agent";
import { Stepper } from "@/components/stepper";
import { type StackView, useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { ImagePayload, StackFacts } from "@/lib/types";
import { cn } from "@/lib/utils";

const BEAM: Record<string, string> = { ELECTRON: "SEM", ION: "FIB" };
const WHOLE: StackView = { z: null, channels: null };

/** SEM / FIB / FM, from the file's own record, else its name. */
export function imageKind(path: string, beam?: unknown): string {
  if (typeof beam === "string" && BEAM[beam]) return BEAM[beam];
  if (/\.ome\.tiff?$/i.test(path)) return "FM";
  if (/_eb\.tif/i.test(path)) return "SEM";
  if (/_ib\.tif/i.test(path)) return "FIB";
  return "Image";
}

function clock(iso: unknown): string {
  if (typeof iso !== "string") return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** One line under a preview, as the Qt image bar says it: the beam or the
 * channels, field width, pixel size, dwell or exposure, the time, and for FM
 * the z planes and their step. */
export function metaLine(path: string, summary: Record<string, unknown> | undefined): string {
  if (!summary) return "";
  const shape = summary.shape as number[] | undefined;
  const pixel = summary.pixel_size as number | null | undefined;
  const width = (summary.hfw as number | null | undefined) ?? (pixel && shape ? pixel * shape[1] : null);
  const fm = summary.fm as StackFacts | undefined;
  const parts: string[] = [];
  if (fm) {
    const exposures = [...new Set(fm.channels.map((c) => c.exposure_time).filter((e): e is number => e != null))];
    parts.push(
      fm.channels.map((c) => (c.excitation_wavelength != null ? `${c.excitation_wavelength} nm` : c.name)).join(", "),
      fmt.length(width),
      pixel ? `${fmt.length(pixel)}/px` : "",
      exposures.length === 1 ? `${fmt.seconds(exposures[0])} exposure` : "",
      `${fm.z_planes} ${fm.z_planes === 1 ? "plane" : "planes"}${fm.z_step != null ? ` × ${fmt.length(fm.z_step)}` : ""}`,
      clock(fm.acquired_at),
    );
  } else {
    const beam = (summary.beam ?? {}) as Record<string, number | null>;
    parts.push(
      fmt.length(width),
      pixel ? `${fmt.length(pixel)}/px` : "",
      beam.dwell_time ? `${fmt.seconds(beam.dwell_time)} dwell` : "",
      clock(summary.acquired_at),
    );
  }
  return [imageKind(path, summary.beam_type), ...parts.filter((p) => p && p !== "–")].join(" · ");
}

function planeLine(fm: StackFacts, z: number | null) {
  if (z == null) return `Max projection of ${fm.z_planes} ${fm.z_planes === 1 ? "plane" : "planes"}${fm.z_step != null ? ` · ${fmt.length(fm.z_step)} step` : ""}`;
  const from = fm.z_positions ? Math.abs(fm.z_positions[z] - fm.z_positions[0]) : fm.z_step != null ? z * fm.z_step : null;
  return `Plane ${z + 1} of ${fm.z_planes}${z > 0 && from != null ? ` · ${fmt.length(from)} from plane 1` : ""}`;
}

/** The stack's channels, and what of it is shown. With *view*, each channel is
 * a checkbox, at least one stays on. */
export function Legend({ fm, view, setView }: { fm: StackFacts; view?: StackView; setView?: (v: StackView) => void }) {
  const shown = (name: string) => !view?.channels || view.channels.includes(name);
  const toggle = (name: string) => {
    if (!view || !setView) return;
    const on = fm.channels.map((c) => c.name).filter((n) => (n === name ? !shown(n) : shown(n)));
    if (on.length) setView({ ...view, channels: on.length === fm.channels.length ? null : on });
  };
  const pick = !!setView && fm.channels.length > 1;
  return (
    <div className="flex flex-col gap-1.5 text-[12px]">
      <div className={cn("flex gap-x-4 gap-y-1", pick ? "flex-col" : "flex-wrap items-center")}>
        {fm.channels.map((c) => {
          const label = (
            <>
              <span className="inline-block size-2.5 rounded-sm" style={{ background: c.color }} />
              {c.name}
              {c.excitation_wavelength != null && <span className="text-muted-foreground">{c.excitation_wavelength} nm</span>}
            </>
          );
          return pick ? (
            <label key={c.name} className="flex cursor-pointer items-center gap-1.5">
              <input type="checkbox" className="m-0 accent-[var(--primary)]" checked={shown(c.name)} onChange={() => toggle(c.name)} />
              {label}
            </label>
          ) : (
            <span key={c.name} className="flex items-center gap-1.5">
              {label}
            </span>
          );
        })}
      </div>
      <span className="text-muted-foreground">{planeLine(fm, view?.z ?? null)}</span>
    </div>
  );
}

/** Max projection or one plane, and which plane: the stack's z, as Qt's FM view steps it. */
function Planes({ fm, view, setView }: { fm: StackFacts; view: StackView; setView: (v: StackView) => void }) {
  const z = view.z ?? fm.channels.find((c) => c.sharpest_plane != null)?.sharpest_plane ?? 0;
  // How sharp each plane is in the channels shown, 1 the sharpest of them.
  const shown = fm.channels.filter((c) => c.sharpness && (!view.channels || view.channels.includes(c.name)));
  const sharpness = Array.from({ length: fm.z_planes }, (_, i) => Math.max(0, ...shown.map((c) => c.sharpness![i] ?? 0)));
  const go = (i: number) => setView({ ...view, z: Math.max(0, Math.min(fm.z_planes - 1, i)) });
  return (
    <div className="flex flex-col gap-2">
      <div role="group" aria-label="Planes" className="flex gap-1">
        {(
          [
            [true, "Max projection"],
            [false, "One plane"],
          ] as const
        ).map(([projection, label]) => (
          <button
            key={label}
            aria-pressed={(view.z == null) === projection}
            onClick={() => setView({ ...view, z: projection ? null : z })}
            className={cn(
              "cursor-pointer border-0 bg-transparent px-2 py-1 text-[12px]",
              (view.z == null) === projection
                ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <Stepper compact at={z} count={fm.z_planes} noun="plane" go={go} />
      {shown.length > 0 && (
        <div className="flex h-6 items-end gap-0.5" title="How sharp each plane is: pick one to show it">
          {sharpness.map((v, i) => (
            <button
              key={i}
              aria-label={`Plane ${i + 1}, sharpness ${v.toFixed(2)}`}
              onClick={() => go(i)}
              className={cn(
                "min-w-0 flex-1 cursor-pointer rounded-[1px] border-0 p-0",
                view.z === i ? "bg-[var(--primary)]" : "bg-border hover:bg-muted-foreground",
              )}
              style={{ height: `${20 + 80 * v}%` }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** A small preview that opens the viewer. */
export function Thumb({ path, onOpen }: { path: string; onOpen: () => void }) {
  const read = useFileRead(path, 480);
  const p = read.data?.preview;
  const kind = imageKind(path, read.data?.summary?.beam_type);
  return (
    <figure className="m-0 flex min-w-0 flex-col gap-1">
      <button
        className="aspect-[3/2] cursor-zoom-in overflow-hidden rounded border-0 bg-black p-0"
        onClick={onOpen}
        title={path.split("/").pop()}
      >
        {p && <img className="block size-full object-contain" alt={kind} src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />}
        {read.error && <span className="text-[12px] text-muted-foreground">Not readable</span>}
      </button>
      <figcaption className="text-[12px] text-muted-foreground">{metaLine(path, read.data?.summary) || `${kind} Image`}</figcaption>
    </figure>
  );
}

function metadata(summary: Record<string, unknown> | undefined, size: number | null | undefined): [string, string][] {
  if (!summary) return [];
  const beam = (summary.beam ?? {}) as Record<string, number | null>;
  const shape = summary.shape as number[] | undefined;
  const rows: [string, string][] = [
    ["Beam", BEAM[String(summary.beam_type)] ?? "–"],
    ["Voltage", fmt.voltage(beam.voltage)],
    ["Current", fmt.current(beam.current)],
    ["Field width", fmt.length(summary.hfw as number)],
    ["Size", shape ? `${shape[1]} × ${shape[0]} px${size ? ` · ${fmt.bytes(size)}` : ""}` : "–"],
    ["Pixel size", fmt.length(summary.pixel_size as number)],
    ["Dwell", fmt.seconds(beam.dwell_time)],
    ["Acquired", fmt.when(summary.acquired_at as string) || "–"],
  ];
  return rows.filter(([, v]) => v && v !== "–");
}

/** The viewer: one image large with its metadata, previous and next through *paths*. */
export function ImageViewer({ paths, at, onClose }: { paths: string[]; at: number | null; onClose: () => void }) {
  // The open image by its path, so images added to *paths* while it is open don't move it.
  const [path, setPath] = useState(at == null ? null : (paths[at] ?? null));
  const [opened, setOpened] = useState(at);
  const last = useRef<ImagePayload | undefined>(undefined);
  if (at !== opened) {
    setOpened(at);
    setPath(at == null ? null : (paths[at] ?? null));
    last.current = undefined;
  }
  const index = Math.max(0, path ? paths.indexOf(path) : 0);
  const [view, setView] = useState<StackView>(WHOLE);
  const [viewOf, setViewOf] = useState(path);
  const [marking, setMarking] = useState(false);
  const [sent, setSent] = useState(false);
  const read = useFileRead(path, 1600, view);
  const p = read.data?.preview;
  // The last image stays up, dimmed, while the next is read, so the box keeps its size.
  if (p) last.current = p;
  const shown = p ?? (read.error ? undefined : last.current);
  const fullWidth = shown?.full_width ?? shown?.width ?? 1;
  const fullHeight = shown?.full_height ?? shown?.height ?? 1;
  const editor = useMarkEditor(fullWidth, fullHeight);
  if (path !== viewOf) {
    setViewOf(path);
    setView(WHOLE);
    setMarking(false);
    setSent(false);
    editor.clear();
  }
  const fm = read.data?.summary?.fm;
  const stack = fm && fm.z_planes > 1 ? fm : undefined;
  const pixelSize = read.data?.summary?.pixel_size as number | null | undefined;
  // a beam image, or an FM stack's max projection: single planes can't be marked yet
  const markable = !!p && (!!fm || imageKind(path ?? "", read.data?.summary?.beam_type) !== "Image");
  const keys = useRef<HTMLDivElement>(null);
  const mark = (on: boolean) => {
    setMarking(on);
    setSent(false);
    if (on && view.z != null) setView({ ...view, z: null });
    // the tools' keys go to the viewer, not the button
    if (on) keys.current?.focus();
  };
  const tool = TOOLS.find((t) => t.tool === editor.tool);
  const step = (by: number) => setPath(paths[(index + by + paths.length) % paths.length]);
  const plane = (by: number) => {
    if (!stack || marking) return;
    const from = view.z ?? stack.channels.find((c) => c.sharpest_plane != null)?.sharpest_plane ?? 0;
    setView({ ...view, z: Math.max(0, Math.min(stack.z_planes - 1, view.z == null ? from : from + by)) });
  };
  const name = path?.split("/").pop() ?? "";
  return (
    <Dialog open={at != null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        title={name}
        className="w-[min(1280px,calc(100vw-32px))]"
        actions={
          markable && (
            <Button aria-pressed={marking} className={cn(marking && "border-primary")} onClick={() => mark(!marking)}>
              Mark for the agent
            </Button>
          )
        }
        onEscapeKeyDown={(e) => {
          if (!marking) return;
          e.preventDefault();
          mark(false);
        }}
      >
        <div
          ref={keys}
          tabIndex={-1}
          className={cn(
            "grid gap-4 outline-none",
            marking ? "md:grid-cols-[minmax(0,1fr)_320px]" : "md:grid-cols-[minmax(0,1fr)_240px]",
          )}
          onKeyDown={(e) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
            const key = e.key.toLowerCase();
            if (marking && TOOLS.some((t) => t.key === key)) editor.setTool(TOOLS.find((t) => t.key === key)!.tool);
            else if (marking && (e.key === "Delete" || e.key === "Backspace") && editor.selected != null)
              editor.remove(editor.selected);
            else if (e.key === "ArrowLeft") step(-1);
            else if (e.key === "ArrowRight") step(1);
            else if (e.key === "ArrowUp" && stack) plane(1);
            else if (e.key === "ArrowDown" && stack) plane(-1);
            else return;
            e.preventDefault();
          }}
        >
          <div className="relative flex min-h-[240px] items-center justify-center self-start bg-black p-2">
            {shown && (!p || read.isPlaceholderData) && (
              <span className="absolute top-3 left-3 z-10 bg-black/70 px-2 py-0.5 text-[12px] text-white">Reading…</span>
            )}
            {shown ? (
              <div
                className={cn("relative", marking && "outline outline-1 outline-[#50a6ff]")}
                style={{
                  aspectRatio: `${fullWidth} / ${fullHeight}`,
                  width: `min(100%, calc(70vh * ${fullWidth / fullHeight}))`,
                }}
              >
                <img
                  className={cn("absolute inset-0 block h-full w-full", (!p || read.isPlaceholderData) && "opacity-50")}
                  alt={name}
                  src={`data:image/jpeg;base64,${shown.image_b64_jpeg}`}
                />
                {marking && <MarkingLayer editor={editor} pixelSize={pixelSize} />}
                {marking && (
                  <span className="pointer-events-none absolute bottom-2 left-2 bg-black/70 px-2 py-0.5 text-[12px] text-white">
                    {tool?.hint} · Esc to stop marking
                  </span>
                )}
              </div>
            ) : (
              <span className="text-[13px] text-muted-foreground">{read.error ? read.error.message : "Reading…"}</span>
            )}
          </div>
          <div className="flex flex-col gap-3 px-4 pb-4 md:pt-2 md:pl-0">
            <Facts rows={metadata(read.data?.summary, read.data?.size)} />
            {fm && <Legend fm={fm} view={view} setView={setView} />}
            {stack && !marking && <Planes fm={stack} view={view} setView={setView} />}
            {marking && path && (
              <MarkPanel
                editor={editor}
                image={path}
                pixelSize={pixelSize}
                channels={fm ? (view.channels ?? fm.channels.map((c) => c.name)) : null}
                onSent={() => {
                  setMarking(false);
                  setSent(true);
                }}
              />
            )}
            {sent && (
              <div className="text-[12px] text-muted-foreground">Sent · the agent reads it on its next turn</div>
            )}
            <div className="text-[12px] break-all text-muted-foreground">{path}</div>
            {paths.length > 1 && (
              <div className="mt-auto flex items-center gap-2">
                <Button onClick={() => step(-1)} aria-label="Previous image">
                  <ChevronLeft size={14} />
                </Button>
                <span className="flex-1 text-center text-[12px] text-muted-foreground tabular-nums">
                  {index + 1} of {paths.length}
                </span>
                <Button onClick={() => step(1)} aria-label="Next image">
                  <ChevronRight size={14} />
                </Button>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
