import { ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { StackFacts } from "@/lib/types";

const BEAM: Record<string, string> = { ELECTRON: "SEM", ION: "FIB" };

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

export function Legend({ fm }: { fm: StackFacts }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
      {fm.channels.map((c) => (
        <span key={c.name} className="flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-sm" style={{ background: c.color }} />
          {c.name}
          {c.excitation_wavelength != null && <span className="text-muted-foreground">{c.excitation_wavelength} nm</span>}
        </span>
      ))}
      <span className="text-muted-foreground">
        Max projection of {fm.z_planes} {fm.z_planes === 1 ? "plane" : "planes"}
        {fm.z_step != null && ` · ${fmt.length(fm.z_step)} step`}
      </span>
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
  const [index, setIndex] = useState(at ?? 0);
  const [opened, setOpened] = useState(at);
  if (at !== opened) {
    setOpened(at);
    if (at != null) setIndex(at);
  }
  const path = at == null ? null : paths[index];
  const read = useFileRead(path, 1600);
  const p = read.data?.preview;
  const fm = read.data?.summary?.fm;
  const step = (by: number) => setIndex((i) => (i + by + paths.length) % paths.length);
  const name = path?.split("/").pop() ?? "";
  return (
    <Dialog open={at != null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent title={name} className="w-[min(1280px,calc(100vw-32px))]">
        <div
          className="grid gap-4 md:grid-cols-[minmax(0,1fr)_240px]"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") step(-1);
            if (e.key === "ArrowRight") step(1);
          }}
        >
          <div className="flex min-h-[240px] items-center justify-center bg-black p-2">
            {p ? (
              <img className="block max-h-[70vh] w-full object-contain" alt={name} src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />
            ) : (
              <span className="text-[13px] text-muted-foreground">{read.error ? read.error.message : "Reading…"}</span>
            )}
          </div>
          <div className="flex flex-col gap-3 px-4 pb-4 md:pt-2 md:pl-0">
            <Facts rows={metadata(read.data?.summary, read.data?.size)} />
            {fm && <Legend fm={fm} />}
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
