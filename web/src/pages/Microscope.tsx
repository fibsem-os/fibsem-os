import { useQueries, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ImageInfoBar, ScaleBar } from "@/components/image-info";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import type {
  Capabilities,
  DeviceListing,
  DeviceSpec,
  DeviceValues,
  DisplayImages,
  ImagePayload,
  MicroscopeInfo,
  ParameterSpec,
} from "@/lib/types";

// Readings move without events (the stage, a beam's current), so they are read
// again on a clock while the page is open.
const READINGS_EVERY_MS = 5000;

type Row = [string, string];
type Values = Record<string, unknown>;

const num = (value: unknown) => (typeof value === "number" ? value : null);

const words = (text: string) => text.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function beamRows(v: Values): Row[] {
  const rows: Row[] = [
    ["Voltage", fmt.voltage(num(v.voltage))],
    ["Current", fmt.current(num(v.current))],
    ["Field width", fmt.length(num(v.hfw))],
    ["Working distance", fmt.length(num(v.working_distance))],
  ];
  if (v.detector_type) rows.push(["Detector", [v.detector_type, v.detector_mode].filter(Boolean).map(String).join(" · ")]);
  if (v.dwell_time != null) rows.push(["Dwell", fmt.seconds(num(v.dwell_time))]);
  if (Array.isArray(v.resolution)) rows.push(["Resolution", v.resolution.join(" × ")]);
  if (v.plasma_gas) rows.push(["Plasma gas", String(v.plasma_gas)]);
  return rows;
}

function stageRows(v: Values): Row[] {
  const p = (v.position ?? {}) as Values;
  return [
    ["X", fmt.length(num(p.x))],
    ["Y", fmt.length(num(p.y))],
    ["Z", fmt.length(num(p.z))],
    ["Rotation", fmt.angle(num(p.r))],
    ["Tilt", fmt.angle(num(p.t))],
  ];
}

function display(value: unknown, spec: ParameterSpec): string | null {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return words(value);
  if (typeof value !== "number") return null;
  const d = spec.display ?? {};
  if (d.scale != null) return `${+(value * d.scale).toFixed(d.decimals ?? 2)}${d.unit ?? ""}`;
  switch (spec.unit) {
    case "m":
      return fmt.length(value);
    case "A":
      return fmt.current(value);
    case "V":
      return fmt.voltage(value);
    case "s":
      return fmt.seconds(value);
    case "rad":
      return fmt.angle(value);
    default:
      return `${+value.toPrecision(4)}${spec.unit ? ` ${spec.unit}` : ""}`;
  }
}

/** Any other device: its plain readings, labelled as the device declares them. */
function genericRows(spec: DeviceSpec, v: Values): Row[] {
  const rows: Row[] = [];
  for (const [name, parameter] of Object.entries(spec.parameters)) {
    if (parameter.display?.advanced || !(name in v)) continue;
    const text = display(v[name], parameter);
    if (text != null) rows.push([parameter.display?.label ?? words(name), text]);
  }
  return rows;
}

// The cards, in this order; the first six show "Not Available" when the
// instrument has none. A device of another type gets a card after these.
const KNOWN: { title: string; match: (s: DeviceSpec) => boolean; always: boolean }[] = [
  { title: "Electron Beam", match: (s) => s.type === "beam" && /electron/i.test(s.name), always: true },
  { title: "Ion Beam", match: (s) => s.type === "beam" && /ion/i.test(s.name), always: true },
  { title: "Stage", match: (s) => s.type === "stage", always: true },
  { title: "Fluorescence Microscope", match: (s) => s.type === "fm", always: false },
  { title: "Gas Injection System", match: (s) => s.type === "gis", always: true },
  { title: "Sputter Coater", match: (s) => s.type === "sputter_coater", always: true },
  { title: "Manipulator", match: (s) => s.type === "manipulator", always: true },
];
// Left out of v1: the chamber and its camera.
const SKIPPED = new Set(["chamber", "chamber_camera"]);

interface Slot {
  title: string;
  spec?: DeviceSpec;
}

function slots(listing: DeviceListing): Slot[] {
  const specs = Object.values(listing).filter((s) => !SKIPPED.has(s.type));
  const used = new Set<string>();
  const out: Slot[] = [];
  for (const known of KNOWN) {
    const found = specs.filter((s) => !used.has(s.name) && known.match(s));
    found.forEach((s) => used.add(s.name));
    if (found.length === 1) out.push({ title: known.title, spec: found[0] });
    else if (found.length > 1) found.forEach((s) => out.push({ title: `${known.title} (${s.name})`, spec: s }));
    else if (known.always) out.push({ title: known.title });
  }
  for (const s of specs) if (!used.has(s.name)) out.push({ title: words(s.name), spec: s });
  return out;
}

function DeviceCard({ slot, values }: { slot: Slot; values?: DeviceValues }) {
  if (!slot.spec)
    return (
      <Card className="flex items-center gap-2 py-3 opacity-60">
        <CardTitle>{slot.title}</CardTitle>
        <span className="ml-auto text-xs text-muted-foreground">Not Available</span>
      </Card>
    );
  const v = values?.values ?? {};
  const rows = !values
    ? []
    : slot.spec.type === "beam"
      ? beamRows(v)
      : slot.spec.type === "stage"
        ? stageRows(v)
        : genericRows(slot.spec, v);
  const sub = slot.spec.type === "beam" && values ? (v.on === false ? "Off" : v.blanked ? "Blanked" : "On") : "";
  return (
    <Card className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <CardTitle className="text-foreground">{slot.title}</CardTitle>
        <span className="ml-auto text-xs text-muted-foreground">{sub}</span>
      </div>
      {values ? <Facts rows={rows} /> : <span className="text-[13px] text-muted-foreground">Reading…</span>}
    </Card>
  );
}

const IMAGE_LABEL: Record<string, string> = { sem: "SEM Image", fib: "FIB Image", fm: "FM Image" };

/** An image in a box exactly its shape, with its scalebar: *height* is the room it
 * has, as a CSS length, and it fits in the width it is given. */
function Shown({ image, alt, height }: { image: ImagePayload; alt: string; height: string }) {
  const ratio = image.width / image.height;
  return (
    <div className="relative" style={{ aspectRatio: `${ratio}`, width: `min(100%, calc(${height} * ${ratio}))` }}>
      <img
        src={`data:image/jpeg;base64,${image.image_b64_jpeg}`}
        alt={alt}
        className="absolute inset-0 block h-full w-full"
      />
      <ScaleBar info={image.info} />
    </div>
  );
}

function LastAcquired() {
  const images = useQuery({ queryKey: ["app/experiment/images"], queryFn: () => api<DisplayImages>("/app/experiment/images") });
  const [shown, setShown] = useState<string | null>(null);
  const keys = images.data?.fm_available ? (["sem", "fib", "fm"] as const) : (["sem", "fib"] as const);
  const pairs = keys.map((key) => [key, images.data?.[key] ?? null] as [string, ImagePayload | null]);
  const big = pairs.find(([key]) => key === shown)?.[1];
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Last acquired</CardTitle>
      <div className={`grid grid-cols-1 gap-3 sm:grid-cols-2 ${keys.length > 2 ? "lg:grid-cols-3" : ""}`}>
        {pairs.map(([key, image]) =>
          image ? (
            <button
              key={key}
              onClick={() => setShown(key)}
              className="flex min-w-0 cursor-pointer flex-col overflow-hidden rounded-sm border-0 bg-transparent p-0 text-left text-foreground"
            >
              {/* A 3:2 frame, the image fitted inside it: the scalebar needs the whole image. */}
              <div className="flex aspect-[3/2] w-full items-center justify-center bg-black [container-type:inline-size]">
                <Shown image={image} alt={`Last ${IMAGE_LABEL[key]}`} height="(100cqw * 2 / 3)" />
              </div>
              <ImageInfoBar info={image.info} />
            </button>
          ) : (
            <div key={key} className="flex flex-col gap-1.5">
              <div className="flex aspect-[3/2] w-full items-center justify-center rounded-sm bg-panel text-[13px] text-muted-foreground">
                None yet
              </div>
              <span className="text-[13px]">{IMAGE_LABEL[key]}</span>
            </div>
          ),
        )}
      </div>
      <Dialog open={!!big} onOpenChange={(open) => !open && setShown(null)}>
        {big && shown && (
          <DialogContent title={IMAGE_LABEL[shown]} className="w-[min(1100px,calc(100vw-32px))]">
            <div className="flex flex-col p-3.5">
              <div className="flex items-center justify-center bg-black">
                <Shown image={big} alt={IMAGE_LABEL[shown]} height="70vh" />
              </div>
              <ImageInfoBar info={big.info} />
            </div>
          </DialogContent>
        )}
      </Dialog>
    </Card>
  );
}

function MicroscopePanel({ info }: { info: MicroscopeInfo }) {
  const s = info.system.info;
  return (
    <Card className="flex flex-col gap-2.5">
      <CardTitle>Microscope</CardTitle>
      <Facts
        rows={[
          ["Name", s.name],
          ["Manufacturer", info.manufacturer],
          ["Model", info.model],
          ["Serial", info.serial_number],
          ["Hardware", s.hardware_version],
          ["Software", info.software_version],
          ["Address", s.port ? `${s.ip_address} : ${s.port}` : s.ip_address],
        ]}
      />
      <CardTitle className="mt-1.5">fibsemOS</CardTitle>
      <Facts
        rows={[
          ["Version", s.fibsem_version],
          ["Revision", s.fibsem_revision ?? "wheel install"],
          ["Driver", info.manufacturer],
        ]}
      />
    </Card>
  );
}

export function MicroscopePage() {
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const served = caps.data?.routers.microscope;
  const info = useQuery({
    queryKey: ["microscope"],
    queryFn: () => api<MicroscopeInfo>("/microscope"),
    enabled: !!served,
  });
  const listing = useQuery({
    queryKey: ["microscope/devices"],
    queryFn: () => api<DeviceListing>("/microscope/devices"),
    enabled: !!served,
  });
  const cards = listing.data ? slots(listing.data) : [];
  const readings = useQueries({
    queries: cards
      .filter((slot) => slot.spec)
      .map((slot) => ({
        queryKey: ["microscope/devices", slot.spec!.name, "values"],
        queryFn: () => api<DeviceValues>(`/microscope/devices/${encodeURIComponent(slot.spec!.name)}/values`),
        refetchInterval: READINGS_EVERY_MS,
      })),
  });
  const valuesOf = new Map(cards.filter((s) => s.spec).map((s, i) => [s.spec!.name, readings[i]?.data]));

  return (
    <div className="mx-auto flex w-full max-w-[1400px] flex-wrap gap-5 p-5">
      <main className="flex min-w-0 flex-[999_1_640px] flex-col gap-5">
        <LastAcquired />
        {caps.data && !served && (
          <Card>
            <p className="m-0 text-[13px] text-muted-foreground">
              This App Server does not serve the microscope's own routes, so its devices are not shown.
            </p>
          </Card>
        )}
        {listing.error && <p className="text-[13px] text-destructive">{(listing.error as Error).message}</p>}
        <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-2">
          {cards.map((slot) => (
            <DeviceCard key={slot.title} slot={slot} values={slot.spec ? valuesOf.get(slot.spec.name) : undefined} />
          ))}
        </div>
      </main>
      {info.data && (
        <aside className="flex min-w-0 flex-[1_1_300px] flex-col gap-5 lg:max-w-[400px]">
          <MicroscopePanel info={info.data} />
        </aside>
      )}
    </div>
  );
}
