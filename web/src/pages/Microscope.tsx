import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ImageInfoBar, ScaleBar } from "@/components/image-info";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, ApiError, reading, type Reading } from "@/lib/api";
import * as fmt from "@/lib/format";
import type {
  Capabilities,
  Connection,
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
  if (v.detector_type)
    rows.push(["Detector", [v.detector_type, v.detector_mode].filter(Boolean).map(String).join(" · ")]);
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
  // A vector or a dict of readings (a shift {x, y}, a size [w, h]): each part
  // shown as the parameter's unit says, so the card isn't empty for them.
  if (Array.isArray(value)) {
    const parts = value.map((part) => display(part, spec));
    return parts.length && parts.every((part) => part != null) ? parts.join(" × ") : null;
  }
  if (value != null && typeof value === "object") {
    const parts = Object.entries(value)
      .map(([key, part]) => {
        const text = display(part, spec);
        return text == null ? null : `${key} ${text}`;
      })
      .filter((part): part is string => part != null);
    return parts.length ? parts.join(" · ") : null;
  }
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

// The cards, in this order; on a FIB/SEM the ones marked always show "Not
// Available" when it has none. A device of another type gets a card after these.
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

/** *fibsem*: whether a FIB/SEM is served, whose missing parts still get a card. */
function slots(listing: DeviceListing, fibsem: boolean): Slot[] {
  const specs = Object.values(listing).filter((s) => !SKIPPED.has(s.type));
  const used = new Set<string>();
  const out: Slot[] = [];
  for (const known of KNOWN) {
    const found = specs.filter((s) => !used.has(s.name) && known.match(s));
    found.forEach((s) => used.add(s.name));
    if (found.length === 1) out.push({ title: known.title, spec: found[0] });
    else if (found.length > 1) found.forEach((s) => out.push({ title: `${known.title} (${s.name})`, spec: s }));
    else if (known.always && fibsem) out.push({ title: known.title });
  }
  for (const s of specs) if (!used.has(s.name)) out.push({ title: words(s.name), spec: s });
  return out;
}

/** When the card's readings are not live: their age while the microscope is
 * busy, or that it is busy with nothing read yet. */
function staleness(read?: Reading<DeviceValues>, error?: unknown): string | null {
  const busy = error instanceof ApiError && error.errorType === "busy";
  if (read?.age != null) return `Busy · read ${Math.round(read.age)} s ago`;
  if (busy) return read ? "Busy · not read again yet" : "Microscope busy";
  return null;
}

function DeviceCard({
  slot,
  read,
  error,
  fibsem,
}: {
  slot: Slot;
  read?: Reading<DeviceValues>;
  error?: unknown;
  fibsem: boolean;
}) {
  const values = read?.value;
  const stale = staleness(read, error);
  if (!slot.spec)
    return (
      <Card className="flex items-center gap-2 py-3 opacity-60">
        <CardTitle>{slot.title}</CardTitle>
        <span className="ml-auto text-xs text-muted-foreground">Not Available</span>
      </Card>
    );
  const v = values?.values ?? {};
  // A FIB/SEM's beams and stage read as their own rows; any other instrument's
  // devices as they describe themselves.
  const rows = !values
    ? []
    : fibsem && slot.spec.type === "beam"
      ? beamRows(v)
      : fibsem && slot.spec.type === "stage"
        ? stageRows(v)
        : genericRows(slot.spec, v);
  const sub = slot.spec.type === "beam" && values ? (v.on === false ? "Off" : v.blanked ? "Blanked" : "On") : "";
  return (
    <Card className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <CardTitle className="text-foreground">{slot.title}</CardTitle>
        <span className="ml-auto text-xs text-muted-foreground">{sub}</span>
      </div>
      {stale && <span className="text-xs text-[var(--fibsem-semantic-warning-color)]">{stale}</span>}
      {values ? <Facts rows={rows} /> : !stale && <span className="text-[13px] text-muted-foreground">Reading…</span>}
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

/** The latest image of each view: a FIB/SEM's SEM, FIB and FM, always; any
 * other instrument's, one for each device that has sent one. */
function LastAcquired({ fibsem }: { fibsem: boolean }) {
  const images = useQuery({
    queryKey: ["app/experiment/images"],
    queryFn: () => api<DisplayImages>("/app/experiment/images"),
  });
  const [shown, setShown] = useState<string | null>(null);
  const keys = images.data?.fm_available ? (["sem", "fib", "fm"] as const) : (["sem", "fib"] as const);
  const pairs: [string, ImagePayload | null][] = fibsem
    ? keys.map((key) => [key, images.data?.[key] ?? null])
    : Object.entries(images.data?.devices ?? {});
  const label = (key: string) => (fibsem ? IMAGE_LABEL[key] : words(key));
  // A device's image is named by its device; a beam's by its metadata.
  const infoOf = (key: string, image: ImagePayload) =>
    fibsem || !image.info ? image.info : { ...image.info, kind: label(key) };
  const big = pairs.find(([key]) => key === shown)?.[1];
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Last acquired</CardTitle>
      {pairs.length === 0 && <span className="text-[13px] text-muted-foreground">None yet</span>}
      <div className={`grid grid-cols-1 gap-3 sm:grid-cols-2 ${pairs.length > 2 ? "lg:grid-cols-3" : ""}`}>
        {pairs.map(([key, image]) =>
          image ? (
            <button
              key={key}
              onClick={() => setShown(key)}
              className="flex min-w-0 cursor-pointer flex-col overflow-hidden rounded-sm border-0 bg-transparent p-0 text-left text-foreground"
            >
              {/* A 3:2 frame, the image fitted inside it: the scalebar needs the whole image. */}
              <div className="flex aspect-[3/2] w-full items-center justify-center bg-black [container-type:inline-size]">
                <Shown image={image} alt={`Last ${label(key)}`} height="(100cqw * 2 / 3)" />
              </div>
              <ImageInfoBar info={infoOf(key, image)} />
            </button>
          ) : (
            <div key={key} className="flex flex-col gap-1.5">
              <div className="flex aspect-[3/2] w-full items-center justify-center rounded-sm bg-panel text-[13px] text-muted-foreground">
                None yet
              </div>
              <span className="text-[13px]">{label(key)}</span>
            </div>
          ),
        )}
      </div>
      <Dialog open={!!big} onOpenChange={(open) => !open && setShown(null)}>
        {big && shown && (
          <DialogContent title={label(shown)} className="w-[min(1100px,calc(100vw-32px))]">
            <div className="flex flex-col p-3.5">
              <div className="flex items-center justify-center bg-black">
                <Shown image={big} alt={label(shown)} height="70vh" />
              </div>
              <ImageInfoBar info={infoOf(shown, big)} />
            </div>
          </DialogContent>
        )}
      </Dialog>
    </Card>
  );
}

// Read often while a connect runs, so its steps show as they happen.
const CONNECTING_EVERY_MS = 1000;
const FIELD = "min-w-0 flex-1 rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";

/** Connect and disconnect the microscope: the operator's, so a token that may
 * not (the agent's) gets no card. The gateway's page passes
 * its own routes to the host's (`base` "/gateway"). */
export function ConnectionCard({ base = "", title = "Connection" }: { base?: string; title?: string }) {
  const client = useQueryClient();
  const key = `${base}/microscope/connection`;
  const state = useQuery({
    queryKey: [key],
    queryFn: () => api<Connection>(key),
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.connecting || query.state.data?.run?.state === "running" ? CONNECTING_EVERY_MS : 5000,
  });
  const [chosen, setChosen] = useState<string | null>(null);
  const act = useMutation({
    mutationFn: (path: string) =>
      api<Connection>(base + path, {
        method: "POST",
        body: JSON.stringify(path.endsWith("/connect") ? { configuration: chosen ?? undefined } : {}),
      }),
    onSettled: () => client.invalidateQueries(),
  });
  const data = state.data;
  // The rest of the page reads the microscope again once it comes or goes.
  const connected = data?.connected;
  useEffect(() => {
    if (connected !== undefined) void client.invalidateQueries({ predicate: (q) => q.queryKey[0] !== key });
  }, [connected, client, key]);
  if (!data) return null;
  const busy = act.isPending || !!data.connecting || data.run?.state === "running";
  const fallback = data.configurations.find((c) => c.default)?.name ?? data.configurations[0]?.name ?? "";
  const configuration = chosen ?? fallback;
  const c = data.configuration;
  const failed = (act.error as Error | null)?.message ?? (!data.connected && !busy ? data.error : null);
  return (
    <Card className="flex flex-col gap-2.5">
      <CardTitle>{title}</CardTitle>
      {data.connected && c ? (
        <>
          <Facts
            rows={(
              [
                ["Configuration", c.name ?? c.path ?? ""],
                ["Microscope", [c.manufacturer, c.model].filter(Boolean).join(" ")],
                ["Address", c.address ?? ""],
              ] as [string, string][]
            ).filter(([, value]) => value)}
          />
          <div className="flex justify-end">
            <Button disabled={busy} onClick={() => act.mutate("/microscope/disconnect")}>
              Disconnect
            </Button>
          </div>
        </>
      ) : (
        <div className="flex items-center gap-2">
          <select
            value={configuration}
            disabled={busy}
            onChange={(event) => setChosen(event.target.value)}
            className={FIELD}
            aria-label="Configuration"
          >
            {data.configurations.map((option) => (
              <option key={option.name} value={option.name}>
                {option.name}
              </option>
            ))}
          </select>
          <Button variant="primary" disabled={busy || !configuration} onClick={() => act.mutate("/microscope/connect")}>
            {busy ? "Connecting…" : "Connect"}
          </Button>
        </div>
      )}
      {busy && data.connecting && <p className="m-0 text-[13px] text-muted-foreground">{data.connecting}</p>}
      {failed && <p className="m-0 text-[13px] text-destructive">{failed}</p>}
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
  // Devices with no microscope are served too, though no microscope is connected.
  const served = caps.data?.routers.microscope && (caps.data.connected !== false || !!caps.data.instruments?.length);
  // What is served first: a FIB/SEM ("fibsem"), or devices with no microscope.
  // A server that does not say serves a FIB/SEM.
  const kind = caps.data?.instruments?.[0];
  const fibsem = kind === undefined || kind === "fibsem";
  const info = useQuery({
    queryKey: ["microscope"],
    queryFn: () => api<MicroscopeInfo>("/microscope"),
    enabled: !!served && fibsem,
  });
  const listing = useQuery({
    queryKey: ["microscope/devices"],
    queryFn: () => api<DeviceListing>("/microscope/devices"),
    enabled: !!served,
  });
  const cards = listing.data ? slots(listing.data, fibsem) : [];
  const readings = useQueries({
    queries: cards
      .filter((slot) => slot.spec)
      .map((slot) => ({
        queryKey: ["microscope/devices", slot.spec!.name, "values"],
        queryFn: () => reading<DeviceValues>(`/microscope/devices/${encodeURIComponent(slot.spec!.name)}/values`),
        refetchInterval: READINGS_EVERY_MS,
      })),
  });
  const readOf = new Map(cards.filter((s) => s.spec).map((s, i) => [s.spec!.name, readings[i]]));

  return (
    <div className="mx-auto flex w-full max-w-[1400px] flex-wrap gap-5 p-5">
      <main className="flex min-w-0 flex-[999_1_640px] flex-col gap-5">
        <LastAcquired fibsem={fibsem} />
        {caps.data && !caps.data.routers.microscope && (
          <Card>
            <p className="m-0 text-[13px] text-muted-foreground">
              This host does not serve the microscope's own routes, so its devices are not shown.
            </p>
          </Card>
        )}
        {listing.error && <p className="text-[13px] text-destructive">{(listing.error as Error).message}</p>}
        <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-2">
          {cards.map((slot) => (
            <DeviceCard
              key={slot.title}
              slot={slot}
              read={slot.spec ? readOf.get(slot.spec.name)?.data : undefined}
              error={slot.spec ? readOf.get(slot.spec.name)?.error : undefined}
              fibsem={fibsem}
            />
          ))}
        </div>
      </main>
      <aside className="flex min-w-0 flex-[1_1_300px] flex-col gap-5 lg:max-w-[400px]">
        <ConnectionCard />
        {fibsem && info.data && <MicroscopePanel info={info.data} />}
      </aside>
    </div>
  );
}
