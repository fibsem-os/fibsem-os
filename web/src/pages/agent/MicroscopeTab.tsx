import { useQueries, useQuery } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";
import { useState, type ReactNode } from "react";
import { ImageInfoBar } from "@/components/image-info";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Popup } from "@/components/ui/popup";
import { api, reading } from "@/lib/api";
import * as fmt from "@/lib/format";
import type { DeviceListing, DeviceSpec, DeviceValues, DisplayImages, ImagePayload, WorkflowStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { IMAGE_LABEL, Shown } from "../Microscope";

// As the Microscope page reads them: the stage and a beam's current move without events.
const READINGS_EVERY_MS = 5000;
const IMAGES_EVERY_MS = 10000;
// Read more often when the images are the main view.
const MAIN_IMAGES_EVERY_MS = 3000;
// The height of an image's label in the main view, with the gap under it.
const LABEL_PX = 22;
const SHOW_KEY = "fibsem.agent.microscope.show";

type Values = Record<string, unknown>;
const num = (value: unknown) => (typeof value === "number" ? value : null);
const words = (text: string) => text.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

/** What the tab shows, chosen in Customise and kept per browser. An image's key
 * is its view ("sem", "fib", "fm") or, on another instrument, its device. */
type ShowChoices = Record<string, boolean>;

function loadShown(): ShowChoices {
  try {
    const kept = JSON.parse(localStorage.getItem(SHOW_KEY) ?? "{}");
    return kept && typeof kept === "object" ? kept : {};
  } catch {
    return {};
  }
}

function keepShown(shown: ShowChoices) {
  try {
    localStorage.setItem(SHOW_KEY, JSON.stringify(shown));
  } catch {
    // A browser that keeps nothing shows the defaults next time.
  }
}

/** The tab's choices, kept per browser; anything not chosen shows. */
export function useMicroscopeShown(): [ShowChoices, (next: ShowChoices) => void] {
  const [shown, setShown] = useState(loadShown);
  return [
    shown,
    (next) => {
      setShown(next);
      keepShown(next);
    },
  ];
}

const on = (shown: ShowChoices, key: string) => shown[key] ?? true;

function clock(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function Section({ title, children, className }: { title: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("border-b border-border/60 pb-3.5 last:border-b-0 last:pb-0", className)}>
      <h4 className="mb-1.5 text-[12px] font-semibold text-muted-foreground">{title}</h4>
      {children}
    </section>
  );
}

function useImages(every = IMAGES_EVERY_MS) {
  return useQuery({
    queryKey: ["app/experiment/images"],
    queryFn: () => api<DisplayImages>("/app/experiment/images"),
    refetchInterval: every,
  });
}

/** The latest image of each view, by its key and label. */
function imagePairs(images: DisplayImages | undefined, fibsem: boolean): [string, string, ImagePayload | null][] {
  if (!fibsem) return Object.entries(images?.devices ?? {}).map(([key, image]) => [key, words(key), image]);
  const keys = images?.fm_available ? (["sem", "fib", "fm"] as const) : (["sem", "fib"] as const);
  return keys.map((key) => [key, IMAGE_LABEL[key].replace(/ Image$/, ""), images?.[key] ?? null]);
}

function LastImages({ fibsem, shown }: { fibsem: boolean; shown: ShowChoices }) {
  const images = useImages();
  const [open, setOpen] = useState<string | null>(null);
  const pairs = imagePairs(images.data, fibsem).filter(([key]) => on(shown, key));
  if (!pairs.length) return null;
  const big = pairs.find(([key]) => key === open);
  return (
    <Section title="Last images">
      <div className="grid grid-cols-2 gap-2.5">
        {pairs.map(([key, label, image]) =>
          image ? (
            <button
              key={key}
              className="flex min-w-0 cursor-pointer flex-col gap-1 border-0 bg-transparent p-0 text-left text-foreground"
              title={`Open the last ${label} image`}
              onClick={() => setOpen(key)}
            >
              <div className="flex aspect-[3/2] w-full items-center justify-center overflow-hidden rounded-sm bg-black [container-type:inline-size]">
                <Shown image={image} alt={`Last ${label}`} height="(100cqw * 2 / 3)" />
              </div>
              <span className="flex gap-1 text-[12px]">
                <b className="font-semibold">{label}</b>
                {image.hfw != null && <span className="text-muted-foreground">· {fmt.length(image.hfw)}</span>}
                <span className="ml-auto text-muted-foreground">{clock(image.acquired_at)}</span>
              </span>
            </button>
          ) : (
            <div key={key} className="flex flex-col gap-1">
              <div className="flex aspect-[3/2] w-full items-center justify-center rounded-sm bg-panel text-[12px] text-muted-foreground">
                None yet
              </div>
              <span className="text-[12px] font-semibold">{label}</span>
            </div>
          ),
        )}
      </div>
      <ImageDialog image={big?.[2] ?? null} label={big?.[1] ?? ""} onClose={() => setOpen(null)} />
    </Section>
  );
}

/** An image opened large, with its metadata under it. */
function ImageDialog({ image, label, onClose }: { image: ImagePayload | null; label: string; onClose: () => void }) {
  return (
    <Dialog open={!!image} onOpenChange={(o) => !o && onClose()}>
      {image && (
        <DialogContent title={label} className="w-[min(1100px,calc(100vw-32px))]">
          <div className="flex flex-col p-3.5">
            <div className="flex items-center justify-center bg-black">
              <Shown image={image} alt={label} height="70vh" />
            </div>
            <ImageInfoBar info={image.info} />
          </div>
        </DialogContent>
      )}
    </Dialog>
  );
}

/** What runs now: a workflow's task and lamella, else a running step's label, with its progress. */
function useRunning() {
  const status = useQuery({
    queryKey: ["app/workflow/status"],
    queryFn: () => api<WorkflowStatus>("/app/workflow/status"),
  });
  const workflow = status.data?.workflow;
  const progress = status.data?.progress;
  const task = workflow?.running
    ? [workflow.current_task ?? "A task", workflow.current_item].filter(Boolean).join(" · ")
    : progress && !progress.outcome
      ? progress.label
      : null;
  return { task, progress };
}

function ProgressBar({ fraction }: { fraction: number }) {
  return (
    <div className="mt-1.5 h-[3px] rounded-sm bg-accent">
      <div
        className="h-full rounded-sm bg-[var(--fibsem-primary-color)]"
        style={{ width: `${Math.min(100, 100 * fraction)}%` }}
      />
    </div>
  );
}

function Running() {
  const { task, progress } = useRunning();
  return (
    <Section title="Running">
      {task ? (
        <>
          <p className="m-0 break-words">{task}</p>
          {progress?.fraction != null && <ProgressBar fraction={progress.fraction} />}
          {progress && (progress.step || progress.numbers) && (
            <p className="m-0 mt-1 flex gap-2 text-muted-foreground">
              <span className="min-w-0 break-words">{progress.step}</span>
              <span className="ml-auto shrink-0">{progress.numbers}</span>
            </p>
          )}
        </>
      ) : (
        <p className="m-0 text-muted-foreground">Nothing running</p>
      )}
    </Section>
  );
}

function beamLine(v: Values): string {
  const state = v.on === false ? "Off" : v.blanked ? "Blanked" : "On";
  return [state, fmt.voltage(num(v.voltage)), fmt.current(num(v.current))].filter((p) => p && p !== "–").join(" · ");
}

function beamName(spec: DeviceSpec): string {
  if (/electron/i.test(spec.name)) return "Electron";
  if (/ion/i.test(spec.name)) return "Ion";
  return words(spec.name);
}

function stageRows(v: Values): [string, string][] {
  const p = (v.position ?? {}) as Values;
  return [
    ["X, Y", `${fmt.length(num(p.x))}, ${fmt.length(num(p.y))}`],
    ["Z", fmt.length(num(p.z))],
    ["Tilt · Rotation", `${fmt.angle(num(p.t))} · ${fmt.angle(num(p.r))}`],
  ];
}

function useDevices(enabled: boolean) {
  const listing = useQuery({
    queryKey: ["/microscope/devices"],
    queryFn: () => api<DeviceListing>("/microscope/devices"),
    enabled,
    retry: false,
  });
  const specs = Object.values(listing.data ?? {}).filter((s) => s.type === "beam" || s.type === "stage");
  const reads = useQueries({
    queries: specs.map((spec) => ({
      queryKey: ["microscope/devices", spec.name, "values"],
      queryFn: () => reading<DeviceValues>(`/microscope/devices/${encodeURIComponent(spec.name)}/values`),
      refetchInterval: READINGS_EVERY_MS,
    })),
  });
  return specs.map((spec, i) => ({ spec, values: reads[i].data?.value?.values as Values | undefined }));
}

/** What the instrument is doing, beside the chat: its last images, what runs,
 * its beams and its stage. A click on an image opens it large. */
export function MicroscopeTab({ fibsem, shown, onOpenPage }: { fibsem: boolean; shown: ShowChoices; onOpenPage: () => void }) {
  const devices = useDevices(on(shown, "beams") || on(shown, "stage"));
  const beams = devices.filter((d) => d.spec.type === "beam");
  const stage = devices.find((d) => d.spec.type === "stage");
  return (
    <div className="flex flex-col gap-3.5 text-[13px] leading-[1.45]">
      <LastImages fibsem={fibsem} shown={shown} />
      {on(shown, "running") && <Running />}
      {on(shown, "beams") && beams.length > 0 && (
        <Section title="Beams">
          <Facts rows={beams.map((b) => [beamName(b.spec), b.values ? beamLine(b.values) : "Reading…"])} />
        </Section>
      )}
      {on(shown, "stage") && stage && (
        <Section title="Stage">
          {stage.values ? <Facts rows={stageRows(stage.values)} /> : <p className="m-0 text-muted-foreground">Reading…</p>}
        </Section>
      )}
      <button
        className="cursor-pointer self-start border-0 bg-transparent p-0 text-[13px] text-link underline underline-offset-2"
        onClick={onOpenPage}
      >
        Open the Microscope page
      </button>
    </div>
  );
}

/** The microscope as the agent page's main view: *live*, the page's line for
 * what runs now (the chat's, with its progress and time), across the top, the last image of each view in a two-column grid (SEM and FIB,
 * then FM), each at its own shape and as large as the room allows, and
 * the beams and stage under them. *switcher* and *menu* sit at the top right. */
export function MicroscopeMain({
  fibsem,
  shown,
  live,
  menu,
  switcher,
  onOpenPage,
}: {
  fibsem: boolean;
  shown: ShowChoices;
  live?: ReactNode;
  menu?: ReactNode;
  switcher?: ReactNode;
  onOpenPage: () => void;
}) {
  const images = useImages(MAIN_IMAGES_EVERY_MS);
  const [open, setOpen] = useState<string | null>(null);
  const devices = useDevices(on(shown, "beams") || on(shown, "stage"));
  const beams = on(shown, "beams") ? devices.filter((d) => d.spec.type === "beam") : [];
  const stage = on(shown, "stage") ? devices.find((d) => d.spec.type === "stage") : undefined;
  const pairs = imagePairs(images.data, fibsem).filter(([key]) => on(shown, key));
  const big = pairs.find(([key]) => key === open);
  return (
    <div className="flex h-full min-w-0 flex-1 flex-col gap-3 px-4 pb-4 text-[13px] leading-[1.45] md:px-5">
      <div className="flex min-h-[30px] items-start gap-3 pt-2.5">
        <div className="min-w-0 flex-1 pt-1">{on(shown, "running") && live}</div>
        <span className="flex shrink-0 items-center gap-2">
          {menu}
          {switcher}
        </span>
      </div>
      {pairs.length > 0 ? (
        <div
          className="grid min-h-0 flex-1 grid-cols-2 gap-3"
          style={{ gridTemplateRows: `repeat(${Math.ceil(pairs.length / 2)}, minmax(0, 1fr))` }}
        >
          {pairs.map(([key, label, image]) => {
            // Each tile is its image's shape, as large as its cell allows, its label above it.
            const ratio = image ? image.width / image.height : 3 / 2;
            const room = `(100cqh - ${LABEL_PX}px)`;
            return (
              <div key={key} className="min-h-0 min-w-0 [container-type:size]">
                <div className="flex flex-col gap-1" style={{ width: `min(100cqw, calc(${room} * ${ratio}))` }}>
                  <span className="flex gap-1 text-[12px]" style={{ height: LABEL_PX - 4 }}>
                    <b className="font-semibold">{label}</b>
                    {image?.hfw != null && <span className="text-muted-foreground">· {fmt.length(image.hfw)}</span>}
                    <span className="ml-auto text-muted-foreground">{clock(image?.acquired_at)}</span>
                  </span>
                  {image ? (
                    <button
                      className="block cursor-pointer overflow-hidden rounded-sm border-0 bg-black p-0"
                      title={`Open the last ${label} image`}
                      onClick={() => setOpen(key)}
                    >
                      <Shown image={image} alt={`Last ${label}`} height={room} />
                    </button>
                  ) : (
                    <div
                      className="flex items-center justify-center rounded-sm bg-panel text-muted-foreground"
                      style={{ aspectRatio: `${ratio}` }}
                    >
                      None yet
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex flex-1 items-center justify-center text-muted-foreground">No images chosen to show</div>
      )}
      <ImageDialog image={big?.[2] ?? null} label={big?.[1] ?? ""} onClose={() => setOpen(null)} />
      {(beams.length > 0 || stage) && (
        <div className="grid shrink-0 grid-cols-1 gap-x-6 gap-y-3 border-t border-border pt-3 md:grid-cols-2">
          {beams.length > 0 && (
            <Section title="Beams" className="border-b-0 pb-0">
              <Facts rows={beams.map((b) => [beamName(b.spec), b.values ? beamLine(b.values) : "Reading…"])} />
            </Section>
          )}
          {stage && (
            <Section title="Stage" className="border-b-0 pb-0">
              {stage.values ? (
                <Facts rows={stageRows(stage.values)} />
              ) : (
                <p className="m-0 text-muted-foreground">Reading…</p>
              )}
            </Section>
          )}
        </div>
      )}
      <button
        className="shrink-0 cursor-pointer self-start border-0 bg-transparent p-0 text-[13px] text-link underline underline-offset-2"
        onClick={onOpenPage}
      >
        Open the Microscope page
      </button>
    </div>
  );
}

/** The Microscope tab's Customise menu: which images and sections show. Stop
 * and the safety buttons always do. */
export function MicroscopeCustomise({
  fibsem,
  shown,
  onChange,
}: {
  fibsem: boolean;
  shown: ShowChoices;
  onChange: (next: ShowChoices) => void;
}) {
  const images = useImages();
  const choices: [string, string][] = [
    ...imagePairs(images.data, fibsem).map(([key, label]): [string, string] => [key, `${label} image`]),
    ["running", "What's running"],
    ["beams", "Beams"],
    ["stage", "Stage"],
  ];
  return (
    <Popup
      label="Customise"
      align="right"
      side="below"
      className="top-[calc(100%+4px)] bottom-auto w-64 p-3"
      trigger={(open, toggle) => (
        <button
          className={cn(
            "cursor-pointer rounded border-0 bg-transparent p-1 text-muted-foreground hover:bg-accent hover:text-foreground",
            open && "bg-accent text-foreground",
          )}
          aria-label="Customise"
          title="Customise what shows"
          aria-expanded={open}
          onClick={toggle}
        >
          <SlidersHorizontal size={15} />
        </button>
      )}
    >
      {(close) => (
        <div className="flex flex-col gap-1.5 text-[13px]">
          <div className="text-[12px] font-semibold text-muted-foreground">Always shown</div>
          <label className="flex items-center gap-2 text-muted-foreground">
            <input type="checkbox" checked disabled />
            Stop and safety buttons
          </label>
          <div className="mt-1.5 text-[12px] font-semibold text-muted-foreground">Show</div>
          {choices.map(([key, label]) => (
            <label key={key} className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={on(shown, key)}
                onChange={(e) => onChange({ ...shown, [key]: e.target.checked })}
              />
              {label}
            </label>
          ))}
          <div className="mt-2 flex items-center justify-between">
            <button
              className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted-foreground hover:text-foreground"
              onClick={() => onChange({})}
            >
              Reset to defaults
            </button>
            <Button onClick={close}>Done</Button>
          </div>
        </div>
      )}
    </Popup>
  );
}
