import { useQuery } from "@tanstack/react-query";
import { Check, ListFilter, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ImageInfoBar, ScaleBar } from "@/components/image-info";
import { ImageViewer } from "@/components/images";
import { StageCard } from "@/components/stage-card";
import { Stepper } from "@/components/stepper";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Popup } from "@/components/ui/popup";
import { ApiError, api } from "@/lib/api";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { ReplayImage, ReplayRow, ReplayTimeline, ReplayView, WorkflowStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const KIND: Record<string, string> = {
  task: "Task",
  prompt: "Prompt",
  decision: "Decision",
  image: "Image",
  fluorescence: "FM",
  stage: "Stage",
  milling: "Milling",
  alignment: "Alignment",
  correlation: "Correlation",
  edit: "Edit",
  message: "Message",
};
const ACTOR: Record<string, string> = { operator: "Operator", agent: "Agent", task: "Workflow" };
const NOT_RECORDED = "-";
const SPEEDS = [1, 10, 60, 600];
// Playback waits the real gap between actions over the speed, within these.
const MIN_STEP_MS = 40;
const MAX_STEP_MS = 1500;

const PROPOSED = "#f0a030";
const DECIDED = "#4caf50";
const STAGE = "#4fc3f7";
const STAGE_NOW = "#ffd54f";
const SPOT = "#e05252";

const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const words = (text: string) =>
  text
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/^./, (c) => c.toUpperCase());

interface Filter {
  item: string | null;
  actor: string | null;
  kind: string | null;
}
const NO_FILTER: Filter = { item: null, actor: null, kind: null };

/** A cross at (x, y), *r* each way. */
function Cross({ x, y, r, colour, w }: { x: number; y: number; r: number; colour: string; w: number }) {
  return (
    <g stroke={colour} strokeWidth={w}>
      <line x1={x - r} y1={y} x2={x + r} y2={y} />
      <line x1={x} y1={y - r} x2={x} y2={y + r} />
    </g>
  );
}

/** The milling, spots and a question's values, where they were, over an image *w* by *h*. */
function Overlay({ image, w, h }: { image: ReplayImage; w: number; h: number }) {
  const r = w / 60;
  const line = w / 400;
  const at = (pt: number[]) => [pt[0] * w, pt[1] * h] as const;
  const rect = (a: number[], colour: string, key: string) => (
    <rect
      key={key}
      x={a[0] * w}
      y={a[1] * h}
      width={a[2] * w}
      height={a[3] * h}
      fill="none"
      stroke={colour}
      strokeWidth={line * 1.5}
    />
  );
  return (
    <svg className="absolute inset-0 size-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {image.milling.map((stage) =>
        stage.outlines.map((ring, i) => (
          <polygon
            key={`${stage.name}-${i}`}
            points={ring.map((pt) => at(pt).join(",")).join(" ")}
            fill={stage.current ? STAGE_NOW : STAGE}
            fillOpacity={0.15}
            stroke={stage.current ? STAGE_NOW : STAGE}
            strokeWidth={line}
          />
        )),
      )}
      {image.spots.map((pt, i) => (
        <circle key={`spot-${i}`} cx={pt[0] * w} cy={pt[1] * h} r={r / 3} fill={SPOT} />
      ))}
      {image.areas_proposed.map((a, i) => rect(a, PROPOSED, `ap-${i}`))}
      {image.areas_decided.map((a, i) => rect(a, DECIDED, `ad-${i}`))}
      {image.proposed.map((pt, i) => (
        <Cross key={`p-${i}`} x={at(pt)[0]} y={at(pt)[1]} r={r} colour={PROPOSED} w={line * 1.5} />
      ))}
      {image.decided.map((pt, i) => (
        <Cross key={`d-${i}`} x={at(pt)[0]} y={at(pt)[1]} r={r} colour={DECIDED} w={line * 1.5} />
      ))}
    </svg>
  );
}

/** One image at this moment, as the Microscope tab's Last acquired shows it: fitted
 * in a 3:2 frame with its scalebar and the bar under it, and what was drawn on it. */
function Tile({
  label,
  path,
  image,
  open,
}: {
  label: string;
  path: string | null;
  image?: ReplayImage | null;
  open: (path: string) => void;
}) {
  const read = useFileRead(path, 1024);
  const p = read.data?.preview;
  if (!path)
    return (
      <div className="flex aspect-[3/2] w-full min-w-0 items-center justify-center rounded-sm bg-panel text-[13px] text-muted-foreground">
        No {label} image yet
      </div>
    );
  const ratio = p ? p.width / p.height : 3 / 2;
  return (
    <button
      onClick={() => open(path)}
      className="flex min-w-0 cursor-pointer flex-col overflow-hidden rounded-sm border-0 bg-transparent p-0 text-left text-foreground"
    >
      <div className="relative flex aspect-[3/2] w-full items-center justify-center bg-black [container-type:inline-size]">
        {p ? (
          <div
            className="relative"
            style={{ aspectRatio: `${ratio}`, width: `min(100%, calc((100cqw * 2 / 3) * ${ratio}))` }}
          >
            <img
              src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
              alt={`${label} image`}
              className={cn("absolute inset-0 block h-full w-full", read.isPlaceholderData && "opacity-60")}
            />
            {image && !read.isPlaceholderData && <Overlay image={image} w={p.width} h={p.height} />}
            <ScaleBar info={read.data?.info} />
          </div>
        ) : (
          <span className="text-[12px] text-muted-foreground">{read.error ? read.error.message : "Reading…"}</span>
        )}
        {!!image?.unsaved_since && (
          <span className="absolute top-1.5 left-1.5 bg-black/70 px-1.5 py-0.5 text-[11px] text-white">
            {image.unsaved_since} newer not saved
          </span>
        )}
      </div>
      <ImageInfoBar info={read.data?.info} />
    </button>
  );
}

function Section<K extends keyof Filter>({
  title,
  field,
  choices,
  value,
  set,
}: {
  title: string;
  field: K;
  choices: [Filter[K], string][];
  value: Filter;
  set: (next: Filter) => void;
}) {
  return (
    <div className="flex flex-col py-1">
      <div className="px-2 py-1 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">{title}</div>
      {choices.map(([choice, label]) => (
        <button
          key={String(choice)}
          className="flex cursor-pointer items-center gap-2 rounded border-0 bg-transparent px-2 py-1 text-left text-[13px] text-foreground hover:bg-accent"
          onClick={() => set({ ...value, [field]: choice })}
        >
          <Check size={13} className={cn(value[field] === choice ? "opacity-100" : "opacity-0")} />
          {label}
        </button>
      ))}
    </div>
  );
}

/** The timeline's filters, behind one button as the History tab has them. */
function FilterMenu({
  value,
  set,
  items,
  actors,
  counts,
}: {
  value: Filter;
  set: (f: Filter) => void;
  items: string[];
  actors: Set<string>;
  counts: Record<string, number>;
}) {
  const n = [value.item, value.actor, value.kind].filter((v) => v != null).length;
  return (
    <Popup
      label="Replay filter"
      align="right"
      side="below"
      className="w-56"
      trigger={(open, toggle) => (
        <Button
          variant="ghost"
          onClick={toggle}
          aria-expanded={open}
          aria-label="Filter"
          title="Filter"
          className={cn(n > 0 && "text-link")}
        >
          <ListFilter size={14} />
          {n > 0 && n}
        </Button>
      )}
    >
      {() => (
        <div className="flex flex-col divide-y divide-border">
          <Section
            title="Item"
            field="item"
            value={value}
            set={set}
            choices={[[null, "All items"], ...items.map((i): [string, string] => [i, i])]}
          />
          <Section
            title="Who acted"
            field="actor"
            value={value}
            set={set}
            choices={[
              [null, "Anyone"],
              ...Object.entries(ACTOR).filter(([a]) => actors.has(a)),
              ...(actors.has(NOT_RECORDED) && actors.size > 1
                ? [[NOT_RECORDED, "Not recorded"] as [string, string]]
                : []),
            ]}
          />
          <Section
            title="Kind"
            field="kind"
            value={value}
            set={set}
            choices={[
              [null, "Every kind"],
              ...Object.entries(KIND)
                .filter(([k]) => counts[k])
                .map(([k, label]): [string, string] => [k, `${label} (${counts[k]})`]),
            ]}
          />
          <div className="p-1">
            <Button variant="ghost" className="w-full justify-start" onClick={() => set(NO_FILTER)} disabled={!n}>
              Reset
            </Button>
          </div>
        </div>
      )}
    </Popup>
  );
}

/** The experiment's recorded actions, stepped through with the images, milling
 * and stage at each. */
export function Replay() {
  const status = useQuery({
    queryKey: ["app/workflow/status"],
    queryFn: () => api<WorkflowStatus>("/app/workflow/status"),
  });
  const running = !!status.data?.workflow.running;
  // The whole record each read: read again only while a run adds to it, and
  // once when it ends.
  const timeline = useQuery({
    queryKey: ["app/experiment/replay"],
    queryFn: () => api<ReplayTimeline>("/app/experiment/replay"),
    refetchInterval: running ? 10_000 : false,
  });
  const refetchTimeline = timeline.refetch;
  const wasRunning = useRef(running);
  useEffect(() => {
    if (wasRunning.current && !running) void refetchTimeline();
    wasRunning.current = running;
  }, [running, refetchTimeline]);
  const events = useMemo(() => timeline.data?.events ?? [], [timeline.data]);
  const [filter, setFilter] = useState<Filter>(NO_FILTER);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(60);
  const [viewing, setViewing] = useState<{ paths: string[]; at: number } | null>(null);

  const items = useMemo(() => [...new Set(events.map((e) => e.item).filter((i): i is string => !!i))], [events]);
  const actors = useMemo(() => new Set(events.map((e) => e.actor ?? NOT_RECORDED)), [events]);
  const shown = useMemo(
    () =>
      events.filter(
        (e) =>
          (filter.item == null || e.item === filter.item) &&
          (filter.actor == null || (e.actor ?? NOT_RECORDED) === filter.actor) &&
          (filter.kind == null || e.kind === filter.kind),
      ),
    [events, filter],
  );
  // Where *index* sits in what is shown: itself, or the last shown before it.
  const at = useMemo(() => {
    let found = 0;
    shown.forEach((e, i) => {
      if (e.index <= index) found = i;
    });
    return found;
  }, [shown, index]);
  const event: ReplayRow | undefined = shown[at];
  const item = filter.item;

  const scene = useQuery({
    queryKey: ["app/experiment/replay", event?.index, item],
    queryFn: () =>
      api<ReplayView>(`/app/experiment/replay/${event!.index}${item ? `?item=${encodeURIComponent(item)}` : ""}`),
    enabled: event !== undefined,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });

  // Play waits the real gap to the next action, sped up.
  useEffect(() => {
    if (!playing || !event) return;
    const next = shown[at + 1];
    if (!next) {
      setPlaying(false);
      return;
    }
    const gap = (Date.parse(next.time) - Date.parse(event.time)) / speed;
    const t = setTimeout(() => setIndex(next.index), Math.min(Math.max(gap, MIN_STEP_MS), MAX_STEP_MS));
    return () => clearTimeout(t);
  }, [playing, event, shown, at, speed]);

  // Keep the current row in view.
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => {
    list.current?.querySelector("[aria-current]")?.scrollIntoView({ block: "nearest" });
  }, [event?.index]);

  if (timeline.isPending) return <div className="text-[13px] text-muted-foreground">Reading the record…</div>;
  // A 404 is an experiment with nothing recorded yet: empty, not an error.
  if (timeline.error instanceof ApiError && timeline.error.status === 404)
    return <div className="text-[13px] text-muted-foreground">{timeline.error.message}</div>;
  // A failed re-read keeps the record on screen.
  if (timeline.error && !timeline.data)
    return <div className="text-[13px] text-destructive">{timeline.error.message}</div>;
  if (!events.length) return <div className="text-[13px] text-muted-foreground">Nothing has been recorded yet.</div>;

  const go = (to: number) => {
    setPlaying(false);
    const next = shown[Math.max(0, Math.min(shown.length - 1, to))];
    if (next) setIndex(next.index);
  };
  const view = scene.data ?? null;
  const paths = [view?.sem?.path, view?.fib?.path, view?.fm].filter((p): p is string => !!p);
  const open = (path: string) => setViewing({ paths, at: paths.indexOf(path) });
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
      <Card className="flex min-w-0 flex-col gap-3">
        {event && (
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="text-[14px] font-semibold">{event.summary}</div>
            <div className="text-[12px] text-muted-foreground tabular-nums">
              {[
                clock(event.time),
                event.item,
                [event.task, event.step && words(event.step)].filter(Boolean).join(" · "),
                event.actor ? (ACTOR[event.actor] ?? event.actor) : null,
                event.duration != null ? fmt.seconds(event.duration) : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </div>
          </div>
        )}
        {scene.error && <div className="text-[13px] text-destructive">{scene.error.message}</div>}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Tile label="SEM" path={view?.sem?.path ?? null} image={view?.sem} open={open} />
          <Tile label="FIB" path={view?.fib?.path ?? null} image={view?.fib} open={open} />
          <Tile label="FM" path={view?.fm ?? null} open={open} />
          <StageCard position={view?.stage_position ?? null} stage={view?.stage ?? null} />
        </div>
        <Stepper
          at={at}
          count={shown.length}
          noun="action"
          go={go}
          between={
            <Button
              className="size-11 shrink-0 p-0"
              aria-label={playing ? "Pause" : "Play"}
              onClick={() => {
                if (!playing && at >= shown.length - 1) setIndex(shown[0]?.index ?? 0);
                setPlaying(!playing);
              }}
              disabled={shown.length < 2}
            >
              {playing ? <Pause size={16} className="mx-auto" /> : <Play size={16} className="mx-auto" />}
            </Button>
          }
          after={
            <select
              aria-label="Speed"
              title="Playback speed, as a multiple of real time"
              className="h-11 rounded border border-input bg-background px-2 text-[13px] text-foreground"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {SPEEDS.map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
          }
        />
      </Card>
      <Card className="flex min-w-0 flex-col gap-1.5 p-3">
        <div className="flex items-center gap-2">
          <span className="text-[12px] text-muted-foreground">
            {shown.length === events.length ? `${events.length} actions` : `${shown.length} of ${events.length}`}
          </span>
          <span className="ml-auto" />
          <FilterMenu value={filter} set={setFilter} items={items} actors={actors} counts={timeline.data.counts} />
        </div>
        <div ref={list} className="max-h-[75vh] overflow-y-auto" role="list" aria-label="Recorded actions">
          {shown.map((e) => (
            <button
              key={e.index}
              role="listitem"
              aria-current={e === event || undefined}
              onClick={() => {
                setPlaying(false);
                setIndex(e.index);
              }}
              className={cn(
                "flex w-full cursor-pointer gap-2 border-0 border-t border-border bg-transparent px-1 py-0.5 text-left text-[12px] text-foreground hover:bg-accent",
                e === event && "bg-accent",
              )}
              title={[KIND[e.kind] ?? e.kind, e.item, e.actor ? ACTOR[e.actor] : null].filter(Boolean).join(" · ")}
            >
              <span className="shrink-0 text-[11px] leading-[18px] text-muted-foreground tabular-nums">
                {clock(e.time)}
              </span>
              <span className="min-w-0 truncate leading-[18px]">{e.summary}</span>
            </button>
          ))}
          {!shown.length && <div className="py-3 text-[12px] text-muted-foreground">Nothing matches.</div>}
        </div>
      </Card>
      <ImageViewer paths={viewing?.paths ?? []} at={viewing ? viewing.at : null} onClose={() => setViewing(null)} />
    </div>
  );
}
