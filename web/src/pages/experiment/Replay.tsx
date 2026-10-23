import { useQuery } from "@tanstack/react-query";
import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Stepper } from "@/components/stepper";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { ReplayImage, ReplayRow, ReplayTimeline, ReplayView } from "@/lib/types";
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
const ACTOR: Record<string, string> = { operator: "Operator", agent: "Agent", task: "Task" };
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

const time = (iso: string) =>
  new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const select = "rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground";

/** A cross at (x, y), *r* pixels each way. */
function Cross({ x, y, r, colour, w }: { x: number; y: number; r: number; colour: string; w: number }) {
  return (
    <g stroke={colour} strokeWidth={w}>
      <line x1={x - r} y1={y} x2={x + r} y2={y} />
      <line x1={x} y1={y - r} x2={x} y2={y + r} />
    </g>
  );
}

/** One beam's image at this moment, with the milling, spots and a question's
 * values drawn where they were. */
function SceneImage({ label, image }: { label: string; image: ReplayImage | null }) {
  const read = useFileRead(image?.path ?? null, 1024);
  const p = read.data?.preview;
  const [w, h] = p ? [p.width, p.height] : [1, 1];
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
    <div className="flex min-w-0 flex-col gap-1.5">
      <div className="flex items-baseline gap-2 text-[12px]">
        <span className="font-semibold">{label}</span>
        {image && <span className="text-muted-foreground tabular-nums">{time(image.time)}</span>}
        {!!image?.unsaved_since && (
          <span className="text-muted-foreground">
            {image.unsaved_since} newer {image.unsaved_since === 1 ? "image wasn't" : "images weren't"} saved
          </span>
        )}
      </div>
      <div className="relative w-full overflow-hidden rounded bg-black">
        {p ? (
          <>
            <img
              className={cn("block w-full", read.isPlaceholderData && "opacity-60")}
              alt={image?.path}
              src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
            />
            {image && !read.isPlaceholderData && (
              <svg className="absolute inset-0 size-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
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
            )}
          </>
        ) : (
          <div className="flex aspect-[3/2] w-full items-center justify-center text-[12px] text-muted-foreground">
            {!image ? "No image yet" : read.error ? read.error.message : "Loading…"}
          </div>
        )}
      </div>
    </div>
  );
}

function FmImage({ path }: { path: string }) {
  const read = useFileRead(path, 512);
  const p = read.data?.preview;
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-[12px] font-semibold">FM</span>
      <div className="w-full max-w-[320px] overflow-hidden rounded bg-black">
        {p ? (
          <img className="block w-full" alt={path} src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />
        ) : (
          <div className="flex aspect-square w-full items-center justify-center text-[12px] text-muted-foreground">
            Loading…
          </div>
        )}
      </div>
      <span className="truncate text-[12px] text-muted-foreground">{path.split("/").pop()}</span>
    </div>
  );
}

function stageLine(pos: ReplayView["stage_position"]) {
  if (!pos) return null;
  const n = (k: string) => (typeof pos[k] === "number" ? (pos[k] as number) : null);
  return [
    `x ${fmt.length(n("x"))}`,
    `y ${fmt.length(n("y"))}`,
    `z ${fmt.length(n("z"))}`,
    `r ${fmt.angle(n("r"))}`,
    `t ${fmt.angle(n("t"))}`,
  ].join(" · ");
}

/** The experiment's recorded actions, stepped through with the images, milling
 * and stage at each. */
export function Replay() {
  const timeline = useQuery({
    queryKey: ["app/experiment/replay"],
    queryFn: () => api<ReplayTimeline>("/app/experiment/replay"),
    refetchInterval: 10_000,
  });
  const events = timeline.data?.events ?? [];
  const [item, setItem] = useState("");
  const [actor, setActor] = useState("");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(60);

  const items = useMemo(() => [...new Set(events.map((e) => e.item).filter((i): i is string => !!i))], [events]);
  const actors = useMemo(() => new Set(events.map((e) => e.actor ?? NOT_RECORDED)), [events]);
  const shown = useMemo(
    () =>
      events.filter(
        (e) => (!item || e.item === item) && (!actor || (e.actor ?? NOT_RECORDED) === actor) && !hidden.has(e.kind),
      ),
    [events, item, actor, hidden],
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
  if (timeline.error) return <div className="text-[13px] text-destructive">{timeline.error.message}</div>;
  if (!timeline.data.available)
    return <div className="text-[13px] text-muted-foreground">{timeline.data.error ?? "Nothing to replay."}</div>;
  if (!events.length) return <div className="text-[13px] text-muted-foreground">Nothing has been recorded yet.</div>;

  const go = (to: number) => {
    setPlaying(false);
    const next = shown[Math.max(0, Math.min(shown.length - 1, to))];
    if (next) setIndex(next.index);
  };
  const view = scene.data?.available ? scene.data : null;
  const counts = timeline.data.counts;
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <Card className="flex min-w-0 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <select aria-label="Item" className={select} value={item} onChange={(e) => setItem(e.target.value)}>
            <option value="">All items</option>
            {items.map((i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
          <select aria-label="Who acted" className={select} value={actor} onChange={(e) => setActor(e.target.value)}>
            <option value="">Anyone</option>
            {Object.entries(ACTOR)
              .filter(([a]) => actors.has(a))
              .map(([a, label]) => (
                <option key={a} value={a}>
                  {label}
                </option>
              ))}
            {actors.has(NOT_RECORDED) && actors.size > 1 && <option value={NOT_RECORDED}>Not recorded</option>}
          </select>
          <span className="ml-auto text-muted-foreground">
            {shown.length === events.length ? `${events.length} actions` : `${shown.length} of ${events.length}`}
          </span>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px]">
          {Object.entries(KIND)
            .filter(([k]) => counts[k])
            .map(([k, label]) => (
              <label key={k} className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  className="accent-[var(--primary)]"
                  checked={!hidden.has(k)}
                  onChange={() => {
                    const next = new Set(hidden);
                    if (next.has(k)) next.delete(k);
                    else next.add(k);
                    setHidden(next);
                  }}
                />
                {label} <span className="text-muted-foreground">{counts[k]}</span>
              </label>
            ))}
        </div>
        <div ref={list} className="max-h-[62vh] overflow-y-auto" role="list" aria-label="Recorded actions">
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
                "grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] gap-x-2 border-0 border-t border-border bg-transparent px-1 py-1.5 text-left text-[13px] text-foreground hover:bg-accent",
                e === event && "bg-accent",
              )}
            >
              <span className="whitespace-nowrap text-muted-foreground tabular-nums">{time(e.time)}</span>
              <span className="truncate">{e.summary}</span>
              <span />
              <span className="truncate text-[12px] text-muted-foreground">
                {[KIND[e.kind] ?? e.kind, e.item, e.actor && e.actor !== "task" ? ACTOR[e.actor] : null]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </button>
          ))}
          {!shown.length && <div className="py-3 text-[13px] text-muted-foreground">Nothing matches.</div>}
        </div>
      </Card>
      <Card className="flex min-w-0 flex-col gap-3">
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
              className={cn(select, "h-11")}
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
        {event && (
          <div className="flex flex-col gap-0.5">
            <div className="text-[13px] font-semibold">{event.summary}</div>
            <div className="text-[12px] text-muted-foreground">
              {[
                fmt.when(event.time),
                event.item,
                [event.task, event.step].filter(Boolean).join(" / "),
                event.actor ? `by ${ACTOR[event.actor]?.toLowerCase() ?? event.actor}` : null,
                event.duration != null ? fmt.seconds(event.duration) : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </div>
          </div>
        )}
        {scene.error && <div className="text-[13px] text-destructive">{scene.error.message}</div>}
        <div className="grid gap-3 md:grid-cols-2">
          <SceneImage label="SEM" image={view?.sem ?? null} />
          <SceneImage label="FIB" image={view?.fib ?? null} />
        </div>
        {view?.fm && <FmImage path={view.fm} />}
        {view?.stage_position && (
          <div className="text-[12px] text-muted-foreground tabular-nums">Stage {stageLine(view.stage_position)}</div>
        )}
      </Card>
    </div>
  );
}
