import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { api, imageUrl } from "@/lib/api";
import type { ItemStack, StackSlice } from "@/lib/types";
import { cn } from "@/lib/utils";
import { clock } from "../lamellae/History";

// A slice loads small while it is scrubbed past, and full width once it rests.
const SMALL = 320;
const FULL = 1600;
const AHEAD = 3;
const FRAME_MS = 250;

const itemPath = (name: string) => `/app/experiment/items/${encodeURIComponent(name)}`;

/** The item's images of one role, as a z-stack. */
export function useStack(name: string, role: string) {
  return useQuery({
    queryKey: ["app/experiment/items", name, "stack", role],
    queryFn: () => api<ItemStack>(`${itemPath(name)}/stack?role=${role}`),
  });
}

function sliceImage(name: string, filename: string, width: number) {
  return {
    queryKey: ["slice image", name, filename, width],
    queryFn: () => imageUrl(`${itemPath(name)}/outputs/${encodeURIComponent(filename)}?width=${width}`),
    staleTime: Infinity,
    gcTime: 5 * 60_000,
  };
}

/** A slice's image: the full one once loaded, else the small one, else the last one shown. */
function useSliceSrc(name: string, slice: StackSlice | undefined, resting: boolean) {
  const small = useQuery({ ...sliceImage(name, slice?.filename ?? "", SMALL), enabled: !!slice });
  const full = useQuery({ ...sliceImage(name, slice?.filename ?? "", FULL), enabled: !!slice && resting });
  const src = full.data ?? small.data;
  const last = useRef<string | undefined>(undefined);
  if (src) last.current = src;
  return slice ? (src ?? last.current) : undefined;
}

/** The slice of *stack* cut at *iteration*, or the nearest before it; the newest for null. */
function indexOf(stack: StackSlice[], iteration: number | null) {
  if (iteration === null) return stack.length - 1;
  let at = 0;
  stack.forEach((s, i) => {
    if (s.iteration <= iteration) at = i;
  });
  return at;
}

function readout(slice: StackSlice, planned: number | undefined) {
  const parts = [`Slice ${slice.iteration + 1}${planned ? ` of ${planned}` : ""}`];
  if (slice.z != null) parts.push(`z ${(slice.z * 1e6).toFixed(1)} µm`);
  if (slice.acquired_at) parts.push(clock(Date.parse(slice.acquired_at) / 1000));
  return parts.join(" · ");
}

function Thumb({ name, slice, on, pick }: { name: string; slice: StackSlice; on: boolean; pick: () => void }) {
  const src = useQuery(sliceImage(name, slice.filename, SMALL)).data;
  return (
    <button
      onClick={pick}
      aria-label={`Slice ${slice.iteration + 1}`}
      aria-current={on || undefined}
      className={cn(
        "relative h-16 w-24 shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black p-0",
        on ? "border-primary" : "border-transparent opacity-70 hover:opacity-100",
      )}
    >
      {src && <img className="block size-full object-cover" alt="" src={src} />}
      <span className="absolute bottom-0.5 left-1 text-[11px] text-white [text-shadow:0_0_3px_#000]">
        {slice.iteration + 1}
      </span>
    </button>
  );
}

/** A position's slices viewed as one stack: step, play, or follow the newest while it runs. */
export function Scrubber({ name, planned, running }: { name: string; planned?: number; running: boolean }) {
  const sem = useStack(name, "slice");
  const fib = useStack(name, "slice_fib");
  const hasFib = (fib.data?.slices.length ?? 0) > 0;
  const [beam, setBeam] = useState<"sem" | "fib">("sem");
  const stack = (beam === "fib" && hasFib ? fib : sem).data?.slices ?? [];
  // The iteration on screen; null follows the newest slice.
  const [iteration, setIteration] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [resting, setResting] = useState(true);
  const at = indexOf(stack, iteration);
  const slice = stack[at];
  const src = useSliceSrc(name, slice, resting && !playing);

  const step = (to: number) => {
    const next = stack[Math.max(0, Math.min(stack.length - 1, to))];
    if (next) setIteration(next.iteration);
  };

  // Full width only once the slice has stayed put a moment.
  useEffect(() => {
    setResting(false);
    const t = setTimeout(() => setResting(true), 300);
    return () => clearTimeout(t);
  }, [slice?.filename]);

  // Fetch the neighbours ahead, so scrubbing doesn't wait.
  const client = useQueryClient();
  useEffect(() => {
    for (let i = at - AHEAD; i <= at + AHEAD; i++) {
      const s = stack[i];
      if (s) void client.prefetchQuery(sliceImage(name, s.filename, SMALL));
    }
  }, [client, name, at, stack]);

  // Play runs through and round again.
  const latest = useRef({ at, stack });
  latest.current = { at, stack };
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      const { at, stack } = latest.current;
      const next = stack[at + 1 < stack.length ? at + 1 : 0];
      if (next) setIteration(next.iteration);
    }, FRAME_MS);
    return () => clearInterval(t);
  }, [playing]);

  // The wheel steps through; not passive, so the page doesn't scroll as well.
  const image = useRef<HTMLDivElement>(null);
  const wheel = useRef(step);
  wheel.current = (d: number) => step(at + d);
  useEffect(() => {
    const el = image.current;
    if (!el) return;
    const on = (e: WheelEvent) => {
      e.preventDefault();
      wheel.current(e.deltaY > 0 ? 1 : -1);
    };
    el.addEventListener("wheel", on, { passive: false });
    return () => el.removeEventListener("wheel", on);
  }, [slice !== undefined]);

  if (sem.isPending) return <div className="text-[13px] text-muted-foreground">Reading the slices…</div>;
  if (sem.error) return <div className="text-[13px] text-destructive">{sem.error.message}</div>;
  if (!slice) return <div className="text-[13px] text-muted-foreground">No slices yet.</div>;

  const user = (to: number) => {
    setPlaying(false);
    step(to);
  };
  const following = iteration === null;
  return (
    <div
      className="flex min-w-0 flex-col gap-3 outline-none"
      tabIndex={0}
      aria-label="Slices"
      onKeyDown={(e) => {
        if (e.target instanceof HTMLInputElement) return;
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") user(at - 1);
        else if (e.key === "ArrowRight" || e.key === "ArrowDown") user(at + 1);
        else return;
        e.preventDefault();
      }}
    >
      <div className="flex flex-wrap items-center gap-2">
        {hasFib && (
          <div role="tablist" aria-label="Beam" className="flex gap-1">
            {(
              [
                ["sem", "SEM Image"],
                ["fib", "FIB Image"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                role="tab"
                aria-selected={beam === key}
                onClick={() => setBeam(key)}
                className={cn(
                  "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[12px]",
                  beam === key
                    ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        <span className="ml-auto text-[13px] tabular-nums">{readout(slice, planned)}</span>
      </div>
      <div ref={image} className="w-full overflow-hidden rounded bg-black">
        {src ? (
          <img className="block max-h-[58vh] w-full object-contain" alt={slice.filename} src={src} />
        ) : (
          <div className="flex aspect-[3/2] w-full items-center justify-center text-[12px] text-muted-foreground">
            Loading slice…
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Button
          className="size-11 shrink-0 p-0"
          aria-label="Previous slice"
          onClick={() => user(at - 1)}
          disabled={at === 0}
        >
          <ChevronLeft size={16} className="mx-auto" />
        </Button>
        <Button
          className="size-11 shrink-0 p-0"
          aria-label={playing ? "Pause" : "Play"}
          onClick={() => setPlaying(!playing)}
          disabled={stack.length < 2}
        >
          {playing ? <Pause size={16} className="mx-auto" /> : <Play size={16} className="mx-auto" />}
        </Button>
        <Button
          className="size-11 shrink-0 p-0"
          aria-label="Next slice"
          onClick={() => user(at + 1)}
          disabled={at === stack.length - 1}
        >
          <ChevronRight size={16} className="mx-auto" />
        </Button>
        <input
          type="range"
          aria-label="Slice"
          className="mx-1.5 h-11 min-w-20 flex-1 accent-[var(--primary)]"
          min={0}
          max={stack.length - 1}
          value={at}
          onChange={(e) => user(Number(e.target.value))}
        />
        {running && (
          <Button
            className={cn("h-11 shrink-0 px-3.5", following && "border-primary text-foreground")}
            aria-pressed={following}
            onClick={() => {
              setPlaying(false);
              setIteration(following ? slice.iteration : null);
            }}
          >
            Follow
          </Button>
        )}
      </div>
      <div className="flex gap-1.5 overflow-x-auto">
        {stack.slice(Math.max(0, at - AHEAD), at + AHEAD + 1).map((s) => (
          <Thumb key={s.filename} name={name} slice={s} on={s === slice} pick={() => user(stack.indexOf(s))} />
        ))}
      </div>
    </div>
  );
}
