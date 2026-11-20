import { useQueries, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, ShieldAlert, Square, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import type { DeviceListing } from "@/lib/types";
import { cn } from "@/lib/utils";

/** One tab of the column, its *icon* beside its label and alone on the folded
 * rail. *menu* sits in the tab row while the tab is open
 * (the Microscope tab's Customise); *closable* tabs show an x that calls *onClose*. */
export interface ColumnTab {
  id: string;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
  closable?: boolean;
  onClose?: () => void;
  menu?: ReactNode;
  render: () => ReactNode;
}

export const COLUMN_MIN = 340;
export const COLUMN_DEFAULT = 400;
const WIDTH_KEY = "fibsem.agent.column.width";
const FOLDED_KEY = "fibsem.agent.column.folded";
const TAB_KEY = "fibsem.agent.column.tab";

function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function keep(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // A browser that keeps nothing starts from the defaults next time.
  }
}

const widest = () => Math.max(COLUMN_MIN, Math.floor(innerWidth / 2));
const clamp = (width: number) => Math.min(widest(), Math.max(COLUMN_MIN, Math.round(width)));

function savedWidth(): number {
  const n = Number(load(WIDTH_KEY));
  return Number.isFinite(n) && n > 0 ? clamp(n) : COLUMN_DEFAULT;
}

/** The tab shown: the one chosen last in this browser while it is still there,
 * else the first. */
function useActiveTab(tabs: ColumnTab[]): [string | undefined, (id: string) => void] {
  const [chosen, setChosen] = useState(() => load(TAB_KEY));
  const active = tabs.find((t) => t.id === chosen)?.id ?? tabs[0]?.id;
  return [
    active,
    (id: string) => {
      setChosen(id);
      keep(TAB_KEY, id);
    },
  ];
}

/** The agent page's right-hand column: tabs over one body, a row pinned at the
 * bottom in every tab (*footer*, the Stop and safety buttons), and the left edge
 * dragged to resize it, from 340 px to half the window. Folded, it is a rail of
 * the tabs' icons that keeps *rail* (Stop) in reach. Width, fold and tab are kept per browser. */
export function SideColumn({
  tabs,
  footer,
  rail,
  focus,
}: {
  tabs: ColumnTab[];
  footer?: ReactNode;
  rail?: ReactNode;
  /** A tab to show now, as when a card in the chat opens its run. */
  focus?: string | null;
}) {
  const [width, setWidth] = useState(savedWidth);
  const [folded, setFolded] = useState(() => load(FOLDED_KEY) === "1");
  const [active, setActive] = useActiveTab(tabs);
  const drag = useRef<{ x: number; width: number } | null>(null);
  useEffect(() => {
    if (focus && tabs.some((t) => t.id === focus)) {
      setActive(focus);
      if (folded) fold(false);
    }
    // Only a new focus opens it; the tab list changing does not.
  }, [focus]);
  // A window made narrower takes the column with it.
  useEffect(() => {
    const fit = () => setWidth((w) => clamp(w));
    addEventListener("resize", fit);
    return () => removeEventListener("resize", fit);
  }, []);
  const fold = (on: boolean) => {
    setFolded(on);
    keep(FOLDED_KEY, on ? "1" : "0");
  };
  const tab = tabs.find((t) => t.id === active);

  if (folded)
    return (
      <aside
        aria-label="Side column, folded"
        className="hidden w-11 shrink-0 flex-col items-center border-l border-border py-2.5 lg:flex"
      >
        <button
          className="cursor-pointer rounded border-0 bg-transparent p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
          aria-label="Open the side column"
          title="Open the side column"
          onClick={() => fold(false)}
        >
          <ChevronLeft size={15} />
        </button>
        <div className="mt-2 flex flex-col items-center gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              className="relative cursor-pointer rounded border-0 bg-transparent p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
              aria-label={t.label}
              title={t.label}
              onClick={() => {
                setActive(t.id);
                fold(false);
              }}
            >
              {t.icon ?? <span className="text-[12px]">{t.label[0]}</span>}
              {t.badge && <span className="absolute -top-0.5 -right-0.5">{t.badge}</span>}
            </button>
          ))}
        </div>
        <div className="mt-auto flex flex-col items-center gap-2">{rail}</div>
      </aside>
    );

  return (
    <aside
      aria-label="Side column"
      className="relative hidden shrink-0 flex-col border-l border-border lg:flex"
      style={{ width }}
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the side column"
        aria-valuemin={COLUMN_MIN}
        aria-valuenow={width}
        title="Drag to resize; double-click for the default width"
        className="absolute inset-y-0 -left-[3px] z-10 w-[6px] cursor-col-resize hover:bg-primary/40"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, width };
        }}
        onPointerMove={(e) => {
          if (drag.current) setWidth(clamp(drag.current.width + drag.current.x - e.clientX));
        }}
        onPointerUp={() => {
          drag.current = null;
          keep(WIDTH_KEY, String(width));
        }}
        onDoubleClick={() => {
          setWidth(clamp(COLUMN_DEFAULT));
          keep(WIDTH_KEY, String(COLUMN_DEFAULT));
        }}
      />
      <div className="flex items-end gap-1 border-b border-border px-4 pt-2.5">
        <div role="tablist" className="flex min-w-0 flex-1 gap-4 overflow-x-auto">
          {tabs.map((t) => (
            <div
              key={t.id}
              className={cn(
                "-mb-px flex shrink-0 items-center gap-1.5 border-b-2 pb-2 text-[13px]",
                t.id === active ? "border-foreground text-foreground" : "border-transparent text-muted-foreground",
              )}
            >
              <button
                role="tab"
                aria-selected={t.id === active}
                className="flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-inherit hover:text-foreground"
                onClick={() => setActive(t.id)}
              >
                {t.icon}
                {t.label}
                {t.badge}
              </button>
              {t.closable && (
                <button
                  className="cursor-pointer border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
                  aria-label={`Close ${t.label}`}
                  onClick={t.onClose}
                >
                  <X size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="flex shrink-0 items-center gap-0.5 pb-1.5">
          {tab?.menu}
          <button
            className="cursor-pointer rounded border-0 bg-transparent p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Fold the side column"
            title="Fold the side column"
            onClick={() => fold(true)}
          >
            <ChevronRight size={15} />
          </button>
        </div>
      </div>
      <div role="tabpanel" className="min-h-0 flex-1 overflow-y-auto px-6 pt-4 pb-4">
        {tab?.render()}
      </div>
      {footer && <div className="border-t border-border px-4 py-3">{footer}</div>}
    </aside>
  );
}

/** A command the instrument declares makes it safe, on each device that has it. */
export interface SafetyAction {
  label: string;
  calls: { path: string; device: string }[];
}

const ROUTES = ["/microscope/devices", "/microscope/services"] as const;

/** The safety actions the served devices and services declare
 * (`command(safety=...)`), one per button label: blanking two beams is one
 * "Blank beams" button. */
export function useSafetyActions(enabled: boolean): SafetyAction[] {
  const listings = useQueries({
    queries: ROUTES.map((route) => ({
      queryKey: [route],
      queryFn: () => api<DeviceListing>(route),
      enabled,
      retry: false,
    })),
  });
  const found = new Map<string, SafetyAction>();
  listings.forEach((listing, i) => {
    for (const [name, spec] of Object.entries(listing.data ?? {})) {
      for (const [command, info] of Object.entries(spec.commands ?? {})) {
        if (!info.safety || info.available === false) continue;
        const action = found.get(info.safety) ?? { label: info.safety, calls: [] };
        action.calls.push({
          path: `${ROUTES[i]}/${encodeURIComponent(name)}/commands/${encodeURIComponent(command)}`,
          device: name.replace(/_/g, " "),
        });
        found.set(info.safety, action);
      }
    }
  });
  return [...found.values()].map((a) =>
    a.calls.length > 1 ? { ...a, label: a.label.replace(/(\w+)$/, "$1s") } : a,
  );
}

const STOP_TIP =
  "Stop the agent: it drops to Watch and stops what it is doing on the microscope. A running workflow keeps its own Stop.";

function useRun() {
  const client = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (paths: string[]) => {
    setBusy(true);
    setError(null);
    const failed: string[] = [];
    // Each device on its own: one that refuses doesn't keep the others from it.
    await Promise.all(
      paths.map((path) =>
        api<unknown>(path, { method: "POST", body: JSON.stringify(path.endsWith("/stop") ? {} : { kwargs: {} }) }).catch(
          (e: unknown) => failed.push(e instanceof Error ? e.message : String(e)),
        ),
      ),
    );
    setBusy(false);
    if (failed.length) setError(failed.join(" "));
    await client.invalidateQueries();
  };
  return { run, busy, error };
}

/** A safety action's button: it asks first, then runs on every device that has it. */
function SafetyButton({ action, compact }: { action: SafetyAction; compact?: boolean }) {
  const { run, busy, error } = useRun();
  return (
    <Popup
      label={action.label}
      align="right"
      className="w-64 p-3"
      trigger={(open, toggle) =>
        compact ? (
          <button
            className="cursor-pointer rounded border border-input bg-transparent p-1.5 text-foreground hover:bg-accent"
            aria-label={action.label}
            title={error ?? action.label}
            aria-expanded={open}
            onClick={toggle}
          >
            <ShieldAlert size={15} />
          </button>
        ) : (
          <Button
            className={cn("min-w-0 flex-1 truncate", error && "border-destructive")}
            aria-expanded={open}
            disabled={busy}
            title={error ?? undefined}
            onClick={toggle}
          >
            {action.label}
          </Button>
        )
      }
    >
      {(close) => (
        <div className="flex flex-col gap-2.5">
          <div className="font-semibold">{action.label}?</div>
          <div className="text-[13px] text-muted-foreground">
            Runs now on {action.calls.map((c) => c.device).join(" and ")}, whatever is running.
          </div>
          {error && <div className="text-[13px] text-destructive">{error}</div>}
          <div className="flex justify-end gap-2">
            <Button onClick={close}>Cancel</Button>
            <Button
              variant="primary"
              disabled={busy}
              onClick={() => {
                close();
                void run(action.calls.map((c) => c.path));
              }}
            >
              {action.label}
            </Button>
          </div>
        </div>
      )}
    </Popup>
  );
}

/** Stop, one click, and the instrument's safety actions, which ask first.
 * *compact* draws them as the folded rail's icons. */
export function SafetyRow({ actions, compact }: { actions: SafetyAction[]; compact?: boolean }) {
  const { run, busy, error } = useRun();
  const stop = () => void run(["/app/agent/stop"]);
  if (compact)
    return (
      <>
        {actions.map((a) => (
          <SafetyButton key={a.label} action={a} compact />
        ))}
        <button
          className="cursor-pointer rounded border border-input bg-transparent p-1.5 text-foreground hover:bg-accent"
          aria-label="Stop"
          title={error ?? STOP_TIP}
          disabled={busy}
          onClick={stop}
        >
          <Square size={14} fill="currentColor" />
        </button>
      </>
    );
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex gap-2">
        <Button className="flex items-center gap-1.5" title={STOP_TIP} disabled={busy} onClick={stop}>
          <Square size={12} fill="currentColor" />
          Stop
        </Button>
        {actions.map((a) => (
          <SafetyButton key={a.label} action={a} />
        ))}
      </div>
      {error && <div className="text-[12px] text-destructive">{error}</div>}
    </div>
  );
}
