import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Info, Lock } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { HoverCard } from "@/components/ui/hover-card";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";

export interface HubSession {
  session_id: string;
  state: string;
  started?: string;
  playbook?: string;
  goal?: string;
  level?: string;
  who?: string;
  budget_usd?: number;
}

export interface Instrument {
  name: string;
  url: string | null;
  ssh: string | null;
  tunnel: { open: boolean; gateway_port: number; app_port: number } | null;
  paired: boolean;
  reachable: boolean | null;
  since: number | null;
  checked: number | null;
  why: string | null;
  gateway: {
    mode?: string;
    version?: string;
    app?: string;
    experiment?: string;
    switching?: boolean;
    since?: number;
    last_experiment?: string;
    /** The apps a hub can start there. */
    apps?: { key: string; name: string }[];
  } | null;
  /** The agent its own PC runs. */
  agent: {
    session_id?: string;
    state?: string;
    playbook?: string;
    goal?: string;
    level?: string;
    started?: string;
  } | null;
  /** The agent this hub runs for it. */
  session: HubSession | null;
  microscope: {
    connected: boolean;
    connecting: boolean;
    configuration?: string;
    model?: string;
  } | null;
  level_cap: string | null;
}

export interface HubState {
  hub: boolean;
  instruments: Instrument[];
}

interface Playbook {
  name: string;
  error?: string | null;
}

export const FIELD =
  "h-[30px] rounded border border-input bg-background px-2.5 text-[13px] text-foreground placeholder:text-muted-foreground/60 disabled:opacity-60";
export const LABEL = "flex flex-col gap-1.5 text-xs text-muted-foreground";
/** Stop, End and Remove: red words, not red buttons. */
export const RED =
  "cursor-pointer border-0 bg-transparent px-1.5 text-[12.5px] text-destructive hover:underline disabled:cursor-default disabled:opacity-50 disabled:no-underline";
export const LEVELS = ["watch", "assist", "drive"];
export const title = (text: string) =>
  text.charAt(0).toUpperCase() + text.slice(1);
export const at = (name: string) =>
  `/hub/instruments/${encodeURIComponent(name)}`;

export function useHub() {
  return useQuery({
    queryKey: ["hub"],
    queryFn: () => api<HubState>("/hub"),
    refetchInterval: 3000,
  });
}

/** An action on the hub: its answer is the list, or the list is read again. */
export function useHubAction<T = void>(
  act: (value: T) => Promise<unknown>,
  onDone?: () => void,
) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: act,
    onSuccess: (answer) => {
      if (answer && typeof answer === "object" && "instruments" in answer)
        client.setQueryData(["hub"], answer);
      else client.invalidateQueries({ queryKey: ["hub"] });
      onDone?.();
    },
  });
}

export function Dot({
  colour,
  className,
}: {
  colour: "ok" | "warn" | "off" | "none";
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-2 shrink-0 rounded-full",
        colour === "ok" && "bg-[var(--fibsem-ok-color)]",
        colour === "warn" && "bg-[var(--fibsem-semantic-warning-color)]",
        colour === "off" && "bg-[var(--fibsem-pending-color)]",
        colour === "none" &&
          "border-[1.5px] border-[var(--fibsem-pending-color)]",
        className,
      )}
    />
  );
}

/** An icon whose words show on hover: where a line of hint text would go. */
export function Hint({
  children,
  lock,
}: {
  children: ReactNode;
  lock?: boolean;
}) {
  const Icon = lock ? Lock : Info;
  return (
    <HoverCard
      align="start"
      trigger={
        <button
          type="button"
          aria-label="More"
          onClick={(event) => event.stopPropagation()}
          className="inline-flex cursor-help border-0 bg-transparent p-0 align-[-2px] text-muted-foreground"
        >
          <Icon size={13} />
        </button>
      }
    >
      <div className="text-xs leading-snug text-foreground">{children}</div>
    </HoverCard>
  );
}

export function Failed({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p className="m-0 text-[13px] text-destructive">
      {(error as Error).message}
    </p>
  );
}

/** The last part of an experiment's folder. */
export const leaf = (path?: string) =>
  path?.split(/[\\/]/).filter(Boolean).pop() ?? "";

/** "3 h 12 m" since *since* (seconds since the epoch). */
export function uptime(since?: number | null): string {
  if (!since) return "";
  const minutes = Math.max(0, Math.floor((Date.now() / 1000 - since) / 60));
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours} h ${minutes % 60} m` : `${minutes} m`;
}

/** The dot beside an instrument's name. */
export function health(i: Instrument): "ok" | "warn" | "off" {
  if (!i.reachable) return "off";
  return i.paired && i.gateway ? "ok" : "warn";
}

/** An instrument in two or three words, for a list. */
export function brief(i: Instrument): string {
  if (i.reachable === null) return "Checking";
  if (!i.reachable) return "Not reachable";
  if (!i.paired) return "Not paired";
  if (!i.gateway) return "Pair again";
  if (i.session || i.agent) return "Agent running";
  if (i.gateway.switching) return "App switching";
  return i.gateway.mode === "running" ? "App up" : "App off";
}

/** Whether an agent can be started for it from here. */
export const ready = (i: Instrument) =>
  !!i.reachable &&
  i.paired &&
  i.gateway?.mode === "running" &&
  !i.session &&
  !i.agent;

/** Whether its PC lets a hub set *level*. */
export function allowed(level: string, cap: string | null): boolean {
  return !cap || LEVELS.indexOf(level) <= LEVELS.indexOf(cap);
}

export function capWords(cap: string): string {
  return `Capped at ${title(cap)} on the microscope PC.`;
}

/** Watch, Assist and Drive side by side; a level above the PC's cap is locked. */
export function LevelControl({ instrument: i }: { instrument: Instrument }) {
  const level = useHubAction((value: string) =>
    api(`${at(i.name)}/level`, {
      method: "PUT",
      body: JSON.stringify({ level: value }),
    }),
  );
  const now = i.session?.level ?? "watch";
  return (
    <span className="flex items-center gap-3">
      <span className="inline-flex h-[26px] overflow-hidden rounded border border-input">
        {LEVELS.map((l) => {
          const locked = !allowed(l, i.level_cap);
          return (
            <button
              key={l}
              type="button"
              aria-pressed={l === now}
              disabled={locked || level.isPending || l === now}
              title={locked && i.level_cap ? capWords(i.level_cap) : undefined}
              onClick={() => level.mutate(l)}
              className={cn(
                "flex items-center gap-1.5 border-0 border-r border-input px-[11px] text-xs last:border-r-0",
                l === now
                  ? "bg-accent text-foreground"
                  : "bg-transparent text-muted-foreground",
                locked
                  ? "opacity-60"
                  : l !== now && "cursor-pointer hover:text-foreground",
              )}
            >
              {title(l)}
              {locked && <Lock size={11} />}
            </button>
          );
        })}
      </span>
      {level.error ? (
        <span className="text-xs text-destructive">
          {(level.error as Error).message}
        </span>
      ) : null}
    </span>
  );
}

/** The goal, playbook, level and budget of a new agent, and its Start. */
export function StartForm({
  instrument: i,
  disabled,
  onDone,
  onCancel,
  children,
}: {
  instrument: Instrument | undefined;
  disabled?: string;
  onDone?: () => void;
  onCancel?: () => void;
  /** Fields above the goal: the instrument picker. */
  children?: ReactNode;
}) {
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
    retry: false,
  });
  const usable = (playbooks.data ?? []).filter((p) => !p.error);
  const [playbook, setPlaybook] = useState("");
  const [goal, setGoal] = useState("");
  const [level, setLevel] = useState("watch");
  const [budget, setBudget] = useState("");
  const chosen =
    playbook ||
    usable.find((p) => p.name === "fibsem-agent")?.name ||
    usable[0]?.name ||
    "";
  const cap = i?.level_cap ?? null;
  const start = useHubAction(
    () =>
      api<HubState>(`${at(i?.name ?? "")}/agent/start`, {
        method: "POST",
        body: JSON.stringify({
          playbook: chosen,
          goal: goal.trim(),
          level: allowed(level, cap) ? level : (cap ?? "watch"),
          ...(budget.trim() ? { budget_usd: budget.trim() } : {}),
        }),
      }),
    onDone,
  );
  return (
    <div className="flex flex-col gap-3">
      {children}
      <label className={LABEL}>
        Goal
        <input
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          placeholder="What should it do?"
          className={FIELD}
        />
      </label>
      <div className="grid grid-cols-[1.6fr_1fr_110px] gap-3">
        <label className={LABEL}>
          Playbook
          <select
            value={chosen}
            onChange={(event) => setPlaybook(event.target.value)}
            className={FIELD}
          >
            {usable.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className={LABEL}>
          <span className="flex items-center gap-1.5">
            Level
            {cap && <Hint>{capWords(cap)}</Hint>}
          </span>
          <select
            value={allowed(level, cap) ? level : (cap ?? "watch")}
            onChange={(event) => setLevel(event.target.value)}
            className={FIELD}
          >
            {LEVELS.map((l) => (
              <option key={l} value={l} disabled={!allowed(l, cap)}>
                {title(l)}
              </option>
            ))}
          </select>
        </label>
        <label className={LABEL}>
          Budget ($)
          <input
            value={budget}
            onChange={(event) => setBudget(event.target.value)}
            inputMode="decimal"
            placeholder="20"
            className={FIELD}
          />
        </label>
      </div>
      <Failed error={start.error} />
      <div className="flex justify-end gap-2">
        {onCancel && (
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button
          variant="primary"
          disabled={!i || !!disabled || !chosen || start.isPending}
          title={disabled}
          onClick={() => start.mutate()}
        >
          Start the agent
        </Button>
      </div>
    </div>
  );
}

/** "playbook · Level · since 7:35 pm", the parts that are known. */
export function agentMeta(i: Instrument, level = true): string {
  const a = i.session ?? i.agent;
  if (!a) return "";
  return [
    a.playbook,
    level && a.level && title(a.level),
    a.started && `since ${fmt.clock(a.started)}`,
  ]
    .filter(Boolean)
    .join(" · ");
}
