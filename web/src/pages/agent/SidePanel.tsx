import { useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronRight, Circle, Minus } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { AgentPlan, ReportedSession } from "@/lib/types";
import { cn } from "@/lib/utils";

// The time left on a deadline turns orange in its last half hour.
const SOON_MIN = 30;

const WARN = "text-[var(--fibsem-warn-color)]";
/** Context sizes, in tokens, from which the panel shows it orange, then red. */
const CONTEXT_WARN = 400_000;
const CONTEXT_HIGH = 800_000;

const dollars = (value: number) => (Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`);

/** 412 tokens as "412", 41_200 as "41k", 1_200_000 as "1.2M". */
function count(n: number): string {
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${+(n / 1000).toFixed(1)}k`;
  return `${n}`;
}

function duration(minutes: number): string {
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours} h ${minutes % 60} min` : `${minutes} min`;
}

function timeLeft(by: string, now: number): { text: string; soon: boolean } {
  const minutes = Math.floor((Date.parse(by) - now) / 60000);
  if (minutes < 0) return { text: "passed", soon: true };
  return { text: `${duration(minutes)} left`, soon: minutes < SOON_MIN };
}

function useNow(everyMs: number) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const clock = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(clock);
  }, [everyMs]);
  return now;
}

function Section({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <section className="border-b border-border/60 pb-3.5 last:border-b-0 last:pb-0">
      <h4 className="mb-1.5 text-[12px] font-semibold text-muted-foreground">{title}</h4>
      {children}
    </section>
  );
}

/** A ticked, open or standing line, the same in To do, Plan and Skills. */
function Tick({ state, children }: { state: "done" | "open" | "always" | "loaded"; children: ReactNode }) {
  return (
    <li className="flex gap-2">
      <span className="mt-[3px] shrink-0 text-muted-foreground">
        {state === "done" ? (
          <Check size={12} />
        ) : state === "always" ? (
          <Minus size={12} />
        ) : (
          <Circle size={11} fill={state === "loaded" ? "currentColor" : "none"} />
        )}
      </span>
      <span className={cn("min-w-0 break-words", state === "done" && "text-muted-foreground")}>{children}</span>
    </li>
  );
}

const SETUP_OPEN = "fibsem-agent-setup-open";

function stored(key: string): boolean {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

function todoTitle(checklist: NonNullable<ReportedSession["checklist"]>): string {
  const counted = checklist.filter((item) => item.state !== "always");
  if (!counted.length) return "To do";
  return `To do · ${counted.filter((item) => item.state === "done").length} of ${counted.length}`;
}

/** Whether the agent has written its plan yet. */
export const hasPlan = (plan: AgentPlan | null | undefined): plan is AgentPlan => Boolean(plan?.markdown);

/** Whether the panel has anything to show: hidden when it would be empty. */
export function hasPanel(session: ReportedSession | null | undefined): session is ReportedSession {
  return Boolean(
    session &&
      (session.playbook?.name ||
        session.skills?.length ||
        session.goal ||
        session.budget_usd != null ||
        session.spent_usd != null ||
        session.context ||
        session.deadlines?.length ||
        session.checklist?.length),
  );
}

type Group = { name: string; steps: AgentPlan["steps"]; done: number };

function groups(steps: AgentPlan["steps"]): Group[] {
  const found: Group[] = [];
  for (const step of steps) {
    let group = found.find((g) => g.name === step.group);
    if (!group) found.push((group = { name: step.group, steps: [], done: 0 }));
    group.steps.push(step);
    if (step.done) group.done += 1;
  }
  return found;
}

/** The plan's steps by group: a finished group folds to one line, the first
 * unfinished one is open, the rest open on a click. */
function PlanSteps({ plan, onOpen }: { plan: AgentPlan; onOpen: () => void }) {
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const found = groups(plan.steps);
  const current = found.find((g) => g.done < g.steps.length);
  const next = current?.steps.find((s) => !s.done);
  const title = plan.steps.length ? `Plan · ${plan.done} of ${plan.steps.length}` : "Plan";
  return (
    <Section
      title={
        <button
          className="flex w-full cursor-pointer items-baseline justify-between border-0 bg-transparent p-0 text-left text-[12px] font-semibold text-muted-foreground hover:text-foreground"
          onClick={onOpen}
          title="Open plan.md"
        >
          <span>{title}</span>
          <span className="font-normal underline underline-offset-2">Open plan</span>
        </button>
      }
    >
      {found.length ? (
        <div className="flex flex-col gap-1.5">
          {found.map((group) => {
            const finished = group.done === group.steps.length;
            const list = (
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
                {group.steps.map((step, i) => (
                  <Tick key={i} state={step.done ? "done" : "open"}>
                    {step.text}
                    {step === next && <span className="text-[12px] text-muted-foreground"> next</span>}
                  </Tick>
                ))}
              </ul>
            );
            // Boxes above any ### line stand on their own.
            if (!group.name) return <div key="">{list}</div>;
            const open = toggled[group.name] ?? group === current;
            return (
              <div key={group.name}>
                <button
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-2 border-0 bg-transparent p-0 text-left text-[13px]",
                    finished || !open ? "text-muted-foreground" : "font-medium text-foreground",
                  )}
                  aria-expanded={open}
                  onClick={() => setToggled({ ...toggled, [group.name]: !open })}
                >
                  <span className="shrink-0">
                    {finished && !open ? (
                      <Check size={12} />
                    ) : open ? (
                      <ChevronDown size={12} />
                    ) : (
                      <ChevronRight size={12} />
                    )}
                  </span>
                  <span className="min-w-0 flex-1 break-words">{group.name}</span>
                  <span className="shrink-0 text-[12px] font-normal text-muted-foreground">
                    {group.done} of {group.steps.length}
                  </span>
                </button>
                {open && <div className="mt-1.5 ml-5">{list}</div>}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="m-0 text-muted-foreground">No steps yet</p>
      )}
    </Section>
  );
}

/** The playbook on one line; a click shows who chose the playbook,
 * its skills and the reminders on now. */
function Setup({ session }: { session: ReportedSession }) {
  const [open, setOpen] = useState(() => stored(SETUP_OPEN));
  const { playbook, hooks, skills } = session;
  const name = playbook?.name;
  if (!name && !skills?.length && !hooks?.length) return null;
  const optional = playbook?.skills ?? [];
  const toggle = () => {
    setOpen(!open);
    try {
      localStorage.setItem(SETUP_OPEN, open ? "0" : "1");
    } catch {
      // A browser without storage opens it closed next time.
    }
  };
  return (
    <section>
      <button
        className="flex w-full cursor-pointer items-center gap-2 border-0 bg-transparent p-0 text-left text-[13px] text-muted-foreground hover:text-foreground"
        aria-expanded={open}
        onClick={toggle}
        title={name ? `Playbook ${name}` : undefined}
      >
        <span className="min-w-0 flex-1 break-words">{name ?? "Setup"}</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-3">
          {name && (
            <p className="m-0 text-muted-foreground">
              Playbook {name}
              {playbook?.chosen_by ? `, ${playbook.chosen_by}` : ""}
            </p>
          )}
          {optional.length ? (
            <div>
              <h5 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">
                Skills · {optional.filter((s) => s.state === "done").length} of {optional.length} done
              </h5>
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
                {optional.map((s) => (
                  <Tick key={s.skill} state={s.state === "done" ? "done" : s.state === "loaded" ? "loaded" : "open"}>
                    {s.skill}
                    {s.state !== "done" && <span className="text-[12px] text-muted-foreground"> {s.state}</span>}
                    {s.summary && <span className="block text-[12px] text-muted-foreground">{s.summary}</span>}
                  </Tick>
                ))}
              </ul>
            </div>
          ) : skills?.length ? (
            <div>
              <h5 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">Skills read</h5>
              <p className="m-0 break-words">{skills.join(", ")}</p>
            </div>
          ) : null}
          {hooks?.length ? (
            <div>
              <h5 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">Reminders on now</h5>
              {hooks.map((hook, i) => (
                <p key={i} className="m-0 mt-1.5 line-clamp-2 break-words first-of-type:mt-0" title={hook.say}>
                  <span className="text-muted-foreground">{hook.label}:</span> {hook.say}
                </p>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

/** The running session's budget, set by the operator: fibsem-agent takes it
 * with its next report, and asks again before it stops on a spent budget. */
function BudgetDialog({
  budget,
  spent,
  open,
  onOpenChange,
}: {
  budget: number | undefined;
  spent: number | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const client = useQueryClient();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (open) {
      setValue(budget != null ? String(budget) : "");
      setError(null);
    }
  }, [open, budget]);
  const amount = Number(value.trim().replace(/^\$/, ""));
  const valid = value.trim() !== "" && Number.isFinite(amount) && amount > 0;
  const below = valid && spent != null && amount <= spent;
  const save = async () => {
    if (!valid || busy) return;
    setBusy(true);
    setError(null);
    try {
      await api("/app/agent/budget", { method: "PUT", body: JSON.stringify({ budget_usd: amount }) });
      // The session and the agent's state (its budget warning) read it again.
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Budget" className="w-[min(420px,calc(100vw-32px))]">
        <div className="flex flex-col gap-3 px-4 py-4 text-[13px]">
          <p className="m-0 text-muted-foreground">
            {spent != null ? `${dollars(spent)} spent` : "Nothing spent yet"}
            {budget != null ? ` of ${dollars(budget)} this run.` : "."} The agent keeps to the new budget from its
            next turn.
          </p>
          <label className="flex items-center gap-2">
            <span className="shrink-0">Budget ($)</span>
            <input
              autoFocus
              inputMode="decimal"
              aria-label="Budget in dollars"
              className="min-w-0 flex-1 rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && save()}
            />
          </label>
          {below && (
            <p className={cn("m-0", WARN)}>
              That is no more than it has spent: the agent stops at the end of its current turn.
            </p>
          )}
          {error && <p className="m-0 text-destructive">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button variant="primary" disabled={!valid || busy} onClick={save}>
              Set budget
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** What the context is, what fills it (when the harness says), and a button
 * that asks the agent for a fresh context (POST /app/agent/context-reset). */
function ContextDialog({
  session,
  open,
  onOpenChange,
}: {
  session: ReportedSession;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (open) setError(null);
  }, [open]);
  const { context, contexts, ended } = session;
  if (!context) return null;
  const now = contexts ?? 1;
  // Asked in this context and not yet done: the agent resets when it can.
  const asked = session.reset_asked_at && (session.reset_asked_in ?? 1) >= now ? session.reset_asked_at : null;
  const parts = [...(context.parts ?? [])].sort((a, b) => b.tokens - a.tokens);
  const largest = parts.length ? parts[0].tokens : 0;
  const ask = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await api("/app/agent/context-reset", { method: "POST" });
      await client.invalidateQueries({ queryKey: ["app/agent"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Context" className="w-[min(460px,calc(100vw-32px))]">
        <div className="flex flex-col gap-3 px-4 py-4 text-[13px] leading-[1.5]">
          <p className="m-0">
            {count(context.used)} tokens
            {context.window ? ` of ${count(context.window)}` : ""}
            {now > 1 ? ` · context ${now} of this session` : ""}
          </p>
          <p className="m-0 text-muted-foreground">
            The context is everything the agent reads again at every step: its instructions, tools and skills, and
            the conversation so far. A larger context costs more per step and the agent works less well in it. The
            panel shows it orange from {count(CONTEXT_WARN)} and red from {count(CONTEXT_HIGH)}.
            {context.compact_at
              ? ` At ${count(context.compact_at)} it is compacted: the conversation is summarised, which can lose detail.`
              : ""}
          </p>
          {parts.length ? (
            <div>
              <h4 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">What fills it</h4>
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
                {parts.map((part) => (
                  <li key={part.name}>
                    <div className="flex justify-between gap-3">
                      <span className="break-words">{part.name}</span>
                      <span className="shrink-0 tabular-nums text-muted-foreground">{count(part.tokens)}</span>
                    </div>
                    <div className="mt-1 h-[3px] rounded-sm bg-accent">
                      <div
                        className="h-full rounded-sm bg-muted-foreground/70"
                        style={{ width: `${(100 * part.tokens) / largest}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="m-0 text-muted-foreground">No breakdown yet: it comes with the agent's next turn.</p>
          )}
          {!ended && (
            <div className="border-t border-border/60 pt-3">
              <h4 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">Fresh context</h4>
              <p className="m-0 text-muted-foreground">
                At its next quiet moment, never during a mill, the agent brings its plan up to date, writes a handoff
                note (what's done, what's running, what's next) and carries on in a fresh context. The session, chat,
                plan, budget and experiment stay as they are.
              </p>
              {asked && (
                <p className={cn("m-0 mt-2", WARN)}>
                  Asked at{" "}
                  {new Date(asked).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}. The agent resets at
                  its next quiet moment.
                </p>
              )}
              {error && <p className="m-0 mt-2 text-destructive">{error}</p>}
              <div className="mt-3 flex justify-end gap-2">
                <Button onClick={() => onOpenChange(false)}>Close</Button>
                <Button variant="primary" disabled={busy || !!asked} onClick={ask}>
                  Reset context
                </Button>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** Goal and deadlines, the plan (or the agent's checklist), budget and context,
 * then the playbook on one line at the bottom. */
export function SidePanel({
  session,
  plan,
  budgetLow,
  onOpenPlan,
}: {
  session: ReportedSession;
  plan?: AgentPlan | null;
  budgetLow: boolean;
  onOpenPlan: () => void;
}) {
  const now = useNow(30000);
  const [budgetOpen, setBudgetOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const {
    started,
    ended,
    goal,
    budget_usd: budget,
    budget_set_usd: budgetSet,
    spent_usd: spent,
    tokens,
    context,
    deadlines,
    checklist,
  } = session;
  // The bar spans the window (else the compaction point); the context is
  // orange from CONTEXT_WARN and red from CONTEXT_HIGH tokens.
  const contextFull = context?.window || context?.compact_at;
  const contextShare = contextFull ? context!.used / contextFull : 0;
  const contextLevel =
    !context || ended ? null : context.used >= CONTEXT_HIGH ? "high" : context.used >= CONTEXT_WARN ? "warn" : null;
  const contextText = contextLevel === "high" ? "text-destructive" : contextLevel === "warn" ? WARN : "";
  return (
    <div className="flex flex-col gap-3.5 text-[13px] leading-[1.45]">
      {(goal || deadlines?.length) ? (
        <Section title="Goal">
          {goal && <p className="m-0 break-words">{goal}</p>}
          {started && !ended && (
            <p className="m-0 mt-1 text-muted-foreground">
              Running for {duration(Math.max(0, Math.floor((now - Date.parse(started)) / 60000)))}
            </p>
          )}
          {deadlines?.map((d, i) => {
            const left = timeLeft(d.by, now);
            return (
              <p key={i} className={cn("m-0 mt-1 break-words", left.soon ? WARN : "text-muted-foreground")}>
                {d.text} · {left.text}
              </p>
            );
          })}
        </Section>
      ) : null}
      {hasPlan(plan) ? (
        <PlanSteps plan={plan} onOpen={onOpenPlan} />
      ) : checklist?.length ? (
        <Section title={todoTitle(checklist)}>
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {checklist.map((item, i) => (
              <Tick key={i} state={item.state}>
                {item.text}
                {item.state === "always" && <span className="text-[12px] text-muted-foreground"> always</span>}
              </Tick>
            ))}
          </ul>
        </Section>
      ) : null}
      {(budget != null || spent != null || context) && (
        <Section title="Usage">
          {(budget != null || spent != null) && (
            <p
              className={cn("m-0", budgetLow && WARN)}
              title={
                tokens
                  ? `${count(tokens.input + tokens.cache_read + tokens.cache_write)} tokens in, ${count(tokens.output)} out`
                  : undefined
              }
            >
              <span className="text-muted-foreground">Budget </span>
              {ended ? (
                budget != null ? (
                  `${dollars(spent ?? 0)} of ${dollars(budget)}`
                ) : (
                  `${dollars(spent!)} spent`
                )
              ) : (
                // While it runs, a click opens the dialog that sets its budget.
                <button
                  className="cursor-pointer border-0 bg-transparent p-0 text-left text-inherit underline decoration-dotted underline-offset-2 hover:text-foreground"
                  onClick={() => setBudgetOpen(true)}
                  title="Change the budget"
                >
                  {budget != null ? `${dollars(spent ?? 0)} of ${dollars(budget)}` : `${dollars(spent!)} spent`}
                </button>
              )}
              {budgetSet != null && <span className="text-muted-foreground"> · set by you</span>}
            </p>
          )}
          {budget ? (
            <div className="mt-1.5 h-[3px] rounded-sm bg-accent">
              <div
                className={cn(
                  "h-full rounded-sm",
                  budgetLow ? "bg-[var(--fibsem-warn-color)]" : "bg-muted-foreground/70",
                )}
                style={{
                  width: `${Math.min(100, (100 * (spent ?? 0)) / budget)}%`,
                }}
              />
            </div>
          ) : null}
          {context && (
            <p className={cn("m-0 mt-2.5", contextText)}>
              <span className="text-muted-foreground">Context </span>
              {/* A click explains it, breaks it down and offers a fresh context. */}
              <button
                className="cursor-pointer border-0 bg-transparent p-0 text-left text-inherit underline decoration-dotted underline-offset-2 hover:text-foreground"
                onClick={() => setContextOpen(true)}
                title="What's in the context"
              >
                {count(context.used)}
                {context.window ? ` of ${count(context.window)}` : " tokens"}
              </button>
            </p>
          )}
          {contextFull ? (
            <div className="relative mt-1.5 h-[3px] rounded-sm bg-accent">
              <div
                className={cn(
                  "h-full rounded-sm",
                  contextLevel === "high"
                    ? "bg-destructive"
                    : contextLevel === "warn"
                      ? "bg-[var(--fibsem-warn-color)]"
                      : "bg-muted-foreground/70",
                )}
                style={{ width: `${Math.min(100, 100 * contextShare)}%` }}
              />
              {/* Ticks where the context turns orange, then red. */}
              {[CONTEXT_WARN, CONTEXT_HIGH].map(
                (mark, i) =>
                  mark < contextFull && (
                    <div
                      key={mark}
                      className={cn(
                        "absolute -top-[2px] h-[7px] w-[2px] rounded-sm",
                        i ? "bg-destructive/80" : "bg-[var(--fibsem-warn-color)]/80",
                      )}
                      style={{ left: `${(100 * mark) / contextFull}%` }}
                      title={`${count(mark)}: ${i ? "red" : "orange"} from here`}
                    />
                  ),
              )}
            </div>
          ) : null}
        </Section>
      )}
      <Setup session={session} />
      <BudgetDialog budget={budget} spent={spent} open={budgetOpen} onOpenChange={setBudgetOpen} />
      {context && <ContextDialog session={session} open={contextOpen} onOpenChange={setContextOpen} />}
    </div>
  );
}
