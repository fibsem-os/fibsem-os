import { Check, ChevronDown, ChevronRight, Circle, Minus } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { AgentPlan, ReportedSession } from "@/lib/types";
import { cn } from "@/lib/utils";

// The time left on a constraint turns orange in its last half hour.
const SOON_MIN = 30;

const WARN = "text-[var(--fibsem-warn-color)]";

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
      (session.profile ||
        session.playbook?.name ||
        session.skills?.length ||
        session.goal ||
        session.budget_usd != null ||
        session.spent_usd != null ||
        session.context ||
        session.constraints?.length ||
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

/** The profile and playbook on one line; a click shows who chose the playbook,
 * its skills and the reminders on now. */
function Setup({ session }: { session: ReportedSession }) {
  const [open, setOpen] = useState(() => stored(SETUP_OPEN));
  const { profile, playbook, hooks, skills } = session;
  const name = playbook?.name;
  if (!profile && !name && !skills?.length && !hooks?.length) return null;
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
        title={[profile && `Profile ${profile}`, name && `Playbook ${name}`].filter(Boolean).join(", ")}
      >
        <span className="min-w-0 flex-1 break-words">
          {profile && <span className="text-foreground">{profile}</span>}
          {profile && name && " · "}
          {name ?? (profile ? null : "Setup")}
        </span>
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

/** Goal, constraints, the plan (or the agent's checklist), budget and context,
 * then the profile and playbook on one line at the bottom. */
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
  const {
    started,
    ended,
    goal,
    budget_usd: budget,
    spent_usd: spent,
    tokens,
    context,
    constraints,
    checklist,
  } = session;
  return (
    <div className="flex flex-col gap-3.5 text-[13px] leading-[1.45]">
      {goal && (
        <Section title="Goal">
          <p className="m-0 break-words">{goal}</p>
          {started && !ended && (
            <p className="m-0 mt-1 text-muted-foreground">
              Running for {duration(Math.max(0, Math.floor((now - Date.parse(started)) / 60000)))}
            </p>
          )}
        </Section>
      )}
      {constraints?.length ? (
        <Section title="Constraints">
          {constraints.map((c, i) => {
            const left = c.by ? timeLeft(c.by, now) : null;
            return (
              <p key={i} className="m-0 break-words">
                {c.text}
                {left && <span className={left.soon ? WARN : "text-muted-foreground"}> · {left.text}</span>}
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
        <Section title="Budget">
          {(budget != null || spent != null) && (
            <p
              className={cn("m-0", budgetLow && WARN)}
              title={
                tokens
                  ? `${count(tokens.input + tokens.cache_read + tokens.cache_write)} tokens in, ${count(tokens.output)} out`
                  : undefined
              }
            >
              {budget != null ? `${dollars(spent ?? 0)} of ${dollars(budget)}` : `${dollars(spent!)} spent`}
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
            <p
              className="m-0 mt-1.5 text-muted-foreground"
              title={context.compact_at ? `Compacts at ${count(context.compact_at)}` : undefined}
            >
              Context {count(context.used)}
              {context.window ? ` of ${count(context.window)}` : " tokens"}
            </p>
          )}
        </Section>
      )}
      <Setup session={session} />
    </div>
  );
}
