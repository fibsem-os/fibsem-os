import { Check, Circle, Minus } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { ReportedSession } from "@/lib/types";
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

function Bar({ share }: { share: number }) {
  return (
    <div className="mt-1.5 h-[3px] rounded-sm bg-accent">
      <div className="h-full rounded-sm bg-muted-foreground/70" style={{ width: `${Math.min(100, 100 * share)}%` }} />
    </div>
  );
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

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-b border-border/60 pb-3.5 last:border-b-0 last:pb-0">
      <h4 className="mb-1.5 text-[12px] font-semibold text-muted-foreground">{title}</h4>
      {children}
    </section>
  );
}

function todoTitle(checklist: NonNullable<ReportedSession["checklist"]>): string {
  const counted = checklist.filter((item) => item.state !== "always");
  if (!counted.length) return "To do";
  return `To do · ${counted.filter((item) => item.state === "done").length} of ${counted.length}`;
}

/** Whether the panel has anything to show: hidden when it would be empty. */
export function hasPanel(session: ReportedSession | null | undefined): session is ReportedSession {
  return Boolean(
    session &&
      (session.profile ||
        session.goal ||
        session.budget_usd != null ||
        session.spent_usd != null ||
        session.context ||
        session.constraints?.length ||
        session.checklist?.length),
  );
}

/** The agent's profile, goal, budget, the operator's constraints and its checklist. */
export function SidePanel({ session, budgetLow }: { session: ReportedSession; budgetLow: boolean }) {
  const now = useNow(30000);
  const {
    profile,
    profile_hash: hash,
    hooks,
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
      {profile && (
        <Section title="Profile">
          <p className="m-0 break-words">
            {profile}
            {hash && <span className="text-muted-foreground"> · {hash.slice(0, 7)}</span>}
          </p>
          {hooks?.map((hook, i) => (
            <p key={i} className="m-0 mt-1.5 break-words">
              <span className="text-muted-foreground">{hook.label}:</span> {hook.say}
            </p>
          ))}
        </Section>
      )}
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
      {(budget != null || spent != null) && (
        <Section title="Budget">
          <p className={cn("m-0", budgetLow && WARN)}>
            {budget != null ? `${dollars(spent ?? 0)} of ${dollars(budget)}` : `${dollars(spent!)} spent`}
          </p>
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
          {tokens && (
            <p className="m-0 mt-1.5 text-muted-foreground">
              {count(tokens.input + tokens.cache_read + tokens.cache_write)} tokens in, {count(tokens.output)} out
            </p>
          )}
        </Section>
      )}
      {context && (
        <Section title="Context">
          <p className="m-0">
            {count(context.used)}
            {context.window ? ` of ${count(context.window)}` : " tokens"}
          </p>
          {context.window ? <Bar share={context.used / context.window} /> : null}
          {context.compact_at ? (
            <p className="m-0 mt-1.5 text-muted-foreground">Compacts at {count(context.compact_at)}</p>
          ) : null}
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
      {checklist?.length ? (
        <Section title={todoTitle(checklist)}>
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {checklist.map((item, i) => (
              <li key={i} className="flex gap-2">
                <span className="mt-[3px] shrink-0 text-muted-foreground">
                  {item.state === "done" ? (
                    <Check size={12} />
                  ) : item.state === "always" ? (
                    <Minus size={12} />
                  ) : (
                    <Circle size={11} />
                  )}
                </span>
                <span className={cn("min-w-0 break-words", item.state === "done" && "text-muted-foreground")}>
                  {item.text}
                  {item.state === "always" && <span className="text-[12px] text-muted-foreground"> always</span>}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}
