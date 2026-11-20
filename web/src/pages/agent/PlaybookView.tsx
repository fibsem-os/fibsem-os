import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, Circle } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import type { McpServer, PlaybookAbout, ReportedSession } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { LibraryFile, LibraryItem } from "./Library";

// The card names up to this many servers; more are counted.
const NAMED_SERVERS = 3;
const OK = "var(--fibsem-ok-color)";
const DOWN = "var(--fibsem-warn-color)";
const SOURCE: Record<string, string> = {
  shipped: "Shipped with fibsemOS",
  lab: "The lab's own copy",
  plugin: "From a plugin",
};

const up = (server: McpServer) => server.status === "connected";

function Dot({ colour }: { colour: string }) {
  return <span className="inline-block size-[7px] shrink-0 rounded-full" style={{ background: colour }} />;
}

/** "fibsem, lab-notebook", or "4 of 5" once there are more than three; a
 * server that isn't connected is always named. */
function Connected({ servers }: { servers: McpServer[] }) {
  const down = servers.filter((s) => !up(s));
  const named = servers.length <= NAMED_SERVERS;
  return (
    <p
      className="m-0 mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-muted-foreground"
      title={servers.map((s) => `${s.name}: ${s.status}`).join("\n")}
    >
      <span>Connected:</span>
      {named ? (
        servers.map((s) => (
          <span key={s.name} className={cn("flex items-center gap-1.5", !up(s) && "text-[var(--fibsem-warn-color)]")}>
            <Dot colour={up(s) ? OK : DOWN} />
            {s.name}
            {!up(s) && " down"}
          </span>
        ))
      ) : (
        <>
          <span className="flex items-center gap-1.5">
            <Dot colour={down.length ? DOWN : OK} />
            {down.length ? `${servers.length - down.length} of ${servers.length}` : `${servers.length} servers`}
          </span>
          {down.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5 text-[var(--fibsem-warn-color)]">
              <Dot colour={DOWN} />
              {s.name} down
            </span>
          ))}
        </>
      )}
    </p>
  );
}

/** The playbook under the goal: its name, who chose it and its servers;
 * Open shows the whole of it. */
export function PlaybookLine({ session, onOpen }: { session: ReportedSession; onOpen: () => void }) {
  const name = session.playbook_about?.name ?? session.playbook?.name;
  const servers = session.servers ?? [];
  if (!name && !servers.length) return null;
  return (
    <div className="mt-3 border-t border-dashed border-border pt-3">
      <p className="m-0 flex items-baseline gap-2">
        <span className="min-w-0 flex-1 break-words">
          <span className="text-muted-foreground">Playbook </span>
          <b className="font-semibold">{name ?? "none"}</b>
          {session.playbook?.chosen_by && (
            <span className="text-muted-foreground"> · {session.playbook.chosen_by}</span>
          )}
        </span>
        <button
          className="shrink-0 cursor-pointer border-0 bg-transparent p-0 text-muted-foreground underline underline-offset-2 hover:text-foreground"
          onClick={onOpen}
          title="The playbook this session runs"
        >
          Open
        </button>
      </p>
      {servers.length > 0 && <Connected servers={servers} />}
    </div>
  );
}

function Part({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <section className="border-t border-border py-3.5 first:border-t-0 first:pt-0">
      <h4 className="m-0 mb-1.5 text-[12px] font-semibold text-muted-foreground">{title}</h4>
      {children}
    </section>
  );
}

function Link({ children, onClick, title }: { children: ReactNode; onClick: () => void; title?: string }) {
  return (
    <button
      className="cursor-pointer border-0 bg-transparent p-0 text-left font-medium text-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
      onClick={onClick}
      title={title}
    >
      {children}
    </button>
  );
}

type Detail = { kind: "skills" | "agents"; name: string };

function useLibrary(kind: "skills" | "agents", enabled: boolean) {
  return useQuery({
    queryKey: ["app/library", kind],
    queryFn: () => api<{ items: LibraryItem[] }>(`/app/library/${kind}`).then((a) => a.items),
    enabled,
    staleTime: 60_000,
  });
}

const STATE: Record<string, string> = { done: "Done", loaded: "In use" };

/** The whole playbook the session runs, as it loaded it, with what is on now. */
function Overview({
  session,
  about,
  descriptions,
  edited,
  onDetail,
}: {
  session: ReportedSession;
  about: PlaybookAbout | undefined;
  descriptions: Record<string, string | null>;
  edited: boolean;
  onDetail: (d: Detail) => void;
}) {
  const states = Object.fromEntries((session.playbook?.skills ?? []).map((s) => [s.skill, s]));
  const optional: { skill: string; when?: string; agents?: string[] }[] =
    about?.optional ?? (session.playbook?.skills ?? []).map((s) => ({ skill: s.skill }));
  const loaded = (skill: string) => ["loaded", "done"].includes(states[skill]?.state ?? "");
  // Each helper, and whether one of the skills that brings it is in use.
  const helpers = new Map<string, { on: boolean; from: string[] }>();
  for (const agent of about?.agents ?? []) helpers.set(agent, { on: true, from: [] });
  for (const o of optional) {
    for (const agent of o.agents ?? []) {
      const row = helpers.get(agent) ?? { on: false, from: [] };
      row.on ||= states[o.skill]?.state === "loaded";
      row.from.push(o.skill);
      helpers.set(agent, row);
    }
  }
  // Where each reminder on now comes from: the playbook or one of its skills.
  const from = new Map<string, string>();
  for (const r of about?.reminders ?? []) from.set(r.say, about!.name);
  for (const o of about?.optional ?? []) for (const r of o.reminders ?? []) from.set(r.say, o.skill);
  const later = (about?.optional ?? []).filter((o) => o.reminders?.length && !loaded(o.skill));
  const servers = session.servers ?? [];
  return (
    <div className="flex flex-col">
      {edited && (
        <p className="m-0 mb-3 rounded border border-border bg-panel px-3 py-2 text-[12px] text-muted-foreground">
          The playbook's file has changed since this session started. The agent keeps to the copy below until a new
          session starts.
        </p>
      )}
      {about?.rules?.length ? (
        <Part title="Rules">
          <ul className="m-0 flex list-disc flex-col gap-1 pl-4">
            {about.rules.map((rule, i) => (
              <li key={i}>{rule}</li>
            ))}
          </ul>
        </Part>
      ) : null}
      {about?.skills?.length || optional.length ? (
        <Part title="Skills">
          {about?.skills?.length ? (
            <>
              <h5 className="m-0 mb-1 text-[12px] font-normal text-muted-foreground">Always read</h5>
              <p className="m-0 flex flex-wrap gap-x-1.5">
                {about.skills.map((skill, i) => (
                  <span key={skill}>
                    <Link onClick={() => onDetail({ kind: "skills", name: skill })} title={descriptions[skill] ?? undefined}>
                      {skill}
                    </Link>
                    {i < about.skills!.length - 1 && <span className="text-muted-foreground"> ·</span>}
                  </span>
                ))}
              </p>
            </>
          ) : null}
          {optional.length ? (
            <>
              <h5 className="m-0 mt-3 mb-1 text-[12px] font-normal text-muted-foreground">Read when needed</h5>
              <ul className="m-0 flex list-none flex-col gap-2 p-0">
                {optional.map((o) => {
                  const state = states[o.skill];
                  const when = o.when;
                  return (
                    <li key={o.skill} className="grid grid-cols-[14px_1fr_auto] gap-x-2">
                      <span className="pt-[3px] text-muted-foreground">
                        {state?.state === "done" ? (
                          <Check size={12} />
                        ) : (
                          <Circle size={10} className={state?.state === "loaded" ? "fill-current" : undefined} />
                        )}
                      </span>
                      <span className="min-w-0">
                        <Link onClick={() => onDetail({ kind: "skills", name: o.skill })}>{o.skill}</Link>
                        {state?.summary ? (
                          <span className="block text-muted-foreground">{state.summary}</span>
                        ) : when ? (
                          <span className="block text-muted-foreground">{readWhen(when)}</span>
                        ) : null}
                      </span>
                      <span className="text-[12px] text-muted-foreground">{STATE[state?.state ?? ""] ?? "Not yet"}</span>
                    </li>
                  );
                })}
              </ul>
            </>
          ) : null}
        </Part>
      ) : null}
      {helpers.size > 0 && (
        <Part title="Helpers">
          <p className="m-0 mb-1.5 text-muted-foreground">Subagents the agent can call.</p>
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {[...helpers].map(([name, row]) => (
              <li key={name} className="grid grid-cols-[minmax(0,180px)_1fr] items-baseline gap-x-3">
                <span className="flex items-center gap-2">
                  <Dot colour={row.on ? OK : "var(--fibsem-gray-secondary-color)"} />
                  <Link onClick={() => onDetail({ kind: "agents", name })}>{name}</Link>
                </span>
                <span className="line-clamp-2 text-muted-foreground" title={descriptions[name] ?? undefined}>
                  {descriptions[name] ? oneLine(descriptions[name]!) : null}
                  {!row.on && row.from.length ? ` Comes with ${row.from.join(" or ")}.` : ""}
                </span>
              </li>
            ))}
          </ul>
        </Part>
      )}
      {session.hooks?.length || later.length ? (
        <Part title="Reminders on now">
          {session.hooks?.map((hook, i) => (
            <p key={i} className="m-0 mb-1 grid grid-cols-[150px_1fr] gap-x-3">
              <span className="text-muted-foreground">{hook.label}</span>
              <span>
                {hook.say}
                {from.get(hook.say) && <span className="text-[12px] text-muted-foreground"> · {from.get(hook.say)}</span>}
              </span>
            </p>
          ))}
          {later.map((o) => (
            <p key={o.skill} className="m-0 mt-1 text-[12px] text-muted-foreground">
              {o.reminders!.length} more come on with {o.skill}.
            </p>
          ))}
        </Part>
      ) : null}
      {servers.length > 0 && (
        <Part title={`Connected · ${servers.filter(up).length} of ${servers.length}`}>
          {servers.map((s) => (
            <p key={s.name} className="m-0 mb-1 grid grid-cols-[minmax(0,180px)_1fr_auto] items-baseline gap-x-3">
              <span className="flex items-center gap-2">
                <Dot colour={up(s) ? OK : DOWN} />
                {s.name}
              </span>
              <span className="text-muted-foreground">{s.about}</span>
              <span className={cn("text-[12px]", up(s) ? "text-muted-foreground" : "text-[var(--fibsem-warn-color)]")}>
                {up(s) ? `${s.tools ?? 0} tools` : s.status}
              </span>
            </p>
          ))}
        </Part>
      )}
      {about && (
        <Part title="Details">
          <dl className="m-0 grid grid-cols-[150px_1fr] gap-x-3 gap-y-1">
            {about.tools?.length ? (
              <>
                <dt className="text-muted-foreground">Loaded at start</dt>
                <dd className="m-0 break-words">{about.tools.join(", ")}</dd>
              </>
            ) : null}
            {about.instrument && (
              <>
                <dt className="text-muted-foreground">Instrument</dt>
                <dd className="m-0">{about.instrument}</dd>
              </>
            )}
            {about.hash && (
              <>
                <dt className="text-muted-foreground">Version</dt>
                <dd className="m-0 font-mono text-[12px]">{about.hash}</dd>
              </>
            )}
            {about.source && (
              <>
                <dt className="text-muted-foreground">From</dt>
                <dd className="m-0">{SOURCE[about.source] ?? about.source}</dd>
              </>
            )}
          </dl>
        </Part>
      )}
    </div>
  );
}

const oneLine = (text: string) => text.replace(/\s+/g, " ").trim();

/** An optional skill's *when* as a line: "Read before the first polishing
 * pass", "Read when the operator asks". */
const readWhen = (when: string) =>
  /^(before|after|at|once|while|when|if|for|on)\b/i.test(when) ? `Read ${when}` : `Read when ${when}`;

/** A file's front matter and body, split. */
function parts(text: string): { front: string; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text);
  return match ? { front: match[1], body: match[2] } : { front: "", body: text };
}

/** A skill or subagent, as the library has it now. */
function DetailView({
  detail,
  description,
  summary,
}: {
  detail: Detail;
  description: string | null | undefined;
  summary?: string;
}) {
  const file = useQuery({
    queryKey: ["app/library", detail.kind, detail.name],
    queryFn: () => api<LibraryFile>(`/app/library/${detail.kind}/${encodeURIComponent(detail.name)}`),
  });
  const tools = Array.isArray(file.data?.data?.tools) ? (file.data!.data!.tools as string[]) : [];
  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="m-0 text-[15px] font-semibold">{detail.name}</h3>
        <p className="m-0 text-muted-foreground">
          {detail.kind === "agents" ? "Subagent" : "Skill"}
          {file.data && ` · ${file.data.source === "lab" ? "the lab's own copy" : "shipped with fibsemOS"}`}
        </p>
      </div>
      {summary && (
        <p className="m-0 rounded border border-border bg-panel px-3 py-2">
          <span className="text-muted-foreground">Done: </span>
          {summary}
        </p>
      )}
      {description && <p className="m-0">{oneLine(description)}</p>}
      {tools.length > 0 && (
        <div>
          <h4 className="m-0 mb-1 text-[12px] font-semibold text-muted-foreground">Tools it may use · {tools.length}</h4>
          <p className="m-0 font-mono text-[12px] break-words text-muted-foreground">
            {tools.map((t) => t.split("__").pop()).join(", ")}
          </p>
        </div>
      )}
      <div className="border-t border-border pt-3">
        {file.isLoading ? (
          <p className="m-0 text-muted-foreground">Reading…</p>
        ) : file.isError ? (
          <p className="m-0 text-muted-foreground">Not in the library on this computer.</p>
        ) : (
          <Markdown text={parts(file.data!.text).body} doc />
        )}
      </div>
    </div>
  );
}

/** Open: the playbook the session runs, as it loaded it. A skill or
 * helper's name opens what the library has for it. */
export function PlaybookDialog({
  session,
  open,
  onOpenChange,
}: {
  session: ReportedSession;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const about = session.playbook_about;
  const name = about?.name ?? session.playbook?.name;
  const file = useQuery({
    queryKey: ["app/agent/playbook-file", about?.hash ?? session.playbook?.hash],
    queryFn: () => api<{ name: string | null; hash: string | null; edited: boolean }>("/app/agent/playbook-file"),
    enabled: open && !!name,
  });
  const skills = useLibrary("skills", open);
  const agents = useLibrary("agents", open);
  const descriptions = Object.fromEntries(
    [...(skills.data ?? []), ...(agents.data ?? [])].map((item) => [item.name, item.description]),
  );
  const close = (o: boolean) => {
    if (!o) setDetail(null);
    onOpenChange(o);
  };
  const summary = detail ? session.playbook?.skills?.find((s) => s.skill === detail.name)?.summary : undefined;
  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent title={name ? `Playbook · ${name}` : "Playbook"} className="w-[min(780px,calc(100vw-32px))]">
        <div className="min-h-0 overflow-y-auto px-5 py-4 text-[13px] leading-[1.5]">
          {detail ? (
            <>
              <button
                className="mb-3 flex cursor-pointer items-center gap-1 border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
                onClick={() => setDetail(null)}
              >
                <ArrowLeft size={13} /> {name ?? "Playbook"}
              </button>
              <DetailView detail={detail} description={descriptions[detail.name]} summary={summary} />
            </>
          ) : (
            <>
              {(about?.description || about?.extends?.length || session.playbook?.chosen_by) && (
                <p className="m-0 mb-3.5 text-muted-foreground">
                  {about?.description}
                  {about?.extends?.length ? ` Builds on ${about.extends.join(", ")}.` : ""}
                  {session.playbook?.chosen_by
                    ? ` ${session.playbook.chosen_by.replace(/^./, (c) => c.toUpperCase())}.`
                    : ""}
                </p>
              )}
              <Overview
                session={session}
                about={about}
                descriptions={descriptions}
                edited={!!file.data?.edited}
                onDetail={setDetail}
              />
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
