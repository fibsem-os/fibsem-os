import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bell, ListFilter, Pencil, Plus, Search, Trash2 } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Popup } from "@/components/ui/popup";
import { api, imageUrl } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";
import { LibraryBrowser, type LibraryKind } from "./Library";

export type KnowledgeSection = "instrument" | "sample" | "playbook" | "experiment";
export type KnowledgeLasts = "until fixed" | "ongoing" | "record";
type Writer = "agent" | "operator";

/** What an entry is tagged with: names, never paths. */
export interface KnowledgeTags {
  instrument?: string;
  sample?: string;
  playbook?: string;
  /** The experiment it was learned in, by name. */
  session?: string;
}

/** One entry in the knowledge index. */
export interface KnowledgeRow {
  section: KnowledgeSection;
  id: string;
  title: string;
  written: string;
  by: Writer;
  status: "to review" | "kept";
  lasts: KnowledgeLasts;
  tags: KnowledgeTags;
  remind?: string | null;
  edited?: boolean;
  resolved?: boolean;
  /** An old fault the agent should ask about again. */
  ask_again?: boolean;
  /** How many images it has. */
  images: number;
}

/** One entry, opened. */
export interface KnowledgeEntry {
  section: KnowledgeSection;
  id: string;
  title: string;
  text: string;
  written: string;
  by: Writer;
  status: "to review" | "kept";
  lasts: KnowledgeLasts;
  tags: KnowledgeTags;
  remind?: string | null;
  /** Where it came from: ids and names, kept so it can be found again. */
  source?: { item?: string; images?: Record<string, string> };
  /** Its own copies of its images, by name. */
  images: string[];
  edited?: boolean;
  edited_by?: Writer | null;
  checked?: string | null;
  resolved?: string | null;
  /** Its file, as the operator would find it. */
  path: string;
}

const SHOW = [
  ["all", "Everything"],
  ["review", "To review"],
  ["until fixed", "Until fixed"],
  ["ongoing", "Ongoing"],
  ["record", "Past runs"],
  ["resolved", "Fixed"],
] as const;
type Kind = (typeof SHOW)[number][0];


const day = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });

function kindOf(row: Pick<KnowledgeRow, "section" | "tags">): string {
  if (row.section === "sample") return row.tags.sample ?? "Sample";
  if (row.section === "playbook") return row.tags.playbook ?? "Playbook";
  return row.section === "instrument" ? "Instrument" : "Past run";
}

const reviewing = (row: { status: string; resolved?: unknown }) =>
  row.status === "to review" && !row.resolved;

/** The folder a sample or playbook entry sits in, as the routes ask for it. */
function group(row: Pick<KnowledgeRow, "section" | "tags">): string {
  if (row.section === "sample") return `?sample=${encodeURIComponent(row.tags.sample ?? "")}`;
  if (row.section === "playbook") return `?playbook=${encodeURIComponent(row.tags.playbook ?? "")}`;
  return "";
}

const key = (row: KnowledgeRow) => `${row.section}/${group(row)}/${row.id}`;

const entryUrl = (row: KnowledgeRow, rest = "") =>
  `/app/knowledge/${row.section}/${encodeURIComponent(row.id)}${rest}${group(row)}`;

function shows(row: KnowledgeRow, kind: Kind): boolean {
  if (kind === "resolved") return !!row.resolved;
  if (row.resolved) return false;
  if (kind === "review") return row.status === "to review";
  return kind === "all" || row.lasts === kind;
}

function KindFilter({ kind, onChange }: { kind: Kind; onChange: (kind: Kind) => void }) {
  const label = SHOW.find(([k]) => k === kind)![1];
  return (
    <Popup
      label="Show"
      align="right"
      className="top-[calc(100%+4px)] bottom-auto w-40"
      trigger={(open, toggle) => (
        <button
          className="flex shrink-0 cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground"
          aria-haspopup="menu"
          aria-expanded={open}
          onClick={toggle}
        >
          <ListFilter size={14} />
          {label}
        </button>
      )}
    >
      {(close) => (
        <div role="menu" className="flex flex-col">
          {SHOW.map(([k, word]) => (
            <button
              key={k}
              role="menuitemradio"
              aria-checked={k === kind}
              className={cn(
                "cursor-pointer rounded border-0 bg-transparent px-2 py-1.5 text-left text-[13px] text-foreground hover:bg-accent",
                k === kind && "bg-accent",
              )}
              onClick={() => {
                onChange(k);
                close();
              }}
            >
              {word}
            </button>
          ))}
        </div>
      )}
    </Popup>
  );
}

function EntryImage({
  src,
  name,
  file,
}: {
  src: (name: string) => Promise<string>;
  name: string;
  file: string;
}) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    src(file).then((u) => live && setUrl(u), () => undefined);
    return () => {
      live = false;
    };
  }, [src, file]);
  return url ? (
    <img src={url} alt={name} title={name} className="max-h-52 max-w-full rounded" />
  ) : (
    <div className="h-40 w-60 rounded bg-accent" />
  );
}

function Editor({
  entry,
  onSave,
  onCancel,
  busy,
}: {
  entry: KnowledgeEntry;
  onSave: (title: string, text: string) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [title, setTitle] = useState(entry.title);
  const [text, setText] = useState(entry.text);
  return (
    <div className="flex flex-1 flex-col gap-2.5">
      <input
        aria-label="Title"
        className="rounded border border-input bg-background px-2 py-1.5 text-[14px] font-semibold"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        aria-label="Text"
        rows={10}
        className="min-h-40 flex-1 resize-y rounded border border-input bg-background px-2 py-1.5 text-[13px] leading-[1.5]"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex justify-end gap-1">
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button variant="primary" disabled={busy || !title.trim()} onClick={() => onSave(title.trim(), text)}>
          Save
        </Button>
      </div>
    </div>
  );
}

function Entry({
  entry,
  image,
  onSave,
  onDelete,
  onKeep,
  onResolve,
  busy,
}: {
  entry: KnowledgeEntry;
  image: (name: string) => Promise<string>;
  onSave: (title: string, text: string) => void;
  onDelete: () => void;
  onKeep: () => void;
  onResolve: () => void;
  busy: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  useEffect(() => {
    setEditing(false);
    setDeleting(false);
  }, [entry.path]);
  const meta = [
    kindOf(entry),
    entry.lasts === "until fixed" && (entry.resolved ? `fixed ${day(entry.resolved)}` : "until fixed"),
    `${entry.by === "agent" ? "agent" : "operator"}, ${day(entry.written)}`,
  ].filter(Boolean);
  const from = [entry.tags.session, entry.source?.item].filter(Boolean).join(", ");
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto px-6 py-5">
      {editing ? (
        <Editor
          entry={entry}
          busy={busy}
          onCancel={() => setEditing(false)}
          onSave={(title, text) => {
            onSave(title, text);
            setEditing(false);
          }}
        />
      ) : (
        <>
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <h3 className="m-0 text-[15px] font-semibold" title={entry.path}>
                {entry.title}
              </h3>
              <p className="m-0 mt-0.5 text-[12px] text-muted-foreground">{meta.join(" · ")}</p>
            </div>
            {deleting ? (
              <span className="flex shrink-0 gap-1">
                <Button variant="danger" className="py-1" disabled={busy} onClick={onDelete}>
                  Delete
                </Button>
                <Button variant="ghost" className="py-1" onClick={() => setDeleting(false)}>
                  Cancel
                </Button>
              </span>
            ) : (
              <span className="flex shrink-0 items-center text-muted-foreground">
                {reviewing(entry) && (
                  <Button
                    variant="ghost"
                    className="py-1 text-foreground"
                    disabled={busy}
                    title="The agent already uses it. Keep it to say it's right."
                    onClick={onKeep}
                  >
                    Keep
                  </Button>
                )}
                {entry.lasts === "until fixed" && !entry.resolved && (
                  <Button variant="ghost" className="py-1" disabled={busy} onClick={onResolve}>
                    Mark fixed
                  </Button>
                )}
                <Button variant="ghost" className="px-2" aria-label="Edit" title="Edit" onClick={() => setEditing(true)}>
                  <Pencil size={14} />
                </Button>
                <Button variant="ghost" className="px-2" aria-label="Delete" title="Delete" onClick={() => setDeleting(true)}>
                  <Trash2 size={14} />
                </Button>
              </span>
            )}
          </div>
          {entry.remind && (
            <p className="m-0 mt-3 flex items-center gap-1.5 text-[13px] text-muted-foreground">
              <Bell size={13} className="shrink-0" />
              {entry.remind}
            </p>
          )}
          <div className="mt-3 text-[13px] leading-[1.6]">
            <Markdown text={entry.text} />
          </div>
          {entry.images.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-3">
              {entry.images.map((name) => (
                <EntryImage key={name} src={image} name={entry.source?.images?.[name] ?? name} file={name} />
              ))}
            </div>
          )}
          {from && <p className="m-0 mt-4 text-[12px] text-muted-foreground">From {from}</p>}
        </>
      )}
    </div>
  );
}

/** What the operator writes as a new entry. */
export interface NewEntryFields {
  section: KnowledgeSection;
  sample?: string;
  playbook?: string;
  lasts: KnowledgeLasts;
  title: string;
  text: string;
}

const ABOUT: [KnowledgeSection, string][] = [
  ["instrument", "This microscope"],
  ["sample", "A sample"],
  ["playbook", "A playbook"],
];

function NewEntry({
  samples,
  playbooks,
  busy,
  onSave,
  onCancel,
}: {
  samples: string[];
  playbooks: string[];
  busy: boolean;
  onSave: (fields: NewEntryFields) => void;
  onCancel: () => void;
}) {
  const [section, setSection] = useState<KnowledgeSection>("instrument");
  const [group, setGroup] = useState("");
  const [lasts, setLasts] = useState<KnowledgeLasts>("ongoing");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const input = "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
  const named = section === "instrument" || group.trim();
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2.5 overflow-y-auto px-6 py-5">
      <h3 className="m-0 text-[15px] font-semibold">New entry</h3>
      <div className="flex flex-wrap gap-2">
        <select
          aria-label="About"
          className={input}
          value={section}
          onChange={(e) => {
            setSection(e.target.value as KnowledgeSection);
            setGroup("");
          }}
        >
          {ABOUT.map(([s, word]) => (
            <option key={s} value={s}>
              {word}
            </option>
          ))}
        </select>
        {section !== "instrument" && (
          <>
            <input
              aria-label={section === "sample" ? "Sample" : "Playbook"}
              list="knowledge-new-group"
              placeholder={section === "sample" ? "Sample" : "Playbook"}
              className={cn(input, "w-48")}
              value={group}
              onChange={(e) => setGroup(e.target.value)}
            />
            <datalist id="knowledge-new-group">
              {(section === "sample" ? samples : playbooks).map((g) => (
                <option key={g} value={g} />
              ))}
            </datalist>
          </>
        )}
        <select aria-label="Lasts" className={input} value={lasts} onChange={(e) => setLasts(e.target.value as KnowledgeLasts)}>
          <option value="ongoing">Ongoing</option>
          <option value="until fixed">Until fixed</option>
        </select>
      </div>
      <input
        aria-label="Title"
        placeholder="Title"
        className={cn(input, "font-semibold")}
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        aria-label="Text"
        rows={10}
        placeholder="What the agent should know"
        className={cn(input, "min-h-40 flex-1 resize-y leading-[1.5]")}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex justify-end gap-1">
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          variant="primary"
          disabled={busy || !title.trim() || !named}
          onClick={() =>
            onSave({
              section,
              sample: section === "sample" ? group.trim() : undefined,
              playbook: section === "playbook" ? group.trim() : undefined,
              lasts,
              title: title.trim(),
              text,
            })
          }
        >
          Save
        </Button>
      </div>
    </div>
  );
}

/** The knowledge base: a filtered list on the left, the open entry on the right. */
export function KnowledgeBrowser({
  rows,
  entry,
  selected,
  onSelect,
  image,
  onSave,
  onDelete,
  onKeep,
  onResolve,
  busy = false,
  adding = null,
  onNew,
}: {
  rows: KnowledgeRow[];
  entry: KnowledgeEntry | null;
  selected: string | null;
  onSelect: (row: KnowledgeRow) => void;
  image: (name: string) => Promise<string>;
  onSave: (title: string, text: string) => void;
  onDelete: () => void;
  onKeep: () => void;
  onResolve: () => void;
  busy?: boolean;
  /** The new entry's form, in place of the open entry. */
  adding?: ReactNode;
  onNew?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<Kind>("all");
  const words = query.trim().toLowerCase();
  const matching = rows.filter(
    (row) =>
      shows(row, kind) &&
      (!words || `${row.title} ${Object.values(row.tags).join(" ")}`.toLowerCase().includes(words)),
  );
  // To review first, newest first within each
  const shown = [...matching.filter(reviewing), ...matching.filter((row) => !reviewing(row))];
  return (
    <div className="flex h-[min(560px,76vh)] min-h-0">
      <div className="flex w-[280px] shrink-0 flex-col border-r border-border">
        <div className="flex items-center gap-1 border-b border-border px-2.5 py-2">
          <label className="flex min-w-0 flex-1 items-center gap-1.5 rounded px-1.5 text-muted-foreground focus-within:bg-accent">
            <Search size={14} className="shrink-0" />
            <input
              aria-label="Filter"
              placeholder="Filter"
              className="min-w-0 flex-1 border-0 bg-transparent py-1 text-[13px] text-foreground outline-none placeholder:text-muted-foreground"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <KindFilter kind={kind} onChange={setKind} />
          {onNew && (
            <Button variant="ghost" className="shrink-0 px-2 py-1 text-[12px]" title="New entry" onClick={onNew}>
              <Plus size={14} />
              New
            </Button>
          )}
        </div>
        <ul className="m-0 min-h-0 flex-1 list-none overflow-y-auto p-1.5">
          {shown.map((row) => (
            <li key={key(row)}>
              <button
                className={cn(
                  "flex w-full cursor-pointer items-baseline gap-2 rounded border-0 bg-transparent px-2.5 py-2 text-left hover:bg-accent",
                  selected === key(row) && "bg-accent",
                )}
                title={reviewing(row) ? "To review" : undefined}
                onClick={() => onSelect(row)}
              >
                <span
                  className={cn(
                    "size-1.5 shrink-0 translate-y-[-2px] rounded-full",
                    reviewing(row) ? "bg-primary" : "bg-transparent",
                  )}
                />
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className={cn("line-clamp-2 text-[13px]", row.resolved ? "text-muted-foreground" : "text-foreground")}>
                    {row.title}
                  </span>
                  <span className="truncate text-[12px] text-muted-foreground">
                    {row.ask_again ? "Is this still true?" : `${kindOf(row)} · ${day(row.written)}`}
                  </span>
                </span>
              </button>
            </li>
          ))}
          {!shown.length && (
            <li className="px-2.5 py-2 text-[13px] text-muted-foreground">
              {rows.length ? "Nothing matches." : "Nothing yet. The agent adds what it learns here."}
            </li>
          )}
        </ul>
      </div>
      {adding ? (
        adding
      ) : entry ? (
        <Entry
          entry={entry}
          image={image}
          onSave={onSave}
          onDelete={onDelete}
          onKeep={onKeep}
          onResolve={onResolve}
          busy={busy}
        />
      ) : (
        <div className="flex flex-1 items-center justify-center text-[13px] text-muted-foreground">
          {rows.length ? "Pick an entry." : null}
        </div>
      )}
    </div>
  );
}

type Tab = "entries" | LibraryKind;
const TABS: [Tab, string][] = [
  ["entries", "Entries"],
  ["profiles", "Profiles"],
  ["playbooks", "Playbooks"],
  ["skills", "Skills"],
];

/** The Knowledge window: what the agent and the operator have recorded about
 * this microscope, its samples and playbooks, and past runs; the agent's own
 * entries to review first; and the agent's profiles, playbooks and skills,
 * the shipped and the lab's own. */
export function KnowledgeDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const client = useQueryClient();
  const [tab, setTab] = useState<Tab>("entries");
  const index = useQuery({
    queryKey: ["knowledge"],
    queryFn: () =>
      api<{ entries: KnowledgeRow[]; samples: string[]; playbooks: string[] }>("/app/knowledge?resolved=true"),
    enabled: open,
  });
  const [row, setRow] = useState<KnowledgeRow | null>(null);
  const entry = useQuery({
    queryKey: ["knowledge", row && key(row)],
    queryFn: () => api<{ entry: KnowledgeEntry }>(entryUrl(row!)).then((answer) => answer.entry),
    enabled: open && row !== null,
  });
  const done = () => client.invalidateQueries({ queryKey: ["knowledge"] });
  const save = useMutation({
    mutationFn: ({ title, text }: { title: string; text: string }) =>
      api(`/app/knowledge`, {
        method: "POST",
        body: JSON.stringify({
          section: row!.section,
          sample: row!.tags.sample,
          playbook: row!.tags.playbook,
          id: row!.id,
          title,
          text,
        }),
      }),
    onSuccess: done,
  });
  const mark = useMutation({
    mutationFn: (what: "keep" | "resolve") => api(entryUrl(row!, `/${what}`), { method: "POST" }),
    onSuccess: done,
  });
  const [adding, setAdding] = useState(false);
  const add = useMutation({
    mutationFn: (fields: NewEntryFields) =>
      api<{ entry: KnowledgeEntry }>(`/app/knowledge`, { method: "POST", body: JSON.stringify(fields) }),
    onSuccess: ({ entry }) => {
      setAdding(false);
      setRow({ ...entry, images: entry.images.length, resolved: !!entry.resolved });
      return done();
    },
  });
  const remove = useMutation({
    mutationFn: () => api(entryUrl(row!), { method: "DELETE" }),
    onSuccess: () => {
      setRow(null);
      return done();
    },
  });
  const image = (name: string) => imageUrl(entryUrl(row!, `/images/${encodeURIComponent(name)}`));
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Knowledge" className="w-[min(960px,calc(100vw-32px))]">
        <div role="tablist" className="flex gap-4 border-b border-border px-4">
          {TABS.map(([t, word]) => (
            <button
              key={t}
              role="tab"
              aria-selected={t === tab}
              className={cn(
                "-mb-px cursor-pointer border-0 border-b-2 bg-transparent px-0.5 py-2 text-[13px]",
                t === tab ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
              )}
              onClick={() => setTab(t)}
            >
              {word}
            </button>
          ))}
        </div>
        {tab === "entries" ? (
          <>
            <KnowledgeBrowser
              rows={index.data?.entries ?? []}
              entry={entry.data ?? null}
              selected={row && key(row)}
              onSelect={(r) => {
                setAdding(false);
                setRow(r);
              }}
              onNew={() => {
                add.reset();
                setAdding(true);
              }}
              adding={
                adding && (
                  <NewEntry
                    samples={index.data?.samples ?? []}
                    playbooks={index.data?.playbooks ?? []}
                    busy={add.isPending}
                    onSave={(fields) => add.mutate(fields)}
                    onCancel={() => setAdding(false)}
                  />
                )
              }
              image={image}
              busy={save.isPending || remove.isPending || mark.isPending}
              onSave={(title, text) => save.mutate({ title, text })}
              onDelete={() => remove.mutate()}
              onKeep={() => mark.mutate("keep")}
              onResolve={() => mark.mutate("resolve")}
            />
            {(index.error || save.error || remove.error || mark.error || add.error) && (
              <p className="m-0 border-t border-border px-4 py-2 text-[13px] text-destructive">
                {((index.error || save.error || remove.error || mark.error || add.error) as Error).message}
              </p>
            )}
          </>
        ) : (
          <LibraryBrowser kind={tab} enabled={open} />
        )}
      </DialogContent>
    </Dialog>
  );
}
