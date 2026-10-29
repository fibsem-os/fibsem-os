import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ListFilter, Pencil, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Popup } from "@/components/ui/popup";
import { api, imageUrl } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";

export type KnowledgeSection = "instrument" | "sample" | "experiment";

/** One entry in the knowledge index. */
export interface KnowledgeRow {
  section: KnowledgeSection;
  id: string;
  title: string;
  written: string;
  by: "agent" | "operator";
  /** The sample's slug, for a sample entry. */
  sample?: string | null;
  edited?: boolean;
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
  by: "agent" | "operator";
  sample?: string | null;
  /** The experiment it was learned in, if any. */
  experiment?: string | null;
  /** Its image files, by name. */
  images: string[];
  edited?: boolean;
  edited_by?: "agent" | "operator" | null;
  /** Its file, as the operator would find it. */
  path: string;
}

const KINDS = [
  ["all", "Everything"],
  ["instrument", "Instrument"],
  ["sample", "Sample"],
  ["experiment", "Experiment"],
] as const;
type Kind = (typeof KINDS)[number][0];

const BY = { agent: "the agent", operator: "the operator" };

const day = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });

function kindOf(row: Pick<KnowledgeRow, "section" | "sample">): string {
  if (row.section === "sample") return row.sample ? `Sample: ${row.sample}` : "Sample";
  return row.section === "instrument" ? "Instrument" : "Experiment";
}

const key = (row: KnowledgeRow) => `${row.section}/${row.sample ?? ""}/${row.id}`;

function entryPath(row: KnowledgeRow): string {
  const id = encodeURIComponent(row.id);
  const sample = row.sample ? `?sample=${encodeURIComponent(row.sample)}` : "";
  return `/app/knowledge/${row.section}/${id}${sample}`;
}

function KindFilter({ kind, onChange }: { kind: Kind; onChange: (kind: Kind) => void }) {
  const label = KINDS.find(([k]) => k === kind)![1];
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
          {KINDS.map(([k, word]) => (
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

function EntryImage({ src, name }: { src: (name: string) => Promise<string>; name: string }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    src(name).then((u) => live && setUrl(u), () => undefined);
    return () => {
      live = false;
    };
  }, [src, name]);
  return url ? (
    <img src={url} alt={name} className="max-h-64 max-w-full rounded border border-border" />
  ) : (
    <div className="h-40 w-60 rounded border border-border bg-accent" />
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
  busy,
}: {
  entry: KnowledgeEntry;
  image: (name: string) => Promise<string>;
  onSave: (title: string, text: string) => void;
  onDelete: () => void;
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
    `Written by ${BY[entry.by]}, ${day(entry.written)}`,
    entry.edited && `edited by ${BY[entry.edited_by ?? "operator"]}`,
    entry.experiment && `in ${entry.experiment}`,
  ].filter(Boolean);
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-border px-5">
        <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground" title={entry.path}>
          {entry.path}
        </span>
        {deleting ? (
          <span className="flex gap-1">
            <Button variant="danger" className="py-1" disabled={busy} onClick={onDelete}>
              Delete
            </Button>
            <Button variant="ghost" className="py-1" onClick={() => setDeleting(false)}>
              Cancel
            </Button>
          </span>
        ) : (
          !editing && (
            <span className="flex">
              <Button variant="ghost" className="px-2" aria-label="Edit" title="Edit" onClick={() => setEditing(true)}>
                <Pencil size={14} />
              </Button>
              <Button variant="ghost" className="px-2" aria-label="Delete" title="Delete" onClick={() => setDeleting(true)}>
                <Trash2 size={14} />
              </Button>
            </span>
          )
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-5 py-4">
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
            <h3 className="m-0 text-[15px] font-semibold">{entry.title}</h3>
            <p className="m-0 mt-1 text-[12px] text-muted-foreground">{meta.join(" · ")}</p>
            <div className="mt-3 text-[13px] leading-[1.55]">
              <Markdown text={entry.text} />
            </div>
            {entry.images.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-3">
                {entry.images.map((name) => (
                  <figure key={name} className="m-0 flex flex-col gap-1">
                    <EntryImage src={image} name={name} />
                    <figcaption className="text-[12px] text-muted-foreground">{name}</figcaption>
                  </figure>
                ))}
              </div>
            )}
          </>
        )}
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
  busy = false,
}: {
  rows: KnowledgeRow[];
  entry: KnowledgeEntry | null;
  selected: string | null;
  onSelect: (row: KnowledgeRow) => void;
  image: (name: string) => Promise<string>;
  onSave: (title: string, text: string) => void;
  onDelete: () => void;
  busy?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<Kind>("all");
  const words = query.trim().toLowerCase();
  const shown = rows.filter(
    (row) =>
      (kind === "all" || row.section === kind) &&
      (!words || `${row.title} ${row.sample ?? ""}`.toLowerCase().includes(words)),
  );
  return (
    <div className="flex h-[min(560px,76vh)] min-h-0">
      <div className="flex w-[290px] shrink-0 flex-col border-r border-border">
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
        </div>
        <ul className="m-0 min-h-0 flex-1 list-none overflow-y-auto p-1.5">
          {shown.map((row) => (
            <li key={key(row)}>
              <button
                className={cn(
                  "flex w-full cursor-pointer flex-col gap-0.5 rounded border-0 bg-transparent px-2.5 py-2 text-left hover:bg-accent",
                  selected === key(row) && "bg-accent",
                )}
                onClick={() => onSelect(row)}
              >
                <span className="text-[13px] text-foreground">{row.title}</span>
                <span className="text-[12px] text-muted-foreground">
                  {kindOf(row)} · {day(row.written)}
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
      {entry ? (
        <Entry entry={entry} image={image} onSave={onSave} onDelete={onDelete} busy={busy} />
      ) : (
        <div className="flex flex-1 items-center justify-center text-[13px] text-muted-foreground">
          {rows.length ? "Pick an entry." : null}
        </div>
      )}
    </div>
  );
}

/** The Knowledge window: what the agent and the operator have recorded about
 * this microscope and its samples, and the completed experiments' summaries. */
export function KnowledgeDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const client = useQueryClient();
  const index = useQuery({
    queryKey: ["knowledge"],
    queryFn: () => api<{ entries: KnowledgeRow[]; samples: string[] }>("/app/knowledge"),
    enabled: open,
  });
  const [row, setRow] = useState<KnowledgeRow | null>(null);
  const entry = useQuery({
    queryKey: ["knowledge", row && key(row)],
    queryFn: () => api<{ entry: KnowledgeEntry }>(entryPath(row!)).then((answer) => answer.entry),
    enabled: open && row !== null,
  });
  const done = () => client.invalidateQueries({ queryKey: ["knowledge"] });
  const save = useMutation({
    mutationFn: ({ title, text }: { title: string; text: string }) =>
      api(`/app/knowledge`, {
        method: "POST",
        body: JSON.stringify({ section: row!.section, sample: row!.sample, id: row!.id, title, text }),
      }),
    onSuccess: done,
  });
  const remove = useMutation({
    mutationFn: () => api(entryPath(row!), { method: "DELETE" }),
    onSuccess: () => {
      setRow(null);
      return done();
    },
  });
  const image = (name: string) => imageUrl(`${entryPath(row!).split("?")[0]}/images/${encodeURIComponent(name)}${row!.sample ? `?sample=${encodeURIComponent(row!.sample)}` : ""}`);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Knowledge" className="w-[min(960px,calc(100vw-32px))]">
        <KnowledgeBrowser
          rows={index.data?.entries ?? []}
          entry={entry.data ?? null}
          selected={row && key(row)}
          onSelect={setRow}
          image={image}
          busy={save.isPending || remove.isPending}
          onSave={(title, text) => save.mutate({ title, text })}
          onDelete={() => remove.mutate()}
        />
        {(index.error || save.error || remove.error) && (
          <p className="m-0 border-t border-border px-4 py-2 text-[13px] text-destructive">
            {((index.error || save.error || remove.error) as Error).message}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
