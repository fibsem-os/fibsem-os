import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";

export type LibraryKind = "profiles" | "playbooks" | "skills";

export interface LibraryItem {
  name: string;
  description: string | null;
  /** The lab has its own; it is the one the agent uses. */
  lab: boolean;
  /** fibsemOS ships one by that name. */
  shipped: boolean;
}

export interface LibraryFile {
  name: string;
  source: "lab" | "shipped";
  text: string;
  path: string;
  lab: boolean;
  shipped: boolean;
}

const EMPTY: Record<LibraryKind, string> = {
  profiles: "No profiles.",
  playbooks: "No playbooks.",
  skills: "No skills.",
};

function origin(item: Pick<LibraryItem, "lab" | "shipped">): string {
  if (item.lab && item.shipped) return "Lab's copy";
  return item.lab ? "Lab's own" : "Shipped";
}

/** A skill's front matter and body, split. */
function skillParts(text: string): { front: string; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text);
  return match ? { front: match[1], body: match[2] } : { front: "", body: text };
}

function FileView({
  kind,
  item,
  file,
  busy,
  onCopy,
  onSave,
  onRemove,
}: {
  kind: LibraryKind;
  item: LibraryItem;
  file: LibraryFile;
  busy: boolean;
  onCopy: () => void;
  onSave: (text: string) => void;
  onRemove: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [text, setText] = useState(file.text);
  useEffect(() => {
    setEditing(false);
    setRemoving(false);
    setText(file.text);
  }, [file.path, file.text]);
  const meta =
    file.source === "shipped"
      ? "Shipped with fibsemOS. Copy it to make the lab's own."
      : item.shipped
        ? "The lab's copy, used in place of the shipped one."
        : "The lab's own.";
  const skill = kind === "skills" ? skillParts(file.text) : null;
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto px-6 py-5">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="m-0 text-[15px] font-semibold" title={file.path}>
            {file.name}
          </h3>
          <p className="m-0 mt-0.5 text-[12px] text-muted-foreground">{meta}</p>
        </div>
        {editing ? null : removing ? (
          <span className="flex shrink-0 gap-1">
            <Button variant="danger" className="py-1" disabled={busy} onClick={onRemove}>
              {item.shipped ? "Use shipped" : "Delete"}
            </Button>
            <Button variant="ghost" className="py-1" onClick={() => setRemoving(false)}>
              Cancel
            </Button>
          </span>
        ) : file.source === "shipped" ? (
          <Button variant="ghost" className="shrink-0 py-1 text-foreground" disabled={busy} onClick={onCopy}>
            Copy to edit
          </Button>
        ) : (
          <span className="flex shrink-0 items-center text-muted-foreground">
            {item.shipped && (
              <Button
                variant="ghost"
                className="py-1"
                title="Remove the lab's copy and use the shipped one again"
                onClick={() => setRemoving(true)}
              >
                Use shipped
              </Button>
            )}
            <Button variant="ghost" className="px-2" aria-label="Edit" title="Edit" onClick={() => setEditing(true)}>
              <Pencil size={14} />
            </Button>
            {!item.shipped && (
              <Button
                variant="ghost"
                className="px-2"
                aria-label="Delete"
                title="Delete"
                onClick={() => setRemoving(true)}
              >
                <Trash2 size={14} />
              </Button>
            )}
          </span>
        )}
      </div>
      {editing ? (
        <div className="mt-3 flex flex-1 flex-col gap-2.5">
          <textarea
            aria-label="Text"
            spellCheck={false}
            className="min-h-72 flex-1 resize-y rounded border border-input bg-background px-2.5 py-2 font-mono text-[12px] leading-[1.55]"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="flex items-center justify-end gap-1">
            <span className="mr-auto text-[12px] text-muted-foreground">The agent uses it from its next start.</span>
            <Button variant="ghost" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={busy || !text.trim() || text === file.text}
              onClick={() => onSave(text)}
            >
              Save
            </Button>
          </div>
        </div>
      ) : skill ? (
        <>
          {item.description && <p className="m-0 mt-3 text-[13px] text-muted-foreground">{item.description}</p>}
          <div className="mt-3 text-[13px] leading-[1.6]">
            <Markdown text={skill.body.replace(/^\s*# .*\n/, "")} />
          </div>
        </>
      ) : (
        <>
          {item.description && <p className="m-0 mt-3 text-[13px] text-muted-foreground">{item.description}</p>}
          <pre className="m-0 mt-3 overflow-x-auto rounded bg-accent px-3 py-2.5 font-mono text-[12px] leading-[1.55] whitespace-pre-wrap">
            {file.text}
          </pre>
        </>
      )}
    </div>
  );
}

/** The profiles, playbooks or skills: what fibsemOS ships and the lab's own,
 * on the left; the open one's file on the right. */
export function LibraryBrowser({ kind, enabled }: { kind: LibraryKind; enabled: boolean }) {
  const client = useQueryClient();
  const index = useQuery({
    queryKey: ["library", kind],
    queryFn: () => api<{ items: LibraryItem[] }>(`/app/library/${kind}`).then((answer) => answer.items),
    enabled,
  });
  const [name, setName] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    setName(null);
    setQuery("");
  }, [kind]);
  const url = (rest = "") => `/app/library/${kind}/${encodeURIComponent(name!)}${rest}`;
  const file = useQuery({
    queryKey: ["library", kind, name],
    queryFn: () => api<LibraryFile>(url()),
    enabled: enabled && name !== null,
  });
  const done = () => client.invalidateQueries({ queryKey: ["library", kind] });
  const copy = useMutation({
    mutationFn: () => api(url("/copy"), { method: "POST" }),
    onSuccess: done,
  });
  const save = useMutation({
    mutationFn: (text: string) => api(url(), { method: "PUT", body: JSON.stringify({ text }) }),
    onSuccess: done,
  });
  const remove = useMutation({
    mutationFn: () => api<LibraryItem>(url(), { method: "DELETE" }),
    onSuccess: (item) => {
      if (!item.shipped) setName(null);
      return done();
    },
  });
  const items = index.data ?? [];
  const words = query.trim().toLowerCase();
  const shown = items.filter(
    (item) => !words || `${item.name} ${item.description ?? ""}`.toLowerCase().includes(words),
  );
  const item = items.find((i) => i.name === name) ?? null;
  const error = index.error || file.error || copy.error || save.error || remove.error;
  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex h-[min(560px,76vh)] min-h-0">
        <div className="flex w-[280px] shrink-0 flex-col border-r border-border">
          <div className="flex items-center border-b border-border px-2.5 py-2">
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
          </div>
          <ul className="m-0 min-h-0 flex-1 list-none overflow-y-auto p-1.5">
            {shown.map((row) => (
              <li key={row.name}>
                <button
                  className={cn(
                    "flex w-full cursor-pointer flex-col gap-0.5 rounded border-0 bg-transparent px-2.5 py-2 text-left hover:bg-accent",
                    name === row.name && "bg-accent",
                  )}
                  onClick={() => {
                    copy.reset();
                    save.reset();
                    remove.reset();
                    setName(row.name);
                  }}
                >
                  <span className="flex w-full items-baseline gap-2">
                    <span className="min-w-0 flex-1 truncate text-[13px] text-foreground">{row.name}</span>
                    {row.lab && <span className="shrink-0 text-[11px] text-muted-foreground">{origin(row)}</span>}
                  </span>
                  {row.description && (
                    <span className="line-clamp-2 text-[12px] text-muted-foreground">{row.description}</span>
                  )}
                </button>
              </li>
            ))}
            {index.isSuccess && !shown.length && (
              <li className="px-2.5 py-2 text-[13px] text-muted-foreground">
                {items.length ? "Nothing matches." : EMPTY[kind]}
              </li>
            )}
          </ul>
        </div>
        {item && file.data ? (
          <FileView
            kind={kind}
            item={item}
            file={file.data}
            busy={copy.isPending || save.isPending || remove.isPending}
            onCopy={() => copy.mutate()}
            onSave={(text) => save.mutate(text)}
            onRemove={() => remove.mutate()}
          />
        ) : (
          <div className="flex flex-1 items-center justify-center text-[13px] text-muted-foreground">
            {items.length ? `Pick one. The lab's own are used in place of the shipped ones.` : null}
          </div>
        )}
      </div>
      {error && (
        <p className="m-0 border-t border-border px-4 py-2 text-[13px] text-destructive">{(error as Error).message}</p>
      )}
    </div>
  );
}
