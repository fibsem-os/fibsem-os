import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";
import { type Data, type LibraryOptions, PlaybookSettings, ProfileSettings, SubagentSettings } from "./Settings";

export type LibraryKind = "profiles" | "playbooks" | "skills" | "agents";

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
  /** A profile's, playbook's or subagent's settings; null for a skill. */
  data: Data | null;
}

/** What a save sends: the form's settings, or the file's text. */
export type Saved = { data: Data } | { text: string };

const ONE: Record<LibraryKind, string> = {
  profiles: "profile",
  playbooks: "playbook",
  skills: "skill",
  agents: "subagent",
};
const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;

/** A skill's front matter and body, split. */
function skillParts(text: string): { front: string; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text);
  return match ? { front: match[1], body: match[2] } : { front: "", body: text };
}

function FileView({
  kind,
  item,
  file,
  options,
  fresh,
  busy,
  onSave,
  onRemove,
}: {
  kind: LibraryKind;
  item: LibraryItem;
  file: LibraryFile;
  options?: LibraryOptions;
  /** Just made: open in the editor. */
  fresh: boolean;
  busy: boolean;
  onSave: (saved: Saved) => void;
  onRemove: () => void;
}) {
  const form = kind !== "skills" && file.data !== null;
  const [editing, setEditing] = useState(fresh);
  const [asText, setAsText] = useState(!form);
  const [removing, setRemoving] = useState(false);
  const [text, setText] = useState(file.text);
  const [data, setData] = useState<Data>(file.data ?? {});
  useEffect(() => {
    setEditing(fresh);
    setAsText(!form);
    setRemoving(false);
    setText(file.text);
    setData(file.data ?? {});
  }, [file.path, file.text, file.data, fresh, form]);
  const skill = kind === "skills" ? skillParts(file.text) : null;
  const description = typeof data.description === "string" ? data.description : item.description;
  const settings = (onChange?: (d: Data) => void) =>
    kind === "profiles" ? (
      <ProfileSettings data={data} options={options} onChange={onChange} />
    ) : kind === "agents" ? (
      <SubagentSettings data={data} options={options} onChange={onChange} />
    ) : (
      <PlaybookSettings name={file.name} data={data} options={options} onChange={onChange} />
    );
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto px-6 py-5">
      <div className="flex items-start gap-2">
        <h3 className="m-0 min-w-0 flex-1 text-[15px] font-semibold" title={file.path}>
          {file.name}
        </h3>
        {editing ? null : removing ? (
          <span className="flex shrink-0 gap-1">
            <Button variant="danger" className="py-1" disabled={busy} onClick={onRemove}>
              {item.shipped ? "Use shipped" : "Delete"}
            </Button>
            <Button variant="ghost" className="py-1" onClick={() => setRemoving(false)}>
              Cancel
            </Button>
          </span>
        ) : (
          <span className="flex shrink-0 items-center text-muted-foreground">
            {item.shipped && item.lab && (
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
            {!item.shipped && item.lab && (
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
          {asText ? (
            <textarea
              aria-label="Text"
              spellCheck={false}
              className="min-h-72 flex-1 resize-y rounded border border-input bg-background px-2.5 py-2 font-mono text-[12px] leading-[1.55]"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          ) : (
            settings(setData)
          )}
          <div className="flex items-center justify-end gap-1">
            {form && (
              <Button
                variant="ghost"
                className="mr-auto px-1.5 text-[12px]"
                onClick={() => {
                  setText(file.text);
                  setData(file.data ?? {});
                  setAsText(!asText);
                }}
              >
                {asText ? "Edit as form" : "Edit as text"}
              </Button>
            )}
            <Button
              variant="ghost"
              onClick={() => {
                setEditing(false);
                setText(file.text);
                setData(file.data ?? {});
              }}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={busy || (asText && (!text.trim() || text === file.text))}
              onClick={() => onSave(asText ? { text } : { data })}
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
      ) : form ? (
        <>
          {description && <p className="m-0 mt-2 mb-2 text-[13px] text-muted-foreground">{description}</p>}
          {settings()}
        </>
      ) : (
        <pre className="m-0 mt-3 overflow-x-auto rounded bg-accent px-3 py-2.5 font-mono text-[12px] leading-[1.55] whitespace-pre-wrap">
          {file.text}
        </pre>
      )}
    </div>
  );
}

/** The profiles, playbooks, skills or subagents: what fibsemOS ships and the lab's own,
 * on the left; the open one on the right. Editing a shipped one saves the
 * lab's copy. */
export function LibraryBrowser({ kind, enabled }: { kind: LibraryKind; enabled: boolean }) {
  const client = useQueryClient();
  const index = useQuery({
    queryKey: ["library", kind],
    queryFn: () => api<{ items: LibraryItem[] }>(`/app/library/${kind}`).then((answer) => answer.items),
    enabled,
  });
  const options = useQuery({
    queryKey: ["library", "options"],
    queryFn: () => api<LibraryOptions>("/app/library/options"),
    enabled: enabled && kind !== "skills",
  });
  const [name, setName] = useState<string | null>(null);
  const [fresh, setFresh] = useState<string | null>(null);
  const [naming, setNaming] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    setName(null);
    setNaming(null);
    setQuery("");
  }, [kind]);
  const url = (n = name!) => `/app/library/${kind}/${encodeURIComponent(n)}`;
  const file = useQuery({
    queryKey: ["library", kind, name],
    queryFn: () => api<LibraryFile>(url()),
    enabled: enabled && name !== null,
  });
  const done = () => client.invalidateQueries({ queryKey: ["library"] });
  const save = useMutation({
    mutationFn: (saved: Saved) => api(url(), { method: "PUT", body: JSON.stringify(saved) }),
    onSuccess: () => {
      setFresh(null);
      return done();
    },
  });
  const create = useMutation({
    mutationFn: (n: string) => api<LibraryFile>(url(n), { method: "POST" }),
    onSuccess: (made) => {
      setNaming(null);
      setFresh(made.name);
      setName(made.name);
      return done();
    },
  });
  const remove = useMutation({
    mutationFn: () => api<LibraryItem>(url(), { method: "DELETE" }),
    onSuccess: (item) => {
      if (!item.shipped) setName(null);
      return done();
    },
  });
  const reset = () => {
    save.reset();
    create.reset();
    remove.reset();
  };
  const items = index.data ?? [];
  const words = query.trim().toLowerCase();
  const shown = items.filter(
    (item) => !words || `${item.name} ${item.description ?? ""}`.toLowerCase().includes(words),
  );
  const item = items.find((i) => i.name === name) ?? null;
  const error = index.error || file.error || save.error || create.error || remove.error;
  const nameOk = naming !== null && NAME.test(naming) && !naming.endsWith(".");
  return (
    <div className="flex min-h-0 flex-col">
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
            <Button
              variant="ghost"
              className="shrink-0 px-2 py-1 text-[12px]"
              title={`New ${ONE[kind]}`}
              onClick={() => {
                reset();
                setNaming("");
              }}
            >
              <Plus size={14} />
              New
            </Button>
          </div>
          {naming !== null && (
            <form
              className="flex flex-col gap-1.5 border-b border-border px-2.5 py-2.5"
              onSubmit={(e) => {
                e.preventDefault();
                if (nameOk) create.mutate(naming);
              }}
            >
              <input
                autoFocus
                aria-label="Name"
                placeholder={`New ${ONE[kind]}'s name`}
                className="rounded border border-input bg-background px-2 py-1 text-[13px]"
                value={naming}
                onChange={(e) => setNaming(e.target.value)}
                onKeyDown={(e) => e.key === "Escape" && setNaming(null)}
              />
              <span className="flex justify-end gap-1">
                <Button variant="ghost" className="py-1" type="button" onClick={() => setNaming(null)}>
                  Cancel
                </Button>
                <Button variant="primary" className="py-1" type="submit" disabled={!nameOk || create.isPending}>
                  Make
                </Button>
              </span>
            </form>
          )}
          <ul className="m-0 min-h-0 flex-1 list-none overflow-y-auto p-1.5">
            {shown.map((row) => (
              <li key={row.name}>
                <button
                  className={cn(
                    "flex w-full cursor-pointer flex-col gap-0.5 rounded border-0 bg-transparent px-2.5 py-2 text-left hover:bg-accent",
                    name === row.name && "bg-accent",
                  )}
                  onClick={() => {
                    reset();
                    setFresh(null);
                    setName(row.name);
                  }}
                >
                  <span className="truncate text-[13px] text-foreground">{row.name}</span>
                  {row.description && (
                    <span className="line-clamp-2 text-[12px] text-muted-foreground">{row.description}</span>
                  )}
                </button>
              </li>
            ))}
            {index.isSuccess && !shown.length && (
              <li className="px-2.5 py-2 text-[13px] text-muted-foreground">
                {items.length ? "Nothing matches." : "Nothing yet."}
              </li>
            )}
          </ul>
        </div>
        {item && file.data ? (
          <FileView
            kind={kind}
            item={item}
            file={file.data}
            options={options.data}
            fresh={fresh === item.name}
            busy={save.isPending || remove.isPending}
            onSave={(saved) => save.mutate(saved)}
            onRemove={() => remove.mutate()}
          />
        ) : (
          <div className="flex-1" />
        )}
      </div>
      {error && (
        <p className="m-0 border-t border-border px-4 py-2 text-[13px] text-destructive">{(error as Error).message}</p>
      )}
    </div>
  );
}
