import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Filter } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
import { ImageViewer } from "@/components/images";
import { type MarkEditor, useMarkEditor } from "@/components/mark-for-agent";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import { type Mark, listMarks, toMarks } from "@/lib/marks";
import type { EvidenceAnswer, ReviewItem, ReviewList } from "@/lib/types";
import { cn } from "@/lib/utils";

type Show = "all" | "to_score" | ReviewItem["state"];
const SHOWS: { show: Show; name: string }[] = [
  { show: "all", name: "All" },
  { show: "waiting", name: "Waiting on you" },
  { show: "to_check", name: "To check" },
  { show: "to_score", name: "To score" },
  { show: "answered", name: "Answered" },
  { show: "done", name: "Done" },
];

const toScore = (i: ReviewItem) => !!i.score && !i.answer && !i.closed;
const shows = (i: ReviewItem, show: Show) => show === "all" || (show === "to_score" ? toScore(i) : i.state === show);

// What a fail asks the operator to mark, so the right answer is kept with it.
const MARK_ON_FAIL: Record<NonNullable<ReviewItem["score"]>, string> = {
  "site-selection": "Say why in the note, and mark a better square if there is one.",
  "site-coverage": "Mark the good squares it missed.",
  "site-setup": "Mark where the lamella should go.",
  target: "Mark the right target.",
  polishing: "Say why in the note: too thick, too thin, damaged.",
};

const KINDS: NonNullable<ReviewItem["score"]>[] = [
  "site-selection",
  "site-coverage",
  "site-setup",
  "target",
  "polishing",
];

// A grid's or lamella's name as the server compares it ("Lamella 01" is "lamella-01").
const same = (name?: string) => (name ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");

/** The agent's text, folded to a few lines when it is long. */
function Folded({ text }: { text: string }) {
  const [all, setAll] = useState(false);
  const long = text.length > 280;
  return (
    <div className="flex flex-col items-start gap-1">
      <p className={cn("m-0 leading-snug whitespace-pre-line", long && !all && "line-clamp-5")}>{text}</p>
      {long && (
        <button
          className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted-foreground underline hover:text-foreground"
          onClick={() => setAll(!all)}
        >
          {all ? "Less" : "More"}
        </button>
      )}
    </div>
  );
}

const KIND_NAME: Record<string, string> = {
  "site-selection": "Site",
  "site-coverage": "Coverage",
  "site-setup": "Setup",
  target: "Target",
  polishing: "Polishing",
};

type Result = "pass" | "fail" | "open" | "missing" | "closed" | "redo" | "none";
const RESULT_WORD: Record<Result, string> = {
  pass: "Pass",
  fail: "Fail",
  open: "To score",
  missing: "No item · Fail",
  closed: "Closed · Fail",
  redo: "After help",
  none: "",
};

/** A line of the scoring list: a review item, or a scored result with no item. */
type Line = { id: string; item?: ReviewItem; kind: string; name: string; grid: string; result: Result };

type Group = "item" | "type";
type ShowScores = "all" | "fail" | "open";

/** Scores: one modal to go through the run's results. The list on the left,
 * grouped by grid and lamella or by kind, each line with its result; the
 * picked item on the right, with its pictures, text and answer. */
function Scores({
  items,
  counts,
  onClose,
}: {
  items: ReviewItem[];
  counts: NonNullable<ReviewList["counts"]["scores"]>;
  onClose: () => void;
}) {
  const [group, setGroup] = useState<Group>("item");
  const [show, setShow] = useState<ShowScores>("all");
  const oldest = [...items].reverse();
  const nameOf = (i: ReviewItem) => i.item_name ?? i.about ?? "";
  // What an item scores, as the server groups it; its first answer is the score.
  const key = (i: ReviewItem) => `${i.score}:${i.item_id ?? same(i.about)}`;
  const firstAnswer = new Map<string, string>();
  const anyOpen = new Set<string>();
  for (const i of oldest) {
    if (!i.score) continue;
    if (i.answer && !firstAnswer.has(key(i))) firstAnswer.set(key(i), i.id);
    if (!i.answer && !i.closed) anyOpen.add(key(i));
  }
  const resultOf = (i: ReviewItem): Result => {
    if (!i.score) return "none";
    if (i.answer) return firstAnswer.get(key(i)) !== i.id ? "redo" : i.answer.choice === "Accept" ? "pass" : "fail";
    if (!i.closed) return "open";
    return firstAnswer.has(key(i)) || anyOpen.has(key(i)) ? "none" : "closed";
  };
  // Each lamella's grid: the about of its site-selection item.
  const gridOf = new Map<string, string>();
  for (const i of oldest) if (i.score === "site-selection" && i.about) gridOf.set(same(nameOf(i)), i.about);
  const lines: Line[] = oldest.map((i) => ({
    id: i.id,
    item: i,
    kind: i.score ?? "",
    name: i.score === "site-coverage" ? "" : nameOf(i),
    grid: i.score === "site-coverage" ? i.about : (gridOf.get(same(nameOf(i))) ?? ""),
    result: resultOf(i),
  }));
  for (const kind of KINDS)
    for (const name of counts[kind]?.missing ?? [])
      lines.push({ id: `${kind}:${name}`, kind, name, grid: gridOf.get(same(name)) ?? "", result: "missing" });
  const shown = lines.filter((l) =>
    show === "all"
      ? true
      : show === "open"
        ? l.result === "open"
        : l.result === "fail" || l.result === "missing" || l.result === "closed",
  );
  // Groups in the order first met: grid, then lamella; or kind, in the run's order.
  if (group === "type") {
    const rank = (k: string) => (k ? KINDS.indexOf(k as (typeof KINDS)[number]) : KINDS.length);
    shown.sort((a, b) => rank(a.kind) - rank(b.kind));
  }
  const sections = new Map<string, { title: string; sub: Map<string, Line[]> }>();
  for (const l of shown) {
    const [top, sub] =
      group === "item"
        ? [l.grid || "No grid named", l.name || (l.kind === "site-coverage" ? "Grid" : "Other")]
        : [l.kind ? (KIND_NAME[l.kind] ?? l.kind) : "Not scored", ""];
    if (!sections.has(top)) sections.set(top, { title: top, sub: new Map() });
    const s = sections.get(top)!.sub;
    if (!s.has(sub)) s.set(sub, []);
    s.get(sub)!.push(l);
  }
  const [chosen, setPicked] = useState<string | null>(null);
  // The pick follows the filter: a line hidden by it gives way to the first one shown.
  const order = [...sections.values()].flatMap((s) => [...s.sub.values()].flat());
  const picked = order.some((l) => l.id === chosen) ? chosen : (order.find((l) => l.item)?.id ?? null);
  const current = items.find((i) => i.id === picked) ?? null;
  // After an answer, the next line still to score.
  const next = () => {
    const at = order.findIndex((l) => l.id === picked);
    const after = [...order.slice(at + 1), ...order.slice(0, at)].find((l) => l.result === "open" && l.item);
    setPicked(after?.id ?? null);
  };
  // One line for the whole run; each kind's count is on its own header in By kind.
  const total = KINDS.reduce(
    (t, kind) => {
      const c = counts[kind] ?? { passed: 0, failed: 0, open: 0 };
      return { passed: t.passed + c.passed, failed: t.failed + c.failed, open: t.open + c.open };
    },
    { passed: 0, failed: 0, open: 0 },
  );
  const sum = (ls: Line[]) => {
    const scored = ls.filter((l) => l.kind && l.result !== "redo" && l.result !== "none");
    const fails = scored.filter((l) => l.result === "fail" || l.result === "missing" || l.result === "closed").length;
    const open = scored.filter((l) => l.result === "open").length;
    const passed = scored.filter((l) => l.result === "pass").length;
    return [passed + fails ? `${passed}/${passed + fails}` : "", open ? `${open} to score` : ""]
      .filter(Boolean)
      .join(" · ");
  };
  const [folded, setFolded] = useState<Set<string>>(new Set());
  const fold = (title: string) =>
    setFolded((f) => {
      const n = new Set(f);
      if (n.has(title)) n.delete(title);
      else n.add(title);
      return n;
    });
  const option = (on: boolean, name: string, pick: () => void) => (
    <button
      key={name}
      className={cn(
        "cursor-pointer border-0 bg-transparent px-3 py-1.5 text-left text-[13px] whitespace-nowrap hover:bg-muted",
        on && "font-semibold",
      )}
      onClick={pick}
    >
      {name}
    </button>
  );
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent title="Scores" className="h-[min(860px,calc(100vh-32px))] w-[min(1400px,calc(100vw-32px))]">
        <div className="flex items-center gap-3 border-b border-border px-4 py-2 text-[12.5px] tabular-nums">
          <span className="flex-1">
            {total.passed}/{total.passed + total.failed} passed
            {total.open > 0 && <span className="text-muted-foreground"> · {total.open} to score</span>}
          </span>
          <span className="text-muted-foreground">
            {group === "item" ? "By lamella" : "By kind"}
            {show !== "all" && ` · ${show === "fail" ? "Fails" : "To score"}`}
          </span>
          <Popup
            label="Group and show"
            align="right"
            side="below"
            trigger={(on, toggle) => (
              <button
                className={cn(
                  "flex size-6 cursor-pointer items-center justify-center rounded border-0 bg-transparent text-muted-foreground hover:text-foreground",
                  (on || show !== "all" || group !== "item") && "text-foreground",
                )}
                aria-label="Group and show"
                onClick={toggle}
              >
                <Filter size={14} />
              </button>
            )}
          >
            {(close) => (
              <div className="flex flex-col py-1">
                <span className="px-3 pt-1 pb-0.5 text-[11.5px] text-muted-foreground">Group</span>
                {(
                  [
                    ["item", "By lamella"],
                    ["type", "By kind"],
                  ] as [Group, string][]
                ).map(([v, name]) =>
                  option(group === v, name, () => {
                    setGroup(v);
                    close();
                  }),
                )}
                <span className="mt-1 border-t border-border px-3 pt-2 pb-0.5 text-[11.5px] text-muted-foreground">
                  Show
                </span>
                {(
                  [
                    ["all", "All"],
                    ["fail", "Fails"],
                    ["open", "To score"],
                  ] as [ShowScores, string][]
                ).map(([v, name]) =>
                  option(show === v, name, () => {
                    setShow(v);
                    close();
                  }),
                )}
              </div>
            )}
          </Popup>
        </div>
        <div className="flex min-h-0 flex-1">
          <div className="w-[360px] shrink-0 overflow-y-auto border-r border-border text-[13px]">
            {[...sections.values()].map((sec) => (
              <section key={sec.title}>
                <button
                  className="sticky top-0 flex w-full cursor-pointer items-baseline gap-2 border-0 bg-[var(--panel)] px-4 pt-3 pb-1 text-left"
                  aria-expanded={!folded.has(sec.title)}
                  onClick={() => fold(sec.title)}
                >
                  <span className="text-[11px] text-muted-foreground">{folded.has(sec.title) ? "▸" : "▾"}</span>
                  <span className="flex-1 font-semibold">{sec.title}</span>
                  <span className="text-[12px] text-muted-foreground tabular-nums">
                    {sum([...sec.sub.values()].flat())}
                  </span>
                </button>
                {!folded.has(sec.title) && [...sec.sub.entries()].map(([sub, ls]) => (
                  <div key={sub} className="pb-1">
                    {sub && <div className="px-4 pt-1.5 text-[12px] text-muted-foreground">{sub}</div>}
                    {ls.map((l) => (
                      <button
                        key={l.id}
                        disabled={!l.item}
                        className={cn(
                          "flex w-full items-baseline gap-2 border-0 bg-transparent px-4 py-1 text-left",
                          l.item ? "cursor-pointer hover:bg-muted/40" : "cursor-default",
                          l.id === picked && "bg-muted/60",
                        )}
                        onClick={() => setPicked(l.id)}
                      >
                        <span
                          className={cn(
                            "shrink-0 truncate text-[12px] text-muted-foreground",
                            group === "item" ? "w-[72px]" : "w-[110px]",
                          )}
                        >
                          {group === "item" ? (KIND_NAME[l.kind] ?? "") : l.name || l.grid}
                        </span>
                        <span className="min-w-0 flex-1 truncate">{l.item?.title ?? "No item filed"}</span>
                        <span
                          className={cn(
                            "shrink-0 text-[12px]",
                            l.result === "fail" || l.result === "missing" || l.result === "closed"
                              ? "text-destructive"
                              : l.result === "pass"
                                ? "text-foreground"
                                : "text-muted-foreground",
                          )}
                        >
                          {l.kind ? RESULT_WORD[l.result] : l.item?.answer ? l.item.answer.choice : "Not scored"}
                        </span>
                      </button>
                    ))}
                  </div>
                ))}
              </section>
            ))}
            {sections.size === 0 && <p className="m-0 px-4 py-3 text-muted-foreground">Nothing here.</p>}
          </div>
          <div className="min-w-0 flex-1 overflow-y-auto">
            {current ? (
              <Picked key={current.id} item={current} onDone={next} />
            ) : (
              <p className="m-0 px-4 py-3 text-[13px] text-muted-foreground">Pick a line to see it and score it.</p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** The picked item in Scores: the image viewer drawn in place, previous and
 * next through its pictures and marking on them, with the answer beside; an
 * item with no pictures shows the answer alone. */
function Picked({ item, onDone }: { item: ReviewItem; onDone: () => void }) {
  const marked = useEvidence(item.evidence);
  const paths = imagesOf(item, marked?.image ?? null);
  const editor = useMarkEditor(1, 1);
  if (item.evidence && !marked && paths.length) return null;
  return (
    <div className="flex flex-col gap-3">
      <h3 className="m-0 px-4 pt-3 text-[14px] font-semibold">{item.title}</h3>
      {paths.length ? (
        <ImageViewer
          inline
          paths={paths}
          at={0}
          onClose={() => {}}
          review={{
            title: item.title,
            agentMarks: (path: string): Mark[] | undefined =>
              marked && path === marked.image ? marked.marks : undefined,
            panel: ({ editor, path, channels }) => (
              <Answer key={item.id} item={item} editor={editor} path={path} channels={channels} onDone={onDone} />
            ),
          }}
        />
      ) : (
        <div className="max-w-[560px] px-4">
          <Answer item={item} editor={editor} path="" channels={null} onDone={onDone} />
        </div>
      )}
    </div>
  );
}

// A small dot says what an item wants: orange waits on the operator, grey asks for a look.
const DOT: Partial<Record<ReviewItem["state"], string>> = { waiting: "#e0a030", to_check: "#8a8f98" };
const SAYS: Record<ReviewItem["state"], string> = {
  waiting: "waiting on you",
  to_check: "to check",
  answered: "",
  closed: "closed",
  done: "",
};

/** GET /app/agent/review, read again on any news about the agent. */
export function useReview() {
  return useQuery({ queryKey: ["app/agent", "review"], queryFn: () => api<ReviewList>("/app/agent/review") });
}

/** The evidence picture's image and the agent's marks on it, when the item has one. */
function useEvidence(id: string | null) {
  const answer = useQuery({
    queryKey: ["app/agent/evidence", id],
    queryFn: () => api<EvidenceAnswer>(`/app/agent/evidence/${encodeURIComponent(id!)}`),
    enabled: !!id,
    staleTime: Infinity,
  });
  const data = answer.data;
  if (!data?.source) return null;
  const marks = data.source === data.path ? listMarks(data.marks) : listMarks(data.source_marks);
  return { image: data.source, marks };
}

/** The item's images, its marked picture's image first. */
function imagesOf(item: ReviewItem, marked: string | null): string[] {
  return marked ? [marked, ...item.images.filter((p) => p !== marked)] : item.images;
}

function Picture({ path }: { path: string }) {
  const read = useFileRead(path, 480);
  const p = read.data?.preview;
  return (
    <span className="block aspect-[3/2] min-w-0 flex-1 overflow-hidden rounded bg-black">
      {p && <img className="block size-full object-cover" alt="" src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />}
    </span>
  );
}

function Row({ item, selected, onOpen }: { item: ReviewItem; selected: boolean; onOpen: () => void }) {
  const marked = useEvidence(item.evidence);
  const shown = imagesOf(item, marked?.image ?? null).slice(0, 2);
  const choice = item.answer?.choice;
  const line = [item.about, item.score, SAYS[item.state], choice && `${choice}${item.answer?.note ? `: “${item.answer.note}”` : ""}`]
    .filter(Boolean)
    .join(" · ");
  return (
    <button
      className={cn(
        "flex w-full cursor-pointer flex-col gap-1.5 border-0 border-t border-border bg-transparent px-4 py-3 text-left hover:bg-muted/40",
        selected && "bg-muted/40",
      )}
      onClick={onOpen}
    >
      {shown.length > 0 && (
        <span className="flex w-full gap-1.5">
          {shown.map((p) => (
            <Picture key={p} path={p} />
          ))}
          {shown.length === 1 && <span className="flex-1" />}
        </span>
      )}
      <span className="flex w-full items-baseline gap-2 text-[13px]">
        <span
          className="size-[7px] shrink-0 self-center rounded-full"
          style={{ background: DOT[item.state] ?? "transparent" }}
        />
        <span className="min-w-0 flex-1">{item.title}</span>
        <span className="text-[11.5px] text-muted-foreground tabular-nums">{fmt.clock(item.at)}</span>
      </span>
      {line && <span className="pl-[15px] text-[12px] text-muted-foreground">{line}</span>}
    </button>
  );
}

/** The side panel of the viewer: what the agent says, its facts, and the answer:
 * one of its options, a note, and the marks drawn on the image shown. */
function Answer({
  item,
  editor,
  path,
  channels,
  onDone,
}: {
  item: ReviewItem;
  editor: MarkEditor;
  path: string;
  channels: string[] | null;
  onDone: () => void;
}) {
  const client = useQueryClient();
  const [choice, setChoice] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  // An answer given may be changed: the new one counts, the one before is kept.
  const [changing, setChanging] = useState(false);
  const open = item.state === "waiting" || item.state === "to_check" || item.state === "done" || changing;
  const send = async () => {
    setError(null);
    setSending(true);
    try {
      const marks = editor.marks.length
        ? { evidence: { image: path, marks: toMarks(editor.marks), ...(channels ? { channels } : {}) } }
        : {};
      await api(`/app/agent/review/${encodeURIComponent(item.id)}/answer`, {
        method: "POST",
        body: JSON.stringify({ choice, note: note.trim(), ...marks }),
      });
      editor.clear();
      setChanging(false);
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
    }
  };
  return (
    <section className="flex flex-col gap-3 text-[13px]">
      <span className="text-[12px] text-muted-foreground">
        {[item.about, item.score && `scores ${item.score}`, SAYS[item.state], fmt.clock(item.at)]
          .filter(Boolean)
          .join(" · ")}
      </span>
      {item.text && <Folded text={item.text} />}
      {item.facts.length > 0 && (
        <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12.5px]">
          {item.facts.map((f) => (
            <div key={f.label} className="contents">
              <dt className="text-muted-foreground">{f.label}</dt>
              <dd className="m-0">{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {item.answer && (
        <p className="m-0 flex items-baseline gap-2 border-t border-border pt-3">
          <span className="flex-1">
            You said <b className="font-semibold">{item.answer.choice}</b>
            {item.answer.note ? `: “${item.answer.note}”` : ""}
          </span>
          {!item.closed && !changing && (
            <button
              className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted-foreground underline hover:text-foreground"
              onClick={() => setChanging(true)}
            >
              Change
            </button>
          )}
        </p>
      )}
      {item.closed && <p className="m-0 text-muted-foreground">Closed by the agent: {item.closed.reason}</p>}
      {open && (
        <div className="flex flex-col gap-2 border-t border-border pt-3">
          <div className="flex flex-wrap gap-1.5">
            {item.options.map((o) => (
              <Button
                key={o}
                aria-pressed={choice === o}
                variant={choice === o ? "primary" : "default"}
                className="px-3"
                onClick={() => setChoice(choice === o ? null : o)}
              >
                {o}
              </Button>
            ))}
          </div>
          {item.score && (
            <p className="m-0 text-[12px] text-muted-foreground">
              {choice === "Reject" ? MARK_ON_FAIL[item.score] : "Accept is a pass, Reject a fail."}
            </p>
          )}
          <textarea
            rows={3}
            aria-label="Note for the agent"
            placeholder={item.score ? "Why (optional)" : "Note for the agent (optional)"}
            className="resize-y rounded border border-input bg-background px-2 py-1.5 text-[13px]"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          {error && <p className="m-0 text-[12px] text-destructive">{error}</p>}
          <div className="flex items-center gap-2">
            <span className="flex-1 text-[12px] text-muted-foreground">
              {editor.marks.length ? `With ${editor.marks.length} mark${editor.marks.length > 1 ? "s" : ""}` : ""}
            </span>
            {changing && (
              <Button onClick={() => setChanging(false)} disabled={sending}>
                Cancel
              </Button>
            )}
            <Button variant="primary" disabled={!choice || sending} onClick={send}>
              {changing ? "Change" : "Send"}
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}

/** An item with no picture: the answer alone. */
function Plain({ item, onClose }: { item: ReviewItem; onClose: () => void }) {
  const editor = useMarkEditor(1, 1);
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent title={item.title} className="w-[min(480px,calc(100vw-32px))]">
        <div className="px-4 pb-4">
          <Answer item={item} editor={editor} path="" channels={null} onDone={onClose} />
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Viewer({ item, onClose }: { item: ReviewItem; onClose: () => void }) {
  const marked = useEvidence(item.evidence);
  const paths = imagesOf(item, marked?.image ?? null);
  if (!paths.length) return item.evidence && !marked ? null : <Plain item={item} onClose={onClose} />;
  return (
    <ImageViewer
      paths={paths}
      at={0}
      onClose={onClose}
      review={{
        title: item.title,
        agentMarks: (path: string): Mark[] | undefined => (marked && path === marked.image ? marked.marks : undefined),
        panel: ({ editor, path, channels }) => (
          <Answer key={item.id} item={item} editor={editor} path={path} channels={channels} onDone={onClose} />
        ),
      }}
    />
  );
}

/** The Review tab: the agent's finished work, newest first, pictures on top; an
 * item opens in the image viewer, where the operator answers it. */
export function ReviewTab({ open }: { open?: { id: string } | null }) {
  const review = useReview();
  const [show, setShow] = useState<Show>("all");
  const [openId, setOpenId] = useState<string | null>(null);
  // An item picked elsewhere (the quick search) opens here; a new object each time, so picking it again reopens it.
  useEffect(() => {
    if (open) setOpenId(open.id);
  }, [open]);
  const items = [...(review.data?.items ?? [])].reverse();
  const shown = items.filter((i) => shows(i, show));
  const [scores, setScores] = useState(false);
  const missing = Object.values(review.data?.counts.scores ?? {}).some((t) => t.missing?.length);
  const anyScored = missing || items.some((i) => i.score);
  const opened = items.find((i) => i.id === openId);
  const count = review.data?.counts.all ?? 0;
  return (
    <div className="-mx-4 -my-3 flex flex-col">
      <div className="flex items-center gap-2 px-4 py-2 text-[12.5px] text-muted-foreground">
        <span className="flex-1">
          {count} item{count === 1 ? "" : "s"}
          {show !== "all" && ` · ${SHOWS.find((s) => s.show === show)?.name.toLowerCase()}`}
        </span>
        {anyScored && (
          <button
            className="cursor-pointer rounded border-0 bg-transparent px-1.5 py-0.5 text-[12.5px] text-muted-foreground hover:text-foreground"
            onClick={() => setScores(true)}
          >
            Scores
          </button>
        )}
        <Popup
          label="Show"
          align="right"
          side="below"
          trigger={(on, toggle) => (
            <button
              className={cn(
                "flex size-6 cursor-pointer items-center justify-center rounded border-0 bg-transparent text-muted-foreground hover:text-foreground",
                (on || show !== "all") && "text-foreground",
              )}
              aria-label="Filter"
              onClick={toggle}
            >
              <Filter size={14} />
            </button>
          )}
        >
          {(close) => (
            <div className="flex flex-col py-1">
              {SHOWS.map((s) => (
                <button
                  key={s.show}
                  className={cn(
                    "cursor-pointer border-0 bg-transparent px-3 py-1.5 text-left text-[13px] whitespace-nowrap hover:bg-muted",
                    s.show === show && "font-semibold",
                  )}
                  onClick={() => {
                    setShow(s.show);
                    close();
                  }}
                >
                  {s.name}
                </button>
              ))}
            </div>
          )}
        </Popup>
      </div>
      {shown.length === 0 && (
        <p className="m-0 px-4 py-3 text-[13px] text-muted-foreground">
          {count ? "Nothing here." : "The agent's finished work shows here, with its pictures."}
        </p>
      )}
      {shown.map((item) => (
        <Row key={item.id} item={item} selected={item.id === openId} onOpen={() => setOpenId(item.id)} />
      ))}
      {opened && <Viewer item={opened} onClose={() => setOpenId(null)} />}
      {scores && !opened && (
        <Scores items={items} counts={review.data?.counts.scores ?? {}} onClose={() => setScores(false)} />
      )}
    </div>
  );
}
