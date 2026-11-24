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
  sites: "Mark the good squares it missed.",
  target: "Mark the right target.",
  polishing: "Say why in the note: too thick, too thin, damaged.",
};

const KINDS: NonNullable<ReviewItem["score"]>[] = ["sites", "target", "polishing"];
const RESULT: Record<string, string> = { Accept: "Pass", Reject: "Fail" };

// A grid's or lamella's name as the server compares it ("Lamella 01" is "lamella-01").
const same = (name?: string) => (name ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");

/** The run's scored items: the server's tally per kind (one score per grid or
 * lamella, its first answer), then each item; one opens in the viewer. */
function Scores({
  items,
  counts,
  onOpen,
  onClose,
}: {
  items: ReviewItem[];
  counts: NonNullable<ReviewList["counts"]["scores"]>;
  onOpen: (id: string) => void;
  onClose: () => void;
}) {
  const scored = items.filter((i) => i.score && !(i.closed && !i.answer));
  // A later answer for the same lamella (a redo after a Reject) isn't its score.
  const first = new Set<string>();
  const redo = new Set<string>();
  for (const i of [...scored].reverse()) {
    if (!i.answer) continue;
    const key = `${i.score}:${same(i.about)}`;
    if (first.has(key)) redo.add(i.id);
    else first.add(key);
  }
  const none = { passed: 0, failed: 0, open: 0 };
  const tally = KINDS.map((kind) => ({ kind, ...(counts[kind] ?? none) })).filter(
    (t) => t.passed + t.failed + t.open > 0,
  );
  const missing = KINDS.flatMap((kind) => (counts[kind]?.missing ?? []).map((about) => ({ kind, about })));
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent title="Scores" className="w-[min(560px,calc(100vw-32px))]">
        <div className="flex flex-col gap-4 overflow-y-auto px-4 py-3 text-[13px]">
          <table className="w-full border-collapse tabular-nums">
            <thead className="text-[12px] text-muted-foreground">
              <tr>
                <th className="py-1 text-left font-normal" />
                <th className="py-1 text-right font-normal">Pass</th>
                <th className="py-1 text-right font-normal">Fail</th>
                <th className="py-1 text-right font-normal">To score</th>
              </tr>
            </thead>
            <tbody>
              {tally.map((t) => (
                <tr key={t.kind} className="border-t border-border">
                  <td className="py-1.5 capitalize">{t.kind}</td>
                  <td className="py-1.5 text-right">{t.passed}</td>
                  <td className="py-1.5 text-right">{t.failed}</td>
                  <td className="py-1.5 text-right text-muted-foreground">{t.open || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-col">
            {scored.map((i) => (
              <button
                key={i.id}
                className="flex cursor-pointer items-baseline gap-3 border-0 border-t border-border bg-transparent px-0 py-2 text-left hover:bg-muted/40"
                onClick={() => onOpen(i.id)}
              >
                <span className="w-24 shrink-0 text-muted-foreground">{i.about}</span>
                <span className="w-20 shrink-0 text-muted-foreground">{i.score}</span>
                <span className="min-w-0 flex-1">
                  {i.title}
                  {i.answer?.note && <span className="block text-[12px] text-muted-foreground">“{i.answer.note}”</span>}
                </span>
                <span className={cn("shrink-0", (!i.answer || redo.has(i.id)) && "text-muted-foreground")}>
                  {i.answer ? (RESULT[i.answer.choice] ?? i.answer.choice) : "To score"}
                  {redo.has(i.id) && " after help"}
                </span>
              </button>
            ))}
            {missing.map((m) => (
              <div key={`${m.kind}:${m.about}`} className="flex items-baseline gap-3 border-t border-border py-2">
                <span className="w-24 shrink-0 text-muted-foreground">{m.about}</span>
                <span className="w-20 shrink-0 text-muted-foreground">{m.kind}</span>
                <span className="min-w-0 flex-1 text-muted-foreground">No item filed</span>
                <span className="shrink-0">Fail</span>
              </div>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
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
  const open = item.state === "waiting" || item.state === "to_check" || item.state === "done";
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
      {item.text && <p className="m-0 leading-snug">{item.text}</p>}
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
        <p className="m-0 border-t border-border pt-3">
          You said <b className="font-semibold">{item.answer.choice}</b>
          {item.answer.note ? `: “${item.answer.note}”` : ""}
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
                className={cn("px-3", choice === o && "border-primary")}
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
            <Button variant="primary" disabled={!choice || sending} onClick={send}>
              Send
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
              {anyScored && (
                <button
                  className="cursor-pointer border-0 border-t border-border bg-transparent px-3 py-1.5 text-left text-[13px] whitespace-nowrap hover:bg-muted"
                  onClick={() => {
                    setScores(true);
                    close();
                  }}
                >
                  Scores…
                </button>
              )}
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
        <Scores items={items} counts={review.data?.counts.scores ?? {}} onOpen={(id) => setOpenId(id)} onClose={() => setScores(false)} />
      )}
    </div>
  );
}
