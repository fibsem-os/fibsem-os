import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Filter } from "lucide-react";
import { useState } from "react";
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

type Show = "all" | ReviewItem["state"];
const SHOWS: { show: Show; name: string }[] = [
  { show: "all", name: "All" },
  { show: "waiting", name: "Waiting on you" },
  { show: "to_check", name: "To check" },
  { show: "answered", name: "Answered" },
  { show: "done", name: "Done" },
];

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
  const line = [item.about, SAYS[item.state], choice && `${choice}${item.answer?.note ? `: “${item.answer.note}”` : ""}`]
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
        {[item.about, SAYS[item.state], fmt.clock(item.at)].filter(Boolean).join(" · ")}
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
          <textarea
            rows={3}
            aria-label="Note for the agent"
            placeholder="Note for the agent (optional)"
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
export function ReviewTab() {
  const review = useReview();
  const [show, setShow] = useState<Show>("all");
  const [openId, setOpenId] = useState<string | null>(null);
  const items = [...(review.data?.items ?? [])].reverse();
  const shown = show === "all" ? items : items.filter((i) => i.state === show);
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
    </div>
  );
}
