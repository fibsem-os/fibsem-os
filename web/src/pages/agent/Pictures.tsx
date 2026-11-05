import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil } from "lucide-react";
import { useState, type ReactNode } from "react";
import { ImageViewer } from "@/components/images";
import { KeepTick, usePlanSample } from "@/components/mark-for-agent";
import { MarkLayer } from "@/components/marks";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { AGENT_MARK, BAD_MARK, GOOD_MARK, OPERATOR_MARK, VERDICT_SIGN, type Verdict, listMarks } from "@/lib/marks";
import type { EvidenceAnswer } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Click to see it large. */
function Enlargeable({ title, children }: { title: string; children: (big: boolean) => ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        className="block w-full max-w-[420px] cursor-zoom-in rounded border-0 bg-black p-0"
        onClick={() => setOpen(true)}
        title={title}
      >
        {children(false)}
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent title={title} className="w-[min(1200px,calc(100vw-32px))]">
          <div className="overflow-auto bg-black p-2">{children(true)}</div>
        </DialogContent>
      </Dialog>
    </>
  );
}

const SAID: Record<Verdict, string> = { good: "good", bad: "bad" };

/** ✓ good, ✗ bad and ✎ mark on it, in a picture's bottom-right corner. */
function FeedbackIcons({
  said,
  asking,
  canMark,
  onVerdict,
  onMark,
}: {
  said?: Verdict;
  asking?: Verdict;
  canMark: boolean;
  onVerdict: (v: Verdict) => void;
  onMark: () => void;
}) {
  const icon = (lit: boolean, color?: string) =>
    cn(
      "flex size-6 cursor-pointer items-center justify-center rounded border-0 p-0 text-[13px] leading-none",
      lit ? "bg-black/85" : "bg-black/55 text-white/75 hover:bg-black/80 hover:text-white",
      lit && !color && "text-white",
    );
  return (
    <div className="absolute right-1.5 bottom-1.5 flex gap-1">
      {(["good", "bad"] as const).map((v) => {
        const lit = said === v || asking === v;
        const color = v === "good" ? GOOD_MARK : BAD_MARK;
        return (
          <button
            key={v}
            title={v === "good" ? "Good" : "Bad"}
            aria-label={v === "good" ? "Good" : "Bad"}
            aria-pressed={lit}
            className={icon(lit, color)}
            style={lit ? { color, boxShadow: `inset 0 0 0 1px ${color}` } : undefined}
            onClick={() => onVerdict(v)}
          >
            {VERDICT_SIGN[v]}
          </button>
        );
      })}
      {canMark && (
        <button title="Mark on it" aria-label="Mark on it" className={icon(false)} onClick={onMark}>
          <Pencil size={12} />
        </button>
      )}
    </div>
  );
}

/** Good or bad of the whole picture, with an optional reason, sent to the agent. */
function VerdictField({
  id,
  verdict,
  onSent,
  onCancel,
}: {
  id: string;
  verdict: Verdict;
  onSent: (reason: string) => void;
  onCancel: () => void;
}) {
  const client = useQueryClient();
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [keep, setKeep] = useState<boolean | null>(null);
  const sample = usePlanSample();
  const kept = !!sample && (keep ?? true);
  const send = async () => {
    setSending(true);
    setError(null);
    try {
      await api("/app/agent/messages", {
        method: "POST",
        body: JSON.stringify({
          text: reason.trim(),
          evidence: { answers: id, verdict, ...(kept ? { keep: true } : {}) },
        }),
      });
      onSent(reason.trim());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
      if (kept) await client.invalidateQueries({ queryKey: ["app/knowledge"] });
    }
  };
  return (
    <div className="mt-1.5 flex max-w-[420px] flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <input
          autoFocus
          aria-label="Why"
          placeholder={verdict === "good" ? "Why good? (optional)" : "Why bad? (optional)"}
          className="min-w-0 flex-1 rounded border border-input bg-background px-2 py-1 text-[13px]"
          value={reason}
          maxLength={500}
          onChange={(e) => setReason(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
            if (e.key === "Escape") onCancel();
          }}
        />
        <Button variant="primary" onClick={send} disabled={sending}>
          Send
        </Button>
        <Button onClick={onCancel}>Cancel</Button>
      </div>
      <KeepTick said sample={sample} keep={keep} setKeep={setKeep} />
      {error && <p className="m-0 text-[12px] text-destructive">{error}</p>}
    </div>
  );
}

/** The image the agent judged from, with its marks drawn over it. The agent's has
 * ✓ ✗ ✎ in its corner: good or bad of the whole picture, or mark on the image it
 * came from, in the viewer over the chat. */
export function EvidenceImage({ id }: { id: string }) {
  const [replying, setReplying] = useState(false);
  const [asking, setAsking] = useState<Verdict | undefined>(undefined);
  const [said, setSaid] = useState<{ verdict: Verdict; reason: string } | undefined>(undefined);
  const answer = useQuery({
    queryKey: ["app/agent/evidence", id],
    queryFn: () => api<EvidenceAnswer>(`/app/agent/evidence/${encodeURIComponent(id)}`),
    staleTime: Infinity,
  });
  const p = answer.data?.preview;
  if (!p) return null;
  const w = p.full_width ?? p.width;
  const h = p.full_height ?? p.height;
  const { by, source, path, plane, source_marks, verdict, judged } = answer.data!;
  const marks = listMarks(answer.data?.marks);
  const agents = by !== "operator";
  const sign = (v?: string | null) =>
    v === "good" || v === "bad" ? (
      <span style={{ color: v === "good" ? GOOD_MARK : BAD_MARK }}>{VERDICT_SIGN[v]}</span>
    ) : null;
  // the operator's feedback with nothing drawn: what was said, not the picture again
  if (!agents && verdict && !marks.length && !judged?.length)
    return (
      <p className="m-0 mt-1 text-[12px] text-muted-foreground">
        The agent's picture: {sign(verdict)} {SAID[verdict as Verdict]}
      </p>
    );
  // the picks on the image the picture was drawn on: its own, or the ones drawn there
  const picks = source === path ? marks : source_marks ? listMarks(source_marks) : [];
  return (
    <>
      <div className="relative max-w-[420px]">
        <Enlargeable title="Evidence">
          {() => (
            <div className="relative">
              <img
                className="block w-full rounded"
                alt="The image the agent judged from"
                src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
              />
              <MarkLayer marks={marks} width={w} height={h} color={agents ? AGENT_MARK : OPERATOR_MARK} />
            </div>
          )}
        </Enlargeable>
        {agents && (
          <FeedbackIcons
            said={said?.verdict}
            asking={asking}
            canMark={!!source}
            onVerdict={(v) => setAsking(asking === v ? undefined : v)}
            onMark={() => setReplying(true)}
          />
        )}
        {!agents && verdict && (
          <span
            className="absolute right-1.5 bottom-1.5 rounded bg-black/85 px-1.5 py-0.5 text-[12px]"
            style={{ color: verdict === "good" ? GOOD_MARK : BAD_MARK }}
          >
            {VERDICT_SIGN[verdict as Verdict]} {SAID[verdict as Verdict]}
          </span>
        )}
      </div>
      {asking && (
        <VerdictField
          id={id}
          verdict={asking}
          onSent={(reason) => {
            setSaid({ verdict: asking, reason });
            setAsking(undefined);
          }}
          onCancel={() => setAsking(undefined)}
        />
      )}
      {!agents && judged && judged.length > 0 && (
        <p className="m-0 mt-1 text-[12px] text-muted-foreground">
          The agent's picks:{" "}
          {judged.map((j, i) => (
            <span key={i}>
              {i > 0 && ", "}
              {j.label || `${j.kind.replace(/s$/, "")} ${j.index + 1}`} {sign(j.verdict)}
              {j.moved_to && " moved"}
            </span>
          ))}
        </p>
      )}
      {said && !asking && (
        <p className="m-0 mt-1 text-[12px] text-muted-foreground">
          You: {SAID[said.verdict]}
          {said.reason && `, ${said.reason}`}
        </p>
      )}
      {agents && source && (
        <ImageViewer
          paths={[source]}
          at={replying ? 0 : null}
          onClose={() => setReplying(false)}
          // an upload's marks are on its own pixels, not the image it was drawn from
          reply={{ marks: picks, plane, answers: id }}
        />
      )}
    </>
  );
}

/** The window's drawing of the request or suggestion, values on the live image. */
export function CardPicture({ kind, version }: { kind: "request" | "suggestion"; version: number }) {
  const answer = useQuery({
    queryKey: ["app/agent/pictures", kind, version],
    queryFn: () => api<{ image_b64_png: string }>(`/app/agent/pictures/${kind}`),
    staleTime: Infinity,
    // A redrawn picture replaces the last one once it has loaded.
    placeholderData: (previous, last) => (last?.queryKey[1] === kind ? previous : undefined),
  });
  if (!answer.data) return null;
  return (
    <Enlargeable title={kind === "request" ? "What the agent asks for" : "What the agent suggests"}>
      {() => (
        <img
          className="block w-full rounded"
          alt="What the agent asks for, on the image"
          src={`data:image/png;base64,${answer.data.image_b64_png}`}
        />
      )}
    </Enlargeable>
  );
}
