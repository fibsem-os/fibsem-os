import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { MarkLayer } from "@/components/marks";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { AGENT_MARK, listMarks } from "@/lib/marks";
import type { EvidenceAnswer } from "@/lib/types";

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

/** The image the agent judged from, with its marks drawn over it. */
export function EvidenceImage({ id }: { id: string }) {
  const answer = useQuery({
    queryKey: ["app/agent/evidence", id],
    queryFn: () => api<EvidenceAnswer>(`/app/agent/evidence/${encodeURIComponent(id)}`),
    staleTime: Infinity,
  });
  const p = answer.data?.preview;
  if (!p) return null;
  const w = p.full_width ?? p.width;
  const h = p.full_height ?? p.height;
  return (
    <Enlargeable title="Evidence">
      {() => (
        <div className="relative">
          <img
            className="block w-full rounded"
            alt="The image the agent judged from"
            src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
          />
          <MarkLayer marks={listMarks(answer.data?.marks)} width={w} height={h} color={AGENT_MARK} />
        </div>
      )}
    </Enlargeable>
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
