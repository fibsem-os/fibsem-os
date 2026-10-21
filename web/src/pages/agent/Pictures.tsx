import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { EvidenceAnswer } from "@/lib/types";

// The agent's purple, light enough to read on a grey overview.
const MARK = "#a78bfa";

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
  const marks = answer.data?.marks ?? {};
  // Thin marks sized for the picture a few hundred pixels wide; a faint halo keeps them clear of the image.
  const r = Math.max(w / 60, 4);
  const sw = Math.max(w / 320, 1.5);
  const size = Math.max(w / 30, 10);
  const halo = {
    stroke: "#000",
    strokeOpacity: 0.5,
    strokeWidth: sw * 2,
    fill: "none",
  };
  const line = { stroke: MARK, strokeWidth: sw, fill: "none" };
  const label = (x: number, y: number, text?: string) =>
    text && (
      <text
        x={x}
        y={y}
        fill={MARK}
        fontWeight={600}
        fontSize={size}
        stroke="#000"
        strokeOpacity={0.5}
        strokeWidth={sw * 1.5}
        strokeLinejoin="round"
        paintOrder="stroke"
      >
        {text}
      </text>
    );
  return (
    <Enlargeable title="Evidence">
      {() => (
        <div className="relative">
          <img
            className="block w-full rounded"
            alt="The image the agent judged from"
            src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
          />
          <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
            {(marks.points ?? []).map((m, i) => (
              <g key={`p${i}`}>
                <circle cx={m.x} cy={m.y} r={r} {...halo} />
                <circle cx={m.x} cy={m.y} r={r} {...line} />
                {label(m.x + r * 1.4, m.y, m.label)}
              </g>
            ))}
            {(marks.areas ?? []).map((m, i) => (
              <g key={`a${i}`}>
                <rect x={m.x} y={m.y} width={m.width} height={m.height} {...halo} />
                <rect x={m.x} y={m.y} width={m.width} height={m.height} {...line} />
                {label(m.x, m.y - sw * 2, m.label)}
              </g>
            ))}
          </svg>
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
