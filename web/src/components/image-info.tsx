import type { ImageInfo } from "@/lib/types";

/** The bar under an image, as the Qt view bar reads it: the kind, the detector or
 * objective, the chosen values with their short labels, and when it was taken. */
export function ImageInfoBar({ info }: { info: ImageInfo | null | undefined }) {
  if (!info) return null;
  return (
    <div className="flex min-h-[26px] flex-wrap items-center gap-x-3 gap-y-0.5 border-t border-border bg-background px-2 py-1 text-[11px]">
      <span className="text-[12px] font-bold text-foreground">{info.kind}</span>
      {info.header.length > 0 && (
        <span className="text-muted-foreground" title={info.header.map((f) => `${f.title}: ${f.value}`).join("\n")}>
          {info.header.map((f) => f.value).join(" · ")}
        </span>
      )}
      {info.fields.map((f) => (
        <span key={f.key} className="tabular-nums" title={`${f.title}: ${f.value}`}>
          <span className="text-muted-foreground">{f.label}</span> {f.value}
        </span>
      ))}
      {info.time && <span className="ml-auto text-muted-foreground tabular-nums">{info.time}</span>}
    </div>
  );
}

/** A round length in the image's bottom right, laid over a box exactly the image's size. */
export function ScaleBar({ info }: { info: ImageInfo | null | undefined }) {
  const bar = info?.scalebar;
  if (!bar) return null;
  // A container the image's width, so the bar can span its fraction of it.
  return (
    <div className="pointer-events-none absolute inset-0 [container-type:inline-size]">
      <div className="absolute right-2 bottom-2 flex flex-col items-center gap-1 rounded-sm bg-black/60 px-1.5 py-1 text-[11px] leading-none text-white">
        <span>{bar.label}</span>
        <span className="block h-1 bg-white" style={{ width: `calc(${bar.fraction} * (100cqw))` }} />
      </div>
    </div>
  );
}
