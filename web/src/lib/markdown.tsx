import { Check } from "lucide-react";
import type { ReactNode } from "react";

// A small Markdown: paragraphs, "-" or "*" lists, **bold** and
// `code`. React elements throughout, so nothing in a message is ever HTML.
// A document (`doc`, the agent's plan) also has # headings, "- [ ]" boxes,
// [links](url) and pictures on their own line, ![caption](path).

const LINK = /(\[[^\]]+\]\([^)\s]+\))/g;

function code(text: string, key: number) {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") && part.length > 1 ? (
      <code key={`${key}-${i}`} className="rounded bg-accent px-1 text-[12px]">
        {part.slice(1, -1)}
      </code>
    ) : part.startsWith("**") && part.endsWith("**") && part.length > 3 ? (
      <b key={`${key}-${i}`}>{part.slice(2, -2)}</b>
    ) : (
      part
    ),
  );
}

function inline(text: string, doc = false): ReactNode[] {
  if (!doc) return code(text, 0);
  return text.split(LINK).flatMap((part, i) => {
    const link = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
    if (!link) return code(part, i);
    const [, label, href] = link;
    // Only web links open; a path or a knowledge entry is named on hover.
    return /^https?:\/\//.test(href) ? (
      <a key={i} href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
        {label}
      </a>
    ) : (
      <span key={i} title={href} className="underline decoration-muted-foreground underline-offset-2">
        {label}
      </span>
    );
  });
}

const HEADINGS = [
  "mt-1 mb-2 text-[17px] font-semibold",
  "mt-5 mb-1.5 border-b border-border pb-1 text-[14px] font-semibold",
  "mt-3 mb-1 text-[13px] font-semibold text-muted-foreground",
];

type Item = { text: string; box?: boolean };

export function Markdown({
  text,
  doc = false,
  picture,
}: {
  text: string;
  doc?: boolean;
  /** How a document draws ![caption](path). */
  picture?: (path: string, caption: string) => ReactNode;
}) {
  const out: ReactNode[] = [];
  let list: Item[] | null = null;
  let para: string[] = [];
  let pictures: ReactNode[] = [];
  const flush = () => {
    if (para.length) {
      const lines = para;
      out.push(
        <p key={out.length} className="my-1">
          {lines.map((line, i) => (
            <span key={i}>
              {i > 0 && <br />}
              {inline(line, doc)}
            </span>
          ))}
        </p>,
      );
      para = [];
    }
    if (list) {
      const items = list;
      const boxes = items.some((item) => item.box !== undefined);
      out.push(
        <ul key={out.length} className={boxes ? "my-1 flex list-none flex-col gap-1 p-0" : "my-1 list-disc pl-5"}>
          {items.map((item, i) =>
            item.box === undefined ? (
              <li key={i} className={boxes ? "ml-5 list-disc" : undefined}>
                {inline(item.text, doc)}
              </li>
            ) : (
              <li key={i} className="flex gap-2">
                <span
                  role="checkbox"
                  aria-checked={item.box}
                  aria-label={item.text}
                  className="mt-[3px] flex size-[14px] shrink-0 items-center justify-center rounded-[3px] border border-muted-foreground/70 data-[on=true]:bg-muted-foreground/70 data-[on=true]:text-card"
                  data-on={item.box}
                >
                  {item.box && <Check size={11} strokeWidth={3} />}
                </span>
                <span className={item.box ? "text-muted-foreground" : undefined}>{inline(item.text, doc)}</span>
              </li>
            ),
          )}
        </ul>,
      );
      list = null;
    }
    if (pictures.length) {
      out.push(
        <div key={out.length} className="my-2 flex flex-wrap gap-2">
          {pictures}
        </div>,
      );
      pictures = [];
    }
  };
  for (const line of text.split("\n")) {
    const heading = doc && line.match(/^(#{1,3})\s+(.+?)\s*$/);
    const image = doc && picture && line.match(/^\s*!\[([^\]]*)\]\(([^)\s]+)\)\s*$/);
    const item = line.match(/^\s*[-*]\s+(.*)$/);
    if (heading) {
      flush();
      const Tag = (["h2", "h3", "h4"] as const)[heading[1].length - 1];
      out.push(
        <Tag key={out.length} className={HEADINGS[heading[1].length - 1]}>
          {inline(heading[2], doc)}
        </Tag>,
      );
    } else if (image) {
      if (para.length || list) flush();
      pictures.push(<span key={pictures.length}>{picture!(image[2], image[1])}</span>);
    } else if (item) {
      if (para.length || pictures.length) flush();
      const box = doc ? item[1].match(/^\[( |x|X)\]\s+(.*)$/) : null;
      (list ??= []).push(box ? { text: box[2], box: box[1] !== " " } : { text: item[1] });
    } else if (!line.trim()) flush();
    else {
      if (list || pictures.length) flush();
      para.push(line);
    }
  }
  flush();
  return <>{out}</>;
}
