import type { ReactNode } from "react";

// A small Markdown: paragraphs, "-" or "*" lists, **bold** and
// `code`. React elements throughout, so nothing in a message is ever HTML.

function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") && part.length > 1 ? (
      <code key={i} className="rounded bg-accent px-1 text-[12px]">
        {part.slice(1, -1)}
      </code>
    ) : part.startsWith("**") && part.endsWith("**") && part.length > 3 ? (
      <b key={i}>{part.slice(2, -2)}</b>
    ) : (
      part
    ),
  );
}

export function Markdown({ text }: { text: string }) {
  const out: ReactNode[] = [];
  let list: string[] | null = null;
  let para: string[] = [];
  const flush = () => {
    if (para.length) {
      const lines = para;
      out.push(
        <p key={out.length} className="my-1">
          {lines.map((line, i) => (
            <span key={i}>
              {i > 0 && <br />}
              {inline(line)}
            </span>
          ))}
        </p>,
      );
      para = [];
    }
    if (list) {
      const items = list;
      out.push(
        <ul key={out.length} className="my-1 list-disc pl-5">
          {items.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </ul>,
      );
      list = null;
    }
  };
  for (const line of text.split("\n")) {
    const item = line.match(/^\s*[-*]\s+(.*)$/);
    if (item) {
      if (para.length) flush();
      (list ??= []).push(item[1]);
    } else if (!line.trim()) flush();
    else {
      if (list) flush();
      para.push(line);
    }
  }
  flush();
  return <>{out}</>;
}
