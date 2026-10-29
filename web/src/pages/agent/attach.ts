import { useRef, useState } from "react";
import { getToken } from "@/lib/api";
import * as fmt from "@/lib/format";
import type { FileEntry } from "@/lib/types";

// What the agent reads (experiment_read_file), as POST /app/experiment/upload takes.
export const TAKEN = ".md .txt .csv .yaml .yml .json .pdf .png .jpg .jpeg .tif .tiff".split(" ");
const LIMIT = 50 * 1024 ** 2;

export interface Attachment {
  id: number;
  name: string;
  size: number;
  image: boolean;
  /** 0 to 1 while it uploads. */
  progress: number;
  /** Its path in the experiment, once saved. */
  path?: string;
  error?: string;
}

export const isImage = (name: string) => /\.(png|jpe?g|tiff?)$/i.test(name);

/** Why *file* isn't taken, before it is sent. */
function refused(file: File): string | undefined {
  const name = file.name.toLowerCase();
  if (!TAKEN.some((suffix) => name.endsWith(suffix))) {
    if (/\.docx?$/.test(name)) return "Word files aren't taken yet. Save it as text or Markdown.";
    if (/\.xlsx?$/.test(name)) return "Excel files aren't taken yet. Save it as CSV.";
    return "The agent can't read this kind of file.";
  }
  if (file.size > LIMIT) return "Over the 50 MB limit";
}

/** *file* into the open experiment's notes folder, *onProgress* as it goes. */
function upload(file: File, onProgress: (done: number) => void): Promise<FileEntry> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/app/experiment/upload?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Authorization", `Bearer ${getToken()}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      const body = JSON.parse(xhr.responseText || "{}");
      if (xhr.status < 300) resolve(body);
      else reject(new Error(body?.detail?.message ?? xhr.statusText));
    };
    xhr.onerror = () => reject(new Error("The upload didn't reach the server."));
    xhr.send(file);
  });
}

/** The files attached to the next message: each uploads as soon as it is added. */
export function useAttachments() {
  const [items, setItems] = useState<Attachment[]>([]);
  const next = useRef(0);
  const change = (id: number, values: Partial<Attachment>) =>
    setItems((all) => all.map((item) => (item.id === id ? { ...item, ...values } : item)));
  const add = (files: Iterable<File>) => {
    for (const file of files) {
      const id = next.current++;
      const error = refused(file);
      setItems((all) => [
        ...all,
        { id, name: file.name, size: file.size, image: isImage(file.name), progress: 0, error },
      ]);
      if (error) continue;
      upload(file, (progress) => change(id, { progress }))
        .then((entry) => change(id, { path: entry.path, progress: 1 }))
        .catch((e: Error) => change(id, { error: e.message }));
    }
  };
  return {
    items,
    add,
    remove: (id: number) => setItems((all) => all.filter((item) => item.id !== id)),
    clear: () => setItems([]),
    uploading: items.some((item) => !item.path && !item.error),
    /** The lines the message ends with, one per saved file. */
    lines: items.filter((item) => item.path).map((item) => attachedLine(item.path!, item.size)),
  };
}

export type Attachments = ReturnType<typeof useAttachments>;

const attachedLine = (path: string, size: number) => `Attached ${path} (${fmt.bytes(size)})`;
const ATTACHED = /^Attached (\S+) \(([^)]+)\)$/;

/** A sent message's text and the files it ends with. */
export function splitAttached(text: string): { text: string; files: { path: string; size: string }[] } {
  const lines = text.split("\n");
  const files = [];
  while (lines.length && ATTACHED.test(lines[lines.length - 1])) {
    const [, path, size] = lines.pop()!.match(ATTACHED)!;
    files.unshift({ path, size });
  }
  return { text: lines.join("\n").trimEnd(), files };
}

/** Whether a drag carries files. */
export const carriesFiles = (e: React.DragEvent) => e.dataTransfer.types.includes("Files");
