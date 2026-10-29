import { File as FileIcon, FileImage, X } from "lucide-react";
import { useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useFileRead } from "@/lib/files";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";
import { FilePreview } from "../Experiment";
import { isImage, type Attachment, type Attachments } from "./attach";

const viewable = (name: string) => /\.tiff?$/i.test(name);

function Chip({ item, remove }: { item: Attachment; remove: () => void }) {
  const Icon = item.image ? FileImage : FileIcon;
  const uploading = !item.path && !item.error;
  return (
    <div
      className={cn(
        "flex max-w-80 flex-col gap-1 rounded-md border bg-background px-2 py-1 text-[12px]",
        item.error ? "border-destructive" : "border-input",
      )}
    >
      <div className={cn("flex items-center gap-1.5", item.error && "text-destructive")}>
        <Icon size={13} className="shrink-0" />
        <span className="min-w-0 truncate">{item.name}</span>
        <span className="shrink-0 text-muted-foreground">{fmt.bytes(item.size)}</span>
        <button
          className="ml-1 flex cursor-pointer rounded border-0 bg-transparent p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
          aria-label={`Remove ${item.name}`}
          onClick={remove}
        >
          <X size={12} />
        </button>
      </div>
      {item.error && <div className="text-destructive">{item.error}</div>}
      {uploading && (
        <>
          <div className="h-[3px] rounded-sm bg-accent">
            <div className="h-full rounded-sm bg-primary" style={{ width: `${Math.round(item.progress * 100)}%` }} />
          </div>
          <div className="text-muted-foreground">Uploading · {Math.round(item.progress * 100)}%</div>
        </>
      )}
    </div>
  );
}

/** The files attached to the message being written. */
export function AttachedChips({ attach }: { attach: Attachments }) {
  if (!attach.items.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 px-2.5 pt-2">
      {attach.items.map((item) => (
        <Chip key={item.id} item={item} remove={() => attach.remove(item.id)} />
      ))}
    </div>
  );
}

function Thumbnail({ path }: { path: string }) {
  const read = useFileRead(path, 360);
  const p = read.data?.preview;
  if (!p) return <div className="h-[110px] w-[180px] bg-accent" />;
  return (
    <img
      className="block h-[110px] w-[180px] object-cover"
      alt={path}
      src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
    />
  );
}

/** The files a sent message carried: each opens as Experiment Files shows it. */
export function SentFiles({ files }: { files: { path: string; size: string }[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!files.length) return null;
  const name = (path: string) => path.split("/").pop() ?? path;
  return (
    <div className="mt-1.5 flex flex-wrap items-end gap-2">
      {files.map(({ path, size }) => {
        const image = isImage(path);
        const Icon = image ? FileImage : FileIcon;
        return (
          <button
            key={path}
            className="cursor-pointer overflow-hidden rounded-md border border-input bg-transparent p-0 text-left text-[12px] text-foreground hover:bg-accent"
            onClick={() => setOpen(path)}
          >
            {image && <Thumbnail path={path} />}
            <span className="flex items-center gap-1.5 px-2 py-1">
              <Icon size={13} className="shrink-0" />
              {path}
              <span className="text-muted-foreground">{size}</span>
            </span>
          </button>
        );
      })}
      <Dialog open={open != null} onOpenChange={(o) => !o && setOpen(null)}>
        {open && (
          <DialogContent title={name(open)}>
            <div className="overflow-auto p-4">
              <FilePreview
                entry={{
                  name: name(open),
                  path: open,
                  kind: isImage(open) ? "image" : "text",
                  size: null,
                  modified: "",
                }}
                images={viewable(open) ? [open] : []}
              />
            </div>
          </DialogContent>
        )}
      </Dialog>
    </div>
  );
}
