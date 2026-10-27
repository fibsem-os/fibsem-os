import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Download, FileImage, FileText, File as FileIcon, Folder } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle, Facts } from "@/components/ui/card";
import { ImageViewer, Legend, metaLine } from "@/components/images";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, getToken } from "@/lib/api";
import { useOpenExperiment } from "@/lib/experiment";
import { encodePath } from "@/lib/files";
import * as fmt from "@/lib/format";
import type { Capabilities, FileEntry, FileRead, FolderListing } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Replay } from "./experiment/Replay";

const ICON = { folder: Folder, image: FileImage, text: FileText, other: FileIcon };

// TIFFs open in the image viewer; a PNG or JPEG is a picture, shown as it is, in colour.
const viewable = (entry: FileEntry) => entry.kind === "image" && /\.tiff?$/i.test(entry.name);

// The folder and the view are in the address, so a reload or a link opens them again.
const param = (key: string) => new URLSearchParams(location.search).get(key);
function setParam(key: string, value: string) {
  const params = new URLSearchParams(location.search);
  if (value) params.set(key, value);
  else params.delete(key);
  const search = params.toString();
  history.replaceState(null, "", `${location.pathname}${search ? `?${search}` : ""}`);
}

/** The file as it is, through the token: a link can't carry the header. */
async function download(entry: FileEntry) {
  const response = await fetch(`/app/experiment/files/${encodePath(entry.path)}?raw=true`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!response.ok) return;
  const url = URL.createObjectURL(await response.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = entry.name;
  a.click();
  URL.revokeObjectURL(url);
}

function Metadata() {
  const { open, entry, loading } = useOpenExperiment();
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const history = useQuery({
    queryKey: ["app/experiment/task_history"],
    queryFn: () => api<{ items: Record<string, unknown>[] }>("/app/experiment/task_history"),
    enabled: !!open,
  });
  const ends = (history.data?.items ?? [])
    .map((r) => String(r.end_timestamp ?? r.completed_at ?? ""))
    .filter(Boolean)
    .sort();
  const last = ends.length ? ends[ends.length - 1] : null;
  if (!open) return null;
  if (loading || caps.isPending || history.isPending)
    return (
      <Card className="flex flex-col gap-3">
        <CardTitle>Experiment</CardTitle>
        <div className="text-[13px] text-muted-foreground">Reading the experiment…</div>
      </Card>
    );
  const microscope = [entry?.instrument_model, entry?.instrument_serial].filter(Boolean).join(" · ");
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle>Experiment</CardTitle>
      <Facts
        className="[&_dd]:text-left"
        rows={[
          ["Name", open.name],
          ["App", [caps.data?.app_name, caps.data?.fibsem?.version && `fibsemOS ${caps.data.fibsem.version}`].filter(Boolean).join(" · ") || "–"],
          ["Created", fmt.when(entry?.created_at) || "–"],
          ["Created with", entry?.software_version ? `fibsemOS ${entry.software_version}` : "–"],
          ["Microscope", microscope || "–"],
          ["Operator", entry?.operator || "–"],
          ["Items", String(open.num_items)],
          ["Last run", last ? `${fmt.when(last)} (${fmt.ago(last)})` : "None yet"],
          ["Folder", <span className="break-all">{open.path}</span>],
        ]}
      />
    </Card>
  );
}

/** The picked file: an image's preview, which opens it large, previous and next
 * through the folder's *images*; a text file's text. */
function Preview({ entry, images }: { entry: FileEntry; images: string[] }) {
  const [big, setBig] = useState(false);
  const read = useQuery({
    queryKey: ["app/experiment/files", entry.path, "read"],
    queryFn: () => api<FileRead>(`/app/experiment/files/${encodePath(entry.path)}?width=1024`),
    enabled: entry.kind === "image" || entry.kind === "text",
  });
  const p = read.data?.preview;
  const fm = read.data?.summary?.fm;
  return (
    <div className="flex min-w-0 flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-semibold">{entry.name}</span>
        <Button onClick={() => download(entry)} title="Download">
          <Download size={14} />
          Download
        </Button>
      </div>
      <div className="text-[12px] text-muted-foreground">
        {[fmt.bytes(entry.size), fmt.when(entry.modified)].filter(Boolean).join(" · ")}
      </div>
      {read.isLoading && <div className="text-[13px] text-muted-foreground">Reading…</div>}
      {read.error && <div className="text-[13px] text-destructive">{read.error.message}</div>}
      {p && (
        <>
          <button className="cursor-zoom-in rounded border-0 bg-black p-0" onClick={() => setBig(true)}>
            <img className="block w-full rounded" alt={entry.name} src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />
          </button>
          <div className="text-[12px] text-muted-foreground">{metaLine(entry.path, read.data?.summary)}</div>
          {fm && <Legend fm={fm} />}
          {viewable(entry) ? (
            <ImageViewer paths={images} at={big ? images.indexOf(entry.path) : null} onClose={() => setBig(false)} />
          ) : (
            <Dialog open={big} onOpenChange={setBig}>
              <DialogContent title={entry.name} className="w-[min(1200px,calc(100vw-32px))]">
                <div className="overflow-auto bg-black p-2">
                  <img className="block w-full" alt={entry.name} src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />
                </div>
              </DialogContent>
            </Dialog>
          )}
        </>
      )}
      {read.data?.text != null && (
        <pre className="m-0 max-h-[60vh] overflow-auto rounded bg-background p-3 text-[12px] whitespace-pre-wrap">
          {read.data.text}
          {read.data.truncated && "\n…"}
        </pre>
      )}
    </div>
  );
}

function Files() {
  const [dir, setDir] = useState(() => param("dir") ?? "");
  const [picked, setPicked] = useState<FileEntry | null>(null);
  const [filter, setFilter] = useState("");
  const listing = useQuery({
    queryKey: ["app/experiment/files", dir],
    queryFn: () => api<FolderListing>(`/app/experiment/files?dir=${encodeURIComponent(dir)}`),
  });
  const parts = dir && dir !== "." ? dir.split("/") : [];
  const go = (to: string) => {
    setDir(to);
    setParam("dir", to === "." ? "" : to);
    setPicked(null);
  };
  const shown = (listing.data?.entries ?? []).filter((e) => e.name.toLowerCase().includes(filter.toLowerCase()));
  return (
    <Card className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <nav aria-label="Folder" className="flex min-w-0 flex-1 flex-wrap items-center gap-1 text-[13px]">
          {[["Experiment", ""] as const, ...parts.map((part, i) => [part, parts.slice(0, i + 1).join("/")] as const)].map(
            ([name, to], i, all) => (
              <span key={to} className="flex items-center gap-1">
                {i > 0 && <ChevronRight size={12} className="text-muted-foreground" />}
                {i === all.length - 1 ? (
                  <span className="font-semibold">{name}</span>
                ) : (
                  <button className="cursor-pointer border-0 bg-transparent p-0 text-link" onClick={() => go(to)}>
                    {name}
                  </button>
                )}
              </span>
            ),
          )}
        </nav>
        <input
          aria-label="Filter files"
          placeholder="Filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-40 rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground"
        />
      </div>
      {listing.error && <div className="text-[13px] text-destructive">{listing.error.message}</div>}
      <div className={cn("grid items-start gap-4", picked && "lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]")}>
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="text-left text-[12px] text-muted-foreground">
              <th className="py-1 font-normal">Name</th>
              <th className="hidden py-1 font-normal sm:table-cell">Kind</th>
              <th className="py-1 text-right font-normal">Size</th>
              <th className="hidden py-1 pl-3 text-right font-normal sm:table-cell">Modified</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => {
              const Icon = ICON[e.kind];
              return (
                <tr
                  key={e.path}
                  className={cn("cursor-pointer border-t border-border hover:bg-accent", picked?.path === e.path && "bg-accent")}
                  onClick={() => (e.kind === "folder" ? go(e.path) : setPicked(e))}
                >
                  <td className="py-1.5 pr-2">
                    <span className="flex min-w-0 items-center gap-2">
                      <Icon size={14} className="shrink-0 text-muted-foreground" />
                      <span className="truncate">{e.name}</span>
                    </span>
                  </td>
                  <td className="hidden py-1.5 text-muted-foreground capitalize sm:table-cell">
                    {e.kind === "folder" && e.images ? `folder · ${e.images} images` : e.kind}
                  </td>
                  <td className="py-1.5 text-right text-muted-foreground tabular-nums">{fmt.bytes(e.size)}</td>
                  <td className="hidden py-1.5 pl-3 text-right whitespace-nowrap text-muted-foreground sm:table-cell">
                    {fmt.ago(e.modified)}
                  </td>
                </tr>
              );
            })}
            {listing.data && !shown.length && (
              <tr>
                <td colSpan={4} className="py-3 text-muted-foreground">
                  {filter ? "Nothing matches." : "This folder is empty."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {picked && <Preview key={picked.path} entry={picked} images={shown.filter(viewable).map((e) => e.path)} />}
      </div>
    </Card>
  );
}

const VIEWS = [
  ["files", "Files"],
  ["replay", "Replay"],
] as const;

export function ExperimentPage() {
  const { status, open } = useOpenExperiment();
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  const [view, setView] = useState<"files" | "replay">(() => (param("view") === "replay" ? "replay" : "files"));
  if (status.data && !open)
    return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const replay = caps.data?.parts?.includes("replay");
  const shown = replay ? view : "files";
  return (
    <div className="flex flex-col gap-3 p-4 md:p-5">
      {replay && (
        <div role="tablist" aria-label="View" className="flex gap-1">
          {VIEWS.map(([key, label]) => (
            <button
              key={key}
              role="tab"
              aria-selected={shown === key}
              onClick={() => {
                setView(key);
                setParam("view", key === "files" ? "" : key);
              }}
              className={cn(
                "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[13px]",
                shown === key
                  ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {shown === "files" ? (
        <div className="grid items-start gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <Metadata />
          {open && <Files key={open.path} />}
        </div>
      ) : (
        open && <Replay key={open.path} />
      )}
    </div>
  );
}
