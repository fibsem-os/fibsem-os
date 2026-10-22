import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { FileRead } from "./types";

export const encodePath = (path: string) => path.split("/").map(encodeURIComponent).join("/");

/** Which of a fluorescence stack to show: plane *z* (null, the max projection)
 * in *channels* (null, all). */
export interface StackView {
  z: number | null;
  channels: string[] | null;
}

/** A file in the experiment folder, read through the files route: an image's
 * preview at *width* (a fluorescence stack's *view* of it), or a text file's text. */
export function useFileRead(path: string | null, width = 1024, view?: StackView) {
  const query = new URLSearchParams({ width: String(width) });
  if (view?.z != null) query.set("z", String(view.z));
  if (view?.channels) query.set("channels", view.channels.join(","));
  return useQuery({
    queryKey: ["app/experiment/files", path, "read", query.toString()],
    queryFn: () => api<FileRead>(`/app/experiment/files/${encodePath(path!)}?${query}`),
    enabled: !!path,
    staleTime: 60_000,
    placeholderData: (previous, last) => (last?.queryKey[1] === path ? previous : undefined),
  });
}
