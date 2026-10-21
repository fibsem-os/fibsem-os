import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { FileRead } from "./types";

export const encodePath = (path: string) => path.split("/").map(encodeURIComponent).join("/");

/** A file in the experiment folder, read through the files route: an image's
 * preview at *width*, or a text file's text. */
export function useFileRead(path: string | null, width = 1024) {
  return useQuery({
    queryKey: ["app/experiment/files", path, "read", width],
    queryFn: () => api<FileRead>(`/app/experiment/files/${encodePath(path!)}?width=${width}`),
    enabled: !!path,
    staleTime: 60_000,
  });
}
