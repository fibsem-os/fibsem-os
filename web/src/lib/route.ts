import { useEffect, useState } from "react";

// Pages are paths under /ui/; the server answers index.html for each.
export const PAGES = ["microscope", "experiment", "lamellae", "grids", "agent"] as const;
export type Page = (typeof PAGES)[number];

export function pageOf(path: string): Page {
  const first = path.replace(/^\/ui\/?/, "").split("/")[0];
  return (PAGES as readonly string[]).includes(first) ? (first as Page) : PAGES[0];
}

export type Go = (page: Page, item?: string) => void;

/** The item a page was opened on: /ui/lamellae?item=<name>. */
export const itemOf = () => new URLSearchParams(location.search).get("item");

export function usePage(): [Page, Go] {
  const [page, setPage] = useState<Page>(() => pageOf(location.pathname));
  useEffect(() => {
    const back = () => setPage(pageOf(location.pathname));
    addEventListener("popstate", back);
    return () => removeEventListener("popstate", back);
  }, []);
  const go: Go = (next, item) => {
    history.pushState(null, "", `/ui/${next}${item ? `?item=${encodeURIComponent(item)}` : ""}`);
    setPage(next);
  };
  return [page, go];
}
