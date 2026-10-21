import { useEffect, useState } from "react";

// Pages are paths under /ui/; the server answers index.html for each.
export const PAGES = ["microscope", "experiment", "agent"] as const;
export type Page = (typeof PAGES)[number];

export function pageOf(path: string): Page {
  const first = path.replace(/^\/ui\/?/, "").split("/")[0];
  return (PAGES as readonly string[]).includes(first) ? (first as Page) : PAGES[0];
}

export function usePage(): [Page, (page: Page) => void] {
  const [page, setPage] = useState<Page>(() => pageOf(location.pathname));
  useEffect(() => {
    const back = () => setPage(pageOf(location.pathname));
    addEventListener("popstate", back);
    return () => removeEventListener("popstate", back);
  }, []);
  const go = (next: Page) => {
    history.pushState(null, "", `/ui/${next}`);
    setPage(next);
  };
  return [page, go];
}
