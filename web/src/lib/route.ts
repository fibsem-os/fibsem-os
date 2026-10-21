import { useEffect, useState } from "react";

// Pages are paths under /ui/; the server answers index.html for each.
export const PAGES = ["microscope"] as const;
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

const SETTINGS = "/ui/settings";
const atSettings = () => location.pathname.replace(/\/$/, "") === SETTINGS;

/** Settings is a dialog over the page with its own path, so a link or a reload keeps it open. */
export function useSettingsPath(page: Page): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(atSettings);
  const [pushed, setPushed] = useState(false);
  useEffect(() => {
    const back = () => setOpen(atSettings());
    addEventListener("popstate", back);
    return () => removeEventListener("popstate", back);
  }, []);
  const set = (next: boolean) => {
    if (next === open) return;
    if (next) {
      history.pushState(null, "", SETTINGS);
      setPushed(true);
    } else if (pushed) {
      history.back();
      setPushed(false);
    } else {
      history.replaceState(null, "", `/ui/${page}`);
    }
    setOpen(next);
  };
  return [open, set];
}
