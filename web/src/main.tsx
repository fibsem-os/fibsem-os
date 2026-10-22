import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ApiError } from "./lib/api";
import "./index.css";

// A 4xx is an answer, not a hiccup: asking again gets the same one.
const retry = (failures: number, error: unknown) =>
  failures < 1 && !(error instanceof ApiError && error.status >= 400 && error.status < 500);

const client = new QueryClient({
  defaultOptions: { queries: { retry, refetchOnWindowFocus: false, staleTime: 2000 } },
});

// An image read is an object URL (imageUrl). It is freed when the cache drops
// or replaces it, never by a component: others may be showing the same one.
const blobs = new Map<string, string>();
client.getQueryCache().subscribe(({ type, query }) => {
  if (type !== "updated" && type !== "removed") return;
  const held = blobs.get(query.queryHash);
  const now = type === "removed" ? undefined : query.state.data;
  if (held && held !== now) {
    URL.revokeObjectURL(held);
    blobs.delete(query.queryHash);
  }
  if (typeof now === "string" && now.startsWith("blob:")) blobs.set(query.queryHash, now);
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
