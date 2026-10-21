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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
