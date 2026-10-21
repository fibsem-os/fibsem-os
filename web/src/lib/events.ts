import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { api, getToken } from "./api";

interface EventsAnswer {
  latest_seq?: number;
}

/** The app's one clock: a long-poll on /app/events. Any news marks every read
 * stale, and the pages on screen read again. */
export function useEventLoop() {
  const client = useQueryClient();
  useEffect(() => {
    let stopped = false;
    let since = 0;
    (async () => {
      while (!stopped && getToken()) {
        try {
          const answer = await api<EventsAnswer>(`/app/events?since=${since}&timeout=25`);
          const latest = answer.latest_seq ?? 0;
          if (latest !== since) {
            since = latest;
            await client.invalidateQueries();
          }
        } catch {
          await new Promise((resolve) => setTimeout(resolve, 3000));
        }
      }
    })();
    return () => {
      stopped = true;
    };
  }, [client]);
}
