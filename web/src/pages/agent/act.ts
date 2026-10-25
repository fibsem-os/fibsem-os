import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api } from "@/lib/api";

/** An operator call that re-reads the agent after, and keeps its refusal to show.
 * One at a time: a second click while the first is answered sends nothing. */
export function useAct() {
  const client = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sending = useRef(false);
  const act = async (path: string, init: RequestInit = { method: "POST" }) => {
    if (sending.current) return undefined;
    sending.current = true;
    setBusy(true);
    setError(null);
    try {
      return await api<unknown>(path, init);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      sending.current = false;
      setBusy(false);
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
    }
  };
  return { act, busy, error, setError };
}
