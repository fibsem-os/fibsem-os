import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";

/** An operator call that re-reads the agent after, and keeps its refusal to show. */
export function useAct() {
  const client = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const act = async (path: string, init: RequestInit = { method: "POST" }) => {
    setError(null);
    try {
      return await api<unknown>(path, init);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      await client.invalidateQueries({ queryKey: ["app/agent"] });
      await client.invalidateQueries({ queryKey: ["app/agent/recent"] });
    }
  };
  return { act, error, setError };
}
