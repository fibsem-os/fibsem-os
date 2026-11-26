import { useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { browserName, setToken, unauthed } from "@/lib/api";

interface Paired {
  token: string;
  id: string;
}

/** This browser's own id, made once and kept: pairing it again replaces its
 * old row in Settings rather than adding one. */
function browserId(): string {
  try {
    let id = localStorage.getItem("fibsem-browser-id");
    if (!id) {
      id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
      localStorage.setItem("fibsem-browser-id", id);
    }
    return id;
  } catch {
    return "";
  }
}

/** No token: pair this browser with the code the microscope PC shows. */
export function Pair({ note }: { note?: string }) {
  const can = useQuery({
    queryKey: ["pair"],
    queryFn: () => unauthed<{ open: boolean; request: boolean }>("/pair"),
  });
  const [code, setCode] = useState("");
  const [name, setName] = useState(browserName);
  const [operatorToken, setOperatorToken] = useState("");
  const [said, setSaid] = useState(note ?? "");

  const pair = async (event: FormEvent) => {
    event.preventDefault();
    if (!code.trim()) return;
    try {
      const answer = await unauthed<Paired>("/pair", { code: code.trim(), name: name.trim(), browser: browserId() });
      try {
        localStorage.setItem("fibsem-paired-id", answer.id);
      } catch {
        // storage off
      }
      setToken(answer.token, "paired");
    } catch (error) {
      setSaid((error as Error).message);
    }
  };

  const ask = async () => {
    try {
      const answer = await unauthed<{ opened: boolean }>("/pair/request", {});
      setSaid(
        answer.opened
          ? "A code is in the server's terminal."
          : "A code is already open, or pairing is resting after wrong codes. Use the code in the terminal.",
      );
    } catch (error) {
      setSaid((error as Error).message);
    }
  };

  return (
    <main className="flex min-h-full items-center justify-center p-4">
      <form
        onSubmit={pair}
        className="flex w-[360px] max-w-full flex-col gap-4 rounded-lg border border-border bg-card p-6"
      >
        <h1 className="m-0 text-lg font-semibold">Pair this browser</h1>
        {can.data?.request ? (
          <Button type="button" variant="ghost" className="self-start px-0 text-link" onClick={ask}>
            Show a code on the server
          </Button>
        ) : (
          <p className="m-0 text-[13px] text-muted-foreground">
            At the microscope PC: Tools → App Server → Pair a computer.
          </p>
        )}
        <label className="flex flex-col gap-1.5 text-[13px]">
          Code
          <input
            value={code}
            onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 4))}
            inputMode="numeric"
            autoComplete="off"
            autoFocus
            className="rounded border border-input bg-background p-2 text-center text-[28px] tracking-[0.3em] outline-none focus:border-primary"
          />
        </label>
        <label className="flex flex-col gap-1.5 text-[13px]">
          Name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="rounded border border-border bg-background p-2 outline-none focus:border-primary"
          />
        </label>
        <Button type="submit" variant="primary" className="py-2.5" disabled={code.length !== 4}>
          Pair
        </Button>
        {said && <p className="m-0 text-[13px] text-muted-foreground">{said}</p>}
        <details className="text-[13px] text-muted-foreground">
          <summary className="cursor-pointer">Use the operator token instead</summary>
          <div className="mt-2 flex gap-2">
            <input
              type="password"
              value={operatorToken}
              onChange={(e) => setOperatorToken(e.target.value)}
              aria-label="Operator token"
              className="min-w-0 flex-1 rounded border border-border bg-background p-2 text-foreground outline-none focus:border-primary"
            />
            <Button
              type="button"
              onClick={() => {
                const value = operatorToken.trim();
                if (value) setToken(value, "operator");
              }}
            >
              Open
            </Button>
          </div>
        </details>
      </form>
    </main>
  );
}
