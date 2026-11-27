import { useEffect } from "react";
import { ArrowLeft, Info } from "lucide-react";
import { useEventLoop } from "@/lib/events";
import { AgentPage } from "@/pages/Agent";

const AT = "/ui/hub/agent/";
const SESSION_AT = "/ui/hub/session/";

function named(path: string, at: string): string | null {
  if (!path.startsWith(at)) return null;
  const name = decodeURIComponent(path.slice(at.length).replace(/\/$/, ""));
  return name || null;
}

/** The instrument an Agent page at the Hub is for, from its path; null on any other page. */
export const agentInstrument = (path: string) => named(path, AT);

/** Where the Hub serves *name*'s Agent page. */
export const agentPagePath = (name: string) => AT + encodeURIComponent(name);

/** The session with no instrument an Agent page at the Hub is for, from its
 * path; null on any other page. */
export const agentSession = (path: string) => named(path, SESSION_AT);

/** Where the Hub serves the Agent page of a session with no instrument. */
export const sessionPagePath = (id: string) => SESSION_AT + encodeURIComponent(id);

/** Where the calls of that session's Agent page go: its own Agent Server on the hub PC. */
export const sessionApiBase = (id: string) => `/hub/agent-server/${encodeURIComponent(id)}`;

const INSTRUMENT_HINT =
  "This is the instrument's own Agent page, read through the Hub. From here you can message the agent, answer it, stop it and change its level, as far as the microscope PC allows. Everything else is done at the microscope PC.";
const SESSION_HINT =
  "This agent has no instrument. It runs on the hub PC, and its chat, plan and record are kept there. End it on the Hub's Agents page.";

/** An Agent page served by the Hub: the same page a microscope PC shows,
 * with every call sent where `setApiBase` says (set before this renders).
 * With *name*, an instrument's, passed through the Hub to its gateway, which
 * decides what is done from here. Without, a session with no instrument,
 * served by its own Agent Server on the hub PC. */
export function InstrumentAgentPage({ name, session }: { name?: string; session?: string }) {
  useEventLoop();
  const here = name !== undefined ? agentPagePath(name) : sessionPagePath(session ?? "");
  useEffect(() => {
    // A link on the page to another page (the microscope, a lamella): those
    // are not served here, so the address stays this page's.
    const stay = () => {
      if (location.pathname !== here) history.replaceState(null, "", here);
    };
    addEventListener("popstate", stay);
    return () => removeEventListener("popstate", stay);
  }, [here]);
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2.5 border-b border-border px-4 py-2 text-[13px]">
        <a
          href="/ui/hub"
          className="flex items-center gap-1 text-muted-foreground no-underline hover:text-foreground"
        >
          <ArrowLeft size={14} />
          Agent Hub
        </a>
        <span className="font-semibold">{name ?? "No instrument"}</span>
        <span className="text-muted-foreground">Agent</span>
        <span
          className="inline-flex text-muted-foreground"
          title={name !== undefined ? INSTRUMENT_HINT : SESSION_HINT}
        >
          <Info size={13} aria-label="About this page" />
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <AgentPage />
      </div>
    </div>
  );
}
