import { useEffect } from "react";
import { ArrowLeft, Info } from "lucide-react";
import { useEventLoop } from "@/lib/events";
import { AgentPage } from "@/pages/Agent";

const AT = "/ui/hub/agent/";

/** The instrument an Agent page at the Hub is for, from its path; null on any other page. */
export function agentInstrument(path: string): string | null {
  if (!path.startsWith(AT)) return null;
  const name = decodeURIComponent(path.slice(AT.length).replace(/\/$/, ""));
  return name || null;
}

/** Where the Hub serves *name*'s Agent page. */
export const agentPagePath = (name: string) => AT + encodeURIComponent(name);

/** An instrument's Agent page, served by the Hub: the microscope PC's own
 * page, with every call passed through the Hub to that instrument's gateway
 * (`setApiBase`, set before this renders). Its gateway decides what is done
 * from here; the other pages stay at the microscope PC. */
export function InstrumentAgentPage({ name }: { name: string }) {
  useEventLoop();
  useEffect(() => {
    // A link on the page to another page (the microscope, a lamella): those
    // are not served here, so the address stays this page's.
    const stay = () => {
      if (agentInstrument(location.pathname) === null) history.replaceState(null, "", agentPagePath(name));
    };
    addEventListener("popstate", stay);
    return () => removeEventListener("popstate", stay);
  }, [name]);
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
        <span className="font-semibold">{name}</span>
        <span className="text-muted-foreground">Agent</span>
        <span
          className="inline-flex text-muted-foreground"
          title="This is the instrument's own Agent page, read through the Hub. From here you can message the agent, answer it, stop it and change its level, as far as the microscope PC allows. Everything else is done at the microscope PC."
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
