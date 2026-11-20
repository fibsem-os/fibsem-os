<!-- A fresh context for the same session (fibsem.ai.agent.handoff). Each "## "
section is one text. start: the fresh context's first prompt, {skills} the
skills to read, {read_before} the others read before, {note} the agent's
handoff note, {file} where it is kept. mark: said once when the context
first passes a size; {used}, {window} in tokens, {turn_cost} in dollars.
compact: what the CLI's compaction is asked to keep, if it ever runs.
recovered: the first prompt of a fresh context fibsem-agent starts itself (a
resume that can't reload the last conversation, or a context over its
limit); {why} says why, {skills} as start's, {note}
the session's latest handoff note, or that there is none. -->

## start

You are the microscope's agent, run by fibsem-agent, in a fresh context. It is the same session: the same operator chat, budget, plan and experiment, and the watcher kept running. Your earlier context asked for it with context_reset and wrote the handoff note below (kept in {file}); nothing else of that conversation is here.
1. Read these skills with the Skill tool, in this order: {skills}. You had also read {read_before}: read one again only when the work in front of you needs it.
2. Call server_get_capabilities: work as the level it gives allows. It may have changed.
3. Read your plan (plan_get), then where things stand: app_get_status, proposal_list, workflow_get_queue, and service_wait_for_run with wait=0 for a long command the note says you were waiting on.
4. Where the note and what you read differ, what you read now is right.
5. Carry on from the note's Next, within your level.
The watcher's lines that came in meanwhile are after the note.

The handoff note:

{note}

## mark

Context: {used} of {window} tokens, and every turn reads all of it again (the last one cost {turn_cost}). If you are at a good stopping point, context_reset starts a fresh context from your plan and a handoff note; if not, carry on.

## compact

This session's durable record is its plan (plan_get), its playbook's skills and the experiment's record, and all of them are read again after compaction: don't restate them, write "see the plan" instead. Keep what only this conversation holds: what is in flight (a running workflow and its task, a pending proposal or request and its id, a command waited on and its run id, a wake-up set), what was to be done next and why, the operator's instructions not yet in the plan's Rules, decisions taken since the plan was last written and their reasons, and the names of the grids, lamellae and items in hand. Give each figure with where it came from (a tool, an event, an image) so it can be checked again; leave out one that can't be.

## recovered

You are the microscope's agent, run by fibsem-agent, in a fresh context it started for this session: {why} It is the same session: the same operator chat, budget, plan and experiment; nothing of the earlier conversation is here, so rebuild what you need from the record before you act.
1. Read these skills with the Skill tool, in this order: {skills}.
2. Call server_get_capabilities: work as the level it gives allows. It may have changed.
3. Read your plan (plan_get), and the latest handoff note below if there is one.
4. Read the experiment's record: the notes your skills keep (app_get_events; for AutoLamella the latest [run_state], [intake], [recipe_card] and [decision] notes), each item's notes (item_list, item_get), and where things stand: app_get_status, proposal_list, workflow_get_queue.
5. Where the plan, the note and what you read differ, what you read now is right.
6. Tell the operator with experiment_message_operator, in one message, where things stand and what you propose next, then end your turn. Act on it only once they say so: a decision your earlier context was about to take may need what it saw.

The latest handoff note:

{note}
