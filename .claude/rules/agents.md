# Subagents — never spin one out without explicit approval

**This is a hard rule and it exists to protect usage. Treat it as binding, not
advisory.**

## The rule

**Never spawn a subagent unless the user has explicitly asked for one in the
current conversation.** This covers every mechanism that creates one:

- the `Agent` / `Task` tool, in foreground or background
- the `Workflow` tool, which can spawn dozens at once
- any skill or command whose effect is to fan work out to agents
- parallel agents launched "to save time"

Default to doing the work yourself, in this session, with ordinary tools.

## What counts as approval

Only an explicit request from the user, in their own words, in this
conversation. For example: *"use a subagent"*, *"spawn an agent to..."*,
*"fan this out"*, *"run a workflow"*, *"use ultracode"*.

## What does NOT count as approval

- The task being large, slow, or parallelisable.
- The task spanning many files, so a search agent "would be more efficient".
- Having spawned an agent earlier in the session. Approval is per-request and
  does not carry forward.
- Judging that the user would probably want it.
- A skill's own instructions suggesting it, absent a user request.

If a subagent looks genuinely worthwhile, **say so in one line and ask**. Do
not launch it and explain afterwards.

## The one exception

Where a repo defines agents in `.claude/agents/`, those agents **may** run when
their own trigger conditions fire — that is the point of defining them, and
adding a definition constitutes standing approval for that specific agent under
its stated trigger.

Nothing else is covered by that exception. An agent that is not defined in
`.claude/agents/` still requires explicit per-request approval, and a defined
agent invoked outside its trigger conditions does too.

## Why

Subagents multiply token usage — a fan-out can cost many times a direct
answer, and a workflow can cost orders of magnitude more. The user is managing
a finite budget across a long project. An unrequested agent spends that budget
on their behalf without their knowledge, which is the specific outcome this
rule prevents.
