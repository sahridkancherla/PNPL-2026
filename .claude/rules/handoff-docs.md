# Handoff docs — only when Kamil asks

**Never create or update anything in `handoff-docs/` unless Kamil asks for it
in the current conversation.**

This includes adding a section to an existing doc, recording a decision, and
logging something as deferred. "This seems worth not losing" is not approval.

## What counts as approval

Only an explicit request, in Kamil's own words, in this conversation — *"write
a handoff doc"*, *"update the handoff"*, *"record this in handoff-docs"*, or
invoking the `handoff` skill.

**Not** approval: the task being large, a decision feeling significant,
context filling up, having been asked to update one earlier in the session, or
a rule elsewhere saying context should survive compaction.

## Where things go instead

| what | where |
|---|---|
| a temporary or v1 choice | ⚠️ note in the **docstring of the code it governs** |
| the maths or the argument behind a choice | that folder's `info/maths/<topic>.md`, and only once the code is agreed |
| what currently exists in a folder | that folder's `info/CONTEXT.md` — **current state only, no history** |
| anything else | say it in the conversation and let Kamil decide |

## Why

Handoff docs are Kamil's record of where the project is. A doc written
unprompted states decisions he has not made in words he did not choose, and it
is harder to notice and remove than code is. The repo already carries more
unread prose than it needs.
