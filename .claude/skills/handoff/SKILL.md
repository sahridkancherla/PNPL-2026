---
name: handoff
description: Write or update a handoff document so a feature's full context and direction survive context compaction. Use after a long brainstorm or design discussion, at a checkpoint in a large task, before pausing work, or whenever context is filling up and the reasoning behind decisions would otherwise be lost.
---

# Handoff documents

A handoff document exists for one reason: **a future Claude with no memory of
this conversation must be able to pick the work up and take the right next
action without re-asking, re-deciding, or re-discovering.**

Compaction destroys reasoning first and facts second. Facts are usually
recoverable from the repo; the *why* behind a decision is not. That asymmetry
determines what goes in.

---

## Rules

- **One document per feature**, at `handoff-docs/big/<feature>.md` for work
  spanning many sessions, `handoff-docs/small/<feature>.md` for work that
  finishes in one. Most features are big; small is the rare case.
- **Updated in place.** It is the *current state* of the feature, not a journal.
  Rewrite stale sections; do not append a new dated entry each session.
- **Never duplicate content that lives elsewhere.** Link to it. If the
  run configuration lives in `configs/`, point there — a handoff that restates it will
  drift out of sync and become actively misleading.
- **Direction is mandatory, state is not sufficient.** A document that says what
  exists but not where it is going has failed its purpose.

---

## When to write or update one

- After a substantial brainstorm or design discussion — **before** the reasoning
  gets compacted away.
- At a checkpoint in a long task, especially before launching something slow.
- Before pausing or handing off.
- When a decision is made that closes off alternatives.

---

## What it must capture

**1. Goal** — one or two sentences. What "done" looks like, concretely.

**2. Current state** — what exists, what works, what is broken, what is
untested. Distinguish *verified* from *assumed*; say which is which.

**3. Direction** — the plan, and the **next concrete action**. Not "continue
improving the model" but the actual next step, specific enough to start on.

**4. Decisions made, with the reason** — and, critically, **the alternatives
rejected and why**. This is the highest-value section. Without it a fresh
Claude will confidently re-propose an option that was already considered and
ruled out, and the loop can repeat indefinitely.

**5. Constraints** — what must hold. Environment, hardware, data location,
rules, anything that invalidates work if violated.

**6. Open questions** — what is genuinely undecided, and **what would resolve
it**. An open question without a resolution path is just anxiety.

**7. Pointers** — where the detail lives.

Order these by what a reader needs first. Goal and next action near the top.

---

## Template

```markdown
# <feature>

**Goal**: <one or two sentences — what done looks like>

**Status**: <not started | in progress | blocked | done> — <one line>

## Next action
<the specific next step, concrete enough to begin immediately>

## Current state
- <what exists and works — verified how>
- <what is broken or untested — say which>

## Decisions
| decision | why | rejected alternative | why not |
|---|---|---|---|
| ... | ... | ... | ... |

## Constraints
- <things that must hold>

## Open questions
- <question> — resolved by: <what would settle it>

## Pointers
- <where the detail lives>
```

Drop sections that are genuinely empty. Do not pad.

---

## Quality bar

Before saving, read it as though you know nothing else about the project and
ask:

1. Do I know what to do next, specifically enough to start?
2. Do I know why the current approach was chosen over the obvious alternatives?
3. Would I avoid re-proposing something already rejected?
4. Can I tell what is measured from what is assumed?

If any answer is no, the document is not finished.

---

## Anti-patterns

- **Narrating history** — "first I tried X, then Y, then Z". Record the
  conclusion and the reason, not the path.
- **Restating content from elsewhere** — it will drift and mislead. Link.
- **Vague next steps** — "keep working on it" carries no information.
- **Recording only what worked.** Negative results are the cheapest thing to
  lose and the most expensive to rediscover. A rejected alternative with its
  reason is worth more than another paragraph on the chosen one.
- **Stating assumptions as facts.** If it was not verified, say so; a confident
  wrong fact is worse than an acknowledged gap.
