# Git workflow

**`main` is home.** Branches are short detours, cut only when asked, merged
back the same day.

⚠️ **Overrides two Claude Code defaults.** The built-in behaviour is "commit or
push only when asked" and "if on the default branch, branch first". Here:
**commit freely, never push unasked, never branch unasked.**

## Commits

### ⚠️ The unit is a working thing, not an attempt

A step is an increment that **works** — one module, one stage, one measurement
reproduced. The tries on the way are the *inside* of that step. A few commits
per session is right; 10+ logs progress instead of outcomes. Size never
triggers a commit.

```
<type>(<scope>): <imperative summary, ≤72 chars>

<what changed, and what state it is in — name anything currently broken>
```

Scope is the area, not the file. `wip` is not a description.

| type | use it when |
|---|---|
| **`fall`** | **FALLBACK** — a savepoint you may want back: before a merge or rebase, and before a change the user might reject on taste |
| **`check`** | **CHECKPOINT** — a real step of a big feature: works for one module, one stage, one config |
| **`feat`** | the feature works, whole. Lands **once** per feature |
| **`fix`** | something that USED TO WORK stopped |
| **`refactor`** | behaviour identical, structure better |
| **`test`** | tests only |
| **`docs`** | prose only — records, `info/`, `.claude/`, comments |
| **`chore`** | deps, config, tooling |
| **`perf`** | same behaviour, measurably faster — say what you measured |

⚠️ **`fix` is not for your own fresh code.** One question: *was there ever a
version of this that worked?* No → it is a `check`, or nothing yet.

⚠️ **`feat` is a claim.** One needing a `fix` twenty minutes later was a
`check` misnamed.

⚠️ **`fall` is the one that gets forgotten, and the expensive one.** Commit
*before* a change whose outcome is uncertain. Scope names the thing at risk;
the message says what is being preserved:
`fall(block): the four-branch design, before trying plain sequential`.
Claude makes it unprompted. **Never `--amend` one** — it exists to be reverted to.

## Pushing and merging

**Claude prepares the push; the user fires it.**

⚠️ **Check the remote's actual name first.** It is not always `origin`, and a
branch with no upstream fails with a message that does not say so.
`git remote -v` and `git branch -vv` answer both in seconds.

⚠️ **A merge can delete work with no conflict at all.** A stale branch that
predates a change takes the older side cleanly, and nothing is flagged.

```bash
git diff --diff-filter=D --name-only HEAD    # deletions — read every one
```

Every deletion must be yours or explained — an unexplained one is an incident,
so stop and ask. A conflict in a file you do not understand is not yours to
decide. **Never commit a merge you have not diffed.** `git merge --abort` is
free and always correct.

## Common sense

- **Never open a branch unless asked.** A branch cut on Claude's own initiative
  is one nobody else knows exists. When asked: `git checkout -b feat/<subject>`
  off a fresh `main`, merge back when it **works**, delete it.
- **After a merge, move to the branch you merged INTO**, and say which branch
  you are on. Staying on the absorbed one is how it silently forks again.
- **Before anything risky:** `git branch backup/pre-<thing>-$(date +%F)`.
- **Never commit bulk data** — caches, checkpoints, weights, datasets, logs.
  If no `.gitignore` covers them, that is the first thing to fix.
