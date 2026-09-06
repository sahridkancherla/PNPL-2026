# Folder structure

**The standing convention.** Apply it to any folder you create or grow, at any
depth — not a description of what happens to exist.

## Code at the root, prose in `info/`

A folder keeps code (`.py`) at its root and its `.md` files in an `info/`
subfolder. **Only folders that actually have prose get one** — never create an
empty `info/`. A subfolder inside a folder that already has one gets its own;
that is normal, not duplication.

```
<any-folder>/
  thing.py                  code at the root
  info/
    CONTEXT.md              ALWAYS. the router
    <topic>/                optional, only once volume earns the split
      <one>.md
```

`info/` **stays flat until volume earns a split.** Start with `CONTEXT.md` plus
loose files; make a subfolder only when a category is worth naming.

⚠️ **The router is a file, not a folder.** The tempting alternative is sibling
folders by category — architecture / overviews / mechanisms. The middle one
never has a predictable boundary, so a reader who cannot guess opens all three,
which is the cost the split was meant to remove.

## `CONTEXT.md`

- **Router first, prose second.** Lead with a table of what is where.
- **Under 30 lines.** Growth is a signal that content belongs in a document the
  router points at, not in the router.
- **Never restate what lives elsewhere.** Link. A router that duplicates its
  destinations drifts and becomes misleading.
- A reader should enter any folder, open it, and know where to go without
  reading code.

## Applying it

- Creating a folder with prose? Make `info/CONTEXT.md` in the same change.
- Adding a `.md`? It goes in its own folder's `info/` — never beside the code,
  never in a sibling's.
- Adding to an existing `info/`? Update `CONTEXT.md` in the same change, or the
  router silently stops being one.

### Exceptions

- **`README.md` stays at the folder root**, not in `info/`. It is the front
  door someone lands on from a file browser; keep it short and point inward.
- **Repo-level record stores** keep whatever layout the repo defines for them.
- **Vendored third-party material** keeps the layout it shipped with.
- **Bulk data outside the repo** uses a plain `README.md` per folder. It holds
  no code, so there is nothing for an `info/` to sit beside.
