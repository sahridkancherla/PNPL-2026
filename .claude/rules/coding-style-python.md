---
description: Python coding style — naming, docstrings, comments, PEP 8
globs: "**/*.py"
---

# Python style

PEP 8, with three things weighted above it: **names, docstrings, comments.**

## Names carry the explanation

A reader should understand a name without reading its definition. Spell it out
— `centre_hz` not `f0`, `build_filters` not `mk`. Length is cheap.

- Name for **what it holds or does**, never its shape or type: `windows`,
  not `arr3d`.
- **No single letters** except a loop index, or `x` threaded through a forward
  pass.
- Functions are verbs (`cut_window`); values are nouns. Booleans read as
  assertions: `with_phonemes`, `is_valid`.
- `snake_case`, `PascalCase` for classes, `UPPER_SNAKE` for constants.
  Never shadow builtins (`id`, `type`, `input`, `list`).

## Docstrings

Every function, class and module. Drop `ARGS`/`RETURNS` only when there are
genuinely none.

```python
def cut_window(data, onset_s, sample_rate=250.0, bounds=(-0.32, 1.28)):
    """Cut one word window out of a continuous recording.

    Bounds are relative to word onset. Windows falling off either end of the
    recording are rejected, not zero-padded.

    ARGS:
        data (np.ndarray): the continuous recording, [n_channels, n_times]
        onset_s (float): word onset in seconds from the start of the run
        sample_rate (float): samples per second of `data`

    RETURNS:
        window : [n_channels, n_samples] - one trial, not normalised
    """
```

Brief in the imperative. For arrays, give the shape in brackets then say what
the axes mean.

## Comments

**One at least every five lines**, even where it looks obvious as you write
it — it will not be obvious to the next reader, or to you in six weeks.

**Own line, directly above the code. Never trailing.**

**One line each.** Two-liners are rare, for something genuinely subtle. If a
comment needs three, it belongs in the docstring or in a record.

```python
# Whiten to the effective rank; past ~73 components is amplified noise.
whitened = whitener @ centred
```

Not `whitened = whitener @ centred  # whiten`. Complete sentences, capitalised.
Prefer *why*, but restating *what* beats no comment.

## The rest is PEP 8

- 4 spaces, 88-character lines, two blank lines between top-level definitions.
- Imports at top, grouped stdlib / third-party / local. Never `import *`.
- `if x is None`, not `== None`. `if not items` for emptiness — different
  tests, since `0`, `""` and empty arrays are all falsy.
- **No type annotations in signatures** -- not `-> np.ndarray` on the end, not
  `data: np.ndarray` on a parameter. The docstring already gives the type and
  the shape, which is the part worth reading. Never a mutable default argument.
- f-strings, double quotes, context managers for files.
- Annotate array shapes at function boundaries: `# [batch, n_features, n_times]`.
- **Always pass an explicit `dim` / `axis`** to `mean`, `sum`, `softmax`,
  `argmax`, `squeeze`. An implicit reduction over everything is almost never
  meant, and will not raise.
