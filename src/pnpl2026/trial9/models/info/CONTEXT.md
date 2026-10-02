# trial9/models/ — the network

Trial 8's network with a fixed high-pass ahead of the instance norm.

## Router

| where | what |
|---|---|
| `../frontend.py` | `WindowHighpass` — **the one change**; the rest is trial 7's front end |
| `../encoder.py` | `PerWordEncoder(highpass_hz=...)` — runs the filter first, before the norm |
| `../net.py`, `../transformer.py`, `../heads.py` | unchanged from trial 8 |
| `../trunk.py`, `../norms.py` | unchanged from trial 6 |
| `../__init__.py` | `REGISTRY` (`trial9_net`) and `build_model` |

## Easy to get wrong here

**Filter before the norm.** A drift left in would enter the robust norm's
median and MAD, which is the thing the filter is there to stop.

**Real samples only.** The reflection is built from each window's own real
length; garbage past the end provably never reaches the output.

**No parameters.** The gain per FFT bin is a non-persistent buffer, so
checkpoints are unchanged in shape and `highpass_hz: null` reproduces trial 8
E0 bit for bit.
