# trial11/models/ — the network

## Router

| where | what |
|---|---|
| `../net.py` | `StreamZipformerNet` — front end, trunk, encoder, per-word readout, heads |
| `../zipformer.py` | `Downsample`, `upsample`, `Bypass`, `ZipformerStack`, `ZipformerEncoder` — **the new part** |
| `../transformer.py` | `TransformerBlock` and `attention_mask`, unchanged from trial 8 |
| `../frontend.py` | robust norm, low-pass decimation, noise, component dropout — trial 7's |
| `../trunk.py`, `../norms.py` | unchanged from trial 6 |
| `../heads.py` | `AttentionPool` (the word readout), `ClosedSetHead`, `SemanticHead` |
| `../__init__.py` | `REGISTRY` (`trial11_net`) and `build_model` |

## Easy to get wrong here

**Every stream is padded to the SAME length.** The trunk's GroupNorm sees the
padding, so padding to each batch's longest stream would make a stream's
output depend on its batch-mates. Fixed 4.8 s makes it identical alone or in
any batch (checked: difference 1.8e-7).

**Every stack factor must tile the frame count.** 240 frames divide by 1, 2,
4, 8 and 16; the config refuses a layout that does not.

**Downsampling skips padding.** Each token is a weighted average of its real
frames only, renormalised; a token is real if any of its frames is.

**Words never cross streams.** The readout gathers from the word's own stream
by index; perturbing one stream moves only its own words (checked).
