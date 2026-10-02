"""Trial 11: one stream per sentence, a Zipformer-style multi-rate encoder.

Each sentence's MEG stays one timeline: trial 7's front end and dilated conv
trunk run over it once, a stack of encoders at 50/25/12.5/6.25/12.5/25 Hz
works across the whole sentence, and every word is read off the shared
timeline at the end. Z0, the control, runs the same stacks all at 50 Hz.

Self-contained by design; nothing here imports from another trial. It reads
trial 7's continuous cache, which trial 7's builder writes.
"""
