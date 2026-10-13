"""Test-time augmentation for lone words: P7 of the 13 Oct research note.

The 92 lone words are scored only on the private board, from one 1 s window
each. The note's low-risk upgrade averages predictions over:

    onset jitter      shifts of -1, 0 and +1 samples at 50 Hz (+/-20 ms). A
                      lone epoch starts at its onset, so a negative shift
                      repeats its first sample (`pad_start`)
    sensor dropout    passes with the spatial merger's training-time disc
                      dropout switched on -- and NOTHING else: BatchNorm,
                      dropout layers and the rest stay in eval mode

Each pass's scores are z-scored per row and averaged, as cli.ensemble
averages files.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def merger_modules(module):
    """Every spatial merger inside a network, of either family.

    ARGS:
        module (torch.nn.Module)

    RETURNS:
        list of nn.Module whose class is ChannelMerger
    """
    return [child for child in module.modules() if type(child).__name__ == "ChannelMerger"]


@contextmanager
def sensor_dropout(module, seed):
    """Inside the block, only the mergers' sensor-disc dropout runs as in training.

    ARGS:
        module (torch.nn.Module): in eval mode
        seed (int): the draw of this pass's discs

    YIELDS:
        int: how many mergers were switched
    """
    mergers = merger_modules(module)
    states = [merger.training for merger in mergers]
    torch.manual_seed(seed)
    try:
        #train mode on the mergers alone: their forward reads `self.training` for the disc
        for merger in mergers:
            merger.train()
        yield len(mergers)
    finally:
        for merger, state in zip(mergers, states):
            merger.train(state)


def pad_start(epoch, samples):
    """An epoch with its first sample repeated in front, for a negative onset shift.

    ARGS:
        epoch (np.ndarray): [306, n]
        samples (int): how many samples to prepend

    RETURNS:
        [306, n + samples]
    """
    if samples <= 0:
        return epoch
    return np.pad(epoch, ((0, 0), (samples, 0)), mode="edge")
