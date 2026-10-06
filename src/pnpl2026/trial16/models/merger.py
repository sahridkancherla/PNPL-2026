"""The spatial front end: 306 sensors to 270 channels, three ways.

    ChannelMerger   the reference's (Défossez et al. 2023, brainmagick
                    `ChannelMerger`): every output channel is a softmax-weighted
                    mix of sensors, the weights a learned function of each
                    sensor's 2D POSITION. Training drops a random disc of sensors
    LinearSpatial   a learned 1x1 map, the fallback if the h5 files carry no
                    channel names. For one subject the merger is a constrained
                    linear map anyway, so this is a superset of it
    FixedPCA        F3: trial 1-8's input -- a fixed whitening PCA to 73
                    components, fitted on the training partition at build time

Plus `SubjectLayer`, brainmagick's per-subject linear map. With one subject it
is a single learned matrix, identity at initialisation.

⚠️ Channels FIRST throughout, `[batch, channels, time]`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """brainmagick's `FourierEmb`: 2D positions to cosine and sine features.

    ATTRS:
        n_freqs (int): frequencies per axis; the output is 2 * n_freqs^2 wide
        margin (float): positions are padded by this before the frequencies
    """

    def __init__(self, dimension=2048, margin=0.2):
        """
        ARGS:
            dimension (int): output width; dimension / 2 must be a square
            margin (float): brainmagick's 0.2
        """
        super().__init__()
        n_freqs = int(round((dimension // 2) ** 0.5))
        if 2 * n_freqs**2 != dimension:
            raise ValueError(f"dimension {dimension} is not 2 * a square number")
        self.n_freqs = n_freqs
        self.margin = margin

    def forward(self, positions):
        """
        ARGS:
            positions (torch.Tensor): [n_sensors, 2] in [0, 1]

        RETURNS:
            [n_sensors, 2 * n_freqs^2]
        """
        freqs_y = torch.arange(self.n_freqs, device=positions.device, dtype=positions.dtype)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        shifted = positions + self.margin

        #one phase per (fx, fy) pair: 2 pi (fx x + fy y) / width
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        shifted = shifted[..., None, None, :]
        phase = (shifted[..., 0] * p_x + shifted[..., 1] * p_y).reshape(*positions.shape[:-1], -1)

        #dim=-1 is the feature axis: cosines then sines
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class ChannelMerger(nn.Module):
    """Mix sensors into virtual channels by learned attention over position.

    ATTRS:
        positions (torch.Tensor): [n_sensors, 2] buffer
        position_embedding (torch.Tensor): [n_sensors, pos_dim] buffer
        heads (nn.Parameter): [out_channels, pos_dim], one query per channel
        dropout (float): radius of the disc of sensors banned per step
    """

    def __init__(self, positions, out_channels=270, pos_dim=2048, dropout=0.2):
        """
        ARGS:
            positions (torch.Tensor): [n_sensors, 2] in [0, 1]
            out_channels (int): 270, brainmagick's merger width
            pos_dim (int): Fourier features, 2048 in the reference config
            dropout (float): ban radius in position units; 0 disables it
        """
        super().__init__()
        positions = torch.as_tensor(positions, dtype=torch.float32)
        self.register_buffer("positions", positions)

        #fixed positions, so their embedding is computed once
        self.register_buffer("position_embedding", FourierEmbedding(pos_dim)(positions))
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim) / pos_dim**0.5)
        self.dropout = dropout

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_sensors, n_times]

        RETURNS:
            [batch, out_channels, n_times]
        """
        #[out, sensors]: how much each virtual channel wants each sensor
        scores = self.heads @ self.position_embedding.T

        #brainmagick's spatial dropout: one random disc, the same for the batch
        if self.training and self.dropout > 0:
            centre = torch.rand(2, device=x.device)
            banned = (self.positions - centre).norm(dim=-1) <= self.dropout
            scores = scores.masked_fill(banned.unsqueeze(0), float("-inf"))

        #dim=1 is the SENSOR axis: each channel's weights sum to one
        weights = scores.softmax(dim=1)
        return torch.einsum("bct,oc->bot", x, weights)


class LinearSpatial(nn.Module):
    """A learned 1x1 map off the sensors: the merger's position-free fallback."""

    def __init__(self, n_sensors, out_channels=270):
        """
        ARGS:
            n_sensors (int): 306
            out_channels (int): 270
        """
        super().__init__()
        self.project = nn.Conv1d(n_sensors, out_channels, 1)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_sensors, n_times]

        RETURNS:
            [batch, out_channels, n_times]
        """
        return self.project(x)


class FixedPCA(nn.Module):
    """F3's input: a fixed PCA-whitening map fitted on the training partition.

    ATTRS:
        mean (torch.Tensor): [n_sensors] buffer
        components (torch.Tensor): [n_components, n_sensors] buffer
        scale (torch.Tensor): [n_components] buffer, 1/sqrt(eigenvalue)
    """

    def __init__(self, mean, components, scale):
        """
        ARGS:
            mean (array): [n_sensors]
            components (array): [n_components, n_sensors]
            scale (array): [n_components]
        """
        super().__init__()

        #buffers: fitted, never learned, and carried in the checkpoint
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    @property
    def out_channels(self):
        """Components kept, 73."""
        return self.components.shape[0]

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_sensors, n_times]

        RETURNS:
            [batch, n_components, n_times] - unit variance per component
        """
        #dim 1 is the sensor axis; subtract, rotate, rescale
        centred = x - self.mean.view(1, -1, 1)
        return torch.einsum("kc,bct->bkt", self.components, centred) * self.scale.view(1, -1, 1)


class SubjectLayer(nn.Module):
    """brainmagick's subject-specific linear map, for a single subject.

    ATTRS:
        weight (nn.Parameter): [channels, channels], identity at start
    """

    def __init__(self, channels):
        """
        ARGS:
            channels (int): width in and out
        """
        super().__init__()

        #identity, so with one subject the layer starts as a no-op
        self.weight = nn.Parameter(torch.eye(channels))

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, channels, n_times]

        RETURNS:
            same shape
        """
        return torch.einsum("bct,cd->bdt", x, self.weight)
