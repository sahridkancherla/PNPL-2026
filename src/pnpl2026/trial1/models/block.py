"""
to do list:

1. add like a @staticmethod validation function to the encoder block. to make sure the data fits
what we need. 
2. add more error check/dimension check functions
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

"""
hyperparameters
"""

#hyperparameters
BRANCH_KERNELS = (7, 19, 25)
BRANCH_DILATIONS = (1, 1, 4)


@dataclass(eq=False)
class EncoderBlock(nn.Module):

    n_channels: int
    kernels: tuple = BRANCH_KERNELS
    dilations: tuple = BRANCH_DILATIONS
    ffn_expansion: int = 2
    spans: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """
        Assemble the block
        """
        #nn.Module refuses submodule assignment before its registries exist
        super().__init__()

        self.branches = self.build_branches(
            self.n_channels, self.kernels, self.dilations
        )
        self.mixer = self.build_mixer(self.n_channels, len(self.kernels))
        self.feed_forward = self.build_feed_forward(
            self.n_channels, self.ffn_expansion
        )

    @staticmethod
    def conv_depthwise(n_channels, kernel_size, dilation):
        #odd kernel plus this padding is exactly `same`, at any dilation
        padding = dilation * (kernel_size - 1) // 2
        return nn.Conv1d(

            #in and out channels remain the same
            n_channels,
            n_channels,

            #kernel size will vary
            kernel_size=kernel_size,
            padding=padding,

            #dilation will vary
            dilation=dilation,

            #for weight efficiency
            groups=n_channels,
            bias=False,
        )

    @classmethod
    def build_branches(cls, n_channels, kernels, dilations):
        """
        Establish the branches for a single neural net block.
        at the moment only builds 3 branches
        """
        #make a list of branches. every branch will have different values of kernels and dilations
        branches = nn.ModuleList(
            cls.conv_depthwise(n_channels, k, d)
            for k, d in zip(kernels, dilations))

        return branches
        

    @staticmethod
    def build_mixer(n_channels, n_branches):
        """
        mixer that is applied once the channels are concatenated 
        """
        return nn.Conv1d(n_branches * n_channels, n_channels, kernel_size=1)

    @staticmethod
    def build_feed_forward(n_channels, expansion):
       
        #hidden layer size
        hidden = expansion * n_channels
        
        return nn.Sequential(
            #still debating the normalisation function
            nn.BatchNorm1d(n_channels),

            #MLP. potentially add more layers
            nn.Conv1d(n_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, n_channels, kernel_size=1),
        )


    @staticmethod
    def forward(self, windows):
        
        #concatenate every branch into one
        concatenated_branch = torch.cat([branch(windows) for branch in self.branches], dim=1)

        #apply the mixer
        mixed = self.mixer(concatenated_branch)

        #residual 1
        windows = windows + mixed

        #impelements the build_feed_forward function
        y = self.feed_forward(windows)

        #residual 2
        output = windows + y
        
        return output
