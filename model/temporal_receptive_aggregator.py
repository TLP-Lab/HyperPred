import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from functools import partial
from layers import HypLinear
from manifolds import Euclidean, Lorentzian, PoincareDisk
from layers import EfficientMLHA

class TemporalReceptiveAaggregator(nn.Module):
    def __init__(self,k_hidden,in_channels, out_channels, args=None):
        super(TemporalReceptiveAaggregator, self).__init__()
        if args.manifold == 'PoincareDisk':
            self.manifold = PoincareDisk()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.c = k_hidden
        self.device = args.device
        self.in_channels = in_channels
        self.hidden_channels = out_channels
        self.receptive_depth = args.receptive_depth
        self.casual_conv_kernel_size = args.casual_conv_kernel_size
        self.scales=args.scales
        self.dilated_stack = nn.ModuleList([EfficientMLHA(manifold=self.manifold,k_hidden=self.c, in_channels=self.in_channels, out_channels=self.hidden_channels,casual_kernel_size=self.casual_conv_kernel_size,dilation=self.casual_conv_kernel_size**layer,heads_ratio=1.0, dim=4, scales=self.scales)
             for layer in range(self.receptive_depth)])

    def forward(self, x):
        skips = []
        for layer in self.dilated_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
        out = torch.cat(skips, dim=0).mean(dim=0)
        out = self.manifold.proj(out, self.c)
        return out
