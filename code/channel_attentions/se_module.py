from typing import Optional, Callable

import numpy as np
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel=None, reduction=16):
        assert channel is not None, "'channel' in kwargs should not be None"
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


def main():
    attention_block = SELayer(64)
    input = np.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
