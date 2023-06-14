# Gated channel transformation for visual recognition (CVPR2020)
from ..utils import use_same_device_as_input_tensor as use_input_device



from torch import nn
import torch
import numpy as np


class GCT(nn.Module):

    def __init__(self, channel=None, epsilon=1e-5, mode='l2', after_relu=False):
        assert channel is not None, "'channel' in kwargs should not be None"
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones((1, channel, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channel, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        self.alpha = self.alpha.to(use_input_device(x))
        self.gamma = self.gamma.to(use_input_device(x))
        self.beta = self.beta.to(use_input_device(x))

        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdims=True).sum(
                3, keepdims=True) * self.alpha
            norm = self.gamma / \
                (np.abs(embedding).mean(dim=1, keepdims=True) + self.epsilon)
        else:
            print('Unknown mode!')
            raise RuntimeError("Unknown mode!")

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


def main():
    attention_block = GCT(64)
    input = np.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
