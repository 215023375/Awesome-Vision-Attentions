# Gated channel transformation for visual recognition (CVPR2020)

from torch import nn
import numpy as np


class GCT(nn.Module):

    def __init__(self, channel=None, epsilon=1e-5, mode='l2', after_relu=False):
        assert channel is not None, "'channel' in kwargs should not be None"
        super(GCT, self).__init__()

        self.alpha = np.ones((1, channel, 1, 1))
        self.gamma = np.zeros((1, channel, 1, 1))
        self.beta = np.zeros((1, channel, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = np.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdims=True).sum(
                3, keepdims=True) * self.alpha
            norm = self.gamma / \
                (np.abs(embedding).mean(dim=1, keepdims=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + np.tanh(embedding * norm + self.beta)

        return x * gate


def main():
    attention_block = GCT(64)
    input = np.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
