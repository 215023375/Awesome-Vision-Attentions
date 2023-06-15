# Context encoding for semantic segmentation (CVPR 2018)

from ..utils import use_same_device_as_input_tensor as use_input_device

from torch import nn
import numpy as np
import torch

class Encoding(nn.Module):
    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(torch.Tensor.uniform_(torch.rand((num_codes, channels)), -std, std))
        # [num_codes]
        self.scale = nn.Parameter(torch.Tensor.uniform_(torch.rand((num_codes,)), -1, 0))

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)

        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.ndim == 4 and x.size(1) == self.channels

        self.codewords = self.codewords.to(use_input_device(x))
        self.scale = self.scale.to(use_input_device(x))

        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).permute(0, 2, 1)
        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = torch.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


class EncModule(nn.Module):
    def __init__(self, channel, num_codes=32):
        assert channel is not None, "'channel' in kwargs should not be None"
        super(EncModule, self).__init__()
        self.encoding_project = nn.Conv2d(channel, channel, 1)
        self.encoding = nn.Sequential(
            Encoding(channels=channel, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel), nn.Sigmoid())

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        return x*gamma.view(batch_size, channels, 1, 1)


def main():
    attention_block = EncModule(64, 32)
    input = np.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
