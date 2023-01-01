from typing import Optional, Callable

import jittor as jt
from torch import nn
from torchvision.models.resnet import BasicBlock


class SELayer(BasicBlock):
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SELayer, self).__init__(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(self.expansion)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // planes, bias=False),
            nn.ReLU(),
            nn.Linear(inplanes // planes, self.expansion * inplanes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


def main():
    attention_block = SELayer(64)
    input = jt.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
