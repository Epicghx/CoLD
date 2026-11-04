import math

import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k, stride=1):
    return nn.Conv1d(cin, cout, k, stride=stride, padding=math.ceil((k - 1) / 2), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, c, stride=1, down=None):
        super().__init__()
        self.conv1, self.bn1 = conv(cin, c, 5, stride), nn.BatchNorm1d(c)
        self.conv2, self.bn2 = conv(c, c, 3), nn.BatchNorm1d(c)
        self.downsample = down

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, stride=1, down=None):
        super().__init__()
        self.conv1, self.bn1 = conv(cin, c, 1), nn.BatchNorm1d(c)
        self.conv2, self.bn2 = conv(c, c, 3, stride), nn.BatchNorm1d(c)
        self.conv3, self.bn3 = conv(c, 4 * c, 1), nn.BatchNorm1d(4 * c)
        self.downsample = down

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(out)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet1D(nn.Module):

    def __init__(self, block, layers, in_channels=12, base=64):
        super().__init__()
        self.inplanes = base
        self.stem = nn.Sequential(conv(in_channels, base, 7, 2), nn.BatchNorm1d(base), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._layer(block, base, layers[0], 1)
        self.layer2 = self._layer(block, 2 * base, layers[1], 2)
        self.layer3 = self._layer(block, 4 * base, layers[2], 2)
        self.layer4 = self._layer(block, 8 * base, layers[3], 2)
        self.embed_dim = 8 * base * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _layer(self, block, c, n, stride):
        down = None
        if stride != 1 or self.inplanes != c * block.expansion:
            down = nn.Sequential(conv(self.inplanes, c * block.expansion, 1, stride), nn.BatchNorm1d(c * block.expansion))
        layers = [block(self.inplanes, c, stride, down)]
        self.inplanes = c * block.expansion
        layers += [block(self.inplanes, c) for _ in range(1, n)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return x.mean(dim=-1)


def resnet1d_18(**kw):
    return ResNet1D(BasicBlock, [2, 2, 2, 2], **kw)


def resnet1d_50(**kw):
    return ResNet1D(Bottleneck, [3, 4, 6, 3], **kw)
