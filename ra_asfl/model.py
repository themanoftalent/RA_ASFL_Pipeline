"""ECG network with five split points between blocks."""
import torch
from torch import nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 4)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1),
                                 nn.GELU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class Stem(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(input_channels, 32, 7, stride=2, padding=3, bias=False),
                                 nn.BatchNorm1d(32), nn.GELU())

    def forward(self, x):
        return self.net(x)


class MorphologyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(32, 32, k, stride=2, padding=k//2,
                                                 groups=32, bias=False) for k in (3, 5, 7)])
        self.project = nn.Sequential(nn.Conv1d(96, 64, 1, bias=False), nn.BatchNorm1d(64))
        self.se = SqueezeExcite(64)
        self.skip = nn.Sequential(nn.Conv1d(32, 64, 1, stride=2, bias=False), nn.BatchNorm1d(64))

    def forward(self, x):
        y = self.project(torch.cat([layer(x) for layer in self.branches], dim=1))
        return F.gelu(self.se(y) + self.skip(x))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.main = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel, stride=2,
                                             padding=pad, dilation=dilation, groups=in_channels,
                                             bias=False),
                                  nn.Conv1d(in_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm1d(out_channels))
        self.se = SqueezeExcite(out_channels)
        self.skip = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=2, bias=False),
                                  nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return F.gelu(self.se(self.main(x)) + self.skip(x))


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Conv1d(192, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)


class ECGNet(nn.Module):
    split_names = ('S1', 'S2', 'S3', 'S4', 'S5')

    def __init__(self, channels=1, classes=5, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            Stem(channels), MorphologyBlock(), ResidualTemporalBlock(64, 96, 5, 2),
            ResidualTemporalBlock(96, 128, 5, 4), ResidualTemporalBlock(128, 192, 3, 8),
        ])
        self.head = nn.Sequential(TemporalAttention(), nn.Linear(192, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, classes))

    def prefix(self, x, split):
        for block in self.blocks[:split]:
            x = block(x)
        return x

    def suffix(self, x, split):
        for block in self.blocks[split:]:
            x = block(x)
        return self.head(x)

    def forward(self, x):
        return self.suffix(self.prefix(x, 5), 5)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
