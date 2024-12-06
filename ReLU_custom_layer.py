import torch
from torch import nn


class DiyReLU(nn.Module):
    def __init__(self):
        super(DiyReLU, self).__init__()

    def forward(self, x):
        return max(0, x)

