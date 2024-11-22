import ReLU_custom_layer
import torch
from torch import nn

class test(nn.Module):
    def __init__(self, in_channels, num_classes=26):
        self.relu = ReLU_custom_layer.DiyReLU()
