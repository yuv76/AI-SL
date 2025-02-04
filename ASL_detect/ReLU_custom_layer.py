from torch import nn


class DiyReLU(nn.Module):
    def __init__(self):
        super(DiyReLU, self).__init__()

    def forward(self, tensor):
        tensor[tensor < 0] = 0
        return tensor
