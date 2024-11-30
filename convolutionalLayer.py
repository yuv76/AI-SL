from torch import nn
import torch.nn.functional
import torch

class DiyConvolution(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1, padding=1):
        super().__init__()
        self.k_size = kernel_size
        self.in_channels = in_size
        self.out_channels = out_size
        self.stride = stride
        self.padding = padding

        # define the weight
        weights = torch.zeros(self.k_size*self.k_size*self.in_channels*self.out_channels)
        weights = weights.view(self.in_channels, self.out_channels, self.k_size, self.k_size)
        self.weights = nn.Parameter(weights)
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)

        # define the bias
        bias = torch.zeros(self.out_channels)
        self.bias = nn.Parameter(bias)
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)

    def forward(self, tensor):
        """
        padded_tensor = F.pad(tensor, (self.padding, self.padding, self.padding, self.padding))

        batch_size, channels, height, width = padded_tensor.shape

        output_height = (height - self.kernel_size) // self.my_stride + 1
        output_width = (width - self.kernel_size) // self.my_stride + 1

        output_tensor = torch.zeros(batch_size, channels, output_height, output_width, device=tensor.device)

        for batch_ind in range(batch_size):
                for out_channel_ind in range(self.out_channels):
                    for i in range(height):
                        for j in range(width):
                            start_from_i = i * self.my_stride
                            start_from_j = j * self.my_stride
                            end_i = start_from_i + self.kernel_size
                            end_j = start_from_j + self.kernel_size

                            patch = padded_tensor[batch_ind, :, start_from_i:end_i, start_from_j:end_j]
                            output[batch_ind, out_channel_ind, i, j] = torch.sum(
                                patch * self.weights[out_channel_ind, :, :, :]
                                    ) + self.bias[out_channel_ind]
        """

        # Pad tensor with zeros
        padded_tensor = F.pad(tensor, (self.padding, self.padding, self.padding, self.padding))

        batch_size, channels, height, width = padded_tensor.shape

        output_height = (height - self.k_size) // self.stride + 1
        output_width = (width - self.k_size) // self.stride + 1

        #output_tensor = torch.zeros(batch_size, channels, output_height, output_width, device=tensor.device)

        # separate the tensor to patches in the size of the kernel, dims batch_size x (channels * kernel_size * kernel_size) x (output_height * output_width)
        unfolded_tensor = torch.nn.functional.unfold(tensor, kernel_size=(self.k_size, self.k_size), stride=self.stride)
        unfolded = unfolded_tensor.view(batch_size, channels, self.k_size*self.k_size, -1)

        output_tensor = torch.sum(unfolded * self.weights) + self.bias

        output_tensor = output_tensor.view(batch_size, channels, output_height, output_width)

        return output_tensor
