from torch import nn
import torch


class DiyMaxPooling(nn.Module):
    def init(self, kernel_size, stride):
        super(DiyMaxPooling, self).init()
        self.my_stride = stride
        self.kernel_size = kernel_size

    def forward(self, tensor):
        """

        """
        batch_size, channels, height, width = tensor.shape()

        output_height = (height - kernel_size) // self.my_stride + 1
        output_width = (width - kernel_size) // self.my_stride + 1

        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        for batch_ind in range(batch_size):
            for channels_ind in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_from_i = i * self.my_stride
                        start_from_j = j * self.my_stride
                        end_i = start_from_i + self.kernel_size
                        end_j = start_from_j + self.kernel_size

                        output_tensor[batch_ind, channels_ind, i, j] = \
                            torch.max(tensor[batch_ind, channels_ind, start_from_i:end_i, start_from_j:end_j])

        return output_tensor
