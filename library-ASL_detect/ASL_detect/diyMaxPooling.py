from torch import nn
import torch

VALUES_DIM=2


class DiyMaxPooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(DiyMaxPooling, self).__init__()
        self.my_stride = stride
        self.kernel_size = kernel_size

    def forward(self, tensor):
        """

        this code is optimized from this code: 
            batch_size, channels, height, width = tensor.shape

            output_height = (height - self.kernel_size) // self.my_stride + 1
            output_width = (width - self.kernel_size) // self.my_stride + 1

            output_tensor = torch.zeros(batch_size, channels, output_height, output_width,device=tensor.device)

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
        """
        batch_size, channels, height, width = tensor.shape

        output_height = (height - self.kernel_size) // self.my_stride + 1
        output_width = (width - self.kernel_size) // self.my_stride + 1

        # separate the tensor to patches in the size of the kernal,  dims batch_size x (channels * kernel_size * kernel_size) x (output_height * output_width)
        unfolded_tensor = torch.nn.functional.unfold(tensor,kernel_size=(self.kernel_size, self.kernel_size),stride=self.my_stride)
        # we want the dims to be: 0- batch , 1- channels, 2 actual values (kernal_size*kernal_size), -1 means put the leftover in another dim automatically, they will be 
        # The -1 will automatically compute the last dimension (output_height*output_width)
        unfolded = unfolded_tensor.view(batch_size,channels,self.kernel_size*self.kernel_size,-1)

        # for each patch take only the max values for dim 2 (the actual values since its 0- batch , 1- channels, 2 actual values(kernal_size*kernal_size))
        res = torch.amax(unfolded,dim=VALUES_DIM)
        # shape it to be one uniform tensor (not in patches)
        return res.view(batch_size,channels,output_height,output_width)
