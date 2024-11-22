from torch import nn


class DiyMaxPooling(nn.Module, kernel_size, stride):
    def __init__(self):
        super(DiyMaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, tensor):
        """

        """
        for i in range tensor.size(:2:):
            curr_pool = tf.stack([tensor[i, :2+(stride*i)], tensor[i+1, :2+(stride*i)]])
            new_pool = tensor.max(curr_pool)
        return tensor
