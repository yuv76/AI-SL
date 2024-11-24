from torch import nn
import torch


class DiySoftMax(nn.Module):
    def __init__(self):
        super(DiySoftMax, self).__init__()

    def forward(self, tensor):
        """
            Compute the exponentials (e^(x1), e^(x2)...)
            Compute the sum of these exponentials sum = (e^(x1) + e^(x2) + e^(x3)...)
            Divide each exponential by the sum ([e^(x1) / sum, e^(x2) / sum,e^(x3) / sum...]

            making the soft max stable explanation here:
            https://stackoverflow.com/questions/42599498/numerically-stable-softmax

        """
        #dim =1 => for each row (dim =0 is for each collum), we want it to be for each row, rather then for each collum because each row represents one
        #  feature (each row is  the result of one filter after the convulotion and the max pooling)
        # keepdim - dont get rid of the collums after this  
        exp_tensor = torch.exp(tensor - torch.max(tensor, dim=1,keepdim=True)[0])
        sum_exp = torch.sum(exp_tensor, dim=1,keepdim=True)

        return torch.div(exp_tensor, sum_exp)


