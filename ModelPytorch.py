import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import SoftMaxDIY
import diyMaxPooling


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=26):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For us, this is 1 (images are black and white).
            num_classes: int
                The number of classes we want to predict, in our case 26 (letters a to z).
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 32 output channels (32 filters), 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = diyMaxPooling.DiyMaxPooling(kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm3 = nn.BatchNorm2d(num_features=128)

        self.flattern = nn.Flatten()

        # Fully connected layer: 128*8*8 input features (after three 2x2 poolings), 26 output features (num_classes)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = SoftMaxDIY.DiySoftMax()


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = self.norm1(x)          # Apply batch norm1

        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = self.norm2(x)          # Apply batch norm2

        x = F.relu(self.conv3(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = self.norm3(x)
        
        x = self.flattern(x)       # Apply flatten layer

        x = F.relu(self.fc1(x))    # Apply fully connected layer 1
        x = self.drop(x)           # Apply dropout layer
        x = F.relu(self.fc2(x))    # Apply fully connected layer 2
        x = self.softmax(x)

        return x





