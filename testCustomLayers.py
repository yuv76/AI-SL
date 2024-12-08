import diyConv
import diyMaxPooling
import ReLU_custom_layer
import SoftMaxDIY

from torch import nn
import torch.nn.functional as F
import torch


def checkReLU():
    print(" --- ReLU --- \n")

    original_tensor = torch.randn(3, 3)
    print(f"original tensor: \n{original_tensor}\n")

    pytorch = original_tensor.clone()
    diy = original_tensor.clone()

    pytorch = F.relu(pytorch)

    diy_relu = ReLU_custom_layer.DiyReLU()
    diy = diy_relu(diy)

    print(f"ReLU on tensor using pytorch: \n{pytorch}\n")
    print(f"ReLU on tensor using custom layer: \n{diy}\n")

    if torch.all(torch.eq(pytorch, diy)):
        status = "pass"
    else:
        status = "fail"
    print(f"check status: {status}")
    return status


def checkSoftmax():
    print(" --- softmax --- \n")

    original_tensor = torch.randn(26)
    print(f"original tensor: \n{original_tensor}")

    pytorch = original_tensor.clone()
    diy = original_tensor.clone()

    pytorch = F.softmax(pytorch)

    diy_softmax = SoftMaxDIY.DiySoftMax()
    diy = diy_softmax(diy)

    print(f"Softmax on tensor using pytorch: \n{pytorch}")
    print(f"pytorch sum: {torch.sum(pytorch)}\n")
    print(f"Softmax on tensor using custom layer: \n{diy}")
    print(f"custom layer sum: {torch.sum(diy)}\n")

    if torch.isclose(torch.sum(pytorch), torch.tensor(1.0), atol=1e-6) and torch.isclose(torch.sum(diy), torch.tensor(1.0), atol=1e-6):  # Rounds the result when comparing as result might be slightly different than 1.
        status = "pass"
    else:
        status = "fail"
    print(f"check status: {status}")
    return status


def checkConv():
    print(" --- Convolution --- \n")

    original_tensor = torch.randn(1, 1, 5, 5)  # 1 image, 1 channel, 5x5 size
    print(f"original tensor: \n{original_tensor}\n")

    pytorch = original_tensor.clone()
    diy = original_tensor.clone()

    weight = torch.randn(1, 1, 3, 3)  # 1 filter, 1 input channel, 3x3 filter size
    weight_diy = weight.view(1, 1, 3 * 3)  # diy requires different shape for weights

    bias = torch.zeros(1)
    bias = nn.Parameter(bias)
    torch.nn.init.normal_(bias, mean=0.0, std=1.0)

    # PyTorch Convolution
    pytorch_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    pytorch_conv.weight.data = weight
    pytorch_conv.bias.data = bias
    pytorch_output = pytorch_conv(pytorch)

    # Custom Convolution
    diy_conv = diyConv.DiyConvolution(1, 1, kernel_size=3, stride=1, padding=1)
    diy_conv.weights = nn.Parameter(weight_diy)
    diy_conv.bias = bias
    diy_output = diy_conv(diy)

    print(f"Convolution output using PyTorch: \n{pytorch_output}\n")
    print(f"Convolution output using custom layer: \n{diy_output}\n")

    # Use torch.allclose to compare with tolerance for floating-point precision
    if torch.allclose(pytorch_output, diy_output, atol=1e-6):
        status = "pass"
    else:
        status = "fail"

    print(f"check status: {status}")
    return status


def checkMaxPooling():
    print(" --- Max Pooling --- \n")

    # Create an example input tensor
    original_tensor = torch.tensor([[[[1, 5, 33, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12],
                                      [13, 0, 153, 16]]]], dtype=torch.float32)
    print(f"Original tensor:\n{original_tensor}\n")

    pytorch = original_tensor.clone()
    diy = original_tensor.clone()

    pytorch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    pytorch_output = pytorch_maxpool(pytorch)

    diy_maxpool = diyMaxPooling.DiyMaxPooling(kernel_size=2, stride=2)
    diy_output = diy_maxpool(diy)

    print(f"Max pooling using PyTorch:\n{pytorch_output}")
    print(f"Max pooling using custom method:\n{diy_output}")

    # Compare the results (using torch.allclose for numerical precision tolerance)
    if torch.allclose(pytorch_output, diy_output):
        status = "pass"
    else:
        status = "fail"
    
    print(f"Check status: {status}")
    return status

def main():
    relu = checkReLU()
    softmax = checkSoftmax()
    conv = checkConv()
    maxpool = checkMaxPooling()

    print(f"\n --- summary ---")
    print(f"relu test: {relu}")
    print(f"softmax test: {softmax}")
    print(f"convolutional test: {conv}")
    print(f"max pooling test: {maxpool}")


if __name__ == "__main__":
    main()

