import glob
import Constants
import ModelPytorch
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ASLDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        self.class_map = {}
        i = 0

        # for each folder
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            self.class_map[class_name] = i
            i += 1

            # add image path and the corr class to the data
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])
        #print(self.data)
        #print(self.class_map)
        self.img_dim = (64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        # load img
        img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)

        # resize it to make sure it is 64*64
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]


        # allows us to change the ordering of the dimensions of a torch tensor so that it will be
        # (Channels, Width, Height) and not (Width, Height, Channels) like in numpy.
        # (Width -> 0), (Height->1), (Channels->2)
        # more explanation here: https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d#:~:text=Torch%20convolutions%20require,2nd%20dimension%20first.
        #img_tensor = img_tensor.permute(1, 0, 2)

        # get image as a tensor (multi dimensional array)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Adds a channel dimension (1, 64, 64)

        return img_tensor.float(), torch.tensor(class_id, dtype=torch.long)



def check_accuracy(loader, model,device):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return accuracy


if __name__ == "__main__":
    TEST_DATA_PATH = "Dataset\\mydata\\test_set\\"
    TRAIN_DATA_PATH = "Dataset\\mydata\\training_set\\"

    # use the gpu (if a gpu is present)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the train and test dataset
    train_dataset = ASLDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Constants.BATCH_SIZE, shuffle=True)

    test_dataset = ASLDataset(TEST_DATA_PATH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Constants.BATCH_SIZE, shuffle=False)

    # initialize model
    model = ModelPytorch.CNN(in_channels=1, num_classes=Constants.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)

    # train model
    for epoch in range(Constants.NUM_EPOCHS):
        model.train()
        print(f"Epoch [{epoch + 1}/{Constants.NUM_EPOCHS}]")

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

        print("Checking accuracy on training data")
        check_accuracy(train_loader, model,device)
        print("Checking accuracy on test data")
        check_accuracy(test_loader, model,device)


    print("FINALE ACCURACY:")
    # Final accuracy check on training and test sets
    check_accuracy(train_loader, model,device)
    check_accuracy(test_loader, model,device)
