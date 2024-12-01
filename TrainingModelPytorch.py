import glob
import Constants
import ModelPytorch
import cv2
import torch
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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


        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),  # randomly flip the image horizontally with a 50% chance
            #transforms.RandomVerticalFlip(p=0.5),    # randomly flip the image vertically with a 50% chance
            transforms.ToTensor(),  # Converts image to a Tensor in the range [0, 1]
            transforms.Normalize((0.5,), (0.5,))   
        ])

        # Apply the transformation to the image
        img_tensor = transform(img)

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


def train_model(model,train_loader,valid_loader, device,optimizer,max_patiance):
    # Will save train and validation accuracies each epoch
    train_accuracies = []
    valid_accuracies = []

    prev_val_acc = -1
    curr_val_acc = -1
    patience_count = 0

    # train model
    for epoch in range(Constants.NUM_EPOCHS):
        if patience_count <= max_patiance:
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
            train_accuracy = check_accuracy(train_loader, model,device)
            train_accuracies.append(train_accuracy)
            print("Checking accuracy on test data")
            curr_val_acc = check_accuracy(valid_loader, model,device)
            valid_accuracies.append(curr_val_acc)

            if(curr_val_acc > prev_val_acc):
                print("Model was saved!\nPatiance reset to 0!")

                # save a checkpint
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, Constants.PATH_CHECKPOINT)
                patience_count = 0
            else:
                print(f"Model didn't improve and therfore, was not saved!\nPatiance increased by 1 and now is : {patience_count} !")
                patience_count +=1

                print("Loading last model!")
                checkpoint = torch.load(Constants.PATH_CHECKPOINT, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']

            prev_val_acc = curr_val_acc




    return train_accuracies, valid_accuracies


if __name__ == "__main__":
    TEST_DATA_PATH = "mydata\\test_set\\"
    TRAIN_DATA_PATH = "mydata\\training_set\\"

    # use the gpu (if a gpu is present)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the train and test dataset
    train_dataset = ASLDataset(TRAIN_DATA_PATH)

    # Calculate sizes of train and valid
    total_size = len(train_dataset)
    train_size = int (total_size * Constants.TRAIN_RATIO)
    validation_size = total_size - train_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    validation_loader = DataLoader(dataset=train_dataset, batch_size=Constants.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Constants.BATCH_SIZE, shuffle=True)

    test_dataset = ASLDataset(TEST_DATA_PATH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Constants.BATCH_SIZE, shuffle=False)

    # initialize model
    model = ModelPytorch.CNN(in_channels=1, num_classes=Constants.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)

    train_accuracies, valid_accuracies = train_model(model,train_loader,test_loader,device,optimizer,Constants.PATIENCE)

    print("FINALE ACCURACY:")
    # Final accuracy check on training and test sets
    check_accuracy(train_loader, model,device)
    check_accuracy(test_loader, model,device)

    # After training, plot the accuracy over epochs
    plt.plot(range(1, Constants.NUM_EPOCHS + 1), train_accuracies, label="Training Accuracy", color='b', marker='o')
    plt.plot(range(1, Constants.NUM_EPOCHS + 1), valid_accuracies, label="Test Accuracy", color='r', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.savefig('accuracy.png')

    # plt.show() # open the saved graph

    torch.save(model.state_dict(), 'my_model_weights.pth')
    print("Model saved")
