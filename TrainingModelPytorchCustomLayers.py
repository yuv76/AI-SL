import Constants
import ModelPytorch
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import ASLTrainDataset,ASLTestDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt

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


def train_model(model, train_loader, valid_loader, device, optimizer, max_patiance):
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
            train_accuracy = check_accuracy(train_loader, model, device)
            train_accuracies.append(train_accuracy)

            print("Checking accuracy on validation data")
            curr_val_acc = check_accuracy(valid_loader, model, device)
            valid_accuracies.append(curr_val_acc)

            if(curr_val_acc > prev_val_acc):
                print("Model was saved!\nPatiance reset to 0!")
                # save a checkpint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, Constants.PATH_CHECKPOINT)
                patience_count = 0
            else:
                patience_count += 1
                print(f"Model didn't improve and therfore, was not saved!\nPatiance increased by 1 and now is : {patience_count} !")

            print("Loading last model!")
            checkpoint = torch.load(Constants.PATH_CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            prev_val_acc = curr_val_acc

    return train_accuracies, valid_accuracies

if __name__ == "__main__":
    # use the gpu (if a gpu is present)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the train and test dataset
    train_dataset = ASLTrainDataset(Constants.TRAIN_DATA_PATH)

    # Calculate sizes of train and valid
    total_size = len(train_dataset)
    train_size = int(total_size * Constants.TRAIN_RATIO)
    validation_size = total_size - train_size
    
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=Constants.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Constants.BATCH_SIZE, shuffle=True)
    
    test_dataset = ASLTestDataset(Constants.TEST_DATA_PATH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Constants.BATCH_SIZE, shuffle=False)

    # initialize model 
    model = ModelPytorch.CNN(in_channels=1, num_classes=Constants.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)

    train_accuracies, valid_accuracies = train_model(model, train_loader, validation_loader, device, optimizer, Constants.PATIENCE)

    print("FINALE ACCURACY:")
    # Final accuracy check on training and test sets
    check_accuracy(train_loader, model, device)
    check_accuracy(test_loader, model, device)

    # Plot training and validation accuracies
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy", color='b', marker='o')
    plt.plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label="Validation Accuracy", color='r', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy.png')

    torch.save(model.state_dict(), Constants.PATH_MODEL)
    print("Model saved") 