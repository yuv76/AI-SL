import cv2
import numpy as np

import ModelPytorchCustomLayers as ModelPytorch
import Constants

import torch
from torchvision import transforms
import torchvision.models as models


def load_model(model_path, num_classes):
    cnn_model = ModelPytorch.CNN(in_channels=1, num_classes=num_classes)
    # Load the saved state dictionary
    cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    cnn_model.eval()
    print("Model loaded from checkpoint.")
    return cnn_model


def cnn_single_image(image, thresh_val):
    """
    Feeds the model with the output of the computer's camera and prints its prediction. Stopped by pressing 'x'.
    in: the initialized camera feed, the loaded model.
    out: none.
    """
    model = load_model(Constants.MODEL_PATH, Constants.NUM_CLASSES)
    sub_model = load_model(Constants.SUB_MODEL_PATH, Constants.NUM_COMBINED_CLASSES)

    # Frame into numpy array
    input_array = np.array(image)

    # Grayscale image
    gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    # add thresholding
    ret, thresh1 = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # flip image (in dataset images are flipped)
    thresh1 = cv2.flip(thresh1, 1)

    # Resize to match model
    resized = cv2.resize(thresh1, (64, 64))

    preprocess = transforms.Compose([
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize assuming single-channel input
    ])

    image_tensor = preprocess(resized).unsqueeze(0)

    # Reshape the array to match the model's input shape
    # img_array = np.reshape(img_array, (-1, 64, 64, 1))

    # Make predictions
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get predicted letter and print it
    predicted_index = np.argmax(prediction)
    class_names = list("#BCDFGHIJKLOPQRUVWXYZ")
    predicted_letter = class_names[predicted_index]
    if predicted_letter == '#':
        # the model recognized a letter from # group, make predictions in sub model
        with torch.no_grad():
            sub_prediction = sub_model(image_tensor)
        # Get predicted letter
        predicted_index = np.argmax(sub_prediction)
        grouped_classes = list("AEMNST")
        predicted_letter = grouped_classes[predicted_index]
    return predicted_letter, np.argmax(prediction)

