import cv2
import numpy as np

from .load_model import loadModel

import torch
from torchvision import transforms

IMAGE_LEN = 64


def cnn_single_image(image, thresh_val=190):
    """
    Feeds the model with the an image and returns the prediction.
    in: the image, optional: threshold value (if smaller than 0, wont threshold).
    out: the model's prediction.
    """
    model = loadModel.get_model()
    sub_model = loadModel.get_sub_model()

    if image is None:
        raise ValueError("Input image is empty or not loaded correctly.")
    # Frame into numpy array
    input_array = np.array(image)

    # Grayscale image
    gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    if thresh_val > 0:
        # add thresholding
        ret, thresh1 = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh1 = gray_image

    # flip image (in dataset images are flipped)
    thresh1 = cv2.flip(thresh1, 1)

    # Resize to match model
    resized = cv2.resize(thresh1, (IMAGE_LEN, IMAGE_LEN))

    preprocess = transforms.Compose([
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize assuming single-channel input
    ])

    image_tensor = preprocess(resized).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get predicted letter and print it
    predicted_index = np.argmax(prediction)
    max_prediction = prediction[0][predicted_index]
    class_names = list("#BCDFGHIJKLOPQRUVWXYZ")
    predicted_letter = class_names[predicted_index]
    if predicted_letter == '#':
        # the model recognized a letter from # group, make predictions in sub model
        with torch.no_grad():
            sub_prediction = sub_model(image_tensor)
        # Get predicted letter
        predicted_index = np.argmax(sub_prediction)
        max_prediction = sub_prediction[0][predicted_index]
        grouped_classes = list("AEMNST")
        predicted_letter = grouped_classes[predicted_index]
    return predicted_letter, max_prediction

