import cv2
import numpy as np

import torch
from torchvision import transforms

from ASL_detect.removeBg import process_hand_frame
from ASL_detect.CNN_single_image import IMAGE_LEN
from .load_model import loadModel


def r_cnn_single_image(image, thresh_val=190, min_size=150, prediction_threshold=0.9, removeBG=False):
    """
    Gets an image and feeds it to the r-cnn model.
    in: image, optionl: threshold value, minimum box size, prediction threshold.
    out: list with bounding boxes, each with coordinates and a prediction.
    """
    model = loadModel.get_model()
    sub_model = loadModel.get_sub_model()

    if image is None:
        raise ValueError("Input image is empty or not loaded correctly.")

    image = cv2.flip(image, 1)
    removed, proposals = process_hand_frame(image)
    if removeBG:
        image = removed
    image = cv2.flip(image, 1)

    # Frame into numpy array
    input_array = np.array(image)

    # Grayscale image
    gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    # add thresholding
    ret, thresh1 = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # flip image (in dataset images are flipped)
    thresh1 = cv2.flip(thresh1, 1)

    boxes = []
    # Make cropped image (where the square is) and show it
    for (x, y, w, h) in proposals:
        len = max(max(w, h), min_size)
        cropped_img = thresh1[max(0,y-10):y + len+10, max(0, x-10):x + len +10]

        # Resize to match model
        resized = cv2.resize(cropped_img, (IMAGE_LEN, IMAGE_LEN))

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
        if max_prediction > prediction_threshold:
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

            boxes += [(x-10, y-10, len + 10, predicted_letter, max_prediction*100)]

    return boxes
