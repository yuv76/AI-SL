import cv2
import numpy as np

import ModelPytorchCustomLayers as ModelPytorch
import Constants

import torch
from torchvision import transforms
import torchvision.models as models

thresh_val = 190


def load_model(model_path, num_classes):
    cnn_model = ModelPytorch.CNN(in_channels=1, num_classes=num_classes)
    # Load the saved state dictionary
    cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    cnn_model.eval()
    print("Model loaded from checkpoint.")
    return cnn_model

def on_trackbar(val):
    global thresh_val
    thresh_val = val


def predict_from_cam(cam, model, sub_model, out):
    """
    Feeds the model with the output of the computer's camera and prints its prediction. Stopped by pressing 'x'.
    in: the initialized camera feed, the loaded model.
    out: none.
    """
    created_trackbar = False
    while True:
        ret, frame = cam.read()

        # Frame into numpy array
        input_array = np.array(frame)

        # Grayscale image
        gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
        # add thresholding
        ret, thresh1 = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # flip image (in dataset images are flipped)
        thresh1 = cv2.flip(thresh1, 1)

        # Make cropped image (where the square is) and show it
        cropped_img = thresh1[100:310, 400:610]
        cv2.imshow('Cropped Image', cropped_img)

        # Draw the rectangle (happens after so it wont be in the cropped image)
        cv2.rectangle(thresh1,(400,100),(610,310),(255,0,0),3)

        # Resize to match model
        resized = cv2.resize(cropped_img, (64, 64))
        cv2.imshow('Image', resized)

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
            print(prediction)

        # Get predicted letter and print it
        predicted_index = np.argmax(prediction)
        class_names = list("#BCDFGHIJKLOPQRUVWXYZ")
        #class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        predicted_letter = class_names[predicted_index]
        if predicted_letter == '#':
            # the model recognized a letter from # group, make predictions in sub model
            with torch.no_grad():
                sub_prediction = sub_model(image_tensor)
                print("sub-prediction: " + str(sub_prediction))
            # Get predicted letter
            predicted_index = np.argmax(sub_prediction)
            grouped_classes = list("AEMNST")
            predicted_letter = grouped_classes[predicted_index]

        print(f"The model predicts the letter: {predicted_letter}\n")

        # Write predicted letter on image
        predicted_text = f"predicted: {predicted_letter}"
        image = cv2.putText(thresh1, predicted_text, (390, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2, cv2.LINE_AA)
        #thresh1 = cv2.flip(thresh1, 1)

        # Display the captured frame
        cv2.imshow('Camera', thresh1)

        # Convert the thresholded frame back to BGR so it can be written to the video
        thresholded_frame_bgr = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        out.write(thresholded_frame_bgr)

        if not created_trackbar:
            cv2.namedWindow("Camera")
            cv2.createTrackbar("Threshold", "Camera", 150, 255, on_trackbar)
            created_trackbar = True

        # Press 'x' to exit the loop
        if cv2.waitKey(1) == ord('x'):
            break


if __name__ == "__main__":
    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    model = load_model('final_model.pth', Constants.NUM_CLASSES)
    sub_model = load_model('sub_model_final.pth', Constants.NUM_COMBINED_CLASSES)

    predict_from_cam(cam, model, sub_model, out)

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
