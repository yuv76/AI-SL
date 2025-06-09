import cv2
import numpy as np

import ModelPytorchCustomLayers as ModelPytorch
import Constants

import torch
from torchvision import transforms

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


def predict_from_cam(cam, model, sub_model):
    """
    Feeds the model with the output of the computer's camera and prints its prediction. Stopped by pressing 'x'.
    in: the initialized camera feed, the loaded model.
    out: none.
    """
    created_trackbar = False
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    right = {chr(i): 0 for i in range(65, 91)}
    wrong = {chr(i): 0 for i in range(65, 91)}

    for expected_letter in class_names:
        for frame in range(0,121):
            ret, frame = cam.read()

            # Frame into numpy array
            input_array = np.array(frame)

            # Grayscale image
            gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)

            # Make cropped image (where the square is) and show it
            cropped_img = gray_image[100:310, 400:610]
            cv2.imshow('Cropped Image', cropped_img)

            # Draw the rectangle (happens after so it wont be in the cropped image)
            cv2.rectangle(gray_image,(400,100),(610,310),(255,0,0),3)

            # Resize to match model
            resized = cv2.resize(cropped_img, (64, 64))
            cv2.imshow('Image', resized)

            preprocess = transforms.Compose([
                transforms.ToTensor(),              # Convert to tensor
                transforms.Normalize((0.5,), (0.5,)) # Normalize assuming single-channel input
            ])

            image_tensor = preprocess(resized).unsqueeze(0)

            # Make predictions
            with torch.no_grad():
                prediction = model(image_tensor)
                print(prediction)

            # Get predicted letter
            predicted_index = np.argmax(prediction)

            class_names_overall = list("#BCDFGHIJKLOPQRUVWXYZ")
            predicted_letter = class_names_overall[predicted_index]

            if predicted_letter == '#':
                # the model recognized a letter from # group, make predictions in sub model
                with torch.no_grad():
                    sub_prediction = sub_model(image_tensor)
                    print("sub-prediction: " + str(sub_prediction))
                # Get predicted letter
                predicted_index = np.argmax(sub_prediction)
                classes = list("AEMNST")
                predicted_letter = classes[predicted_index]


            print(f"predicted: {predicted_letter} Expected: {expected_letter}\n")
            if expected_letter == predicted_letter:
                right[expected_letter] += 1
            else:
                wrong[expected_letter] += 1

            # Write predicted letter on image
            predicted_text = f"predicted: {predicted_letter}"
            image = cv2.putText(gray_image, predicted_text, (390, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 2, cv2.LINE_AA)

            # Display the captured frame
            cv2.imshow('Camera', gray_image)

            # Convert the thresholded frame back to BGR so it can be written to the video
            thresholded_frame_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            if not created_trackbar:
                cv2.namedWindow("Camera")
                cv2.createTrackbar("Threshold", "Camera", 150, 255, on_trackbar)
                created_trackbar = True

            # Press 'x' to exit the loop
            if cv2.waitKey(1) == ord('x'):
                break
    print(f"got {sum(right.values())}/{sum(right.values())+sum(wrong.values())} right")
    for letter in class_names:
        print(f"{letter}: got {right[letter]}/{right[letter]+wrong[letter]} right")


if __name__ == "__main__":
    # Open the default camera
    cam = cv2.VideoCapture("testvid2.mp4")

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    model = load_model('model_second_try.pth', Constants.NUM_CLASSES)
    sub_model = load_model('only_#_improve.pth', Constants.NUM_COMBINED_CLASSES)

    predict_from_cam(cam, model, sub_model)

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()
