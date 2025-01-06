import cv2
import numpy as np
import time

thresh_val = 190

def on_trackbar(val):
    global thresh_val
    thresh_val = val


def predict_from_cam(cam, out):
    """
    Feeds the model with the output of the computer's camera and prints its prediction. Stopped by pressing 'x'.
    in: the initialized camera feed, the loaded model.
    out: none.
    """
    created_trackbar = False
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for letter in class_names:
        time.sleep(5)
        for frame in range(0,121):
            ret, frame = cam.read()

            # Frame into numpy array
            input_array = np.array(frame)

            # Grayscale image
            gray_image = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
            # add thresholding
            ret, thresh1 = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # flip image (in dataset images are flipped)
            thresh1 = cv2.flip(thresh1, 1)

            # Draw the rectangle (happens after so it wont be in the cropped image)
            cv2.rectangle(thresh1,(400,100),(610,310),(255,0,0),3)

            print(f"sign the letter {letter}\n")

            # Write letter to sign
            text = f"expected {letter}"
            image = cv2.putText(thresh1, text, (390,40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2, cv2.LINE_AA)

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
    out = cv2.VideoWriter('testvid.mp4', fourcc, 20.0, (frame_width, frame_height))

    predict_from_cam(cam, out)

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
