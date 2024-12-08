import cv2
import numpy as np
import ModelPytorch
import torch
from torchvision import transforms

# Path to your local image
image_path = "F.png"

# Load the image
frame = cv2.imread(image_path)
frame = cv2.flip(frame,1)

if frame is None:
    print("Error: Could not load the image.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Load the PyTorch model
model = ModelPytorch.CNN(in_channels=1, num_classes=26)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()
print("Model loaded from checkpoint.")

# Preprocess the image for the model
preprocess = transforms.Compose([
    transforms.ToTensor(),               # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize for a single-channel image
])


# Function to update the threshold and process the image
def on_trackbar(val):
    # Apply thresholding
    ret, thresh1 = cv2.threshold(gray_image, val, 255, cv2.THRESH_BINARY_INV)

    # Resize to match the model's input size
    resized = cv2.resize(thresh1, (64, 64))

    # Preprocess for the model
    image_tensor = preprocess(resized).unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get the predicted letter
    predicted_index = torch.argmax(prediction).item()
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    predicted_letter = class_names[predicted_index]

    print(f"Threshold: {val}, Predicted Letter: {predicted_letter}")

    # Display the thresholded image with prediction
    output_image = cv2.putText(thresh1.copy(), f"Predicted: {predicted_letter}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Thresholded Image", output_image)


# Create a window for the trackbar
cv2.namedWindow("Thresholded Image")
cv2.createTrackbar("Threshold", "Thresholded Image", 150, 255, on_trackbar)

# Initialize with the default threshold value
on_trackbar(190)

# Wait until the user presses a key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
