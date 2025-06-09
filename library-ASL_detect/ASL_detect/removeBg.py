import cv2
import mediapipe as mp
import numpy as np

# init media pipe
mp_hands = mp.solutions.hands

# only detect hand if has a high prob of being a hand (70% +)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


def remove_background(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # feed the frame to media pipe
    hand_results = hands.process(frame)

    hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    bounding_boxes = []

    # for each normalized point returned from the model of a hand, get the points in the frame by multiplying it by
    # frame width and height
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]

            # Create a bounding box around the landmarks
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

            # Calculate width and height of the bounding box
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Determine the size of the square (max of width and height)
            box_size = max(box_width, box_height)

            # Calculate the center of the current bounding box
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # Adjust the bounding box to make it a square and center the hand
            x_min = max(0, x_center - box_size // 2)
            x_max = min(w, x_center + box_size // 2)
            y_min = max(0, y_center - box_size // 2)
            y_max = min(h, y_center + box_size // 2)

            bounding_boxes.append((x_min-15, y_min-15, box_size+15, box_size+15))

            cv2.fillConvexPoly(hand_mask, np.array(points, dtype=np.int32), 1)

    return bounding_boxes


def process_hand_frame(frame, bounding_box_padding=15, hand_padding=35):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    h, w, _ = frame.shape
    hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    bounding_boxes = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get points for the hand
            points = [(int(landmark.x * w), int(landmark.y * h))
                      for landmark in hand_landmarks.landmark]

            cv2.fillConvexPoly(hand_mask, np.array(points, dtype=np.int32), 1)

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

            box_width = x_max - x_min
            box_height = y_max - y_min
            box_size = max(box_width, box_height)

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            x_min = max(0, x_center - box_size // 2)
            x_max = min(w, x_center + box_size // 2)
            y_min = max(0, y_center - box_size // 2)
            y_max = min(h, y_center + box_size // 2)

            # Add padding to bounding box

            x_min = max(0, x_min - bounding_box_padding)
            y_min = max(0, y_min - bounding_box_padding)
            box_size = box_size + (bounding_box_padding * 2)

            bounding_boxes.append((int(x_min), int(y_min), int(box_size), int(box_size)))

    # Add padding around the hand
    kernel = np.ones((hand_padding, hand_padding), np.uint8)
    hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)

    # Create white background
    white_frame = np.ones_like(frame) * 255

    # Combine original frame with white background using mask
    output_frame = np.where(hand_mask[..., None] == 1, frame, white_frame)

    return output_frame, bounding_boxes

