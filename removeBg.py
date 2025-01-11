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

    # for each normalized point returned from the model of a hand, get the points in the frame by multiplying it by
    # frame width and height
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
            cv2.fillConvexPoly(hand_mask, np.array(points, dtype=np.int32), 1)

        # add padding around the hand
        kernel = np.ones((35, 35), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)

    white_frame = np.ones_like(frame) * 255

    # If hand_mask == 1: Keep the original (clear) frame (hands)
    # Else: Use the blurred frame
    output_frame = np.where(hand_mask[..., None] == 1, frame, white_frame)

    return output_frame
