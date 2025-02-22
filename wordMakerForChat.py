import cv2
import time
import ASL_detect


class SignLanguageWordMaker:
    def __init__(self):
        self.thresh_val = 141
        self.THRESHOLD = 0.9
        self.MIN_SIZE = 150
        self.countdown_duration = 2
        self.current_word = ""
        self.last_prediction = ""
        self.prediction_start_time = 0

    def predict_once_from_cam(self, cam):
        # created_trackbar = False - should add
        ret, frame = cam.read()
        final_letter = ''

        output_image = frame.copy()

        output_image = cv2.flip(output_image, 1)

        # Get prediction
        predictions = ASL_detect.r_cnn_single_image(frame, thresh_val=self.thresh_val, min_size=self.MIN_SIZE, removeBG=True)  # add parameters for bg removal if needed

        current_time = time.time()

        for (x, y, len, letter, maxPredictionVal) in predictions:
            if letter:
                if letter != self.last_prediction:
                    self.prediction_start_time = current_time
                    self.last_prediction = letter
                elif current_time - self.prediction_start_time >= self.countdown_duration:
                    final_letter = letter
                    self.prediction_start_time = current_time

                if self.prediction_start_time != 0:
                    remaining_time = max(0, self.countdown_duration -
                                        (current_time - self.prediction_start_time))
                    countdown_text = f"Countdown: {remaining_time:.1f}s"
                else:
                    countdown_text = f"Hold sign for {self.countdown_duration}s"

                # Draw information on display frame
                predicted_text = f"{letter} {maxPredictionVal:.2f}%"
                image = cv2.putText(output_image, predicted_text, (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(output_image, (x-10, y-10), (x + len +10, y + len+10), (0, 255, 0), 2)
                cv2.putText(output_image, countdown_text,
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        """
        if not created_trackbar:
            cv2.namedWindow("Sign Language Word Maker")
            cv2.createTrackbar("Threshold", "Sign Language Word Maker",
                                 self.thresh_val, 255, lambda x: setattr(self, 'thresh_val', x))
            created_trackbar = True"""

        return output_image, final_letter
