import cv2
import base64
import threading
from io import BytesIO
from PIL import Image
from wordMakerForChat import SignLanguageWordMaker


class CameraService:
    def __init__(self, new_message, camera_image, page, camera_button, lowercase_button, slider):
        self.camera_button = camera_button
        self.camera_running = False
        self.new_message = new_message
        self.new_message_text = ""
        self.camera_image = camera_image
        self.page = page
        self.lowercase_button = lowercase_button
        self.is_lower = False if lowercase_button.text == "lower" else true
        self.threshold_slider = slider

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        wordMaker = SignLanguageWordMaker()
        while self.camera_running:
            frame, letter = wordMaker.predict_once_from_cam(cap, self.is_lower, self.threshold_slider.value.__round__())

            self.new_message.value += letter
            self.new_message_text += letter
            self.new_message.update()

            # Convert frame to Base64
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame)
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Update Flet UI
            if self.camera_running:
                self.camera_image.src_base64 = img_b64
                self.page.update()

        cap.release()

    def toggle_camera(self, e):
        if not self.camera_running:
            self.camera_running = True
            threading.Thread(target=self.start_camera, daemon=True).start()
            self.camera_button.text = "Stop Camera"
        else:
            self.camera_running = False
            self.camera_button.text = "Start Camera"

            # update image to indicate cam not running
            img = cv2.imread("nocam.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Update Flet UI
            self.camera_image.src_base64 = img_b64
            self.page.update()

    def toggle_lowercase(self, e):
        if self.is_lower:
            self.lowercase_button.text = "lower"
            self.is_lower = False
        else:
            self.lowercase_button.text = "upper"
            self.is_lower = True
