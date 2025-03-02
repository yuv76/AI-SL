import cv2
import base64
import threading
from io import BytesIO
from PIL import Image
from wordMakerForChat import SignLanguageWordMaker
from flet import Icons, colors
from thresholdImage import perform_threshold


class CameraService:
    def __init__(self, new_message, camera_image, page, camera_button, lowercase_button, slider, thresh, remove_back):
        self.camera_button = camera_button
        self.camera_running = False
        self.new_message = new_message
        self.new_message_text = ""
        self.camera_image = camera_image
        self.page = page
        self.lowercase_button = lowercase_button
        self.is_lower = False if lowercase_button.text == "lower" else true
        self.threshold_slider = slider
        self.threshold_button = thresh
        self.is_thresh = False if thresh.text == "Threshold" else true
        self.background_rem = remove_back
        self.is_removebg = False if remove_back.text == "Remove Background" else true

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        wordMaker = SignLanguageWordMaker()
        while self.camera_running:
            frame, letter = wordMaker.predict_once_from_cam(cap, self.is_lower, self.threshold_slider.value.__round__(), self.is_removebg)

            self.new_message.value += letter
            self.new_message_text += letter
            self.new_message.update()

            if self.is_thresh:
                frame = perform_threshold(frame, self.threshold_slider.value.__round__())

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
            self.camera_button.icon = Icons.PAUSE_CIRCLE_FILLED_ROUNDED
            self.camera_button.icon_color = colors.RED_ACCENT_400
            self.camera_button.tooltip = "Stop camera"
        else:
            self.camera_running = False
            self.camera_button.text = "Start Camera"
            self.camera_button.icon = Icons.PLAY_CIRCLE_FILL_OUTLINED
            self.camera_button.icon_color = colors.GREEN_ACCENT_700
            self.camera_button.tooltip = "Start camera"

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

    def toggle_thresh_btn(self, e):
        if self.is_thresh:
            self.threshold_button.text = "Threshold"
            self.is_thresh = False
        else:
            self.threshold_button.text = "Disable Threshold"
            self.is_thresh = True

    def toggle_remove_back(self, e):
        if self.is_removebg:
            self.background_rem.text = "Remove Background"
            self.is_removebg = False
        else:
            self.background_rem.text = "Restore Background"
            self.is_removebg = True

