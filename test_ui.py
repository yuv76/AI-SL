import random
import time
from datetime import datetime

import cv2
import threading
import flet as ft
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tcpClient import TCPClient, parse_server_update_msg, create_client_server_update_msg, SERVER_UPDATE_MSG_NUM
from wordMakerForChat import SignLanguageWordMaker


class Message:
    def __init__(self, user_name: str, text: str, message_type: str):
        self.user_name = user_name
        self.text = text
        self.message_type = message_type
        self.timestamp = datetime.now().strftime("%H:%M")


class ChatApp:
    def __init__(self):
        self.loading_indicator = ft.ProgressRing(visible=False, scale=1.5)
        self.main_chat_view = None
        self.new_message_text = ""
        self.new_message = None
        self.current_chat_user = None
        self.user_list = None
        self.count_curr_chat_msgs = 0
        self.user_name = None
        self.user_wait_for_switch = None
        self.client = TCPClient()
        self.users = set()
        self.online_users_count = 0
        self.camera_running = False
        self.camera_button = None
        self.camera_image = ft.Image(src="nocam.png", width=320, height=240)

    def listen_for_updates(self, page):
        while True:
            try:
                if self.current_chat_user:
                    update_msg = create_client_server_update_msg(self.current_chat_user, "")
                else:
                    update_msg = create_client_server_update_msg("", "")
                self.client.send_message(update_msg)

                response = self.client.receive_message()

                if response and response[:3] == SERVER_UPDATE_MSG_NUM:
                    chat_content, partner_username, usernames = parse_server_update_msg(response[3:])

                    if partner_username == self.user_wait_for_switch:
                        self.user_wait_for_switch = None  # Clear flag/callback
                        self.update_chat_ui(page, chat_content, partner_username, usernames)
                    elif self.user_wait_for_switch is None:
                        self.update_chat_ui(page, chat_content, partner_username, usernames)


            except Exception as e:
                print(f"Error in update listener: {e}")
                break
            time.sleep(0.5)

    def switch_chat(self, username):
        """Switches the current chat view to the specified user."""
        self.count_curr_chat_msgs = 0
        self.current_chat_user = username
        self.user_wait_for_switch = username
        self.main_chat_view.controls.clear()
        self.loading_indicator.visible = True  # Show loading indicator
        self.update_user_list(self.page, username)
        self.page.update()

        # Send empty message to get chat history
        update_msg = create_client_server_update_msg(username, "")
        self.client.send_message(update_msg)

    def add_message(self, page: ft.Page, message: Message):
        self.main_chat_view.controls.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        message.user_name,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Text(message.timestamp, color=ft.colors.GREY_500, size=12),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            ft.Text(message.text, selectable=True),
                        ],
                        spacing=5,
                    ),
                    padding=10,
                ),
                color=ft.colors.BLUE_100 if message.message_type == "user_message" else ft.colors.GREY_800,
            )
        )
        self.main_chat_view.scroll_to(offset=len(self.main_chat_view.controls) * 1000)

    def update_chat_ui(self, page, chat_content, partner_username, usernames):

        if self.users != set(usernames):
            # Update users list
            self.users = set(usernames)
            self.update_user_list(page, partner_username)

        if partner_username not in self.users:
            self.main_chat_view.controls.clear()

        # Update chat content if we're viewing the relevant chat
        elif partner_username == self.current_chat_user:
            self.loading_indicator.visible = False

            if len(chat_content) > self.count_curr_chat_msgs:
                new_count = len(chat_content)
                chat_content = chat_content[-(new_count - self.count_curr_chat_msgs):]
                for author, msg in chat_content:
                    if chat_content:
                        self.add_message(
                            page,
                            Message(
                                user_name=author,
                                text=msg,
                                message_type="partner_message"
                            )
                        )
                self.count_curr_chat_msgs = new_count
        page.update()

    def update_user_list(self, page, current_partner=None):
        self.user_list.controls.clear()
        for username in sorted(self.users):
            if username != page.session.get("user_name"):
                self.user_list.controls.append(
                    ft.ListTile(
                        leading=ft.Icon(ft.icons.PERSON),
                        title=ft.Text(username),
                        selected=(username == current_partner),
                        on_click=lambda e, user=username: self.switch_chat(user)
                    )
                )

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        wordMaker = SignLanguageWordMaker()
        while self.camera_running:
            frame, letter = wordMaker.predict_once_from_cam(cap)

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

    def main(self, page: ft.Page):
        self.page = page  # Store page reference for updates
        page.horizontal_alignment = "center"
        page.title = "Private Chat"
        page.theme_mode = "dark"

        def join_chat_click(e):
            if not self.user_name.value:
                self.user_name.error_text = "Name cannot be empty!"
                page.update()
                return

            if not self.client.connect():
                self.user_name.error_text = "Could not connect to server!"
                page.update()
                return

            login_msg = f"200{len(self.user_name.value):02d}{self.user_name.value}"
            if not self.client.send_message(login_msg):
                self.user_name.error_text = "Login failed!"
                page.update()
                return

            page.session.set("user_name", self.user_name.value)
            page.clean()
            self.init_chat_view(page)

            threading.Thread(target=self.listen_for_updates, args=(page,), daemon=True).start()
            page.update()

        self.user_name = ft.TextField(label="Enter your name to join", autofocus=True, on_submit=join_chat_click)

        page.add(
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            self.user_name,
                            ft.ElevatedButton("Join Chat", on_click=join_chat_click),
                        ],
                        tight=True,
                    ),
                    padding=20,
                )
            )
        )

    def init_chat_view(self, page: ft.Page):
        """Initialize chat interface."""

        def send_message_click(e):
            if self.new_message.value and self.current_chat_user:
                update_msg = create_client_server_update_msg(self.current_chat_user, self.new_message.value)
                self.client.send_message(update_msg)

                self.new_message.value = ""
                page.update()

        # User list
        self.user_list = ft.ListView(expand=1, spacing=10, padding=20, width=200)

        # Chat messages view
        self.main_chat_view = ft.ListView(expand=True, spacing=10, auto_scroll=True, padding=20)

        # Chat input field
        self.new_message = ft.TextField(
            hint_text="Write a message...",
            autofocus=True,
            shift_enter=True,
            min_lines=1,
            max_lines=5,
            filled=True,
            expand=True,
            on_submit=send_message_click,
        )

        # Camera Button
        self.camera_button = ft.ElevatedButton("Start Camera", on_click=self.toggle_camera)

        # Layout
        page.add(
            ft.Row(
                [
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("Users Online", size=20, weight=ft.FontWeight.BOLD),
                                    self.user_list,
                                    self.camera_button,  # Add camera button to the UI
                                    self.camera_image,  # Add camera feed to the UI
                                ],
                                spacing=10,
                            ),
                            padding=10,
                        ),
                        width=350,
                        height=page.height,
                    ),
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("Select a user to start chatting", size=20, weight=ft.FontWeight.BOLD),
                                    self.loading_indicator,
                                    self.main_chat_view,
                                    ft.Row(
                                        [
                                            self.new_message,
                                            ft.IconButton(
                                                icon=ft.icons.SEND_ROUNDED,
                                                tooltip="Send message",
                                                on_click=send_message_click,
                                            ),
                                        ]
                                    ),
                                ],
                                spacing=20,
                            ),
                            padding=20,
                            expand=True,
                        ),
                        expand=True,
                        height=page.height,
                    ),
                ],
                expand=True,
                spacing=10,
            )
        )


if __name__ == "__main__":
    app = ChatApp()
    ft.app(target=app.main, port=random.randint(8550, 8650))
