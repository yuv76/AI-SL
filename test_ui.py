import random

import flet as ft
from datetime import datetime
import threading
from tcpClient import TCPClient, parse_server_update_msg, create_client_server_update_msg, SERVER_UPDATE_MSG_NUM


class Message:
    def __init__(self, user_name: str, text: str, message_type: str):
        self.user_name = user_name
        self.text = text
        self.message_type = message_type
        self.timestamp = datetime.now().strftime("%H:%M")


class ChatApp:
    def __init__(self):
        self.tcpClient_chat_view = None
        self.new_message = None
        self.current_chat_user = None
        self.user_list = None
        self.user_name = None
        self.client = TCPClient()
        self.users = set()

    def tcpClient(self, page: ft.Page):
        page.horizontal_alignment = "center"
        page.title = "Private Chat"
        page.theme_mode = "dark"

        def join_chat_click(e):
            if not self.user_name.value:
                self.user_name.error_text = "Name cannot be empty!"
                page.update()
                return

            # Connect to server and login
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

            # Start listening for server updates in a separate thread
            threading.Thread(target=self.listen_for_updates, args=(page,), daemon=True).start()

            page.update()

        self.user_name = ft.TextField(
            label="Enter your name to join",
            autofocus=True,
            on_submit=join_chat_click
        )

        page.add(
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            self.user_name,
                            ft.ElevatedButton(
                                "Join Chat",
                                on_click=join_chat_click
                            ),
                        ],
                        tight=True,
                    ),
                    padding=20,
                )
            )
        )

    def listen_for_updates(self, page):
        while True:
            try:
                response = self.client.receive_message()
                if response and response[:3] == SERVER_UPDATE_MSG_NUM:
                    chat_content, partner_username, usernames = parse_server_update_msg(response[3:])

                    # Update UI in the tcpClient thread
                    page.invoke_async(lambda: self.update_chat_ui(
                        page, chat_content, partner_username, usernames
                    ))
            except Exception as e:
                print(f"Error in update listener: {e}")
                break

    def update_chat_ui(self, page, chat_content, partner_username, usernames):
        # Update users list
        self.users = set(usernames)
        self.update_user_list(page, partner_username)

        # Update chat content if we're viewing the relevant chat
        if partner_username == self.current_chat_user:
            self.tcpClient_chat_view.controls.clear()
            if chat_content:
                self.add_message(
                    page,
                    Message(
                        user_name=partner_username,
                        text=chat_content,
                        message_type="partner_message"
                    )
                )
        page.update()

    def init_chat_view(self, page: ft.Page):
        def send_message_click(e):
            if self.new_message.value and self.current_chat_user:
                # Send message to server
                update_msg = create_client_server_update_msg(
                    self.current_chat_user,
                    self.new_message.value
                )
                self.client.send_message(update_msg)

                self.new_message.value = ""
                page.update()

        def switch_chat(username):
            self.current_chat_user = username
            self.tcpClient_chat_view.controls.clear()
            # Send empty message to get chat history
            update_msg = create_client_server_update_msg(username, "")
            self.client.send_message(update_msg)
            page.update()

        # Create user list view
        self.user_list = ft.ListView(
            expand=1,
            spacing=10,
            padding=20,
            width=200,
        )

        # Chat messages view
        self.tcpClient_chat_view = ft.ListView(
            expand=True,
            spacing=10,
            auto_scroll=True,
            padding=20,
        )

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

        # Layout
        page.add(
            ft.Row(
                [
                    # Left column - User list
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("Users Online", size=20, weight=ft.FontWeight.BOLD),
                                    self.user_list,
                                ],
                                spacing=10,
                            ),
                            padding=10,
                        ),
                        width=250,
                        height=page.height,
                    ),

                    # Right column - Chat view
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "Select a user to start chatting",
                                        size=20,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    self.tcpClient_chat_view,
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

    def add_message(self, page: ft.Page, message: Message):
        self.tcpClient_chat_view.controls.append(
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
                                alignment=ft.tcpClientAxisAlignment.SPACE_BETWEEN,
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
        self.tcpClient_chat_view.scroll_to(offset=len(self.tcpClient_chat_view.controls) * 1000)


def main():
    chat_app = ChatApp()
    ft.app(target=chat_app.main, port=random.randint(8550, 8650), view=ft.WEB_BROWSER)


if __name__ == "__main__":
    main()
