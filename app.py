import random
import flet as ft
from chat_view import ChatApp


def main():
    app = ChatApp()
    ft.app(target=app.main, port=random.randint(8550, 8650))


if __name__ == "__main__":
    main()
