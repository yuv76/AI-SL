import socket

bind_ip = "127.0.0.1"
bind_port = 8876
LOGIN_MSG_NUM = "200"
SERVER_UPDATE_MSG_NUM = "101"
CLIENT_UPDATE_MSG_NUM = "204"

import socket
import sys


class TCPClient:
    def __init__(self, host='localhost', port=bind_port):
        """Initialize TCP client with host and port."""
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"Connection failed - server might be offline")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def send_message(self, message):
        try:
            self.socket.send(message.encode())
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def receive_message(self, buffer_size=1024):
        try:
            data = self.socket.recv(buffer_size)
            return data.decode()
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None

    def close(self):
        """Close the connection."""
        if self.socket:
            self.socket.close()
            print("Connection closed")


def build_login_msg(name):
    return "200" + "%02d" % len(name) + name


def extract(message, length_size):
    length = int(message[:length_size])
    content = message[length_size:length_size + length]
    remaining = message[length_size + length:]
    return content, remaining


def parse_server_update_msg(msg):
    chat_content, msg = extract(msg, 5)
    partner_username, msg = extract(msg, 2)
    usernames_str, _ = extract(msg, 5)

    return chat_content, partner_username, usernames_str.split('&')


def create_client_server_update_msg(second_username, new_msg):
    return CLIENT_UPDATE_MSG_NUM + "%02d" % len(second_username) + second_username + "%05d" % len(new_msg) + new_msg



def main():
    # Create a TCP client instance
    client = TCPClient()

    # Try to connect to the server
    if not client.connect():
        sys.exit(1)

    try:
        while True:
            name = input("Please enter this user's  name (for login) ")

            # Check if user wants to quit
            if name.lower() == 'quit':
                break

            # Send message to server
            if client.send_message(build_login_msg(name)):
                # Receive response from server
                response = client.receive_message()
                if response[:3] == SERVER_UPDATE_MSG_NUM:
                    chat_content, partner_username, usernames = parse_server_update_msg(response[3:])
                    print(chat_content, partner_username, usernames)

                if response:
                    print(f"Server response: {response}")
                else:
                    print("No response from server")

    except KeyboardInterrupt:
        print("\nClient terminated by user")
    finally:
        client.close()


if __name__ == "__main__":
    main()
