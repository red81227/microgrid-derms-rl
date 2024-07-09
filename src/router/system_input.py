
import os
import socket
from config.project_setting import unixsocket_config
import json
from src.service.system_input_server import SystemInputServer
from config.logger_setting import log


class SystemInput:
    def __init__(self):
        self.system_input_server = SystemInputServer()
        self.unix_socket_path = unixsocket_config.unix_socket_from_ems_path
        # 確保 socket 文件不存在
        if os.path.exists(self.unix_socket_path):
            os.remove(self.unix_socket_path)
    
    def run(self):
        try:
            server_socket = self.create_server()
            while True:
                message, connection = self.receive_system_message(server_socket)
                if message:
                    try:
                        self.save_message(message)
                        connection.sendall("Received your JSON data!".encode('utf-8'))
                    except json.JSONDecodeError:
                        error_message = "Invalid JSON data received"
                        log.info(error_message)
                        connection.sendall(error_message.encode('utf-8'))
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log.info(error_message)

    def create_server(self):
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(self.unix_socket_path)
        server_socket.listen()
        return server_socket
            
    def receive_system_message(self, server_socket):
        connection, client_address = server_socket.accept()
        message = connection.recv(1024)
        return message.decode('utf-8'), connection

    def save_message(self, message):
        self.system_input_server.parse_and_save_message(message)

    



