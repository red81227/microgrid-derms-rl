import time
from config.project_setting import unixsocket_config
import socket
from src.service.derms_controller import DermsController
from config.logger_setting import log


class DermsOutputClient:
    def __init__(self):
        self.unix_socket_to_ems_path = unixsocket_config.unix_socket_to_ems_path
        unixsocket_config.send_frequency
        self.send_frequency = unixsocket_config.send_frequency
        self.derms_controller = DermsController()

    def communicate_with_unix_socket(self, test: bool=False):
        while True:
            try:
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect(self.unix_socket_to_ems_path)
                while True:
                    # 发送数据
                    message = self.derms_controller.send_data()
                    client_socket.send(message.encode())
                    # 每隔0.1秒发送一次消息
                    time.sleep(self.send_frequency)
                    if test:
                        break
            except Exception as e:
                log.info(f"Error:{e}")
                time.sleep(1)
            if test:
                break

