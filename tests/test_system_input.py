

import json
import unittest
from unittest.mock import patch, Mock
from config.project_setting import unixsocket_config
from src.router.system_input import SystemInput
from src.operator.redis import RedisOperator
from src.service.event.redis.input_admin import InputAdmin

class TestSystemInput(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.mock_server = SystemInput()


    def test_server_behavior(self):
        with patch('socket.socket') as mock_socket:
            # Create a mock server socket object
            mock_server_socket = mock_socket.return_value

            # Set up the mock server socket's methods and behavior
            mock_server_socket.bind.return_value = None
            mock_server_socket.listen.return_value = None

            mock_connection = Mock()
            test_message = self.get_test_message()
            mock_connection.recv.return_value  = test_message
            mock_connection.sendall.return_value=None
            mock_server_socket.accept.return_value = (mock_connection, "MockClientAddress")
            
            # Create a mock instance of the MyServer class
            server_socket = self.mock_server.create_server()
            message, connection = self.mock_server.receive_system_message(server_socket)
        self.mock_server.save_message(message)

        # Assertions (for demonstration, you can modify these as needed)
        mock_server_socket.bind.assert_called_once_with(unixsocket_config.unix_socket_from_ems_path)
        mock_server_socket.listen.assert_called_once_with()
        # Make assertions for multiple client connections if needed

        redis_operator = RedisOperator()
        input_data_set = InputAdmin(redis_operator)
        result = input_data_set.hmgetall_input()
        result_dict = {key.decode('utf-8'): float(value.decode('utf-8')) for key, value in result.items()}
        self.assertEqual(result_dict["FREQ_HZ"], 0.6)

    @staticmethod
    def get_test_message():
        message = {
                "type":1,
                "dataTime":"20230801020307321",
                "data":{
                    "FREQ_HZ":60,
                    "U1_V":10000,
                    "U2_V":20000,
                    "U3_V":30000,
                    "I1_I":1111,
                    "I2_I":2222,
                    "I3_I":3333,
                    "P_SUM_W":555555,
                    "Q_SUM_W":777777,
                    "S_SUM_W":999999,
                    "PF_AVG":0.567,
                    "AE_IMP_KWH":2222222,
                    "AE_EXP_KWH":6666666,
                    "AE_TOT_KWH":8888888
                }
            }
        return json.dumps(message).encode()
