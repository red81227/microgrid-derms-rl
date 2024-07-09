

import json
import socket
import unittest
from unittest import mock
from config.project_setting import unixsocket_config
from src.router.derms_output_client import DermsOutputClient
from src.operator.redis import RedisOperator
from src.service.event.redis.input_admin import InputAdmin


class TestDermsOutputClient(unittest.TestCase):

    @classmethod
    def setUp(cls):
        redis_operator = RedisOperator()
        input_data_set = InputAdmin(redis_operator)
        input_mapping = cls.get_test_data()
        input_data_set.hmset_input(mapping=input_mapping)
        cls.mock_server = DermsOutputClient()
    

    def test_client_behavior(self):
        mock_socket = mock.Mock(spec=socket.socket)
        
        message = self.mock_server.derms_controller.send_data()
        json_message = json.loads(message)
        assert json_message["data"]["v"] is not None

        with mock.patch('socket.socket', return_value=mock_socket):
            self.mock_server.communicate_with_unix_socket(test=True)
            mock_socket.connect.assert_called_once_with(unixsocket_config.unix_socket_to_ems_path)
            mock_socket.send.assert_called_once()



    @staticmethod
    def get_test_data():
        message = {
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
        return message
