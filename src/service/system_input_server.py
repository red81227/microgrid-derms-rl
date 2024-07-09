import json

from src.operator.redis import RedisOperator
from src.service.event.redis.input_admin import InputAdmin
from src.util.function_utils import get_current_timestamp
from config.logger_setting import log

class SystemInputServer:
    def __init__(self):
        redis_operator = RedisOperator()
        self.input_data_set = InputAdmin(redis_operator)
    
    def parse_and_save_message(self, message):
        input_mapping = self.parse_mesage(message)
        self.save_data(input_mapping)
    
    def save_data(self, input_mapping: dict):
        """將資料儲存"""
        self.input_data_set.hmset_input(mapping=input_mapping)

    @staticmethod
    def parse_mesage(message)-> dict:
        """解析 JSON 資料為 Python 字典"""

        wang_an_data = json.loads(message)
        # 解析字典資料
        this_datetime = get_current_timestamp()
        FREQ_HZ = wang_an_data["data"]["FREQ_HZ"]       # FREQ頻率，0.01HZ
        U1_V = wang_an_data["data"]["U1_V"]             # U1相電壓，0.1V
        U2_V = wang_an_data["data"]["U2_V"]             # U2相電壓，0.1V
        U3_V = wang_an_data["data"]["U3_V"]             # U3相電壓，0.1V
        I1_I = wang_an_data["data"]["I1_I"]             # I1電流，0.001A
        I2_I = wang_an_data["data"]["I2_I"]             # I2電流，0.001A
        I3_I = wang_an_data["data"]["I3_I"]             # I3電流，0.001A
        P_SUM_W = wang_an_data["data"]["P_SUM_W"]       # PSUM總有效功率，W
        Q_SUM_W = wang_an_data["data"]["Q_SUM_W"]       # QSUM總無效功率，W
        S_SUM_W = wang_an_data["data"]["S_SUM_W"]       # SSUM總視在功率，W
        PF_AVG = wang_an_data["data"]["PF_AVG"]         # PFAVG平均功率因素
        AE_IMP_KWH = wang_an_data["data"]["AE_IMP_KWH"] # kWh-IMP輸入有效電能，0.1kWh
        AE_EXP_KWH = wang_an_data["data"]["AE_EXP_KWH"] # kWh-Exp輸出有效電能，0.1kWh
        AE_TOT_KWH = wang_an_data["data"]["AE_TOT_KWH"] # kWh-Total總有效電能，0.1kWh

        FREQ_HZ = FREQ_HZ / 100        # HZ
        U1_V = U1_V / 10000            # kV
        U2_V = U2_V / 10000            # kV
        U3_V = U3_V / 10000            # kV
        I1_I = I1_I / 1000             # A
        I2_I = I2_I / 1000             # A
        I3_I = I3_I / 1000             # A
        P_SUM_W = P_SUM_W / 1000       # kW
        Q_SUM_W = Q_SUM_W / 1000       # kW
        S_SUM_W = S_SUM_W / 1000       # kW
        AE_IMP_KWH = AE_IMP_KWH / 10   # kWh
        AE_EXP_KWH = AE_EXP_KWH / 10   # kWh
        AE_TOT_KWH = AE_TOT_KWH / 10   # kWh

        input_mapping = {
            "FREQ_HZ": FREQ_HZ,
            "U1_V": U1_V,
            "U2_V": U2_V,
            "U3_V": U3_V,
            "I1_I": I1_I,
            "I2_I": I2_I,
            "I3_I": I3_I,
            "P_SUM_W": P_SUM_W,
            "Q_SUM_W": Q_SUM_W,
            "S_SUM_W": S_SUM_W,
            "PF_AVG": PF_AVG,
            "AE_IMP_KWH": AE_IMP_KWH,
            "AE_EXP_KWH": AE_EXP_KWH,
            "AE_TOT_KWH": AE_TOT_KWH,
            "record_time": this_datetime
        }
        return input_mapping
