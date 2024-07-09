from datetime import *
import json
import random

from src.operator.redis import RedisOperator
from src.service.event.redis.input_admin import InputAdmin
from src.service.event.redis.output_p_admin import OutputPAdmin
from src.service.event.redis.output_q_admin import OutqutQAdmin
from src.service.realtime_controller import RealTimeControl
from src.util.function_utils import get_current_timestamp
from config.logger_setting import log

class DermsController():
    def __init__(self):
        redis_operator = RedisOperator()
        self.input_data_set = InputAdmin(redis_operator)
        self.output_p_set = OutputPAdmin(redis_operator)
        self.output_q_set = OutqutQAdmin(redis_operator)
        self.max_battery_p_output = 500
        self.smooth_rate = 0.1
        self.max_p_charge = self.max_battery_p_output*self.smooth_rate
        self.real_time_control = RealTimeControl()

    def send_data(self):
        system_input = self.get_system_input_from_redis()
        if system_input:
            p, q = self.ems_control(system_input)
        else:
            p = 0
            q = 0
        old_p = self.get_old_p()
        old_q = self.get_old_q()
        smooth_p = self.smooth_control(p, old_p, self.max_p_charge)
        smooth_q = self.smooth_control(q, old_q, self.max_p_charge)
        self.output_p_set.set_output_p(smooth_p, ex=3) #保留3秒
        self.output_q_set.set_outqut_q(smooth_q, ex=3)
        
        return self.package_message(p,q)

    def get_system_input_from_redis(self)->dict:
        # 讀取 redis 資料
        result = self.input_data_set.hmgetall_input()
        return {key.decode('utf-8'): float(value.decode('utf-8')) for key, value in result.items()}
    
    def get_old_p(self)->float:
        # 讀取 redis 資料
        old_p = self.output_p_set.get_output_p()
        if old_p:
            old_p = float(old_p)
        else:
            old_p = 0
        return old_p

    def get_old_q(self)->float:
        # 讀取 redis 資料
        old_q = self.output_q_set.get_output_q()
        if old_q:
            old_q = float(old_q)
        else:
            old_q = 0
        return old_q
    
    def smooth_control(self, control_value:float, old_control_value:float, limit: float)-> float:
        if old_control_value > control_value:
            p_change = old_control_value - control_value
            smooth_p_change = min(p_change, self.max_p_charge)
            smooth_p = old_control_value - smooth_p_change
        elif old_control_value < control_value:
            p_change = control_value - old_control_value
            smooth_p_change = min(p_change, self.max_p_charge)
            smooth_p = old_control_value + smooth_p_change
        else:
            smooth_p = control_value
        return smooth_p
    
    def package_message(self, p:int)-> str:
        """將資料打包成 JSON 格式的文字"""
        this_datetime = get_current_timestamp()
        data = {
            "type": 1,
            "dataTime": this_datetime,
            "data":{
                "m": "ess",
                "d": "0922533e-7b74-11ed-9cf2-0242ac110002",
                "n": "p",
                "v": f"{p}",
                "vt": "INT"
            }
        }
        # 將 JSON 數據轉換為字串
        return json.dumps(data)


    def ems_control(self, system_input: dict)-> int:
        p = self.real_time_control.active_power_controller(
            plan_active_power_output=0,
            realtime_frequence=system_input["FREQ_HZ"],
            Kp=0.5, Ki=0.1, Kd=0, sample_time=0.1
        )
        q = self.real_time_control.reactive_power_controller(
            realtime_voltage=system_input["U1_V"],
            Kp=0.5, Ki=0.1, Kd=0, sample_time=0.1
        )
        
        return p, q

