# -*- coding: utf-8 -*-


import random
import numpy as np
from queue import Queue
import queue



class MicrogridEnv():


    def __init__(self,
                 grid_frequence_param: float,
                 grid_latency_param: float,
                 refrence_frequence: float,
                 init_frequence: float,
                 grid_voltage_param: float,
                 refrence_voltage: float,
                 init_voltage: float):

        self.refrence_frequence = refrence_frequence
        self.init_frequence = init_frequence
        self.grid_frequence_param = grid_frequence_param

        self.refrence_voltage = refrence_voltage
        self.init_voltage = init_voltage
        self.grid_voltage_param = grid_voltage_param

        self.grid_latency_param = grid_latency_param

        self.frequence_queue = Queue(maxsize=int(grid_latency_param/0.1))
        self.voltage_queue = Queue(maxsize=int(grid_latency_param/0.1))
        # self.grid_param = {
        #     "D": 0.12, # pu/HZ
        #     "T": 0.1  # time delay
        #     }
        self.pv_ramp_rate = 0.001 # %/0.1s
        self.wg_ramp_rate = 0.001 # %/0.1s
        self.activate_power_load_ramp_rate = 0.0001 # %/0.1s #負載變化率較小因為資料較即時
        self.reactive_power_load_ramp_rate = 0.0001 # %/0.1s #負載變化率較小因為資料較即時


        self.reset()

    
    #假設下控頻率為0.1秒
    #pv wg資料為一分鐘一筆
    #load 為0.1秒一筆
    def step(self, observation: dict, action: dict, ramp: bool = True):
        """
        observation: {
            active_power:{
                "pv" : 0.0,
                "wg" : 0.0,
                "load" : 0.0,
                "dg" : 0.0
                },
            reactive_power:{
                "load"
            }

        }
        action: {
            "battery_active_power" : 0.0,
            "battery_reactive_power" : 0.0,
        }
        """
        self.frequence_change(observation, action, ramp)
        self.voltage_change(observation, action, ramp)


    def frequence_change(self, observation: dict, action: dict, ramp: bool = True):
        self.delta_frequence = self.get_queue(self.frequence_queue)
        self.frequence = self.frequence + self.delta_frequence
        if self.frequence < 0:
            self.frequence = 0

        reward_frequence = min(self.refrence_frequence, np.abs(self.refrence_frequence - self.frequence))#防止Reward太大，讓AI只敢調小輸出
        reward_frequence = reward_frequence*(reward_frequence/self.refrence_frequence)*100
        self.frequence_reward -= reward_frequence

        #太陽能、風力、負載變化有隨機性
        if ramp:
            ob_pv = observation["active_power"]["pv"]
            ob_wg = observation["active_power"]["wg"]
            ob_load = observation["active_power"]["load"]
            observation["active_power"]["pv"] = round(ob_pv + random.uniform(-ob_pv*self.pv_ramp_rate, ob_pv*self.pv_ramp_rate), 3)
            observation["active_power"]["wg"] = round(ob_wg + random.uniform(-ob_wg*self.wg_ramp_rate, ob_wg*self.wg_ramp_rate), 3)
            observation["active_power"]["load"] = round(ob_load + random.uniform(-ob_load*self.activate_power_load_ramp_rate, ob_load*self.activate_power_load_ramp_rate), 3)
        

        delta_p =  observation["active_power"]["pv"] + observation["active_power"]["wg"] + observation["active_power"]["dg"] + action["battery_active_power"] - observation["active_power"]["load"]
        frequence_affect = delta_p*self.grid_frequence_param
        self.put_queue(self.frequence_queue, frequence_affect)

    def voltage_change(self, observation: dict, action: dict, ramp: bool = True):
        self.delta_voltage = self.get_queue(self.voltage_queue)
        self.voltage = self.voltage + self.delta_voltage
        if self.voltage < 0:
            self.voltage = 0

        reward_voltage = min(self.refrence_voltage, np.abs(self.refrence_voltage - self.voltage))#防止Reward太大，讓AI只敢調小輸出
        reward_voltage = reward_voltage*(reward_voltage/self.refrence_voltage)*100
        self.voltage_reward -= reward_voltage

        if ramp:
            ob_load = observation["reactive_power"]["load"]
            observation["reactive_power"]["load"] = round(ob_load + random.uniform(-ob_load*self.reactive_power_load_ramp_rate, ob_load*self.reactive_power_load_ramp_rate), 3)
        
        delta_q = action["battery_reactive_power"] + observation["reactive_power"]["load"]
        
        voltage_affact = delta_q*self.grid_voltage_param
        self.put_queue(self.voltage_queue, voltage_affact)

    def reset(self):
        self.frequence = self.init_frequence
        self.voltage = self.init_voltage
        self.frequence_reward = 0
        self.voltage_reward = 0
        self.delta_frequence = 0.0
        self.delta_voltage = 0.0
        # round(self.refrence_frequence + random.uniform(-self.refrence_frequence*0.05, self.refrence_frequence*0.05),3)

        self.frequence_queue.queue.clear()
        self.voltage_queue.queue.clear()
        for i in range(int(self.grid_latency_param/0.1)):
            self.put_queue(self.frequence_queue, 0)
            self.put_queue(self.voltage_queue, 0)

    def reward_reset(self):
        self.frequence_reward = 0
        self.voltage_reward = 0

    @staticmethod
    def put_queue(target_queue, value):
        try:
            target_queue.put(value, block=False)
        except queue.Full:
            print("Queue已滿，放棄放入")
    @staticmethod
    def get_queue(target_queue):
        try:
            item = target_queue.get_nowait()  # 嘗試從Queue中取出一個元素
        except queue.Empty:
            item = 0  # 如果Queue為空，就回傳預設值0
        return item


if __name__ == "__main__":

    grid_frequence_param = 0.12
    grid_latency_param = 0.1
    refrence_frequence = 60.0
    microgrid_env = MicrogridEnv(grid_frequence_param, grid_latency_param, refrence_frequence)
