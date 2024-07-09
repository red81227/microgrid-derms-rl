import pandas as pd
import numpy as np
import random

from src.service.secondary_control.rl_model import RLModel
import itertools


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



class RLModelTraining():
    def __init__(self) -> None:
        self.dataset = self.create_test_dataset()

    
    def rl_model_training(self):

        kp_ki_setting = [0, 0.01, 0.03, 0.08, 0.1, -0.01, -0.03, -0.08, -0.1]
        # p_ki_setting = [0, 0.01, 0.05, -0.01, -0.05]
        # q_kp_setting = [0, 0.01, 0.05, -0.01, -0.05]
        # q_ki_setting = [0, 0.01, 0.05, -0.01, -0.05]
        # kp_ki_setting = list(itertools.product(p_kp_setting, p_ki_setting, q_kp_setting, q_ki_setting))

        rl_model = RLModel(self.dataset, kp_ki_setting)
        total_episode_reward_list = rl_model.rl_model_training()
        return total_episode_reward_list

    def create_test_dataset(self):
        raw_data = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)
        total_data = self.min_to_minsceond(raw_data)
        plant_change = total_data["battery_output"] - total_data["battery_output"].shift(1)
        plant_change = plant_change.dropna().tolist()
        battery_plant_change = [total_data["battery_output"].iloc[0]]
        battery_plant_change.extend(plant_change)
        total_data["battery_plant_change"] = battery_plant_change
        datasets = np.array_split(total_data.to_numpy(), 2000)
        return datasets

    @staticmethod
    def min_to_minsceond(test_data):
        # 将索引转换为100毫秒级别\",
        test_data = test_data.resample('100ms').asfreq()

        # 线性插值
        test_data["load"] = test_data["load"].interpolate()
        test_data = test_data.fillna(method='ffill')

        test_data["dg"] = test_data["load"] - test_data["pv"] -test_data["wg"] -test_data["battery_output"]
        
        load_ramp_rate = 0.001
        dg_ramp_rate = 0.01
        real_battery_output_ramp_rate = 0.01
        
            
        test_data.loc[:,"load"] =test_data["load"].map(lambda x : round(x +  random.uniform(-x*load_ramp_rate, x*load_ramp_rate), 3))
        test_data.loc[:,"dg"] =test_data["dg"].map(lambda x : round(x +  random.uniform(-x*dg_ramp_rate, x*dg_ramp_rate), 3))
        test_data.loc[:,"real_battery_output"] =test_data["battery_output"].map(lambda x : round(x +  random.uniform(-x*real_battery_output_ramp_rate, x*real_battery_output_ramp_rate), 3))
        return test_data



