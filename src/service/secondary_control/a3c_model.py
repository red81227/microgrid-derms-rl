
import os
import pickle
import numpy as np
import random
from src.service.realtime_controller import RealTimeControl

import time
from src.service.secondary_control.microgrid_env import MicrogridEnv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from queue import Queue
import tracemalloc
import gc



class RLModel():
    def __init__(self,
                 datasets,
                 kp_ki_setting,
                 num_episodes: int = 1000,
                 model_path: str = "./") -> None:
        self.datasets = datasets
        self.num_episodes = num_episodes
        self.kp_ki_setting = kp_ki_setting
        self.num_actions = len(self.kp_ki_setting)
        self.model_path = model_path
        self.input_data_length = 30

        # Configuration parameters for the whole setup
        self.seed = 42
        self.gamma = 0.2 # Discount factor for past rewards
        self.eps = np.finfo(np.float32).eps.item()  
        
        num_hidden = 64
        features = 12
        timesteps = self.input_data_length
        inputs = layers.Input(shape=(timesteps, features))
        # 使用GRU層
        common = layers.GRU(num_hidden, activation="relu", return_sequences=True)(inputs)
        common = layers.Dropout(0.2)(common)
        common = layers.GlobalAveragePooling1D()(common)
        common = layers.Dense(num_hidden, activation="relu")(common)
        common = layers.Dropout(0.2)(common)
        action1 = layers.Dense(self.num_actions, activation="softmax")(common)

        concatenated = layers.concatenate([action1, common])

        action2 = layers.Dense(self.num_actions, activation="softmax")(concatenated)
        action3 = layers.Dense(self.num_actions, activation="softmax")(common)

        concatenated2 = layers.concatenate([action3, common])
        action4 = layers.Dense(self.num_actions, activation="softmax")(concatenated2)
        critic = layers.Dense(1)(common)
        self.model = keras.Model(inputs=inputs, outputs=[action1, action2, action3, action4, critic])

        if os.path.isfile(f'{model_path}model_weight.h5'):
            self.model.load_weights(f'{model_path}model_weight.h5')
        self.save_nn_model_stru(self.model)

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.huber_loss = keras.losses.Huber()
        self.action1_probs_history = []
        self.action2_probs_history = []
        self.action3_probs_history = []
        self.action4_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0
        battery_limmit = 500
        self.refrence_frequence = 60.0
        self.refrence_voltage = 220.0
        self.real_time_control = RealTimeControl(battery_limmit, self.refrence_frequence, self.refrence_voltage)
        self.p_kp = 0.2
        self.p_ki = 0.1
        self.q_kp = 0.2
        self.q_ki = 0.1
        self.model_input_queue = Queue(maxsize=self.input_data_length)


class worker():
    def __init__(self,
                    datasets,
                    kp_ki_setting,
                    num_episodes: int = 1000,
                    model_path: str = "./") -> None:
        self.datasets = datasets
        self.num_episodes = num_episodes
        self.kp_ki_setting = kp_ki_setting
        self.num_actions = len(self.kp_ki_setting)
        self.model_path = model_path
        self.input_data_length = 30

    def rl_model_training(self):
#         tracemalloc.start()
        
        # running_reward_list = []
        while True:
            gc.enable()
            start_time = time.time()
            init_frequence = self.refrence_frequence*random.uniform(0.5, 1.5)
            init_voltage = self.refrence_voltage*random.uniform(0.5, 1.5)
            microgrid_env = self.get_random_microgrid(self.refrence_frequence, init_frequence, self.refrence_voltage, init_voltage)
            train_data = self.get_random_dataset(self.datasets)
            data_length = len(train_data)
            episode_reward = 0 # 初始化episode的總獎勵
            self.p_kp = 0.2
            self.p_ki = 0.1
            self.q_kp = 0.2
            self.q_ki = 0.1
            self.get_init_input(microgrid_env, train_data)#取得初始化的input

            #加入意外狀況
            test = False
            test_case = 0
            if random.random() < 0.8:
                test_case = random.choice([1,2,3,4])
                test = True
                error_start = random.randint(0, data_length)
                error_stop = random.randint(0,  data_length - error_start)  
#                 for step in range(30, data_length - 630, 600):#3s當作訓練資料 1 min 跑測試 570 = 600-30
            count = self.input_data_length
            with tf.GradientTape() as tape:
                while True:

                    total_ob = list(self.model_input_queue.queue)
                    # input_data = np.diff(np.array(total_ob), axis=0).flatten()
                    input_data = np.array(total_ob)
                    del total_ob
                    
                    state = tf.convert_to_tensor(input_data)
                    state = tf.expand_dims(state, 0)#加入batch維度
                
                    action1_probs, action2_probs, action3_probs, action4_probs, critic_value = self.model(state)
                   
                    self.critic_value_history.append(critic_value[0, 0])
                    del input_data
                    del state
                    # Sample action from action probability distribution
                    action1 = np.random.choice(self.num_actions, p=np.squeeze(action1_probs))
                    self.action1_probs_history.append(tf.math.log(action1_probs[0, action1]))

                    action2 = np.random.choice(self.num_actions, p=np.squeeze(action2_probs))
                    self.action2_probs_history.append(tf.math.log(action2_probs[0, action2]))

                    action3 = np.random.choice(self.num_actions, p=np.squeeze(action3_probs))
                    self.action3_probs_history.append(tf.math.log(action3_probs[0, action3]))

                    action4 = np.random.choice(self.num_actions, p=np.squeeze(action4_probs))
                    self.action4_probs_history.append(tf.math.log(action4_probs[0, action4]))
                    
                    del action1_probs
                    del action2_probs
                    del action3_probs
                    del action4_probs
                    
                    self.p_kp += self.kp_ki_setting[action1]
                    self.p_ki += self.kp_ki_setting[action2]
                    self.q_kp += self.kp_ki_setting[action3]
                    self.q_ki += self.kp_ki_setting[action4]

                    del action1
                    del action2
                    del action3
                    del action4
                    
                    deduct_points = -1000 # 希望不要頻繁下決定
                    
                    params = [self.p_kp, self.p_ki, self.q_kp, self.q_ki]# 希望參數不要太大太小
                    for i in range(len(params)):
                        if params[i] < 0 or params[i] > 20:
                            params[i] = 0
                            deduct_points -= 60000
                    
                    if abs(self.p_kp - self.p_ki) > 10:# 希望kp ki參數不要差距過大
                        deduct_points -= 5000  
                    if abs(self.q_kp - self.q_ki) > 10:
                        deduct_points -= 5000

                    control_step = 0
                    while True:
                        control_step += 1
                        count += 1
                        near_state = train_data[count]
                        observation = self.get_env_observation(near_state)
                        if test:
                            if error_start > count > error_stop:
                                if test_case == 1:
                                    observation["active_power"]["pv"] = 0.0
                                if test_case == 2:
                                    observation["active_power"]["wg"] = 0.0
                                if test_case == 3:
                                    observation["active_power"]["dg"] = 0.0
                                    near_state[6] = 0.0
                                if test_case == 4:
                                    observation["active_power"]["load"] *= 3
                                    near_state[6] *= 3
                        
                        frequence_bios = abs(microgrid_env.refrence_frequence-microgrid_env.frequence)/microgrid_env.refrence_frequence

                        if observation["active_power"]["dg"] == 0:
                            plan_active_power_output =  observation["active_power"]["load"] - observation["active_power"]["pv"] - observation["active_power"]["wg"]

                        elif frequence_bios >0.03:#可能有異常發生，放棄原本的plan
                            plan_active_power_output =  observation["active_power"]["load"] - observation["active_power"]["pv"] - observation["active_power"]["wg"] - observation["active_power"]["dg"]

                        else:
                            plan_active_power_output = near_state[5]#plan battery output
            
                        actural_p_output = self.real_time_control.active_power_controller(plan_active_power_output, microgrid_env.frequence, self.p_kp, self.p_ki)
                        actural_q_output = self.real_time_control.reactive_power_controller(microgrid_env.voltage, self.q_kp, self.q_ki)
                        
                        p_q_action = self.get_action(actural_p_output, actural_q_output)
                                
                        microgrid_env.step(observation, p_q_action, True)
                        
                        del observation
                        del p_q_action
                        del plan_active_power_output

                        if self.model_input_queue.full():
                            self.model_input_queue.get()
                        
                        #collect "real_battery_output", "load", "dg", "frequence" , "diff_frequence", "kp", "ki" for training next model
                        value = self.collect_next_model_input(microgrid_env, actural_p_output, actural_q_output, near_state, self.p_kp, self.p_ki, self.q_kp, self.q_ki)
                        self.model_input_queue.put(value, block=False)

                        del actural_p_output
                        del actural_q_output
                        del near_state
                        del value

                        frequence_bios = abs(microgrid_env.refrence_frequence-microgrid_env.frequence)/microgrid_env.refrence_frequence
                        voltage_bios = abs(microgrid_env.refrence_voltage-microgrid_env.voltage)/microgrid_env.refrence_voltage
                        if frequence_bios >0.03 or voltage_bios >0.03:
                            break
                        if count > (data_length -self.input_data_length):
                            break
                        del frequence_bios
                        del voltage_bios
                        gc.collect()
                    # while not self.model_input_queue.empty():
                    #     total_ob.append(self.model_input_queue.get())
                    
                    reward = (microgrid_env.frequence_reward + microgrid_env.voltage_reward)/control_step# frequence_reward為負值
                    reward -= deduct_points

                    del deduct_points
                    del control_step

                    self.rewards_history.append(reward)

                    episode_reward += reward
                    
                    del reward
                    
                    if count > (data_length - self.input_data_length):
                        break
                    
                
                #go to Backpropagation
                del train_data
                del microgrid_env
                gc.collect()
                

                # Update running reward to check condition for solving
                self.running_reward = 0.05 * episode_reward + (1 - 0.05) * self.running_reward

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action1_probs_history,
                            self.action2_probs_history,
                            self.action3_probs_history,
                            self.action4_probs_history,
                            self.critic_value_history,
                            returns)
                actor_action1_losses = []
                actor_action2_losses = []
                actor_action3_losses = []
                actor_action4_losses = []

                critic_losses = []
                for log1_prob , log2_prob, log3_prob, log4_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_action1_losses.append(-log1_prob * diff)  # actor loss
                    actor_action2_losses.append(-log2_prob * diff)  # actor loss
                    actor_action3_losses.append(-log3_prob * diff)  # actor loss
                    actor_action4_losses.append(-log4_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                actor_losses = sum(actor_action1_losses)
                actor_losses += sum(actor_action2_losses)
                actor_losses += sum(actor_action3_losses)
                actor_losses += sum(actor_action4_losses)

                loss_value = actor_losses/4 + sum(critic_losses)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            del grads
            del actor_losses
            del critic_losses
            del history
            del tape
            del returns

            # Clear the loss and reward history
            self.action1_probs_history.clear()
            self.action2_probs_history.clear()
            self.action3_probs_history.clear()
            self.action4_probs_history.clear()

            self.critic_value_history.clear()
            self.rewards_history.clear()
            

            # Log details
            self.episode_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            # if self.episode_count % 10 == 0:
            print(f"running reward: {self.running_reward} at episode {self.episode_count} , elapsed_time: {elapsed_time}")
                # running_reward_list.append(self.running_reward)
            self.running_reward = 0
            
            if self.episode_count % 50 == 0:
                self.model.save_weights(f"{self.model_path}model_weight.h5")
                # with open('running_reward_list.pkl', 'wb') as f:
                #     pickle.dump(running_reward_list, f)
            
            # if self.episode_count > 1500:
            #         break

                # snapshot2 = tracemalloc.take_snapshot()
                # top_stats = snapshot2.compare_to(snapshot1, "lineno")
                # print("Top 10 memory usage:")
                # for stat in top_stats[:10]:
                #     print(stat)

                # tracemalloc.stop()

        
    @staticmethod
    def get_env_observation(near_state: np.array):

        observation = {
                "active_power":{
                    "pv" : near_state[1],
                    "wg" : near_state[0],
                    "load" : near_state[2],
                    "dg" : near_state[6]      
                },
                "reactive_power":{
                    "load": near_state[3]
                }

            }
        return observation
    
    @staticmethod
    def get_action(actural_p_output, actural_q_output):

        action = {
                "battery_active_power" : actural_p_output,
                "battery_reactive_power" : actural_q_output
                }
        return  action
    

    def save_nn_model_stru(self, model):
        """Save model structure"""
        model_json = model.to_json()
        with open(f'{self.model_path}rl_frequence_model.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close() 

    @staticmethod
    def get_random_dataset(datasets):
        return random.choice(datasets)

    @staticmethod
    def get_random_microgrid(refrence_frequence = 60, init_frequence = 55, refrence_voltage=220, init_voltage=215):
        grid_frequence_param = random.uniform(0.01, 0.03)
        grid_latency_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        grid_latency_param = random.choice(grid_latency_list)
        refrence_frequence = refrence_frequence
        init_frequence = init_frequence
        grid_voltage_param = random.uniform(0.001, 0.01)
        refrence_voltage = refrence_voltage
        init_voltage = init_voltage
        return MicrogridEnv(
            grid_frequence_param,
            grid_latency_param,
            refrence_frequence,
            init_frequence,
            grid_voltage_param,
            refrence_voltage,
            init_voltage)

    def collect_next_model_input(self, microgrid_env, actural_p_output, actural_q_output, near_state, p_kp, p_ki, q_kp, q_ki):
        
        diff_frequence = self.refrence_frequence - microgrid_env.frequence
        diff_voltage = self.refrence_voltage - microgrid_env.voltage
        return [
            actural_p_output, actural_q_output,
            near_state[2], near_state[6],
            microgrid_env.frequence, diff_frequence,
            microgrid_env.voltage, diff_voltage,
             p_kp, p_ki, q_kp, q_ki
             ]

    def get_init_input(self, microgrid_env, train_data):
        self.model_input_queue.queue.clear()
        for i in range(self.input_data_length):

            near_state = train_data[i]
            plan_active_power_output = near_state[5] #battery_output plan output
            
            p_kp = 0.2
            p_ki = 0.1
            q_kp = 0.2
            q_ki = 0.1

            actural_p_output = self.real_time_control.active_power_controller(plan_active_power_output, microgrid_env.frequence, p_kp, p_ki)
            actural_q_output = self.real_time_control.reactive_power_controller(microgrid_env.voltage,q_kp, q_ki)
            
            observation = self.get_env_observation(near_state)
            action = self.get_action(actural_p_output, actural_q_output)
            microgrid_env.step(observation, action, True)

            #collect "real_battery_output", "load", "dg", "frequence" ,"voltage", "diff_frequence", "diff_voltage" for training next model
            value = self.collect_next_model_input( microgrid_env, actural_p_output, actural_q_output, near_state, p_kp, p_ki, q_kp, q_ki)

            self.model_input_queue.put(value, block=False)
            
            del observation
            del action
            del near_state