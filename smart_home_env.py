import gym
from gym import spaces
import numpy as np

class SmartHomeEnv(gym.Env):
    def __init__(self):
        super(SmartHomeEnv, self).__init__()
        self.num_appliances = 2
      
        self.time_slots = 24
        # State: [time (normalized), appliance1_on, appliance2_on, appliance3_on]
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_appliances + 1,), dtype=np.float32)
        # Action: On/off for each appliance
      
        self.action_space = spaces.MultiBinary(self.num_appliances)
        self.reset()

    def reset(self):
        self.time = 0
        self.appliance_states = np.zeros(self.num_appliances+1)
        self.done = False
        return self._get_state()

    def step(self, action):
        fridge_on = 1  # Fridge is always ON

    # Full action now has: [fridge, fan, ac]
        full_action = np.concatenate(([fridge_on], action))
        self.appliance_states = full_action
        power_rates = self._get_power_rates(self.time)
    # Compute energy cost: dot product of power rates and usage
        energy_cost = np.dot(full_action, power_rates)
    
        reward = -energy_cost  # Minimize cost
        self.time += 1
        self.done = self.time >= self.time_slots
    
        return self._get_state(), reward, self.done, {"cost": energy_cost, "action": full_action}

    def _get_state(self):
        normalized_time = self.time / self.time_slots
        return np.concatenate(([normalized_time], self.appliance_states[1:]))

    def _get_power_rates(self, time_step):
        # Dummy dynamic pricing (peak hours 6-10pm)
        if 17 <= time_step <= 21:
            return np.array([1.5, 3.5, 2.5])  # higher cost
        else:
            return np.array([1.5, 2.0, 0.8])  # lower cost

    def render(self, mode='human'):
        print(f"Time: {self.time}, Appliances: {self.appliance_states}")
