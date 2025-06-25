from smart_home_env import SmartHomeEnv
from stable_baselines3 import PPO
import os

env = SmartHomeEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

os.makedirs("models", exist_ok=True
model.save("models/ppo_smart_home")

print("Training complete and model saved!")
