from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv
import os

env = SmartHomeEnv()
model = PPO("MlpPolicy", env, verbose=1) # Create PPO agent

model.learn(total_timesteps=10000) # Train agent

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_smart_home")

print("Training complete and model saved!")
