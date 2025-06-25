from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv

# Load environment and model
env = SmartHomeEnv()
model = PPO.load("models/ppo_smart_home")

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    total_reward += reward

print(f"\nTotal Reward (Cost Inverse): {total_reward}")
