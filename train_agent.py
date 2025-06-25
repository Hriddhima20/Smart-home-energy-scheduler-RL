"""
Trains a PPO agent to schedule Fan and AC while the fridge remains ON 24×7.
"""

from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv
import os

def main():
    env = SmartHomeEnv()          # fridge fixed ON; action_space = MultiBinary(2)

    model = PPO( 
        policy="MlpPolicy", 
        env=env,
        verbose=1,                # set to 0 to silence training log
        tensorboard_log="./logs"   
    )

    model.learn(total_timesteps=10_000)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_smart_home")
    print("✅ Training complete – model saved to models/ppo_smart_home.zip")

if __name__ == "__main__":
    main()
