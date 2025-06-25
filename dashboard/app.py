import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_home_env import SmartHomeEnv

st.set_page_config(page_title="Smart Home RL Scheduler", layout="centered")

st.title("Smart Home Energy Scheduler using RL")
st.markdown("""This dashboard demonstrates how a reinforcement learning agent can schedule appliance usage to minimize electricity cost in a smart home setting.""")

# Load model and environment
model = PPO.load("models/ppo_smart_home")
env = SmartHomeEnv()
obs = env.reset()

timeline = []
actions = []
costs = []
total_cost = 0
step = 0

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    timeline.append(step)
    actions.append(action)
    costs.append(-reward)  # negative reward = cost
    total_cost += -reward
    step += 1

    if done:
        break

st.subheader("Appliance Schedule Over 24 Hours")

# Label the appliances in order — update these names if your config is different
appliance_names = ["Fan", "AC", "Fridge"]

for i in range(env.num_appliances):
    states = [a[i] for a in actions]
    st.markdown(f"### 🔌 {appliance_names[i]}")
    st.caption("State of the appliance at each hour of the day (0 = OFF, 1 = ON)")
    
    # Create a labeled DataFrame for proper axis labels
    df = pd.DataFrame({
        "Hour": list(range(len(states))),
        "State": states
    }).set_index("Hour")

    st.line_chart(df)


st.subheader("Energy Cost Over Time")
st.line_chart(costs)

st.success(f"Total Energy Cost: ₹{round(total_cost, 2)}")
st.success(f"Total Electricity Cost: **₹{total_cost:.2f}**")

