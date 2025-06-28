import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_home_env import SmartHomeEnv

st.set_page_config(page_title="Smart Home RL Scheduler", layout="centered")

st.markdown(
    """
    <style>
        .title   {font-size:2.2em;font-weight:bold;color:#2c3e50;}
        .subtitle{font-size:1.2em;color:#57606f;margin-bottom:6px;}
        .metric  {background:#e8f5e9;padding:12px 8px;border-radius:10px;}
        .divider {height:1px;width:100%;background:#dfe4ea;margin:18px 0;}
    </style>
    """,
    unsafe_allow_html=True,
) 

st.markdown("<div class='title'>Smart Home Energy Scheduler</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Optimise your appliance usage with Reinforcement Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True) 

#LOAD MODEL & ENV 
model = PPO.load("models/ppo_smart_home")
env   = SmartHomeEnv()
obs   = env.reset()

timeline, actions, costs = [], [], []
total_cost, step = 0, 0

#SIMULATE ONE DAY 
while True:
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)

    fan_on, ac_on, fridge_on = int(action[0]), int(action[1]), 1

    # kW ratings
    fan_kw, ac_kw, fridge_kw = 0.07, 1.5, 0.3
    kwh  = (fan_on*fan_kw) + (ac_on*ac_kw) + fridge_on*fridge_kw
    rate = 6.61 if 17 <= step % 24 <= 21 else 5.23
    cost = kwh * rate

    timeline.append(step)
    actions.append([fan_on, ac_on])
    costs.append(cost)
    total_cost += cost

    step += 1
    if done:
        break

#CHARTS
st.markdown("Appliance Usage (0 = OFF, 1 = ON)")
for name, idx in zip(["Fan", "AC"], [0, 1]):
    df = pd.DataFrame({"Hour":range(24),
                       "State":[a[idx] for a in actions]}).set_index("Hour")
    st.line_chart(df, height=160)
# Fridge (always ON)
df_fridge = pd.DataFrame({"Hour":range(24),"State":[1]*24}).set_index("Hour")
st.line_chart(df_fridge, height=160, use_container_width=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

#COSTING
daily_cost    = total_cost
monthly_cost  = daily_cost * 30
col1, col2 = st.columns(2)
col1.metric("Daily Cost",    f"₹{daily_cost:,.2f}",    help="RL‑optimised schedule")
col2.metric("Monthly Cost",  f"₹{monthly_cost:,.2f}")
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)  

#ENERGY COST GRAPH
st.markdown("Hour‑by‑Hour Cost")
st.line_chart(costs, height=180)

#CSV DOWNLOAD
time_labels   = [f"{h:02d}:00" for h in range(24)]
schedule_data = [
    {"Time": time_labels[t],
     "Fan_State": actions[t][0],
     "AC_State": actions[t][1],
     "Fridge_State": 1,
     "Cost": costs[t]}
    for t in range(24)
]

df_schedule = pd.DataFrame(schedule_data)
df_schedule.loc[len(df_schedule)] = ["Total", "", "", "",
                                     round(daily_cost,2)]

csv = df_schedule.to_csv(index=False)
fname = f"schedule_{datetime.today().strftime('%Y-%m-%d')}.csv"
st.download_button("⬇️ Download Daily Schedule CSV", data=csv,
                   file_name=fname, mime="text/csv")
