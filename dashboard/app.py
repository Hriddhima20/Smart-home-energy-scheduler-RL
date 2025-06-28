import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_home_env import SmartHomeEnv

if not os.path.exists("results"):
    os.makedirs("results")

st.set_page_config(page_title="Smart Home RL Scheduler", layout="centered")
st.markdown(
    """
    <style>
        .title   {font-size:2.2em;font-weight:bold;color:#b08607;}
        .subtitle{font-size:1.2em;color:#0b5929;margin-bottom:6px;}
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

st.markdown("Fan")
df_fan = pd.DataFrame({
    "Hour": range(24),
    "State": [a[0] for a in actions]
}).set_index("Hour")
st.line_chart(df_fan, height=160)

st.markdown("Air Conditioner (AC)")
df_ac = pd.DataFrame({
    "Hour": range(24),
    "State": [a[1] for a in actions]
}).set_index("Hour")
st.line_chart(df_ac, height=160)

st.markdown("Fridge")
df_fridge = pd.DataFrame({
    "Hour": range(24),
    "State": [1] * 24  # always ON
}).set_index("Hour")
st.line_chart(df_fridge, height=160)

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
results_path = os.path.join("results", fname)
with open(results_path, "w") as f:
    f.write(combined_csv)

st.download_button("⬇️ Download Daily Schedule CSV", data=csv,
                   file_name=fname, mime="text/csv")

with st.expander("📆 View Monthly Bill Estimation", expanded=False):
    st.markdown("Assumed Daily Appliance Usage")
    daily_usage = {
        "Fridge": {"power": 200, "hours": 24},
        "Fan": {"power": 70, "hours": 16},
        "AC": {"power": 1500, "hours": 6}
    }

    total_daily_kwh = 0
    per_appliance_kwh = {}

    for name, info in daily_usage.items():
        kwh = (info["power"] / 1000) * info["hours"]
        per_appliance_kwh[name] = kwh
        total_daily_kwh += kwh
        st.write(f"🔹 **{name}**: {info['power']}W × {info['hours']}h = {kwh:.2f} kWh/day")

    monthly_kwh = total_daily_kwh * 30
    st.markdown(f"### 🔢 Total Monthly Units: **{monthly_kwh:.2f} kWh**")

    # Slab-based energy bill calculation
    remaining_units = monthly_kwh
    bill = 0
    slabs = [
        (30, 3.34),
        (20, 4.27),
        (100, 5.23),
        (150, 6.61),
        (float('inf'), 6.80)
    ]

    for slab_units, rate in slabs:
        units = min(remaining_units, slab_units)
        slab_cost = units * rate
        bill += slab_cost
        st.write(f"{units:.0f} units @ ₹{rate}/kWh = ₹{slab_cost:.2f}")
        remaining_units -= units
        if remaining_units <= 0:
            break

    # Fixed charges
    fixed_charge = 27 * 30  # for 3kW sanctioned load
    gst = 0.18 * fixed_charge
    duty = 0.06 * monthly_kwh

    total_bill = bill + fixed_charge + gst + duty

    st.markdown("**Monthly Energy Charges Breakdown**")
    st.write(f"🔹 Energy Charge: ₹{bill:.2f}")
    st.write(f"🔹 Fixed Charge (3kW): ₹{fixed_charge}")
    st.write(f"🔹 Electricity Duty: ₹{duty:.2f}")
    st.write(f"🔹 GST on Fixed Charge: ₹{gst:.2f}")
    st.success(f"🧾 **Estimated Total Monthly Bill: ₹{total_bill:.2f}**")

    # Bar chart
    st.markdown("Appliance-wise Monthly Consumption")
    monthly_kwh_per_appliance = {k: v * 30 for k, v in per_appliance_kwh.items()}
    st.bar_chart(pd.DataFrame.from_dict(monthly_kwh_per_appliance, orient='index', columns=["kWh"]))
