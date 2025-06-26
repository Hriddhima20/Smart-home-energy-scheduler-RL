import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
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
    obs, _, done, _ = env.step(action)

    fan_on = action[0]
    ac_on = action[1]
    fridge_on = 1  # Always ON

    # Power ratings (Watts → kW)
    fan_kw = 70 / 1000
    ac_kw = 1500 / 1000
    fridge_kw = 300 / 1000

    # Total energy used in kWh
    energy_kwh = (fan_on * fan_kw) + (ac_on * ac_kw) + (fridge_on * fridge_kw)

    # Get current hour's electricity rate (avg for simplicity)
    rate = np.mean(env._get_power_rates(step))

    # Cost = energy × rate
    cost = energy_kwh * rate

    timeline.append(step)
    actions.append(action)
    costs.append(cost)
    total_cost += cost

    step += 1
    if done:
        break


st.subheader("Appliance Schedule Over 24 Hours")
# Label the appliances in order — update these names if your config is different
appliance_names = ["Fan", "AC"]

for i in range(len(appliance_names)):
    states = [a[i] for a in actions]
    st.markdown(f"### {appliance_names[i]}")
    st.caption("State of the appliance at each hour of the day (0 = OFF, 1 = ON)")
    
    # Create a labeled DataFrame for proper axis labels
    df = pd.DataFrame({
        "Hour": list(range(len(states))),
        "State": states
    }).set_index("Hour")

    st.line_chart(df)

# Plot Fridge (always ON)
st.markdown("### Fridge")
st.caption("Fridge is always ON (State = 1)")
df_fridge = pd.DataFrame({"Hour": list(range(24)), "State": [1]*24}).set_index("Hour")
st.line_chart(df_fridge)

schedule_data = []
for t in range(len(actions)):
    time_labels = [f"{h:02d}:00" for h in range(len(actions))]

    row = {"Time": time_labels[t]}
    for i in range(len(appliance_names)):
        row[f"{appliance_names[i]}_State"] = actions[t][i]
    row["Fridge_State"] = 1
    row["Cost"] = costs[t]
    schedule_data.append(row)

st.subheader("Energy Cost Over Time")
st.line_chart(costs)

st.success(f"Total Energy Cost: ₹{round(total_cost, 2)}")
st.success(f"Total Electricity Cost: **₹{total_cost:.2f}**")

# Generate human-readable time labels for each hour
time_labels = [f"{h:02d}:00" for h in range(len(actions))]

# Build schedule data
schedule_data = []
for t in range(len(actions)):
    row = {"Time": time_labels[t]}
    for i in range(env.num_appliances):
        row[f"{appliance_names[i]}_State"] = actions[t][i]
    row["Cost"] = costs[t]
    schedule_data.append(row)

# Create DataFrame
df_schedule = pd.DataFrame(schedule_data)

# Add total cost row at the end
total_row = {col: "" for col in df_schedule.columns}
total_row["Time"] = "Total"
total_row["Cost"] = round(total_cost, 2)
df_schedule = pd.concat([df_schedule, pd.DataFrame([total_row])], ignore_index=True)

# Add power usage table (based on config)
appliance_power = {
    "Fan": "70 W",
    "AC": "1500 W",
    "Fridge": "300 W"
}
power_data = {"Appliance": list(appliance_power.keys()), "Power": list(appliance_power.values())}
df_power = pd.DataFrame(power_data)

# Combine into one CSV string
combined_csv = df_schedule.to_csv(index=False) + "\n\n" + df_power.to_csv(index=False)

# Generate filename with today's date
date_str = datetime.today().strftime('%Y-%m-%d')
file_name = f"schedule_{date_str}.csv"

# Download button
st.download_button(
    label="⬇️ Download Schedule as CSV",
    data=combined_csv,
    file_name=file_name,
    mime="text/csv"
)

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
