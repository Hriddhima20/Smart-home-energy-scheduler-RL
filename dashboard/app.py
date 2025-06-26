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
    obs, reward, done, _ = env.step(action)

    timeline.append(step)
    actions.append(action)
    costs.append(-reward)  # negative reward = cost
    total_cost += -reward
    step += 1

    if done:
        break

def evaluate_agent(env, model):
    obs = env.reset()
    timeline = []
    actions = []
    costs = []
    step = 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        timeline.append(step)
        actions.append(action)
        costs.append(-reward)
        step += 1

        if done:
            break

    schedule = []
    for t in range(len(actions)):
        hour = t
        fan = actions[t][0]
        ac = actions[t][1]
        fridge = 1  # Always ON
        power_kw = (fan * 70 + ac * 1500 + fridge * 300) / 1000
        cost = costs[t]
        schedule.append([hour, fan, ac, fridge, power_kw, cost])

    return schedule

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

    
with st.expander("RL vs Non-RL Comparison", expanded=True):
    st.markdown("Step 1: Estimate Your Manual (Non-RL) Usage")

    with st.form("manual_usage_form"):
        num_manual_appliances = st.number_input("Number of appliances (excluding fridge):", min_value=1, max_value=5, step=1)
        appliance_inputs = []
        submitted = st.form_submit_button("Calculate Non-RL Cost")
        
        for i in range(num_manual_appliances):
            st.markdown(f"Appliance {i+1}")
            name = st.text_input(f"Appliance {i+1} name:", key=f"name_{i}")
            power = st.number_input(f"{name or 'Appliance'} power rating (Watts):", min_value=10, max_value=5000, key=f"power_{i}")
            intervals = st.text_input(
                f"How many hours do you use {name or 'this appliance'}? (e.g., 8, 16)", key=f"intervals_{i}"
            )
            appliance_inputs.append((name, power, intervals))

        submitted = st.form_submit_button("Calculate Non-RL Cost")

    if submitted:
        non_rl_cost = 0

        for name, power, intervals in appliance_inputs:
            hours_on = set()
            for interval in intervals.split(","):
                try:
                    start, end = map(int, interval.strip().split("-"))
                    hours_on.update(range(start, end))  # end not inclusive
                except:
                    st.error(f"⚠️ Invalid format in '{interval}'")
                    continue

            for h in hours_on:
                if h < 0 or h >= 24:
                    st.warning(f"Hour {h} is invalid. Must be 0-23.")
                    continue
                rate = env._get_power_rates(h)[1]  # Use average controllable appliance rate
                power_kW = power / 1000
                cost = power_kW * rate
                non_rl_cost += cost

        st.success(f"Estimated Cost without RL: ₹{non_rl_cost:.2f}")

        # RL cost (from model's result schedule)
        schedule = evaluate_agent(env, model)
        # define the dataframe for RL results
        schedule_df = pd.DataFrame(schedule, columns=["Hour", "Fan", "AC", "Fridge", "Power (kW)", "Cost (₹)"])

        cost_with_rl = schedule_df["Cost (₹)"].sum()
        savings = non_rl_cost - cost_with_rl
        percent_saved = (savings / non_rl_cost) * 100 if non_rl_cost > 0 else 0

        st.markdown("Step 2: Compare with RL Model")

        col1, col2 = st.columns(2)
        col1.metric("Manual Cost", f"₹{non_rl_cost:.2f}")
        col2.metric("RL Model Cost", f"₹{cost_with_rl:.2f}", f"Saved ₹{savings:.2f} ({percent_saved:.1f}%)")

        st.markdown(f"""
        >**Your smart RL scheduler reduced your cost by ₹{savings:.2f} ({percent_saved:.1f}%) per day**  
        > Try changing the usage pattern above to see how much you can save!
        """)

