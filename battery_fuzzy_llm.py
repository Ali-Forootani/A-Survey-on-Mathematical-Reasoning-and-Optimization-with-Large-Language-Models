import os
from openai import OpenAI


#https://ai.pydantic.dev/#hello-world-example



# Set API key and base URL as environment variables
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["OPENAI_API_BASE"] = "Your API Base"

api_key = "Your API Key"
api_base = "Your API Base"




client = OpenAI(api_key=api_key, base_url=api_base)

# Define the BESS optimization problem and fuzzy rules in the prompt
prompt = """
We want to optimize the operation of a Battery Energy Storage System (BESS). The goal is to minimize the operational cost by determining the charge/discharge schedule for the battery over a 10-hour period.

### Problem Definition:
The problem involves scheduling the charging and discharging of a battery to minimize the operational cost while adhering to constraints.

1. **Battery Parameters**:
   - Maximum capacity (SOC_max): 100 kWh
   - Initial state of charge (SOC_0): 50 kWh
   - Charging efficiency (η): 90%
   - Discharging efficiency (1/η): 90%
   - Maximum charge/discharge power: 20 kW

2. **Electricity Prices and Revenue**:
   - Electricity price (C_t, $/kWh): [20, 25, 15, 10, 30, 35, 40, 20, 15, 10]
   - Revenue from discharging (R_t, $/kWh): [30, 35, 25, 20, 40, 45, 50, 30, 25, 20]

3. **Constraints**:
   - Battery capacity: \(0 \leq SOC(t) \leq SOC_{max}\)
   - Power limits: \(0 \leq P_{charge}(t) \leq P_{charge}^{max}\), \(0 \leq P_{discharge}(t) \leq P_{discharge}^{max}\)
   - Energy balance: \(SOC(t+1) = SOC(t) + \eta \cdot P_{charge}(t) - \frac{1}{\eta} \cdot P_{discharge}(t)\)
   - No simultaneous charging/discharging: \(P_{charge}(t) \cdot P_{discharge}(t) = 0\)

4. **Objective Function**:
   - Minimize \(Z = \sum_{t=1}^{10} (C_t \cdot P_{charge}(t) - R_t \cdot P_{discharge}(t))\)

### Fuzzy Logic Rules:
1. If the electricity price \(C_t\) is high, prioritize discharging the battery.
2. If the battery SOC is below 30%, limit discharging and prioritize charging.
3. If \(C_t\) is low, delay charging unless the SOC is critically low.
4. Avoid frequent switching between charge and discharge states.

### Requirements:
1. Use fuzzy logic rules to guide decisions dynamically during optimization.
2. Reformulate constraints or priorities based on the fuzzy logic rules.
3. Solve the optimization problem and provide:
   - A step-by-step explanation of how fuzzy logic influenced the solution.
   - The final optimal charge/discharge schedule (\(P_{charge}(t)\), \(P_{discharge}(t)\)) and state of charge (\(SOC(t)\)) for each hour.
   - The total minimized operational cost.

### Expected Output:
1. An explanation of how fuzzy logic rules were applied in each time step.
2. The optimal charge/discharge schedule over 10 hours.
3. The total minimized operational cost of the system.
"""


# Call the OpenAI API to generate a response
response = client.chat.completions.create(
    model="alias-fast",
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled in optimization and fuzzy logic for battery systems."},
        {"role": "user", "content": prompt},
    ]
)



# Print the response from the API
print(response.choices[0].message.content)

print("++++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++++")

##########################################
##########################################
##########################################
##########################################

"""

#!pip install pulp  # Uncomment this if you need to install PuLP

import pulp

# -----------------------------
# 1. Problem Data
# -----------------------------
SOC_max = 100.0   # kWh
SOC_0   = 50.0    # initial state of charge (kWh)

# Efficiencies
eta_charge    = 0.90   # charging efficiency
eta_discharge = 0.90   # discharging efficiency
# The problem statement used "1/eta" for discharging, also 90%,
# but typically that means for every 1 kWh discharged from the battery,
# only 0.9 kWh is delivered outside, or the battery loses 1/0.9 kWh from its SOC for every 1 kWh delivered.
# We'll assume the net effect is:
#    SOC(t+1) = SOC(t) + eta_charge * P_charge(t) - (1/eta_discharge)* P_discharge(t)

Pmax = 20.0  # maximum charge or discharge power (kW)
T = 10       # number of hours

# Electricity cost array (C_t) $/kWh
C = [20, 25, 15, 10, 30, 35, 40, 20, 15, 10]

# Revenue from discharging (R_t) $/kWh
R = [30, 35, 25, 20, 40, 45, 50, 30, 25, 20]

# -----------------------------
# 2. Define the MILP problem
# -----------------------------
model = pulp.LpProblem("BESS_Scheduling", sense=pulp.LpMinimize)

# -----------------------------
# 3. Decision Variables
# -----------------------------
# State of Charge at each hour t
SOC = [pulp.LpVariable(f"SOC_{t}", lowBound=0, upBound=SOC_max, cat=pulp.LpContinuous) 
       for t in range(T+1)]

# Charge and Discharge Power at each hour t
P_charge    = [pulp.LpVariable(f"P_charge_{t}", lowBound=0, upBound=Pmax, cat=pulp.LpContinuous)
               for t in range(T)]
P_discharge = [pulp.LpVariable(f"P_discharge_{t}", lowBound=0, upBound=Pmax, cat=pulp.LpContinuous)
               for t in range(T)]

# Binary variables to prevent simultaneous charge/discharge
# b_charge[t] = 1 if battery is charging in hour t, else 0
# b_discharge[t] = 1 if battery is discharging in hour t, else 0
b_charge    = [pulp.LpVariable(f"b_charge_{t}", cat=pulp.LpBinary) for t in range(T)]
b_discharge = [pulp.LpVariable(f"b_discharge_{t}", cat=pulp.LpBinary) for t in range(T)]

# -----------------------------
# 4. Constraints
# -----------------------------

# (a) Initial SOC
model += (SOC[0] == SOC_0), "Initial_SOC"

# (b) Battery capacity constraints
#     0 <= SOC(t) <= SOC_max already covered by variable bounds

# (c) Power limits and logic (no simultaneous charging/discharging)
for t in range(T):
    # No overlap in charging and discharging:
    model += b_charge[t] + b_discharge[t] <= 1, f"NoSimultaneous_{t}"
    
    # Link P_charge[t] to b_charge[t]
    model += P_charge[t] <= Pmax * b_charge[t],    f"ChargePowerLimit_{t}"
    
    # Link P_discharge[t] to b_discharge[t]
    model += P_discharge[t] <= Pmax * b_discharge[t], f"DischargePowerLimit_{t}"

# (d) SOC evolution constraint
#     SOC(t+1) = SOC(t) + η_charge * P_charge(t) - (1/η_discharge) * P_discharge(t)
for t in range(T):
    model += (
        SOC[t+1] 
        == SOC[t] 
           + eta_charge * P_charge[t] 
           - (1.0 / eta_discharge) * P_discharge[t]
    ), f"SOC_balance_{t}"

# -----------------------------
# 5. Objective Function
# -----------------------------
# Minimize sum of cost to buy energy minus revenue gained from discharging
# Z = ∑ [ C[t] * P_charge(t) - R[t] * P_discharge(t) ]
obj_expr = []
for t in range(T):
    cost   = C[t] * P_charge[t]
    revenue = R[t] * P_discharge[t]
    obj_expr.append(cost - revenue)

model += pulp.lpSum(obj_expr), "Total_Operational_Cost"

# -----------------------------
# 6. Solve
# -----------------------------
solution_status = model.solve(pulp.PULP_CBC_CMD(msg=False))

# -----------------------------
# 7. Report the results
# -----------------------------
print(f"Solver Status: {pulp.LpStatus[solution_status]}")
print(f"Optimal Objective Value = {pulp.value(model.objective):.2f} $")

for t in range(T):
    print(f"Hour {t}:")
    print(f"  P_charge     = {pulp.value(P_charge[t]):6.2f} kW")
    print(f"  P_discharge  = {pulp.value(P_discharge[t]):6.2f} kW")
    print(f"  b_charge     = {pulp.value(b_charge[t]):6.0f}")
    print(f"  b_discharge  = {pulp.value(b_discharge[t]):6.0f}")
    print(f"  SOC          = {pulp.value(SOC[t]):6.2f} kWh --> "
          f"{pulp.value(SOC[t+1]):6.2f} kWh next hour\n")


"""




