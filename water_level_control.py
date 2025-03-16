import os
from openai import OpenAI

# Set OpenAI API key and base URL
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "OPENAI_API_BASE"

api_key = "OPENAI_API_KEY"
api_base = "OPENAI_API_BASE"

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url=api_base)


# Define the water level control problem and fuzzy rules in the prompt
prompt = """
We aim to control the water levels in two interconnected tanks to achieve desired levels while minimizing a quadratic cost function. The system is represented as a second-order dynamical system using state-space representation.

### System Description:
The water level in Tank 1 and Tank 2 is governed by the following state-space equations:

1. **State-space representation**:
   - State variables:
     - \(x_1(t)\): Water level in Tank 1 (m)
     - \(x_2(t)\): Water level in Tank 2 (m)
   - Control input:
     - \(u(t)\): Pump flow rate (m³/s)
   - State equations:
     \[
     \begin{bmatrix}
     \dot{x}_1(t) \\
     \dot{x}_2(t)
     \end{bmatrix}
     =
     \begin{bmatrix}
     -a_1 & b_1 \\
     b_2 & -a_2
     \end{bmatrix}
     \begin{bmatrix}
     x_1(t) \\
     x_2(t)
     \end{bmatrix}
     +
     \begin{bmatrix}
     c_1 \\
     0
     \end{bmatrix}
     u(t)
     \]
     - \(y(t) = [x_1(t), x_2(t)]^\top\)

2. **System Parameters**:
   - \(a_1 = 0.1\), \(a_2 = 0.1\): Outflow coefficients for Tank 1 and Tank 2
   - \(b_1 = 0.05\), \(b_2 = 0.05\): Flow interaction between the tanks
   - \(c_1 = 0.1\): Pump-to-Tank 1 flow rate coefficient

3. **Control Objective**:
   - Achieve desired water levels \(x_1^{\text{desired}} = 2.0 \, \text{m}\) and \(x_2^{\text{desired}} = 1.5 \, \text{m}\).

### Quadratic Cost Function:
Minimize:
\[
J = \sum_{t=0}^{T} \left( Q_1 (x_1(t) - x_1^{\text{desired}})^2 + Q_2 (x_2(t) - x_2^{\text{desired}})^2 + R u(t)^2 \right)
\]
where:
   - \(Q_1 = 10\), \(Q_2 = 8\): State penalty weights
   - \(R = 2\): Control effort penalty

### Fuzzy Logic Rules:
1. If x1 > x1_desired, decrease the pump flow rate u(t) proportionally to the error (x1 - x1_desired).
2. If x2 < x2_desired, increase the pump flow rate u(t) proportionally to the error (x2_desired - x2).
3. If x1 and x2 are close to the desired levels, reduce the rate of change in u(t) to avoid oscillations.
4. Balance the control action between x1 and x2 deviations to prioritize the larger deviation.
5. Minimize the quadratic cost function:
   J = Q1 * (x1 - x1_desired)^2 + Q2 * (x2 - x2_desired)^2 + R * u(t)^2.
6. Ensure the control input u(t) stays within the range [-u_max, u_max].
7. Provide the control input u(t) for this time step. Ensure u(t) is within [-u_max, u_max].
8. Avoid rapid changes in \(u(t)\) to prevent oscillations.

### Requirements:
1. Use fuzzy logic rules to guide the control signal \(u(t)\) dynamically.
2. Optimize the system using the quadratic cost function \(J\).
3. Provide the following:
   - A step-by-step explanation of how fuzzy logic influenced the control decision at each time step.
   - The state trajectory \([x_1(t), x_2(t)]\) and control input \(u(t)\) over a simulation period of 10 seconds.
   - The total minimized cost \(J\).

### Expected Output:
1. An explanation of how fuzzy logic rules were applied in each time step.
2. The state trajectory and control input over time.
3. The total minimized cost of the system.
"""



# Call the OpenAI API to generate a response
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled in optimization, control systems, and fuzzy logic."},
        {"role": "user", "content": prompt},
    ]
)

# Print the response from the API
print(response.choices[0].message.content)