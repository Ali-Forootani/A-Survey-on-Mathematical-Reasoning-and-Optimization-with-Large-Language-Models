import os
import numpy as np
from openai import OpenAI


# Set API key and base URL as environment variables
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["OPENAI_API_BASE"] = "Your API Base"


api_key = "Your API Key"
api_base = "Your API Base"

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url=api_base)

# System parameters
A = np.array([[-0.1, 0.05],
              [0.05, -0.1]])
B = np.array([[0.1],
              [0]])
x_desired = np.array([2.0, 1.5])  # Desired water levels
Q = np.array([10, 8])  # State penalty weights
R = 2  # Control penalty weight
dt = 1.0  # Time step
T = 10  # Total simulation time (seconds)
u_max = 0.5  # Maximum pump flow rate

# Initial state
x = np.array([1.0, 0.5])  # Initial water levels
u = 0  # Initial control input

# Log data
trajectory = []  # Store (time, x1, x2, u)

# Feedback loop simulation
for t in range(T):
    # Prompt the LLM for fuzzy logic-based control decision
    prompt = f"""
    The current time step is {t}. The system state is:
    - Tank 1 water level: x1(t) = {x[0]:.2f}
    - Tank 2 water level: x2(t) = {x[1]:.2f}
    
    The desired water levels are:
    - Tank 1: x1_desired = {x_desired[0]:.2f}
    - Tank 2: x2_desired = {x_desired[1]:.2f}
    
    System parameters:
    - State transition matrix A:
      {A}
    - Control input matrix B:
      {B}
    - Quadratic cost function weights:
      Q1 = {Q[0]}, Q2 = {Q[1]}, R = {R}
    - Maximum pump flow rate: u_max = {u_max}

    Based on fuzzy logic rules:
    1. If x1 > x1_desired, reduce the pump flow rate u(t).
    2. If x2 < x2_desired, increase the pump flow rate u(t).
    3. Avoid rapid changes in u(t).
    4. Minimize the quadratic cost function:
       J = Q1 * (x1 - x1_desired)^2 + Q2 * (x2 - x2_desired)^2 + R * u(t)^2.

    Provide the control input u(t) for this time step. Ensure u(t) is within [-u_max, u_max].
    """

    # Call the OpenAI API to get the control input
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in control systems and fuzzy logic."},
            {"role": "user", "content": prompt},
        ]
    )
    
    # Parse the response to extract the control input
    u_response = response.choices[0].message.content.strip()
    try:
        # Extract the numerical value of the control input
        u = float(u_response.split("u(t) =")[1].split()[0])
        u = np.clip(u, -u_max, u_max)  # Ensure u is within bounds
    except (IndexError, ValueError):
        print("Error parsing LLM response, using u(t) = 0 as fallback.")
        u = 0

    # Update the state using the state-space equations
    x_dot = A @ x + B.flatten() * u
    x = x + x_dot * dt

    # Log the results
    trajectory.append((t, x[0], x[1], u))
    print(f"Time {t}: x1 = {x[0]:.2f}, x2 = {x[1]:.2f}, u = {u:.2f}")

# Print the trajectory
print("\nSimulation Results:")
print("Time\t x1\t x2\t u")
for step in trajectory:
    print(f"{step[0]:.2f}\t {step[1]:.2f}\t {step[2]:.2f}\t {step[3]:.2f}")
