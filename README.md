# Using Large Language Models (LLMs) in Control and Optimization 

This repository contains a collection of Python scripts designed to solve Linear Programming (LP) optimization problems in a multi-agent environment using the AutoGen framework. It also includes an asynchronous execution framework to run LP solvers concurrently.
These codes are part of the paper entitled `A Survey on Mathematical Reasoning and Optimization with Large Language Models' by Ali Forootani.


### Key Components:
1. **AutoGen Framework**: Utilizes the AutoGen conversational agents to model the LP optimization task in multiple stages.
2. **LP Optimization**: Solves transportation-type LP problems using the `scipy.optimize.linprog` function.
3. **Asynchronous Execution**: Utilizes the `asyncio` library to run multiple LP solvers concurrently, which improves efficiency when executing large-scale optimizations.
4. **Fuzzy Logic Integration**: Incorporates fuzzy logic rules for decision-making in systems like Battery Energy Storage Systems (BESS), and water level control systems.
5. **Energy Forecasting**: Uses OpenAI's GPT-based models to forecast energy consumption based on historical data.

---

## Requirements
1. **Python 3.x**
2. **Required Libraries**:
   - `autogen`: For conversational agent framework.
   - `numpy`: For mathematical operations.
   - `scipy`: For optimization problems.
   - `asyncio`: For asynchronous execution.
   - `openai`: For interfacing with OpenAI's models.
   - `matplotlib`: For plotting data.
   - `pulp`: For MILP optimization (if required).
   - `nest_asyncio`: For compatibility with Jupyter/Spyder.

   Install dependencies using:

   ```bash
   pip install autogen numpy scipy asyncio openai matplotlib pulp nest_asyncio
   ```

## Structure

### 1. **`auto_gen_lp_optimization.py`**:
   This script demonstrates how to solve an LP optimization problem (such as a transportation problem) with AutoGen agents. It involves:
   - An **Evaluator Agent** that validates the LP constraints.
   - An **LP Executor Agent** that runs the LP solver only if the constraints are valid.
   
   The agents communicate with each other, and the solution is optimized through multiple executions.

### 2. **`async_lp_solver.py`**:
   This script extends the LP solver by using asynchronous execution to solve multiple instances of the LP optimization concurrently. It utilizes `asyncio` and `nest_asyncio` to ensure compatibility in environments like Jupyter or Spyder. The script uses the same LP optimization problem as the main script, but with asynchronous execution for improved performance in large-scale problems.

### 3. **`energy_optimization_with_fuzzy_logic.py`**:
   This script demonstrates how fuzzy logic can be applied to optimize a Battery Energy Storage System (BESS). The goal is to minimize operational costs over a 10-hour period by determining the charge/discharge schedule. Fuzzy logic rules guide the optimization by considering electricity prices, battery state of charge (SOC), and other parameters.

### 4. **`water_level_control.py`**:
   This script integrates fuzzy logic with a water level control system. The goal is to regulate two water tanks based on desired water levels using control inputs. The system utilizes the OpenAI GPT model for decision-making and adjusts control strategies dynamically.

### 5. **`energy_demand_forecasting.py`**:
   This script queries OpenAI's model to forecast energy demand based on historical energy consumption data. It generates forecasts and optimization strategies for the next hour and provides recommendations for optimizing energy consumption.

### 6. **`wind_speed_analysis.py`**:
   This script processes wind speed and pressure data for Germany using various functions to extract, interpolate, and scale wind data. It provides extended statistical features for wind speeds and pressure, and uses them to suggest optimal wind turbine placement and energy forecasting strategies.

---

## How to Run the Scripts

### 1. **Set API Key and Base URL**:
   Before running the scripts, ensure you set your OpenAI API key and base URL as environment variables. For example:

   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
   ```

### 2. **Running the Scripts**:
   - For **LP Optimization with AutoGen**:
     Simply run the script:
     ```bash
     python auto_gen_lp_optimization.py
     ```

   - For **Async LP Solver**:
     Run the script using:
     ```bash
     python async_lp_solver.py
     ```

   - For **Energy Optimization with Fuzzy Logic**:
     Execute the script for BESS optimization:
     ```bash
     python energy_optimization_with_fuzzy_logic.py
     ```

   - For **Water Level Control**:
     Run:
     ```bash
     python water_level_control.py
     ```

   - For **Energy Demand Forecasting**:
     Execute the script:
     ```bash
     python energy_demand_forecasting.py
     ```

   - For **Wind Speed Analysis**:
     Run the script:
     ```bash
     python wind_speed_analysis.py
     ```

### Example Outputs:
- **LP Optimization**: Optimized transportation costs based on supply and demand constraints.
- **Async LP Solver**: Multiple LP solvers executed concurrently for large-scale optimization problems.
- **Energy Optimization with Fuzzy Logic**: Optimal charge/discharge schedules for BESS with dynamic decision-making based on fuzzy logic.
- **Water Level Control**: Control input suggestions for adjusting water levels in two tanks based on fuzzy logic.
- **Energy Demand Forecasting**: Forecast of energy demand and actionable optimization strategies.
- **Wind Speed Analysis**: Statistical summaries and recommendations for optimal wind turbine placement.

---

## Conclusion

This repository provides a flexible and scalable approach to solving optimization problems using AutoGen agents, asynchronous execution, and fuzzy logic integration. It can be extended for more complex real-world optimization tasks, including energy systems, control systems, and forecasting models.

## Contact

Please do not hesitate to contact us via aliforootani@ieee.org/ali.forootani@ufz.de
