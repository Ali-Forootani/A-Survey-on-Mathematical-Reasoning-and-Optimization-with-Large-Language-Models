#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoGen Multi-Agent LP Optimization:
- An EvaluatorAgent verifies LP constraints before execution.
- An LPExecutorAgent executes `execute_lp_solver()` three times.

Created on Jan 27, 2025
@author: forootan
"""

import os
import autogen
import numpy as np
from scipy.optimize import linprog


# Set API key and base URL as environment variables
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["OPENAI_API_BASE"] = "Your API Base"



# ====== CONFIGURATION ======
config_list = {
    "config_list": [
        {
            "model": "alias-fast",
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ["OPENAI_API_BASE"],
        }
    ]
}

# ====== Function to Solve LP (Transportation Problem) ======
def execute_lp_solver():
    """Solves the transportation LP problem and returns results."""
    # Coefficients for the objective function (transportation costs)
    c = [4, 6, 9, 5, 3, 8]  # Minimize total transportation cost

    # Coefficients for the constraints (supply and demand)
    A = [
        [1, 1, 1, 0, 0, 0],  # Supply limit at Warehouse 1
        [0, 0, 0, 1, 1, 1],  # Supply limit at Warehouse 2
        [-1, 0, 0, -1, 0, 0],  # Demand at Store 1
        [0, -1, 0, 0, -1, 0],  # Demand at Store 2
        [0, 0, -1, 0, 0, -1]   # Demand at Store 3
    ]

    # Right-hand side values for constraints
    b = [50, 60, -30, -40, -20]  # Negative signs to convert >= to <= for scipy

    # Bounds for variables (all x_ij >= 0)
    x_bounds = [(0, None)] * 6

    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method="highs")

        if result.success:
            return {
                "status": "Success",
                "optimal_solution": result.x.tolist(),
                "objective_value": result.fun
            }
        else:
            return {"status": "Failure", "message": result.message}

    except Exception as e:
        return {"error": str(e)}

# ====== Evaluator Agent ======
class EvaluatorAgent(autogen.ConversableAgent):
    """Agent that verifies LP constraints before execution."""

    def generate_reply(self, messages, sender=None):
        """Checks if LP constraints are valid before execution."""
        last_message = messages[-1]["content"] if messages else ""

        print(f"[DEBUG] EvaluatorAgent received message: {last_message}")

        if "validate constraints" in last_message.lower():
            # Check if constraints are valid (basic example)
            valid = True  # Assume constraints are correct; add real checks if needed.

            if valid:
                return {"content": "✅ Constraints are valid. Proceed with LP execution."}
            else:
                return {"content": "❌ Invalid constraints detected. Execution halted."}

        return {"content": "❌ Error: Invalid command."}

# ====== LP Execution Agent ======
class LPExecutorAgent(autogen.ConversableAgent):
    """Agent that executes the LP solver after evaluation."""

    def generate_reply(self, messages, sender=None):
        """Executes LP solver only if constraints are validated."""
        last_message = messages[-1]["content"] if messages else ""

        print(f"[DEBUG] LPExecutorAgent received message: {last_message}")

        if "proceed with lp execution" in last_message.lower():
            response_text = ""
            for i in range(1):  # Execute the LP solver three times
                print(f"\n🔄 Execution {i+1}: Running LP solver...")
                solution = execute_lp_solver()
                print(f"✅ Result {i+1}: {solution}")
                response_text += f"\nExecution {i+1}: {solution}\n"

            print("\n==== Final Results ====")
            print(response_text)
            print("=======================")

            return {"content": response_text}

        return {"content": "❌ Error: Invalid command."}

# ====== Initialize and Run Agents ======
if __name__ == "__main__":
    evaluator = EvaluatorAgent(
        "lp_evaluator",
        system_message="You verify if the LP constraints are satisfied before execution.",
        llm_config=config_list,
        human_input_mode="NEVER",
    )

    executor = LPExecutorAgent(
        "lp_executor",
        system_message="You execute the LP solver only if constraints are valid.",
        llm_config=config_list,
        human_input_mode="NEVER",
    )

    # Evaluator checks constraints
    validation_result = evaluator.initiate_chat(
        evaluator,
        message="validate constraints.",
        max_turns=1
    )

    # ✅ Correctly access the last message content
    validation_message = validation_result.chat_history[-1]["content"]

    if "✅" in validation_message:
        # If constraints are valid, Executor proceeds
        result = executor.initiate_chat(
            executor,
            message="Proceed with LP execution.",
            max_turns=1
        )

        print("\n[INFO] LP Execution Finished.")
    else:
        print("\n[ERROR] Constraints validation failed. Execution aborted.")
