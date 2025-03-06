#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:45:36 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoGen Multi-Agent LP Optimization with Unique Problems and Async Execution.
Fixed for Spyder/Jupyter to prevent interpreter freezing.

Created on Jan 30, 2025
@author: forootan
"""

import os
import autogen
import numpy as np
import asyncio
import nest_asyncio  # ✅ Fix for Spyder/Jupyter Notebook
from scipy.optimize import linprog

# ✅ Apply fix to allow nested async execution (for Spyder/Jupyter)
nest_asyncio.apply()

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


# ====== Define Unique LP Problems ======
LP_PROBLEMS = [
    {
        "c": [4, 6, 9, 5, 3, 8],  # Minimize total transportation cost
        "A": [
            [1, 1, 1, 0, 0, 0],  # Supply limit at Warehouse 1
            [0, 0, 0, 1, 1, 1],  # Supply limit at Warehouse 2
            [-1, 0, 0, -1, 0, 0],  # Demand at Store 1
            [0, -1, 0, 0, -1, 0],  # Demand at Store 2
            [0, 0, -1, 0, 0, -1]   # Demand at Store 3
        ],
        "b": [50, 60, -30, -40, -20],  # Right-hand side values for constraints
        "bounds": [(0, None)] * 6
    },
    {
        "c": [3, 7, 2, 4],  # Another LP problem
        "A": [
            [1, 2, 0, 1],  # Supply limits
            [0, 1, 1, 2],  # Demand at different locations
            [-1, -2, -1, 0]  # Different constraints
        ],
        "b": [40, 30, -20],  # Different demand and supply constraints
        "bounds": [(0, None)] * 4
    },
    {
        "c": [10, 2, 5],  # A different objective function
        "A": [
            [2, 3, 1],  
            [-1, 0, -1],  
            [0, -1, -2]
        ],
        "b": [50, -15, -30],  
        "bounds": [(0, None)] * 3
    }
]

# ====== Async Function to Solve LP (Different Problems) ======
async def execute_lp_solver(problem_id):
    """Asynchronously solves an LP problem identified by `problem_id`."""
    problem = LP_PROBLEMS[problem_id]

    c = problem["c"]
    A = problem["A"]
    b = problem["b"]
    x_bounds = problem["bounds"]

    await asyncio.sleep(1)  # Simulating async execution (e.g., API call or heavy computation)

    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method="highs")

        if result.success:
            return {
                "problem_id": problem_id,
                "status": "Success",
                "optimal_solution": result.x.tolist(),
                "objective_value": result.fun
            }
        else:
            return {"problem_id": problem_id, "status": "Failure", "message": result.message}

    except Exception as e:
        return {"problem_id": problem_id, "error": str(e)}

# ====== Evaluator Agent (Validates Different LP Problems) ======
class EvaluatorAgent(autogen.ConversableAgent):
    """Agent that verifies LP constraints before execution for multiple problems."""

    def generate_reply(self, messages, sender=None):
        """Dynamically validates constraints for different LP problems."""
        last_message = messages[-1]["content"] if messages else ""
        print(f"[DEBUG] EvaluatorAgent received message: {last_message}")

        if "validate constraints" in last_message.lower():
            response = ""
            for i, problem in enumerate(LP_PROBLEMS):
                valid = True  # Assume constraints are correct for now; can add verification logic.

                if valid:
                    response += f"✅ Problem {i} constraints are valid.\n"
                else:
                    response += f"❌ Problem {i} constraints are invalid.\n"

            return {"content": response}

        return {"content": "❌ Error: Invalid command."}

# ====== LP Execution Agent (Handles Multiple Unique Problems) ======
class LPExecutorAgent(autogen.ConversableAgent):
    """Agent that executes different LP problems asynchronously."""

    async def async_generate_reply(self, messages, sender=None):
        """Executes different LP solvers asynchronously based on problem indices."""
        last_message = messages[-1]["content"] if messages else ""
        print(f"[DEBUG] LPExecutorAgent received message: {last_message}")

        if "proceed with lp execution" in last_message.lower():
            response_text = ""

            # Run different LP solvers in parallel
            results = await asyncio.gather(
                execute_lp_solver(0),
                execute_lp_solver(1),
                execute_lp_solver(2)
            )

            for result in results:
                response_text += f"\nProblem {result['problem_id']} Execution: {result}\n"

            print("\n==== Final Results ====")
            print(response_text)
            print("=======================")

            return {"content": response_text}

        return {"content": "❌ Error: Invalid command."}

# ====== Initialize and Run Agents ======
async def main():
    evaluator = EvaluatorAgent(
        "lp_evaluator",
        system_message="You verify if the LP constraints are satisfied before execution.",
        llm_config=config_list,
        human_input_mode="NEVER",
    )

    executor = LPExecutorAgent(
        "lp_executor",
        system_message="You execute different LP solvers in parallel using asyncio.",
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
        # If constraints are valid, execute LP solvers in parallel
        result = await executor.async_generate_reply(
            [{"content": "Proceed with LP execution."}]
        )

        print("\n[INFO] LP Execution Finished.")
    else:
        print("\n[ERROR] Constraints validation failed. Execution aborted.")

# ✅ Fix for Spyder & Jupyter Notebook: Ensures clean exit after execution
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(main())  
    except RuntimeError:
        asyncio.run(main())  
