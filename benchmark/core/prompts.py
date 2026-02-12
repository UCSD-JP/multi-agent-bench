"""Shared prompt templates for multi-agent workflows."""


class PromptBuilder:
    SYSTEM_PROMPT = "You are a helpful assistant."

    @staticmethod
    def planner(user_prompt: str) -> str:
        return (
            "You are the Planner agent. Produce a short plan with numbered steps.\n\n"
            f"User request:\n{user_prompt}\n"
        )

    @staticmethod
    def executor(user_prompt: str, plan: str, step_id: str) -> str:
        return (
            "You are an Executor agent. Execute one part of the plan. Be concise but correct.\n\n"
            f"Plan:\n{plan}\n\n"
            f"User request:\n{user_prompt}\n\n"
            f"Executor focus: {step_id}\n"
        )

    @staticmethod
    def aggregator(user_prompt: str, plan: str, exec_outputs: dict) -> str:
        execs = "\n\n".join([f"{k}: {v}" for k, v in exec_outputs.items()])
        return (
            "You are the Aggregator agent. Combine planner + executors into one final answer.\n\n"
            f"User request:\n{user_prompt}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Executor outputs:\n{execs}\n\n"
            "Return the final response."
        )
