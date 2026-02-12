"""ShareGPT / ShareGPT4V dataset loading (text prompts only)."""

import json
from typing import List


def load_sharegpt_dataset(dataset_path: str) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    for item in data:
        if "conversations" in item:
            for msg in item["conversations"]:
                if msg.get("from") == "human" or msg.get("role") == "user":
                    prompts.append(msg.get("value", "") or "")
                    break
        elif "messages" in item:
            for msg in item["messages"]:
                if msg.get("role") == "user":
                    prompts.append(msg.get("content", "") or "")
                    break

    prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    return prompts


def get_prompt_from_dataset(dataset: List[str], task_id: int) -> str:
    if not dataset:
        raise ValueError("Dataset is empty. ShareGPT4V prompts are required.")
    return dataset[task_id % len(dataset)]
