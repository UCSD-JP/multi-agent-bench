"""WorkflowRunner ABC and RunContext — framework-agnostic interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx

from benchmark.core.streaming_client import OpenAIStreamingClient
from benchmark.core.types import TaskRecord


@dataclass
class RunContext:
    """Shared context passed to every runner."""
    model: str
    base_url: str
    api_key: str
    http_client: httpx.AsyncClient
    llm_semaphore: asyncio.Semaphore
    streaming_client: OpenAIStreamingClient
    executors: int = 2
    temperature: float = 0.2
    max_model_len: int = 4096


class WorkflowRunner(ABC):
    """Abstract base for all framework runners (raw, autogen, ...)."""

    @abstractmethod
    async def run_task(
        self,
        task_id: int,
        prompt: str,
        context: RunContext,
    ) -> TaskRecord:
        """Execute a single multi-agent task and return a TaskRecord."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this runner (e.g. 'raw', 'autogen')."""
        ...
