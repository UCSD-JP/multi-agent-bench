"""
InstrumentedModelClient — wraps OpenAIStreamingClient as an AutoGen ChatCompletionClient.

Internally uses streaming to capture TTFT/TPOT, then returns a non-streaming
CreateResult to AutoGen's agent framework.
"""

import asyncio
import time
from typing import (
    Any,
    AsyncGenerator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import httpx

from benchmark.core.streaming_client import OpenAIStreamingClient
from benchmark.core.types import LLMCallMetrics

try:
    from autogen_core import CancellationToken
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        LLMMessage,
        ModelCapabilities,
        ModelInfo,
        RequestUsage,
    )
    from autogen_core.tools import Tool, ToolSchema

    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


def _llm_messages_to_openai(messages: Sequence[Any]) -> list[dict[str, str]]:
    """Convert AutoGen LLMMessage sequence to OpenAI-format dicts."""
    result = []
    for msg in messages:
        # AutoGen messages have .content and .type attributes
        msg_type = getattr(msg, "type", None)
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # Multi-part content — join text parts
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif hasattr(part, "text"):
                    parts.append(part.text)
            content = "\n".join(parts)

        if msg_type == "SystemMessage":
            result.append({"role": "system", "content": str(content)})
        elif msg_type == "UserMessage":
            result.append({"role": "user", "content": str(content)})
        elif msg_type == "AssistantMessage":
            result.append({"role": "assistant", "content": str(content)})
        else:
            # Fallback: use role heuristic
            result.append({"role": "user", "content": str(content)})
    return result


if _AUTOGEN_AVAILABLE:

    class InstrumentedModelClient(ChatCompletionClient):
        """
        AutoGen ChatCompletionClient that wraps our OpenAIStreamingClient.

        - Internally uses streaming to capture TTFT/TPOT per call.
        - Returns CreateResult to AutoGen (non-streaming interface).
        - Accumulates LLMCallMetrics for later retrieval via pop_metrics().
        """

        component_type = "model"
        component_config_schema = None  # not config-loadable
        component_provider_override = None

        def __init__(
            self,
            streaming_client: OpenAIStreamingClient,
            http_client: httpx.AsyncClient,
            model: str,
            semaphore: asyncio.Semaphore,
            temperature: float = 0.2,
            max_model_len: int = 4096,
        ):
            self._streaming_client = streaming_client
            self._http_client = http_client
            self._model = model
            self._semaphore = semaphore
            self._temperature = temperature
            self._max_model_len = max_model_len
            self._call_records: List[LLMCallMetrics] = []
            self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
            self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
            tool_choice: Any = "auto",
            json_output: Optional[Any] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[CancellationToken] = None,
        ) -> CreateResult:
            openai_messages = _llm_messages_to_openai(messages)

            async with self._semaphore:
                metrics = await self._streaming_client.chat_completions_stream(
                    client=self._http_client,
                    model=self._model,
                    messages=openai_messages,
                    temperature=self._temperature,
                    max_model_len=self._max_model_len,
                )

            self._call_records.append(metrics)

            # Update usage tracking
            prompt_tok = int(metrics.prompt_tokens or 0)
            completion_tok = int(metrics.out_tokens or 0)
            self._total_usage = RequestUsage(
                prompt_tokens=self._total_usage.prompt_tokens + prompt_tok,
                completion_tokens=self._total_usage.completion_tokens + completion_tok,
            )
            self._actual_usage = RequestUsage(
                prompt_tokens=self._actual_usage.prompt_tokens + prompt_tok,
                completion_tokens=self._actual_usage.completion_tokens + completion_tok,
            )

            from autogen_core.models import AssistantMessage

            usage = RequestUsage(
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
            )

            return CreateResult(
                finish_reason="stop" if metrics.ok else "unknown",
                content=metrics.out_text or "",
                usage=usage,
                cached=False,
            )

        async def create_stream(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
            tool_choice: Any = "auto",
            json_output: Optional[Any] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[CancellationToken] = None,
        ) -> AsyncGenerator[Union[str, CreateResult], None]:
            # AutoGen calls create_stream for streaming; we delegate to create()
            result = await self.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            yield result.content
            yield result

        async def close(self) -> None:
            pass

        @property
        def actual_usage(self) -> RequestUsage:
            return self._actual_usage

        @property
        def total_usage(self) -> RequestUsage:
            return self._total_usage

        def count_tokens(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
        ) -> int:
            # Rough estimate: ~4 chars per token
            openai_msgs = _llm_messages_to_openai(messages)
            total_chars = sum(len(m.get("content", "")) for m in openai_msgs)
            return max(1, total_chars // 4)

        def remaining_tokens(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
        ) -> int:
            return self._max_model_len - self.count_tokens(messages, tools=tools)

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(
                vision=False,
                function_calling=False,
                json_output=False,
            )

        @property
        def model_info(self) -> ModelInfo:
            return ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family="custom",
            )

        def pop_metrics(self) -> List[LLMCallMetrics]:
            """Retrieve and clear accumulated call metrics."""
            records = list(self._call_records)
            self._call_records.clear()
            return records
