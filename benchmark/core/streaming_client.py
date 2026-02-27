"""OpenAI-compatible async streaming client with TTFT/TPOT measurement."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

import httpx

from .types import LLMCallMetrics


class OpenAIStreamingClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_s)

    @staticmethod
    def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
        """Conservative token estimate: ~2 chars per token.

        Using 2 (not 4) because chat templates, special tokens, and
        token-dense content (code, CJK) make real counts much higher
        than naive char/4.  Over-estimating prompt tokens is safe —
        it just shortens max_tokens, which is fine for benchmarking.
        """
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        return max(1, total_chars // 2)

    async def chat_completions_stream(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ) -> LLMCallMetrics:
        """
        Uses /chat/completions streaming to estimate:
        - TTFT: first streamed token delta
        - TPOT: average inter-token time (from 1st token to end / (tokens-1))
        Also tries to read usage if server provides it via stream_options.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Clamp max_tokens so prompt + completion fits within model context
        if max_model_len is not None:
            prompt_est = self.estimate_prompt_tokens(messages)
            margin = 512  # safety margin for chat template + special tokens + tokenizer variance
            remaining = max_model_len - prompt_est - margin
            if remaining <= 0:
                # Prompt already exceeds context — will get 400, but let server
                # return the real error rather than silently dropping
                remaining = 16
            if max_tokens is None:
                max_tokens = remaining
            else:
                max_tokens = min(max_tokens, remaining)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        start_ts = time.time()
        start_ns = time.monotonic_ns()
        first_token_ts: Optional[float] = None
        first_token_ns: Optional[int] = None
        end_ts: float = start_ts
        end_ns: int = start_ns
        token_timestamps: List[float] = []
        out_chunks: List[str] = []

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        try:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                    else:
                        continue

                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices", [])
                    delta = (
                        choices[0].get("delta", {}).get("content", None)
                        if choices
                        else None
                    )
                    if delta:
                        now = time.time()
                        now_ns = time.monotonic_ns()
                        if first_token_ts is None:
                            first_token_ts = now
                            first_token_ns = now_ns
                        token_timestamps.append(now)
                        out_chunks.append(delta)

                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        total_tokens = usage.get("total_tokens", total_tokens)

            end_ts = time.time()
            end_ns = time.monotonic_ns()
            out_text = "".join(out_chunks)

            if completion_tokens is None:
                completion_tokens = max(1, len(out_text) // 4) if out_text else 0
            if prompt_tokens is None:
                joined = " ".join(m.get("content", "") for m in messages if "content" in m)
                prompt_tokens = max(1, len(joined) // 4) if joined else 0
            if total_tokens is None:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            ttft_ms = (first_token_ts - start_ts) * 1000 if first_token_ts else None

            tpot_ms = None
            if first_token_ts and completion_tokens and completion_tokens > 1:
                tpot_ms = ((end_ts - first_token_ts) * 1000) / (completion_tokens - 1)

            return LLMCallMetrics(
                ok=True,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                out_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                out_text=out_text,
                error=None,
                start_ns=start_ns,
                first_token_ns=first_token_ns,
                end_ns=end_ns,
            )
        except httpx.HTTPStatusError as e:
            end_ts = time.time()
            end_ns = time.monotonic_ns()
            partial_text = "".join(out_chunks)
            body = ""
            try:
                body = e.response.text
            except Exception:
                pass
            err_msg = f"{e} | body={body}" if body else str(e)
            log.warning("HTTP %s: %s", e.response.status_code, err_msg)
            return LLMCallMetrics(
                ok=False,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                ttft_ms=(first_token_ts - start_ts) * 1000 if first_token_ts else None,
                tpot_ms=None,
                out_tokens=None,
                prompt_tokens=None,
                total_tokens=None,
                out_text=partial_text,
                error=err_msg,
                start_ns=start_ns,
                first_token_ns=first_token_ns,
                end_ns=end_ns,
            )
        except Exception as e:
            end_ts = time.time()
            end_ns = time.monotonic_ns()
            partial_text = "".join(out_chunks)
            log.warning("LLM call error: %s", e)
            return LLMCallMetrics(
                ok=False,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                ttft_ms=(first_token_ts - start_ts) * 1000 if first_token_ts else None,
                tpot_ms=None,
                out_tokens=None,
                prompt_tokens=None,
                total_tokens=None,
                out_text=partial_text,
                error=str(e),
                start_ns=start_ns,
                first_token_ns=first_token_ns,
                end_ns=end_ns,
            )
