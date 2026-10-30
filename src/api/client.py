"""Unified chat-completions client for the two supported API protocols.

- openai:     any OpenAI-compatible /chat/completions endpoint (OpenAI,
              DashScope, vLLM, ...). Sends enable_thinking=false when
              disable_thinking is set; if a provider rejects the parameter
              (HTTP 400) it is dropped after a one-time probe, same for
              temperature.
- anthropic:  the official Anthropic Messages API. There is no thinking
              toggle on this protocol, so disable_thinking is a no-op.

Both accept the same neutral message parts:

    {"type": "text",  "text": "..."}
    {"type": "image", "data": bytes, "media_type": "image/png"}
    {"type": "image", "url": "https://..."}

Callers may pass an OpenAI-style json_schema ``response_format`` for
structured output; on the anthropic protocol it is carried as a forced
tool call and the tool input is returned as a JSON string.

``extra_payload`` merges provider-specific keys (for example
``{"reasoning_effort": "none"}``) into the request's ``extra_body`` on the
openai protocol.  The anthropic protocol has no equivalent and rejects it.
"""

from __future__ import annotations

import base64
import json
import sys
from typing import Any

from src.image.encoding import to_data_url


class ImageTooLargeError(RuntimeError):
    """The provider rejected the request because an image payload is too large."""


class InvalidResponseError(RuntimeError):
    """The provider returned a structurally invalid response (no choices,
    abnormal finish/stop reason, or non-string content)."""


def is_retryable_api_error(exc: Exception) -> bool:
    """Frozen retry table: HTTP 408/429/5xx and transport errors retry;
    everything else is terminal. Both SDKs expose .status_code on APIStatusError
    and use the same connection-error class names, so duck-typing covers both."""
    if isinstance(exc, (ImageTooLargeError, InvalidResponseError)):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status in (408, 429) or 500 <= status < 600
    return (type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
            or isinstance(exc, (TimeoutError, OSError)))


def _is_oversize_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    return (
        "exceeds" in text
        or "too large" in lowered
        or "invalid base64" in lowered
        or "validation" in lowered
    )


def _to_openai_content(parts: list[dict]) -> list[dict]:
    content = []
    for part in parts:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif "url" in part:
            content.append({"type": "image_url", "image_url": {"url": part["url"]}})
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": to_data_url(part["data"], part["media_type"])},
            })
    return content


def _to_anthropic_content(parts: list[dict]) -> list[dict]:
    blocks = []
    for part in parts:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif "url" in part:
            blocks.append({"type": "image", "source": {"type": "url", "url": part["url"]}})
        else:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part["media_type"],
                    "data": base64.b64encode(part["data"]).decode("ascii"),
                },
            })
    return blocks


class OpenAICompatClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 600.0):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._thinking_supported: bool | None = None  # None = not probed yet
        self._temperature_supported: bool | None = None

    async def complete(self, *, model: str, parts: list[dict],
                       temperature: float, max_tokens: int | None,
                       disable_thinking: bool,
                       response_format: dict | None = None,
                       require_stop: bool = False,
                       extra_payload: dict | None = None) -> str:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": _to_openai_content(parts)}],
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._temperature_supported is False:
            kwargs.pop("temperature")
        extra_body = dict(extra_payload or {})
        if disable_thinking and self._thinking_supported is not False:
            extra_body["enable_thinking"] = False
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if (disable_thinking and self._thinking_supported is not False
                    and "enable_thinking" in str(exc)):
                self._thinking_supported = False
                print("[WARN] provider rejects enable_thinking; dropping it "
                      "(probed once per process)", file=sys.stderr)
                extra_body.pop("enable_thinking", None)
                if extra_body:
                    kwargs["extra_body"] = extra_body
                else:
                    kwargs.pop("extra_body", None)
                response = await self._client.chat.completions.create(**kwargs)
            elif self._temperature_supported is not False and "temperature" in str(exc):
                self._temperature_supported = False
                print("[WARN] provider rejects temperature; dropping it "
                      "(probed once per process)", file=sys.stderr)
                kwargs.pop("temperature", None)
                response = await self._client.chat.completions.create(**kwargs)
            elif _is_oversize_error(exc):
                raise ImageTooLargeError(str(exc)) from exc
            else:
                raise
        if require_stop:
            choices = getattr(response, "choices", None)
            if not choices:
                raise InvalidResponseError("API response lacks choices")
            choice = choices[0]
            if getattr(choice, "finish_reason", None) != "stop":
                raise InvalidResponseError("API response did not finish normally")
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                raise InvalidResponseError("API response lacks string message.content")
            return content
        return response.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, *, base_url: str | None = None, api_key: str,
                 timeout: float = 600.0):
        import anthropic

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(self, *, model: str, parts: list[dict],
                       temperature: float, max_tokens: int | None,
                       disable_thinking: bool,
                       response_format: dict | None = None,
                       require_stop: bool = False,
                       extra_payload: dict | None = None) -> str:
        if extra_payload:
            raise RuntimeError(
                "the anthropic protocol does not support extra_payload"
            )
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": _to_anthropic_content(parts)}],
        )
        tool_name = None
        if response_format is not None:
            js = response_format["json_schema"]
            tool_name = js["name"]
            kwargs["tools"] = [{"name": tool_name, "input_schema": js["schema"]}]
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}
        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            if _is_oversize_error(exc):
                raise ImageTooLargeError(str(exc)) from exc
            raise
        if require_stop and getattr(response, "stop_reason", None) not in (
                "end_turn", "tool_use"):
            raise InvalidResponseError(
                f"API response stopped abnormally: "
                f"{getattr(response, 'stop_reason', None)}")
        if tool_name is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    return json.dumps(block.input, ensure_ascii=False)
            return ""
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )


def make_client(protocol: str, *, base_url: str | None, api_key: str,
                timeout: float = 600.0):
    if protocol == "openai":
        if not base_url:
            raise ValueError("protocol=openai requires a base_url in the config")
        return OpenAICompatClient(base_url=base_url, api_key=api_key, timeout=timeout)
    if protocol == "anthropic":
        return AnthropicClient(base_url=base_url or None, api_key=api_key,
                               timeout=timeout)
    raise ValueError(f"unsupported protocol: {protocol!r} (expected openai|anthropic)")
