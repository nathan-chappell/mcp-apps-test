from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

OPENAI_LOGS_BASE_URL = "https://platform.openai.com/logs"


@dataclass(frozen=True, slots=True)
class OpenAITraceRefs:
    response_id: str | None = None
    response_log_url: str | None = None
    conversation_id: str | None = None
    conversation_log_url: str | None = None

    @property
    def has_any(self) -> bool:
        return any(
            value is not None
            for value in (
                self.response_id,
                self.response_log_url,
                self.conversation_id,
                self.conversation_log_url,
            )
        )


_ACTIVE_OPENAI_TOOL_TRACE_REFS: ContextVar[OpenAITraceRefs | None] = ContextVar(
    "active_openai_tool_trace_refs",
    default=None,
)


def build_openai_log_url(openai_id: str | None) -> str | None:
    normalized_id = _normalize_openai_id(openai_id)
    if normalized_id is None:
        return None
    return f"{OPENAI_LOGS_BASE_URL}/{normalized_id}"


def build_openai_trace_refs(
    *,
    response_id: str | None = None,
    conversation_id: str | None = None,
) -> OpenAITraceRefs:
    normalized_response_id = _normalize_openai_id(response_id, prefix="resp_")
    normalized_conversation_id = _normalize_openai_id(conversation_id, prefix="conv_")
    return OpenAITraceRefs(
        response_id=normalized_response_id,
        response_log_url=build_openai_log_url(normalized_response_id),
        conversation_id=normalized_conversation_id,
        conversation_log_url=build_openai_log_url(normalized_conversation_id),
    )


def extract_openai_trace_refs(response: object) -> OpenAITraceRefs:
    return build_openai_trace_refs(
        response_id=_normalize_openai_id(_attribute(response, "id"), prefix="resp_"),
        conversation_id=_extract_conversation_id(response),
    )


def latest_openai_trace_refs(responses: list[object]) -> OpenAITraceRefs:
    latest_response_id: str | None = None
    latest_conversation_id: str | None = None
    for response in responses:
        trace_refs = extract_openai_trace_refs(response)
        if trace_refs.response_id is not None:
            latest_response_id = trace_refs.response_id
        if trace_refs.conversation_id is not None:
            latest_conversation_id = trace_refs.conversation_id
    return build_openai_trace_refs(
        response_id=latest_response_id,
        conversation_id=latest_conversation_id,
    )


def get_active_openai_tool_trace_refs() -> OpenAITraceRefs | None:
    return _ACTIVE_OPENAI_TOOL_TRACE_REFS.get()


def set_active_openai_tool_trace_refs(trace_refs: OpenAITraceRefs) -> None:
    _ACTIVE_OPENAI_TOOL_TRACE_REFS.set(trace_refs)


def clear_active_openai_tool_trace_refs() -> None:
    _ACTIVE_OPENAI_TOOL_TRACE_REFS.set(None)


def _attribute(value: object, name: str) -> Any | None:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_conversation_id(response: object) -> str | None:
    conversation = _attribute(response, "conversation")
    if isinstance(conversation, str):
        return _normalize_openai_id(conversation, prefix="conv_")
    if isinstance(conversation, dict):
        return _normalize_openai_id(conversation.get("id"), prefix="conv_")
    return _normalize_openai_id(getattr(conversation, "id", None), prefix="conv_")


def _normalize_openai_id(value: object, *, prefix: str | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if prefix is not None and not normalized.startswith(prefix):
        return None
    return normalized
