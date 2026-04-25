#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.openai_tracing import build_openai_trace_refs, extract_openai_trace_refs  # noqa: E402


class ToolCallSummary(BaseModel):
    type: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    status: str | None = None


class ResponseSummary(BaseModel):
    id: str | None = None
    log_url: str | None = None
    conversation_id: str | None = None
    conversation_log_url: str | None = None
    status: str | None = None
    model: str | None = None
    created_at: float | None = None
    error: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    output_item_types: list[str] = Field(default_factory=list)
    assistant_text: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)


class ConversationItemSummary(BaseModel):
    id: str | None = None
    type: str | None = None
    role: str | None = None
    status: str | None = None
    call_id: str | None = None
    name: str | None = None
    text_fragments: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    id: str | None = None
    log_url: str | None = None
    metadata: dict[str, Any] | None = None
    items: list[ConversationItemSummary] = Field(default_factory=list)


class TraceInspectionResult(BaseModel):
    response: ResponseSummary | None = None
    conversation: ConversationSummary | None = None


def main() -> int:
    args = _parse_args()
    client = OpenAI()

    response_summary: ResponseSummary | None = None
    conversation_id = args.conversation_id
    if args.response_id is not None:
        response = client.responses.retrieve(response_id=args.response_id)
        response_summary = _normalize_response(response)
        if conversation_id is None:
            conversation_id = response_summary.conversation_id

    conversation_summary: ConversationSummary | None = None
    if conversation_id is not None:
        conversation = client.conversations.retrieve(conversation_id)
        items = list(
            client.conversations.items.list(
                conversation_id=conversation_id,
                limit=args.limit,
                order=args.order,
            )
        )
        conversation_summary = _normalize_conversation(conversation, items)

    result = TraceInspectionResult(
        response=response_summary,
        conversation=conversation_summary,
    )
    print(result.model_dump_json(indent=2 if args.pretty else None))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect OpenAI response and conversation state for local debugging.",
    )
    parser.add_argument("--response-id", dest="response_id", help="OpenAI response ID (resp_...).")
    parser.add_argument("--conversation-id", dest="conversation_id", help="OpenAI conversation ID (conv_...).")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum conversation items to fetch when a conversation is available.",
    )
    parser.add_argument(
        "--order",
        choices=("asc", "desc"),
        default="desc",
        help="Conversation item ordering.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()
    if args.response_id is None and args.conversation_id is None:
        parser.error("Provide --response-id, --conversation-id, or both.")
    return args


def _normalize_response(response: object) -> ResponseSummary:
    payload = _model_dump(response)
    trace_refs = extract_openai_trace_refs(response)
    output_items = payload.get("output")
    normalized_output_items = output_items if isinstance(output_items, list) else []

    assistant_text: list[str] = []
    output_item_types: list[str] = []
    tool_calls: list[ToolCallSummary] = []
    for raw_item in normalized_output_items:
        item_payload = _model_dump(raw_item)
        item_type = _string(item_payload.get("type"))
        if item_type is not None:
            output_item_types.append(item_type)
        assistant_text.extend(_extract_text_fragments(item_payload))
        tool_call = _extract_tool_call(item_payload)
        if tool_call is not None:
            tool_calls.append(tool_call)

    output_text = _string(payload.get("output_text"))
    if output_text is not None:
        assistant_text.append(output_text)

    return ResponseSummary(
        id=trace_refs.response_id,
        log_url=trace_refs.response_log_url,
        conversation_id=trace_refs.conversation_id,
        conversation_log_url=trace_refs.conversation_log_url,
        status=_string(payload.get("status")),
        model=_string(payload.get("model")),
        created_at=_float(payload.get("created_at")),
        error=_dict(payload.get("error")),
        usage=_dict(payload.get("usage")),
        output_item_types=output_item_types,
        assistant_text=_deduplicate_strings(assistant_text),
        tool_calls=tool_calls,
    )


def _normalize_conversation(conversation: object, items: list[object]) -> ConversationSummary:
    payload = _model_dump(conversation)
    trace_refs = build_openai_trace_refs(
        conversation_id=_string(payload.get("id")),
    )
    return ConversationSummary(
        id=trace_refs.conversation_id,
        log_url=trace_refs.conversation_log_url,
        metadata=_dict(payload.get("metadata")),
        items=[_normalize_conversation_item(item) for item in items],
    )


def _normalize_conversation_item(item: object) -> ConversationItemSummary:
    payload = _model_dump(item)
    return ConversationItemSummary(
        id=_string(payload.get("id")),
        type=_string(payload.get("type")),
        role=_string(payload.get("role")),
        status=_string(payload.get("status")),
        call_id=_string(payload.get("call_id")),
        name=_string(payload.get("name")),
        text_fragments=_deduplicate_strings(_extract_text_fragments(payload)),
    )


def _extract_tool_call(item_payload: dict[str, Any]) -> ToolCallSummary | None:
    item_type = _string(item_payload.get("type"))
    call_id = _string(item_payload.get("call_id"))
    name = _string(item_payload.get("name"))
    arguments = item_payload.get("arguments")
    status = _string(item_payload.get("status"))
    if item_type is None and call_id is None and name is None and arguments is None:
        return None
    if item_type is not None and "call" not in item_type and call_id is None:
        return None
    return ToolCallSummary(
        type=item_type,
        call_id=call_id,
        name=name,
        arguments=_serialize_jsonish(arguments),
        status=status,
    )


def _extract_text_fragments(payload: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    direct_text = _string(payload.get("text"))
    if direct_text is not None:
        fragments.append(direct_text)
    output_text = _string(payload.get("output_text"))
    if output_text is not None:
        fragments.append(output_text)
    content = payload.get("content")
    if isinstance(content, list):
        for raw_part in content:
            part = _model_dump(raw_part)
            text = _string(part.get("text"))
            refusal = _string(part.get("refusal"))
            transcript = _string(part.get("transcript"))
            if text is not None:
                fragments.append(text)
            if refusal is not None:
                fragments.append(refusal)
            if transcript is not None:
                fragments.append(transcript)
    return fragments


def _serialize_jsonish(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _model_dump(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _dict(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _deduplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduplicated.append(stripped)
    return deduplicated


if __name__ == "__main__":
    raise SystemExit(main())
