from __future__ import annotations

from types import SimpleNamespace

from backend.openai_tracing import build_openai_trace_refs, extract_openai_trace_refs, latest_openai_trace_refs


def test_extract_openai_trace_refs_builds_clickable_urls() -> None:
    response = SimpleNamespace(
        id="resp_123",
        conversation=SimpleNamespace(id="conv_456"),
    )

    trace_refs = extract_openai_trace_refs(response)

    assert trace_refs.response_id == "resp_123"
    assert trace_refs.response_log_url == "https://platform.openai.com/logs/resp_123"
    assert trace_refs.conversation_id == "conv_456"
    assert trace_refs.conversation_log_url == "https://platform.openai.com/logs/conv_456"


def test_latest_openai_trace_refs_prefers_latest_non_empty_ids() -> None:
    trace_refs = latest_openai_trace_refs(
        [
            SimpleNamespace(
                id="resp_001",
                conversation=SimpleNamespace(id="conv_alpha"),
            ),
            SimpleNamespace(
                id="resp_002",
                conversation=SimpleNamespace(id="conv_alpha"),
            ),
        ]
    )

    assert trace_refs == build_openai_trace_refs(
        response_id="resp_002",
        conversation_id="conv_alpha",
    )
