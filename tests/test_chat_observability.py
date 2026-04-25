from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.usage import Usage
import pytest
from chatkit.types import ThreadMetadata, UserMessageItem

from backend import create_services
from backend.chat_store import FileDeskChatContext
from backend.chatkit_server import _ChatRunHooks
from backend.file_library_gateway import OpenAIFileLibraryGateway
from backend.mcp_app import create_dev_mcp_server
from backend.openai_tracing import (
    build_openai_trace_refs,
    clear_active_openai_tool_trace_refs,
    set_active_openai_tool_trace_refs,
)
from backend.schemas import TagListResponse
from backend.settings import AppSettings


class _DummyMCPServer:
    async def __aenter__(self) -> "_DummyMCPServer":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>File Desk</div></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-clerk-secret")
    monkeypatch.setenv("APP_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("CLERK_ISSUER_URL", "https://clerk.example.com")
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'file-desk.db'}",
    )
    monkeypatch.setenv("STATIC_DIR", str(static_dir))
    return AppSettings()


@pytest.mark.asyncio
async def test_chat_server_uses_delta_input_and_persists_openai_trace_metadata(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = create_services(configured_settings)
    captured_run_arguments: dict[str, object] = {}

    async def unexpected_history_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("history replay should not be used once OpenAI conversation state exists")

    def fake_run_streamed(*args: object, **kwargs: object) -> object:
        captured_run_arguments["args"] = args
        captured_run_arguments["kwargs"] = kwargs
        return SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    id="resp_new",
                    conversation=SimpleNamespace(id="conv_new"),
                    usage=Usage(input_tokens=12, output_tokens=7, total_tokens=19),
                )
            ],
            last_response_id="resp_new",
        )

    async def fake_stream_agent_response(*args: object, **kwargs: object):
        del args, kwargs
        if False:
            yield None

    monkeypatch.setattr(services.chat_store, "load_thread_items", unexpected_history_load)
    monkeypatch.setattr(services.chatkit_server, "_build_mcp_server", lambda context: _DummyMCPServer())
    monkeypatch.setattr("backend.chatkit_server.Runner.run_streamed", fake_run_streamed)
    monkeypatch.setattr("backend.chatkit_server.stream_agent_response", fake_stream_agent_response)

    thread = ThreadMetadata.model_validate(
        {
            "id": "thread_1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": {"type": "active"},
            "metadata": {
                "openai_previous_response_id": "resp_prev",
                "openai_conversation_id": "conv_prev",
            },
        }
    )
    input_user_message = UserMessageItem.model_validate(
        {
            "id": "msg_1",
            "thread_id": "thread_1",
            "created_at": datetime.now(UTC).isoformat(),
            "content": [{"type": "input_text", "text": "Find the onboarding doc"}],
            "attachments": [],
            "inference_options": {},
        }
    )
    context = FileDeskChatContext(
        clerk_user_id="user_123",
        user_email="owner@example.com",
        display_name="Owner",
        bearer_token="test-bearer-token",
        selected_file_ids=[],
        thread_origin="interactive",
        request_app=SimpleNamespace(),
    )

    caplog.set_level(logging.INFO, logger="chatkit.server")
    try:
        events = [
            event
            async for event in services.chatkit_server._respond(
                thread=thread,
                input_user_message=input_user_message,
                context=context,
            )
        ]
    finally:
        await services.close()

    assert events == []
    runner_kwargs = captured_run_arguments["kwargs"]
    assert runner_kwargs["previous_response_id"] == "resp_prev"
    assert runner_kwargs["conversation_id"] == "conv_prev"
    assert runner_kwargs["max_turns"] == 20
    assert isinstance(runner_kwargs["hooks"], _ChatRunHooks)
    agent_input = captured_run_arguments["args"][1]
    assert isinstance(agent_input, list)
    assert len(agent_input) == 1
    first_item = agent_input[0]
    assert _input_text(first_item) == "Find the onboarding doc"
    assert thread.metadata["openai_previous_response_id"] == "resp_new"
    assert thread.metadata["openai_conversation_id"] == "conv_new"
    assert thread.metadata["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "cost_usd": 8.1e-05,
    }
    assert "https://platform.openai.com/logs/resp_prev" in caplog.text
    assert "https://platform.openai.com/logs/resp_new" in caplog.text
    assert "https://platform.openai.com/logs/conv_new" in caplog.text


@pytest.mark.asyncio
async def test_chat_server_uses_extended_mcp_client_session_timeout(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = create_services(configured_settings)
    captured_kwargs: dict[str, object] = {}

    class _FakeMCPServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("backend.chatkit_server.MCPServerStreamableHttp", _FakeMCPServer)
    context = FileDeskChatContext(
        clerk_user_id="user_123",
        user_email="owner@example.com",
        display_name="Owner",
        bearer_token="test-bearer-token",
        selected_file_ids=[],
        thread_origin="interactive",
        request_app=SimpleNamespace(),
    )

    try:
        services.chatkit_server._build_mcp_server(context)
    finally:
        await services.close()

    assert captured_kwargs["client_session_timeout_seconds"] == configured_settings.mcp_client_session_timeout_seconds


@pytest.mark.asyncio
async def test_chat_run_hooks_log_tool_calls_with_openai_trace_urls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hooks = _ChatRunHooks(
        thread_id="thread_1",
        model="gpt-5.4-mini",
        trace_refs=build_openai_trace_refs(
            response_id="resp_trace",
            conversation_id="conv_trace",
        ),
    )

    caplog.set_level(logging.INFO, logger="chatkit.server")
    await hooks.on_tool_start(
        SimpleNamespace(
            tool_name="search_files",
            tool_call_id="call_1",
            tool_arguments='{"query":"alpha"}',
        ),
        None,
        SimpleNamespace(name="search_files"),
    )
    await hooks.on_tool_end(
        SimpleNamespace(
            tool_name="search_files",
            tool_call_id="call_1",
        ),
        None,
        SimpleNamespace(name="search_files"),
        '[{"file_id":"node_alpha"}]',
    )

    assert "chat_mcp_tool_started" in caplog.text
    assert "chat_mcp_tool_completed" in caplog.text
    assert "https://platform.openai.com/logs/resp_trace" in caplog.text
    assert "https://platform.openai.com/logs/conv_trace" in caplog.text


@pytest.mark.asyncio
async def test_direct_mcp_tool_calls_are_logged(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = create_services(configured_settings)

    async def fake_list_tags(*, clerk_user_id: str) -> TagListResponse:
        assert clerk_user_id == "user_123"
        return TagListResponse(tags=[])

    monkeypatch.setattr(services.file_library, "list_tags", fake_list_tags)
    monkeypatch.setattr(
        "backend.mcp_app.get_current_clerk_access_token",
        lambda: SimpleNamespace(subject="user_123"),
    )

    server = create_dev_mcp_server(configured_settings, services)
    caplog.set_level(logging.INFO, logger="backend.mcp_app")
    try:
        await server.call_tool("list_tags", {}, run_middleware=False)
    finally:
        await services.close()

    assert "mcp_tool_started tool=list_tags" in caplog.text
    assert "mcp_tool_completed tool=list_tags" in caplog.text


@pytest.mark.asyncio
async def test_chat_driven_mcp_tool_logs_include_openai_trace_urls(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = create_services(configured_settings)

    async def fake_list_tags(*, clerk_user_id: str) -> TagListResponse:
        assert clerk_user_id == "user_123"
        return TagListResponse(tags=[])

    monkeypatch.setattr(services.file_library, "list_tags", fake_list_tags)
    monkeypatch.setattr(
        "backend.mcp_app.get_current_clerk_access_token",
        lambda: SimpleNamespace(subject="user_123"),
    )

    server = create_dev_mcp_server(configured_settings, services)
    caplog.set_level(logging.INFO, logger="backend.mcp_app")
    set_active_openai_tool_trace_refs(
        build_openai_trace_refs(
            response_id="resp_trace",
            conversation_id="conv_trace",
        )
    )
    try:
        await server.call_tool("list_tags", {}, run_middleware=False)
    finally:
        clear_active_openai_tool_trace_refs()
        await services.close()

    assert "mcp_tool_started tool=list_tags" in caplog.text
    assert "mcp_tool_completed tool=list_tags" in caplog.text
    assert "https://platform.openai.com/logs/resp_trace" in caplog.text
    assert "https://platform.openai.com/logs/conv_trace" in caplog.text


@pytest.mark.asyncio
async def test_gateway_arxiv_search_logs_clickable_openai_trace_urls(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = OpenAIFileLibraryGateway(configured_settings)

    async def fake_parse(**kwargs: object) -> object:
        assert kwargs["model"] == configured_settings.openai_agent_model
        return SimpleNamespace(
            id="resp_arxiv",
            conversation=SimpleNamespace(id="conv_arxiv"),
            output_parsed=SimpleNamespace(
                papers=[
                    SimpleNamespace(
                        title="Attention Is All You Need",
                        summary="Transformer architecture paper.",
                        authors=["Ashish Vaswani", "Noam Shazeer"],
                        url="https://arxiv.org/abs/1706.03762",
                    )
                ]
            ),
        )

    monkeypatch.setattr(gateway._client.responses, "parse", fake_parse)
    caplog.set_level(logging.INFO, logger="backend.file_library_gateway")
    try:
        results = await gateway.search_arxiv_papers(query="transformers", max_results=3)
    finally:
        await gateway.close()

    assert len(results) == 1
    assert results[0].arxiv_id == "1706.03762"
    assert "file_library_arxiv_search" in caplog.text
    assert "https://platform.openai.com/logs/resp_arxiv" in caplog.text
    assert "https://platform.openai.com/logs/conv_arxiv" in caplog.text


def _input_text(item: object) -> str | None:
    if isinstance(item, dict):
        content = item.get("content")
    else:
        content = getattr(item, "content", None)
    if not isinstance(content, list) or not content:
        return None
    first_part = content[0]
    if isinstance(first_part, dict):
        text = first_part.get("text")
    else:
        text = getattr(first_part, "text", None)
    return text if isinstance(text, str) else None
