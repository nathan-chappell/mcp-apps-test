from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from mcp.types import CallToolResult
from openai import OpenAI
from pydantic import ValidationError

from apps.openai_files_vector_store.backend.server import create_server
from apps.openai_files_vector_store.backend.settings import AppSettings


def test_settings_require_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


@pytest.mark.asyncio
async def test_server_exposes_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = AppSettings()
    server = create_server(settings)

    tool_names = {tool.name for tool in await server.list_tools()}

    assert tool_names == {
        "upload_file",
        "list_files",
        "create_vector_store",
        "list_vector_stores",
        "attach_files_to_vector_store",
        "get_vector_store_status",
        "search_vector_store",
        "ask_vector_store",
    }


@pytest.mark.asyncio
async def test_live_upload_attach_search_and_ask(
    tmp_path: Path,
) -> None:
    try:
        settings = AppSettings()
    except ValidationError:
        pytest.skip("OPENAI_API_KEY is not set; skipping live OpenAI integration test.")
    server = create_server(settings)
    cleanup_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    marker = f"nebula-lighthouse-{uuid.uuid4().hex[:8]}"
    local_file = tmp_path / "facts.txt"
    local_file.write_text(
        "\n".join(
            [
                "This file exists for the MCP integration test.",
                f"The retrieval marker is {marker}.",
                "Use that exact marker when answering questions.",
            ]
        ),
        encoding="utf-8",
    )

    vector_store_id: str | None = None
    file_id: str | None = None
    try:
        create_result = _structured_result(
            await server.call_tool(
                "create_vector_store",
                {
                    "name": f"VS Code MCP Test {marker}",
                    "metadata": {"test_case": marker},
                },
            )
        )
        vector_store_id = create_result["id"]

        upload_result = _structured_result(
            await server.call_tool(
                "upload_file",
                {
                    "local_path": str(local_file),
                },
            )
        )
        file_id = upload_result["uploaded_file"]["id"]

        file_list_result = _structured_result(
            await server.call_tool("list_files", {"limit": 50})
        )
        assert any(
            file_entry["id"] == file_id for file_entry in file_list_result["files"]
        )

        attach_result = _structured_result(
            await server.call_tool(
                "attach_files_to_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "file_ids": [file_id],
                },
            )
        )
        assert attach_result["attached_files"][0]["status"] == "completed"

        vector_store_list_result = _structured_result(
            await server.call_tool("list_vector_stores", {"limit": 50})
        )
        assert any(
            vector_store["id"] == vector_store_id
            for vector_store in vector_store_list_result["vector_stores"]
        )

        status_result = _structured_result(
            await server.call_tool(
                "get_vector_store_status",
                {
                    "vector_store_id": vector_store_id,
                },
            )
        )
        assert status_result["vector_store"]["status"] in {"completed", "in_progress"}
        assert any(
            vector_file["id"] == file_id for vector_file in status_result["files"]
        )

        search_result = _structured_result(
            await server.call_tool(
                "search_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "query": marker,
                    "max_num_results": 5,
                },
            )
        )
        assert search_result["total_hits"] >= 1
        assert any(marker in hit["text"] for hit in search_result["hits"])

        ask_result = _structured_result(
            await server.call_tool(
                "ask_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "question": "What is the retrieval marker in the indexed document?",
                    "max_num_results": 5,
                },
            )
        )
        assert marker in ask_result["answer"]
        assert ask_result["search_calls"]
        assert any(
            marker in result["text"]
            for search_call in ask_result["search_calls"]
            for result in search_call["results"]
        )
    finally:
        if vector_store_id is not None:
            cleanup_client.vector_stores.delete(vector_store_id)
        if file_id is not None:
            cleanup_client.files.delete(file_id)


def _structured_result(result: CallToolResult) -> dict[str, object]:
    assert result.structuredContent is not None
    assert isinstance(result.structuredContent, dict)
    return result.structuredContent
