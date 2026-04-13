from __future__ import annotations

from collections.abc import Sequence
import subprocess
import uuid
from pathlib import Path
from typing import Any, TypeVar

import pytest
from mcp.types import CallToolResult, ContentBlock
from openai import OpenAI
from pydantic import BaseModel

from apps.openai_files_vector_store.backend.schemas import (
    AskVectorStoreResult,
    AttachFilesResult,
    DeletedFileResult,
    FileListResult,
    OpenVectorStoreConsoleResult,
    SearchVectorStoreResult,
    UploadFileResult,
    VectorStoreFileSummary,
    VectorStoreListResult,
    VectorStoreStatusResult,
    VectorStoreSummary,
)
from apps.openai_files_vector_store.backend.server import (
    CONSOLE_RESOURCE_URI,
    RESOURCE_MIME_TYPE,
    create_server,
)
from apps.openai_files_vector_store.backend.settings import AppSettings

type ToolCallResponse = Sequence[ContentBlock] | dict[str, Any] | CallToolResult

ResultModelT = TypeVar("ResultModelT", bound=BaseModel)

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "apps/openai_files_vector_store/ui"
UI_DIST_PATH = UI_DIR / "dist/mcp-app.html"


def test_settings_load_from_dotenv() -> None:
    settings = AppSettings()
    assert settings.openai_api_key.get_secret_value()
    assert settings.openai_agent_model == "gpt-5.4"
    assert settings.openai_file_search_max_results == 5
    assert settings.log_level == "INFO"


@pytest.mark.asyncio
async def test_server_exposes_expected_tools() -> None:
    server = create_server(AppSettings())

    tool_names = {tool.name for tool in await server.list_tools()}

    assert tool_names == {
        "open_vector_store_console",
        "upload_file",
        "list_files",
        "create_vector_store",
        "list_vector_stores",
        "attach_files_to_vector_store",
        "get_vector_store_status",
        "search_vector_store",
        "ask_vector_store",
        "update_vector_store_file_attributes",
        "delete_file",
    }


@pytest.mark.asyncio
async def test_server_registers_all_tools_as_async() -> None:
    server = create_server(AppSettings())

    tool_names = {
        "open_vector_store_console",
        "upload_file",
        "list_files",
        "create_vector_store",
        "list_vector_stores",
        "attach_files_to_vector_store",
        "get_vector_store_status",
        "search_vector_store",
        "ask_vector_store",
        "update_vector_store_file_attributes",
        "delete_file",
    }

    for tool_name in tool_names:
        tool = server._tool_manager.get_tool(tool_name)
        assert tool is not None
        assert tool.is_async


@pytest.fixture(scope="session")
def built_console_ui() -> Path:
    subprocess.run(
        ["npm", "run", "build"],
        check=True,
        cwd=UI_DIR,
    )
    assert UI_DIST_PATH.is_file()
    return UI_DIST_PATH


@pytest.mark.asyncio
async def test_server_exposes_console_resource(
    built_console_ui: Path,
) -> None:
    server = create_server(AppSettings())

    resources = await server.list_resources()
    resource = next(
        resource_item
        for resource_item in resources
        if str(resource_item.uri) == CONSOLE_RESOURCE_URI
    )
    assert resource.mimeType == RESOURCE_MIME_TYPE

    contents = await server.read_resource(CONSOLE_RESOURCE_URI)
    assert len(contents) == 1
    assert contents[0].mime_type == RESOURCE_MIME_TYPE
    content = (
        contents[0].content
        if isinstance(contents[0].content, str)
        else str(contents[0].content, encoding="utf-8")
    )
    assert "<title>OpenAI Files Vector Store Console</title>" in content
    assert "vector-store-console-root" in content


@pytest.mark.asyncio
async def test_live_retrieval_scoping_metadata_and_delete(
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    server = create_server(settings)
    cleanup_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    marker_primary = f"nebula-lighthouse-{uuid.uuid4().hex[:8]}"
    marker_secondary = f"aurora-capsule-{uuid.uuid4().hex[:8]}"
    primary_file = tmp_path / "primary-facts.txt"
    primary_file.write_text(
        "\n".join(
            [
                "This file exists for the primary MCP integration test flow.",
                f"The primary retrieval marker is {marker_primary}.",
                "Use that exact primary marker when answering questions.",
            ]
        ),
        encoding="utf-8",
    )
    secondary_file = tmp_path / "secondary-facts.txt"
    secondary_file.write_text(
        "\n".join(
            [
                "This file exists for the secondary MCP integration test flow.",
                f"The secondary retrieval marker is {marker_secondary}.",
                "Use that exact secondary marker when answering questions.",
            ]
        ),
        encoding="utf-8",
    )

    vector_store_id: str | None = None
    primary_file_id: str | None = None
    secondary_file_id: str | None = None
    try:
        create_result = _structured_result(
            await server.call_tool(
                "create_vector_store",
                {
                    "name": f"VS Code MCP Test {marker_primary}",
                    "metadata": {"test_case": marker_primary},
                },
            ),
            VectorStoreSummary,
        )
        vector_store_id = create_result.id

        upload_result = _structured_result(
            await server.call_tool(
                "upload_file",
                {
                    "local_path": str(primary_file),
                    "vector_store_id": vector_store_id,
                    "attributes": {"dataset": "primary", "priority": 1},
                },
            ),
            UploadFileResult,
        )
        primary_file_id = upload_result.uploaded_file.id
        assert upload_result.attached_file is not None
        assert upload_result.attached_file.status == "completed"
        assert upload_result.attached_file.attributes is not None
        assert upload_result.attached_file.attributes["openai_file_id"] == primary_file_id
        assert upload_result.attached_file.attributes["filename"] == primary_file.name
        assert upload_result.attached_file.attributes["dataset"] == "primary"
        assert upload_result.attached_file.attributes["priority"] == 1.0

        file_list_result = _structured_result(
            await server.call_tool("list_files", {"limit": 50}),
            FileListResult,
        )
        assert any(
            file_entry.id == primary_file_id for file_entry in file_list_result.files
        )

        attach_result = _structured_result(
            await server.call_tool(
                "attach_files_to_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "local_paths": [str(secondary_file)],
                    "attributes": {"dataset": "secondary", "priority": 2},
                },
            ),
            AttachFilesResult,
        )
        assert attach_result.attached_files[0].status == "completed"
        assert attach_result.attached_files[0].attributes is not None
        assert attach_result.attached_files[0].attributes["filename"] == secondary_file.name
        assert attach_result.attached_files[0].attributes["dataset"] == "secondary"
        assert attach_result.attached_files[0].attributes["priority"] == 2.0
        secondary_file_id = attach_result.attached_files[0].id
        assert attach_result.attached_files[0].attributes["openai_file_id"] == secondary_file_id

        console_result = _structured_result(
            await server.call_tool(
                "open_vector_store_console",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            OpenVectorStoreConsoleResult,
        )
        assert console_result.selected_vector_store_id == vector_store_id
        assert console_result.selected_vector_store_status is not None
        assert (
            console_result.selected_vector_store_status.vector_store.id
            == vector_store_id
        )
        assert console_result.search_panel.query == ""
        assert console_result.search_panel.scope == "vector_store"
        assert console_result.ask_panel.question == ""
        assert console_result.ask_panel.scope == "vector_store"
        assert any(
            vector_store.id == vector_store_id
            for vector_store in console_result.vector_store_list.vector_stores
        )

        vector_store_list_result = _structured_result(
            await server.call_tool("list_vector_stores", {"limit": 50}),
            VectorStoreListResult,
        )
        assert any(
            vector_store.id == vector_store_id
            for vector_store in vector_store_list_result.vector_stores
        )

        status_result = _structured_result(
            await server.call_tool(
                "get_vector_store_status",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            VectorStoreStatusResult,
        )
        assert status_result.vector_store.status in {"completed", "in_progress"}
        primary_status_file = next(
            vector_file
            for vector_file in status_result.files
            if vector_file.id == primary_file_id
        )
        secondary_status_file = next(
            vector_file
            for vector_file in status_result.files
            if vector_file.id == secondary_file_id
        )
        assert primary_status_file.attributes is not None
        assert primary_status_file.attributes["filename"] == primary_file.name
        assert primary_status_file.attributes["openai_file_id"] == primary_file_id
        assert secondary_status_file.attributes is not None
        assert secondary_status_file.attributes["filename"] == secondary_file.name
        assert secondary_status_file.attributes["openai_file_id"] == secondary_file_id

        search_result = _structured_result(
            await server.call_tool(
                "search_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "query": marker_primary,
                    "max_num_results": 5,
                    "file_id": primary_file_id,
                },
            ),
            SearchVectorStoreResult,
        )
        assert search_result.total_hits >= 1
        assert search_result.file_id == primary_file_id
        assert all(hit.file_id == primary_file_id for hit in search_result.hits)
        assert any(marker_primary in hit.text for hit in search_result.hits)
        assert all(marker_secondary not in hit.text for hit in search_result.hits)

        filename_scoped_search_result = _structured_result(
            await server.call_tool(
                "search_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "query": marker_secondary,
                    "max_num_results": 5,
                    "filename": secondary_file.name,
                },
            ),
            SearchVectorStoreResult,
        )
        assert filename_scoped_search_result.filename == secondary_file.name
        assert all(hit.filename == secondary_file.name for hit in filename_scoped_search_result.hits)
        assert any(
            marker_secondary in hit.text
            for hit in filename_scoped_search_result.hits
        )
        assert all(
            marker_primary not in hit.text
            for hit in filename_scoped_search_result.hits
        )

        ask_result = _structured_result(
            await server.call_tool(
                "ask_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "question": "What is the primary retrieval marker in the indexed document?",
                    "max_num_results": 5,
                    "file_id": primary_file_id,
                },
            ),
            AskVectorStoreResult,
        )
        assert ask_result.file_id == primary_file_id
        assert marker_primary in ask_result.answer
        assert ask_result.search_calls
        assert any(
            marker_primary in result.text
            for search_call in ask_result.search_calls
            for result in search_call.results
        )
        assert all(
            marker_secondary not in result.text
            for search_call in ask_result.search_calls
            for result in search_call.results
        )

        filename_scoped_ask_result = _structured_result(
            await server.call_tool(
                "ask_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "question": "What is the secondary retrieval marker in the indexed document?",
                    "max_num_results": 5,
                    "filename": secondary_file.name,
                },
            ),
            AskVectorStoreResult,
        )
        assert filename_scoped_ask_result.filename == secondary_file.name
        assert marker_secondary in filename_scoped_ask_result.answer
        assert any(
            marker_secondary in result.text
            for search_call in filename_scoped_ask_result.search_calls
            for result in search_call.results
        )

        updated_attributes_result = _structured_result(
            await server.call_tool(
                "update_vector_store_file_attributes",
                {
                    "vector_store_id": vector_store_id,
                    "file_id": primary_file_id,
                    "attributes": {
                        "dataset": "primary-updated",
                        "reviewed": True,
                        "openai_file_id": "wrong-on-purpose",
                    },
                },
            ),
            VectorStoreFileSummary,
        )
        assert updated_attributes_result.attributes is not None
        assert updated_attributes_result.attributes["dataset"] == "primary-updated"
        assert updated_attributes_result.attributes["reviewed"] is True
        assert updated_attributes_result.attributes["openai_file_id"] == primary_file_id
        assert updated_attributes_result.attributes["filename"] == primary_file.name

        refreshed_status_result = _structured_result(
            await server.call_tool(
                "get_vector_store_status",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            VectorStoreStatusResult,
        )
        refreshed_primary_file = next(
            vector_file
            for vector_file in refreshed_status_result.files
            if vector_file.id == primary_file_id
        )
        assert refreshed_primary_file.attributes is not None
        assert refreshed_primary_file.attributes["dataset"] == "primary-updated"
        assert refreshed_primary_file.attributes["reviewed"] is True
        assert refreshed_primary_file.attributes["filename"] == primary_file.name

        deleted_file_result = _structured_result(
            await server.call_tool(
                "delete_file",
                {
                    "file_id": secondary_file_id,
                },
            ),
            DeletedFileResult,
        )
        assert deleted_file_result.file_id == secondary_file_id
        assert deleted_file_result.deleted is True
        secondary_file_id = None

        file_list_after_delete = _structured_result(
            await server.call_tool("list_files", {"limit": 50}),
            FileListResult,
        )
        assert all(
            file_entry.id != deleted_file_result.file_id
            for file_entry in file_list_after_delete.files
        )

        status_after_delete = _structured_result(
            await server.call_tool(
                "get_vector_store_status",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            VectorStoreStatusResult,
        )
        assert all(
            vector_file.id != deleted_file_result.file_id
            for vector_file in status_after_delete.files
        )
    finally:
        if vector_store_id is not None:
            cleanup_client.vector_stores.delete(vector_store_id)
        if primary_file_id is not None:
            cleanup_client.files.delete(primary_file_id)
        if secondary_file_id is not None:
            cleanup_client.files.delete(secondary_file_id)


@pytest.mark.asyncio
async def test_server_restart_does_not_require_local_history_or_create_repo_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_status_before = _git_status_snapshot()
    settings = AppSettings()
    cleanup_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    marker = f"stateless-restart-{uuid.uuid4().hex[:8]}"
    input_file = tmp_path / "restart-stateless.txt"
    input_file.write_text(
        "\n".join(
            [
                "This file verifies the MCP server stays near-stateless across restarts.",
                f"The restart verification marker is {marker}.",
                "The second server instance should find this marker using only OpenAI state.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    vector_store_id: str | None = None
    file_id: str | None = None
    try:
        first_server = create_server(settings)

        create_result = _structured_result(
            await first_server.call_tool(
                "create_vector_store",
                {
                    "name": f"Stateless Restart Test {marker}",
                    "metadata": {"test_case": marker, "persistence": "stateless"},
                },
            ),
            VectorStoreSummary,
        )
        vector_store_id = create_result.id

        upload_result = _structured_result(
            await first_server.call_tool(
                "upload_file",
                {
                    "local_path": input_file.name,
                    "vector_store_id": vector_store_id,
                    "attributes": {"dataset": "restart-stateless"},
                },
            ),
            UploadFileResult,
        )
        file_id = upload_result.uploaded_file.id
        assert upload_result.attached_file is not None
        assert upload_result.attached_file.status == "completed"

        restarted_server = create_server(settings)

        status_result = _structured_result(
            await restarted_server.call_tool(
                "get_vector_store_status",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            VectorStoreStatusResult,
        )
        restored_file = next(
            vector_file for vector_file in status_result.files if vector_file.id == file_id
        )
        assert restored_file.attributes is not None
        assert restored_file.attributes["openai_file_id"] == file_id
        assert restored_file.attributes["filename"] == input_file.name

        search_result = _structured_result(
            await restarted_server.call_tool(
                "search_vector_store",
                {
                    "vector_store_id": vector_store_id,
                    "query": marker,
                    "max_num_results": 5,
                    "file_id": file_id,
                },
            ),
            SearchVectorStoreResult,
        )
        assert search_result.file_id == file_id
        assert any(marker in hit.text for hit in search_result.hits)

        console_result = _structured_result(
            await restarted_server.call_tool(
                "open_vector_store_console",
                {
                    "vector_store_id": vector_store_id,
                },
            ),
            OpenVectorStoreConsoleResult,
        )
        assert console_result.selected_vector_store_id == vector_store_id
        assert console_result.selected_vector_store_status is not None
        assert console_result.selected_vector_store_status.vector_store.id == vector_store_id
    finally:
        if vector_store_id is not None:
            cleanup_client.vector_stores.delete(vector_store_id)
        if file_id is not None:
            cleanup_client.files.delete(file_id)

    assert sorted(path.name for path in tmp_path.iterdir()) == [input_file.name]
    assert _git_status_snapshot() == repo_status_before


def _structured_result(
    result: ToolCallResponse,
    result_type: type[ResultModelT],
) -> ResultModelT:
    if isinstance(result, CallToolResult):
        structured_content = result.structuredContent
    else:
        structured_content = result

    assert structured_content is not None
    assert isinstance(structured_content, dict)
    return result_type.model_validate(structured_content)


def _git_status_snapshot() -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )
    return result.stdout
