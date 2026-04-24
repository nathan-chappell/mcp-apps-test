from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import subprocess
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest
from mcp.types import CallToolResult, ContentBlock
from pydantic import BaseModel

from apps.openai_vectorstore_mcp_app.backend.db import DatabaseManager
from apps.openai_vectorstore_mcp_app.backend.models import (
    AppUser,
    DerivedArtifact,
    KnowledgeBase,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeNodeTag,
    KnowledgeTag,
)
from apps.openai_vectorstore_mcp_app.backend.schemas import (
    KnowledgeBaseCommandResult,
    KnowledgeInfoResult,
    KnowledgeQueryResult,
    UpdateKnowledgeBaseResult,
)
from apps.openai_vectorstore_mcp_app.backend.server import (
    DESK_RESOURCE_URI,
    RESOURCE_MIME_TYPE,
    create_server,
)
from apps.openai_vectorstore_mcp_app.backend.settings import AppSettings

type ToolCallResponse = Sequence[ContentBlock] | dict[str, Any] | CallToolResult

ResultModelT = TypeVar("ResultModelT", bound=BaseModel)

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "apps/openai_vectorstore_mcp_app/ui"
UI_DIST_PATH = UI_DIR / "dist/mcp-app.html"
HOST_INDEX_PATH = UI_DIR / "host-dist/dev-host/index.html"
HOST_SANDBOX_PATH = UI_DIR / "host-dist/dev-host/sandbox.html"


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-clerk-secret")
    monkeypatch.setenv("APP_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("CLERK_ISSUER_URL", "https://clerk.example.com")
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'knowledge-base-desk.db'}",
    )
    return AppSettings()


def test_settings_load_from_dotenv(configured_settings: AppSettings) -> None:
    settings = configured_settings
    assert settings.openai_api_key.get_secret_value()
    assert settings.clerk_secret_key.get_secret_value()
    assert settings.app_signing_secret.get_secret_value()
    assert settings.normalized_app_base_url.startswith("http")
    assert settings.normalized_database_url
    assert settings.sync_database_url
    assert settings.openai_agent_model == "gpt-5.4"
    assert settings.openai_audio_transcription_model == "gpt-4o-transcribe-diarize"
    assert settings.mcp_required_scopes == ["user"]


@pytest.mark.asyncio
async def test_server_exposes_expected_tools(configured_settings: AppSettings) -> None:
    server = create_server(configured_settings)
    tool_names = {tool.name for tool in await server.list_tools()}
    assert tool_names == {
        "query_knowledge_base",
        "get_knowledge_base_info",
        "update_knowledge_base",
        "run_knowledge_base_command",
        "confirm_knowledge_base_command",
    }


@pytest.mark.asyncio
async def test_server_registers_all_tools_as_async(
    configured_settings: AppSettings,
) -> None:
    server = create_server(configured_settings)
    for tool_name in {
        "query_knowledge_base",
        "get_knowledge_base_info",
        "update_knowledge_base",
        "run_knowledge_base_command",
        "confirm_knowledge_base_command",
    }:
        tool = server._tool_manager.get_tool(tool_name)
        assert tool is not None
        assert tool.is_async


@pytest.mark.asyncio
async def test_server_only_uses_ui_resource_for_query_knowledge_base(
    configured_settings: AppSettings,
) -> None:
    server = create_server(configured_settings)
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert tools["query_knowledge_base"].meta == {
        "ui": {"resourceUri": DESK_RESOURCE_URI}
    }
    assert tools["get_knowledge_base_info"].meta is None
    assert tools["update_knowledge_base"].meta == {"ui": {"visibility": ["app"]}}
    assert tools["run_knowledge_base_command"].meta == {"ui": {"visibility": ["app"]}}
    assert tools["confirm_knowledge_base_command"].meta == {
        "ui": {"visibility": ["app"]}
    }


@pytest.fixture(scope="session")
def built_desk_ui() -> Path:
    subprocess.run(["npm", "run", "build"], check=True, cwd=UI_DIR)
    assert UI_DIST_PATH.is_file()
    return UI_DIST_PATH


@pytest.fixture(scope="session")
def built_dev_host_ui() -> tuple[Path, Path]:
    subprocess.run(["npm", "run", "host:build"], check=True, cwd=UI_DIR)
    assert HOST_INDEX_PATH.is_file()
    assert HOST_SANDBOX_PATH.is_file()
    return HOST_INDEX_PATH, HOST_SANDBOX_PATH


@pytest.mark.asyncio
async def test_server_exposes_knowledge_base_desk_resource(
    built_desk_ui: Path,
    configured_settings: AppSettings,
) -> None:
    server = create_server(configured_settings)

    resources = await server.list_resources()
    resource = next(
        resource_item
        for resource_item in resources
        if str(resource_item.uri) == DESK_RESOURCE_URI
    )
    assert resource.mimeType == RESOURCE_MIME_TYPE

    contents = await server.read_resource(DESK_RESOURCE_URI)
    assert len(contents) == 1
    assert contents[0].mime_type == RESOURCE_MIME_TYPE
    content = (
        contents[0].content
        if isinstance(contents[0].content, str)
        else str(contents[0].content, encoding="utf-8")
    )
    assert "<title>Knowledge Base Desk</title>" in content
    assert "knowledge-base-desk-root" in content


@pytest.mark.asyncio
async def test_backend_serves_root_host_sandbox_and_auth_config(
    built_dev_host_ui: tuple[Path, Path],
    configured_settings: AppSettings,
) -> None:
    _ = built_dev_host_ui
    server = create_server(configured_settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.streamable_http_app()),
        base_url="http://testserver",
    ) as client:
        host_response = await client.get("/")
        assert host_response.status_code == 200
        assert host_response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
        assert "OpenAI Files Vector Store Dev Host" in host_response.text

        index_response = await client.get("/index.html")
        assert index_response.status_code == 200
        assert "OpenAI Files Vector Store Dev Host" in index_response.text

        sandbox_response = await client.get("/sandbox", params={"csp": '{"connectDomains":["https://api.example.com"]}'})
        assert sandbox_response.status_code == 200
        assert "MCP App Sandbox" in sandbox_response.text
        assert (
            sandbox_response.headers["content-security-policy"]
            == "default-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: data:; "
            "style-src 'self' 'unsafe-inline' blob: data:; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data: blob:; "
            "media-src 'self' data: blob:; "
            "connect-src 'self' https://api.example.com; "
            "worker-src 'self' blob:; "
            "frame-src 'none'; "
            "object-src 'none'; "
            "base-uri 'none'"
        )

        auth_config_response = await client.get("/api/dev-auth-config")
        assert auth_config_response.status_code == 200
        assert auth_config_response.json() == {
            "clerk_publishable_key": configured_settings.clerk_publishable_key,
            "app_name": configured_settings.app_name,
        }


@pytest.mark.asyncio
async def test_query_knowledge_base_without_query_returns_local_dev_state(
    configured_settings: AppSettings,
) -> None:
    server = create_server(configured_settings)

    query_result = _structured_result(
        await server.call_tool("query_knowledge_base", {}),
        KnowledgeQueryResult,
    )

    desk_state = query_result.knowledge_base_state
    assert query_result.kind == "knowledge_base"
    assert desk_state.access.user.clerk_user_id == "local-dev"
    assert desk_state.access.status == "active"
    assert desk_state.access.user.active is True
    assert desk_state.access.user.role == "admin"
    assert desk_state.knowledge_base is not None
    assert desk_state.knowledge_base.nodes == []
    assert desk_state.knowledge_base.tags == []
    assert desk_state.knowledge_base.edges == []
    assert desk_state.capabilities.upload_url.endswith("/api/uploads")
    assert desk_state.capabilities.supports_video_audio_extraction is True


@pytest.mark.asyncio
async def test_get_knowledge_base_info_returns_seeded_graph_detail(
    configured_settings: AppSettings,
) -> None:
    await _seed_graph(configured_settings)
    server = create_server(configured_settings)

    info_result = _structured_result(
        await server.call_tool(
            "get_knowledge_base_info",
            {
                "selected_node_id": "node_alpha",
                "graph_selection_mode": "children",
                "tag_ids": ["tag_ops"],
                "tag_match_mode": "all",
                "detail_node_id": "node_alpha",
            },
        ),
        KnowledgeInfoResult,
    )

    assert info_result.knowledge_base_state.knowledge_base is not None
    assert info_result.knowledge_base_state.knowledge_base.context.selected_node_id == (
        "node_alpha"
    )
    assert sorted(
        info_result.knowledge_base_state.knowledge_base.context.visible_node_ids
    ) == ["node_alpha", "node_beta"]
    assert sorted(
        info_result.knowledge_base_state.knowledge_base.context.scoped_node_ids
    ) == ["node_alpha", "node_beta"]
    assert info_result.node_detail is not None
    assert info_result.node_detail.display_title == "Alpha Brief"
    assert [edge.label for edge in info_result.node_detail.outgoing_edges] == ["supports"]
    assert [tag.name for tag in info_result.node_detail.tags] == ["ops"]


@pytest.mark.asyncio
async def test_update_and_command_confirmation_flow(
    configured_settings: AppSettings,
) -> None:
    await _seed_graph(configured_settings)
    server = create_server(configured_settings)

    rename_result = _structured_result(
        await server.call_tool(
            "update_knowledge_base",
            {
                "action": "rename_node",
                "node_id": "node_alpha",
                "title": "Alpha Renamed",
            },
        ),
        UpdateKnowledgeBaseResult,
    )
    assert rename_result.node is not None
    assert rename_result.node.display_title == "Alpha Renamed"

    command_result = _structured_result(
        await server.call_tool(
            "run_knowledge_base_command",
            {
                "command": "add an edge from Alpha Renamed to Gamma Notes labeled references",
                "selected_node_id": "node_alpha",
            },
        ),
        KnowledgeBaseCommandResult,
    )
    assert command_result.status == "executed"
    assert command_result.edge is not None
    assert command_result.edge.label == "references"
    assert command_result.edge.to_node_title == "Gamma Notes"

    delete_request = _structured_result(
        await server.call_tool(
            "run_knowledge_base_command",
            {
                "command": "delete node Gamma Notes",
            },
        ),
        KnowledgeBaseCommandResult,
    )
    assert delete_request.status == "pending_confirmation"
    assert delete_request.pending_confirmation is not None

    delete_result = _structured_result(
        await server.call_tool(
            "confirm_knowledge_base_command",
            {
                "token": delete_request.pending_confirmation.token,
            },
        ),
        KnowledgeBaseCommandResult,
    )
    assert delete_result.status == "executed"
    assert delete_result.action == "delete_node"

    info_result = _structured_result(
        await server.call_tool("get_knowledge_base_info", {}),
        KnowledgeInfoResult,
    )
    assert info_result.knowledge_base_state.knowledge_base is not None
    node_titles = {
      node.display_title
      for node in info_result.knowledge_base_state.knowledge_base.nodes
    }
    assert "Gamma Notes" not in node_titles


async def _seed_graph(settings: AppSettings) -> None:
    database = DatabaseManager(settings)
    await database.ensure_ready()

    async with database.session() as session:
        now = datetime.now(UTC)
        user = AppUser(
            clerk_user_id="local-dev",
            primary_email="local-dev@example.com",
            display_name="Local Developer",
            active=True,
            role="admin",
            last_seen_at=now,
        )
        session.add(user)
        await session.flush()

        knowledge_base = KnowledgeBase(
            id="kb_seed",
            user_id=user.id,
            title="Local Developer's Knowledge Base",
            description="Seeded graph for integration tests.",
            updated_at=now,
        )
        session.add(knowledge_base)

        tag_ops = KnowledgeTag(
            id="tag_ops",
            knowledge_base_id=knowledge_base.id,
            name="ops",
            slug="ops",
            color="#0f766e",
        )
        tag_research = KnowledgeTag(
            id="tag_research",
            knowledge_base_id=knowledge_base.id,
            name="research",
            slug="research",
            color="#2563eb",
        )
        session.add_all([tag_ops, tag_research])

        node_alpha = KnowledgeNode(
            id="node_alpha",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Alpha Brief",
            original_filename="alpha-brief.md",
            media_type="text/markdown",
            source_kind="document",
            status="ready",
            byte_size=1200,
            original_mime_type="text/markdown",
            updated_at=now,
        )
        node_beta = KnowledgeNode(
            id="node_beta",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Beta Runbook",
            original_filename="beta-runbook.md",
            media_type="text/markdown",
            source_kind="document",
            status="ready",
            byte_size=1600,
            original_mime_type="text/markdown",
            updated_at=now,
        )
        node_gamma = KnowledgeNode(
            id="node_gamma",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Gamma Notes",
            original_filename="gamma-notes.md",
            media_type="text/markdown",
            source_kind="document",
            status="ready",
            byte_size=900,
            original_mime_type="text/markdown",
            updated_at=now,
        )
        session.add_all([node_alpha, node_beta, node_gamma])

        session.add_all(
            [
                KnowledgeNodeTag(node_id=node_alpha.id, tag_id=tag_ops.id),
                KnowledgeNodeTag(node_id=node_beta.id, tag_id=tag_ops.id),
                KnowledgeNodeTag(node_id=node_gamma.id, tag_id=tag_research.id),
            ]
        )
        session.add(
            KnowledgeEdge(
                id="edge_alpha_beta",
                knowledge_base_id=knowledge_base.id,
                from_node_id=node_alpha.id,
                to_node_id=node_beta.id,
                label="supports",
                updated_at=now,
            )
        )
        session.add(
            DerivedArtifact(
                id="artifact_alpha",
                node_id=node_alpha.id,
                kind="document_text",
                openai_file_id=None,
                text_content="Alpha operational brief.",
                structured_payload=None,
                updated_at=now,
            )
        )
        await session.commit()

    await database.close()


def _structured_result(result: ToolCallResponse, model_type: type[ResultModelT]) -> ResultModelT:
    if isinstance(result, CallToolResult):
        if not result.structuredContent:
            raise AssertionError("Expected structuredContent in the tool result.")
        return model_type.model_validate(result.structuredContent)
    raise AssertionError(f"Expected CallToolResult, received {type(result)!r}")
