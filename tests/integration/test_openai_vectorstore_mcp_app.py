from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from chatkit.server import NonStreamingResult
from sqlalchemy import select

from backend.clerk import (
    ClerkUserRecord,
    ClerkVerifiedSessionToken,
)
from backend import create_fastapi_app, create_mcp_server, create_services
from backend.db import DatabaseManager
from backend.file_library_gateway import DownloadedRemoteFile
from backend.mcp_app import create_dev_mcp_server
from backend.models import (
    AppUser,
    DerivedArtifact,
    FileLibrary,
    FileTag,
    FileTagLink,
    LibraryFile,
)
from backend.schemas import ArxivPaperCandidate, SearchHit
from backend.settings import AppSettings


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


def test_settings_load_from_env(configured_settings: AppSettings) -> None:
    assert configured_settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert configured_settings.clerk_secret_key.get_secret_value() == "test-clerk-secret"
    assert configured_settings.app_signing_secret.get_secret_value() == "test-signing-secret"
    assert configured_settings.openai_agent_model == "gpt-5.4-mini"
    assert configured_settings.openai_vision_model == "gpt-5.4-mini"
    assert configured_settings.mcp_required_scopes == ["openid", "email", "profile"]
    assert configured_settings.normalized_static_dir


def test_settings_map_legacy_static_dir_to_root_ui_dist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    fallback_static_dir = repo_root / "ui" / "dist"
    fallback_static_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-clerk-secret")
    monkeypatch.setenv("APP_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("CLERK_ISSUER_URL", "https://clerk.example.com")
    monkeypatch.setenv("STATIC_DIR", "apps/openai_vectorstore_mcp_app/ui/dist")
    monkeypatch.setattr("backend.settings.PROJECT_ROOT", repo_root)

    settings = AppSettings()

    assert settings.normalized_static_dir == str(fallback_static_dir)


@pytest.mark.asyncio
async def test_mcp_server_exposes_file_desk_tools(configured_settings: AppSettings) -> None:
    services = create_services(configured_settings)
    server = create_mcp_server(configured_settings, services)
    try:
        tools = {tool.name: tool for tool in await server.list_tools(run_middleware=False)}
    finally:
        await services.close()

    assert set(tools) == {
        "list_files",
        "list_tags",
        "search_files",
        "search_arxiv_papers",
        "import_arxiv_paper",
        "search_file_branches",
        "get_file_detail",
        "read_file_text",
        "delete_file",
        "files",
        "file_search",
        "branch_search",
    }
    assert tools["files"].meta is not None
    assert tools["files"].meta["ui"]["resourceUri"].startswith("ui://")
    assert tools["file_search"].meta is not None
    assert tools["file_search"].meta["ui"]["resourceUri"].startswith("ui://")
    assert tools["branch_search"].meta is not None
    assert tools["branch_search"].meta["ui"]["resourceUri"].startswith("ui://")


@pytest.mark.asyncio
async def test_dev_mcp_server_exposes_file_desk_tools(configured_settings: AppSettings) -> None:
    services = create_services(configured_settings)
    server = create_dev_mcp_server(configured_settings, services)
    try:
        tools = {tool.name: tool for tool in await server.list_tools(run_middleware=False)}
    finally:
        await services.close()

    assert set(tools) == {
        "list_files",
        "list_tags",
        "search_files",
        "search_arxiv_papers",
        "import_arxiv_paper",
        "search_file_branches",
        "get_file_detail",
        "read_file_text",
        "delete_file",
        "files",
        "file_search",
        "branch_search",
    }
    assert tools["files"].meta is not None
    assert tools["files"].meta["ui"]["resourceUri"].startswith("ui://")
    assert tools["file_search"].meta is not None
    assert tools["file_search"].meta["ui"]["resourceUri"].startswith("ui://")
    assert tools["branch_search"].meta is not None
    assert tools["branch_search"].meta["ui"]["resourceUri"].startswith("ui://")


@pytest.mark.asyncio
async def test_fastapi_redirects_bare_mcp_root_to_trailing_slash(configured_settings: AppSettings) -> None:
    app = create_fastapi_app(configured_settings)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            response = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})

    assert response.status_code == 307
    assert response.headers["location"] == "/mcp/"


@pytest.mark.asyncio
async def test_file_search_ui_tool_accepts_initial_arguments(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_file_library(configured_settings)
    services = create_services(configured_settings)

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    async def fake_search_vector_store(
        _self,
        *,
        vector_store_id: str,
        query: str,
        max_results: int,
        rewrite_query: bool,
        filters,
    ) -> list[SearchHit]:
        assert vector_store_id == "vs_alpha"
        assert query == "alpha"
        assert max_results == 8
        assert rewrite_query is True
        assert filters == {"type": "eq", "key": "tag__operations", "value": True}
        return []

    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.search_vector_store",
        fake_search_vector_store,
    )
    monkeypatch.setattr(
        "backend.mcp_app.get_current_clerk_access_token",
        lambda: SimpleNamespace(subject="user_123"),
    )

    server = create_dev_mcp_server(configured_settings, services)
    try:
        result = await server.call_tool(
            "file_search",
            {
                "query": "alpha",
                "tag_ids": ["tag_ops"],
                "tag_match_mode": "all",
            },
            run_middleware=False,
        )
    finally:
        await services.close()

    assert result.structured_content is not None
    assert result.structured_content["state"]["search_state"]["query"] == "alpha"
    assert result.structured_content["state"]["search_state"]["tag_ids"] == ["tag_ops"]
    assert result.structured_content["state"]["search_state"]["tag_match_mode"] == "all"


@pytest.mark.asyncio
async def test_branch_search_iterates_vector_search_with_tag_filter(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_file_library(configured_settings)
    services = create_services(configured_settings)
    recorded_queries: list[str] = []
    recorded_filters: list[object] = []

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    async def fake_search_vector_store(
        _self,
        *,
        vector_store_id: str,
        query: str,
        max_results: int,
        rewrite_query: bool,
        filters,
    ) -> list[SearchHit]:
        assert vector_store_id == "vs_alpha"
        assert max_results == 2
        assert rewrite_query is True
        recorded_queries.append(query)
        recorded_filters.append(filters)
        if query == "ops handbook":
            return [
                SearchHit(
                    file_id="node_alpha",
                    file_title="Alpha Notes",
                    original_filename="alpha-notes.txt",
                    derived_artifact_id="artifact_alpha",
                    openai_file_id="artifact_file_alpha",
                    original_openai_file_id="file_alpha",
                    media_type="text/plain",
                    source_kind="document",
                    score=0.96,
                    text="Alpha notes explain how the file desk should work.",
                    tags=["Operations"],
                    attributes=None,
                ),
                SearchHit(
                    file_id="node_beta",
                    file_title="Beta Guide",
                    original_filename="beta-guide.txt",
                    derived_artifact_id="artifact_beta",
                    openai_file_id="artifact_file_beta",
                    original_openai_file_id="file_beta",
                    media_type="text/plain",
                    source_kind="document",
                    score=0.88,
                    text="Beta guide covers operational rollout details.",
                    tags=["Operations"],
                    attributes=None,
                ),
            ]
        if "Alpha Notes" in query:
            return [
                SearchHit(
                    file_id="node_gamma",
                    file_title="Gamma Checklist",
                    original_filename="gamma-checklist.txt",
                    derived_artifact_id="artifact_gamma",
                    openai_file_id="artifact_file_gamma",
                    original_openai_file_id="file_gamma",
                    media_type="text/plain",
                    source_kind="document",
                    score=0.81,
                    text="Gamma checklist expands on the same operating model.",
                    tags=["Operations"],
                    attributes=None,
                )
            ]
        if "Beta Guide" in query:
            return [
                SearchHit(
                    file_id="node_delta",
                    file_title="Delta Runbook",
                    original_filename="delta-runbook.txt",
                    derived_artifact_id="artifact_delta",
                    openai_file_id="artifact_file_delta",
                    original_openai_file_id="file_delta",
                    media_type="text/plain",
                    source_kind="document",
                    score=0.79,
                    text="Delta runbook follows the same rollout approach.",
                    tags=["Operations"],
                    attributes=None,
                )
            ]
        return []

    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.search_vector_store",
        fake_search_vector_store,
    )
    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )

    try:
        branch_result = await services.file_library.search_file_branches(
            clerk_user_id="user_123",
            query="ops handbook",
            tag_ids=["tag_ops"],
            tag_match_mode="all",
            descend=1,
            max_width=2,
        )
    finally:
        await services.close()

    assert branch_result.query == "ops handbook"
    assert branch_result.tag_ids == ["tag_ops"]
    assert [level.depth for level in branch_result.levels] == [0, 1]
    assert [hit.file_id for hit in branch_result.levels[0].hits] == ["node_alpha", "node_beta"]
    assert [hit.file_id for hit in branch_result.levels[1].hits] == ["node_gamma", "node_delta"]
    assert recorded_queries[0] == "ops handbook"
    assert any("Alpha Notes" in query for query in recorded_queries[1:])
    assert any("Beta Guide" in query for query in recorded_queries[1:])
    assert recorded_filters == [
        {"type": "eq", "key": "tag__operations", "value": True},
        {"type": "eq", "key": "tag__operations", "value": True},
        {"type": "eq", "key": "tag__operations", "value": True},
    ]


@pytest.mark.asyncio
async def test_fastapi_routes_cover_health_static_files_and_chat(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_file_library(configured_settings)

    async def verify_session_token(_self, token: str):
        if token != "test-session":
            return None
        return ClerkVerifiedSessionToken(
            subject="user_123",
            session_id="sess_123",
            token_id="tok_123",
            expiration=None,
        )

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    async def delete_file_noop(_self, *, file_id: str) -> None:
        return None

    async def fake_chat_process(_self, request: str | bytes | bytearray, context):
        assert context.clerk_user_id == "user_123"
        assert request
        return NonStreamingResult(b'{"ok":true}')

    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.verify_session_token",
        verify_session_token,
    )
    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.delete_file",
        delete_file_noop,
    )
    monkeypatch.setattr(
        "backend.chatkit_server.FileDeskChatKitServer.process",
        fake_chat_process,
    )

    app = create_fastapi_app(configured_settings)
    headers = {"Authorization": "Bearer test-session"}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            root = await client.get("/")
            assert root.status_code == 200
            assert "File Desk" in root.text

            auth_response = await client.get("/api/auth/me", headers=headers)
            assert auth_response.status_code == 200
            assert auth_response.json() == {
                "clerk_user_id": "user_123",
                "display_name": "File Desk Owner",
                "primary_email": "owner@example.com",
                "active": True,
                "role": "admin",
            }

            files_response = await client.get("/api/files", headers=headers)
            assert files_response.status_code == 200
            payload = files_response.json()
            assert payload["total_count"] == 1
            assert payload["files"][0]["display_title"] == "Alpha Notes"
            assert "outgoing_edge_count" not in payload["files"][0]
            assert "incoming_edge_count" not in payload["files"][0]
            assert "/api/files/node_alpha/content?token=" in payload["files"][0]["download_url"]

            detail_response = await client.get("/api/files/node_alpha", headers=headers)
            assert detail_response.status_code == 200
            detail_payload = detail_response.json()
            assert detail_payload["derived_artifacts"][0]["kind"] == "document_text"
            assert "outgoing_edges" not in detail_payload
            assert "incoming_edges" not in detail_payload

            tags_response = await client.get("/api/tags", headers=headers)
            assert tags_response.status_code == 200
            assert tags_response.json()["tags"][0]["name"] == "Operations"

            chat_response = await client.post(
                "/api/chatkit",
                headers=headers,
                content=b'{"type":"threads.create","params":{"input":{"content":[],"attachments":[],"inference_options":{}}},"metadata":{"selected_file_ids":["node_alpha"]}}',
            )
            assert chat_response.status_code == 200
            assert chat_response.json() == {"ok": True}

            delete_response = await client.delete("/api/files/node_alpha", headers=headers)
            assert delete_response.status_code == 200
            assert delete_response.json() == {"deleted_file_id": "node_alpha"}

            files_after_delete = await client.get("/api/files", headers=headers)
            assert files_after_delete.status_code == 200
            assert files_after_delete.json()["total_count"] == 0


@pytest.mark.asyncio
async def test_files_api_lists_library_items_with_explicit_browser_sort_and_metadata_filter(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_file_library(configured_settings)

    database = DatabaseManager(configured_settings)
    await database.ensure_ready()
    async with database.session() as session:
        owner = await session.scalar(select(AppUser).where(AppUser.clerk_user_id == "user_123"))
        assert owner is not None
        alpha_file = await session.get(LibraryFile, "node_alpha")
        assert alpha_file is not None
        alpha_file.created_at = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        alpha_file.updated_at = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

        beta_file = LibraryFile(
            id="node_beta",
            file_library_id="kb_alpha",
            uploaded_by_user_id=owner.id,
            display_title="Project Brief",
            original_filename="beta-brief.txt",
            media_type="text/plain",
            source_kind="document",
            status="ready",
            byte_size=256,
            original_mime_type="text/plain",
            openai_original_file_id="file_beta",
            created_at=datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
        )
        gamma_file = LibraryFile(
            id="node_gamma",
            file_library_id="kb_alpha",
            uploaded_by_user_id=owner.id,
            display_title="Gamma Memo",
            original_filename="gamma-memo.txt",
            media_type="application/pdf",
            source_kind="document",
            status="ready",
            byte_size=512,
            original_mime_type="application/pdf",
            openai_original_file_id="file_gamma",
            created_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
            updated_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
        )
        session.add(beta_file)
        session.add(gamma_file)
        await session.flush()

        session.add(
            DerivedArtifact(
                id="artifact_beta",
                file_id=beta_file.id,
                kind="document_text",
                openai_file_id="artifact_file_beta",
                text_content="Quarterly workflow notes live only in extracted text.",
                structured_payload=None,
                created_at=datetime(2026, 1, 2, 9, 5, tzinfo=UTC),
                updated_at=datetime(2026, 1, 2, 9, 5, tzinfo=UTC),
            )
        )
        await session.commit()
    await database.close()

    async def verify_session_token(_self, token: str):
        if token != "test-session":
            return None
        return ClerkVerifiedSessionToken(
            subject="user_123",
            session_id="sess_123",
            token_id="tok_123",
            expiration=None,
        )

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.verify_session_token",
        verify_session_token,
    )
    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )

    app = create_fastapi_app(configured_settings)
    headers = {"Authorization": "Bearer test-session"}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            newest_response = await client.get("/api/files", headers=headers)
            assert newest_response.status_code == 200
            assert [item["id"] for item in newest_response.json()["files"]] == ["node_gamma", "node_beta", "node_alpha"]

            filename_response = await client.get("/api/files?sort=filename", headers=headers)
            assert filename_response.status_code == 200
            assert [item["original_filename"] for item in filename_response.json()["files"]] == [
                "alpha-notes.txt",
                "beta-brief.txt",
                "gamma-memo.txt",
            ]

            metadata_filter_response = await client.get("/api/files?query=workflow", headers=headers)
            assert metadata_filter_response.status_code == 200
            assert metadata_filter_response.json()["total_count"] == 0


@pytest.mark.asyncio
async def test_arxiv_search_api_returns_normalized_candidates(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_file_library(configured_settings)

    async def verify_session_token(_self, token: str):
        if token != "test-session":
            return None
        return ClerkVerifiedSessionToken(
            subject="user_123",
            session_id="sess_123",
            token_id="tok_123",
            expiration=None,
        )

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    async def fake_search_arxiv_papers(
        _self,
        *,
        query: str,
        max_results: int,
    ) -> list[ArxivPaperCandidate]:
        assert query == "attention is all you need"
        assert max_results == 2
        return [
            ArxivPaperCandidate(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                summary="The original transformer paper.",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                abs_url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            )
        ]

    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.verify_session_token",
        verify_session_token,
    )
    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.search_arxiv_papers",
        fake_search_arxiv_papers,
    )

    app = create_fastapi_app(configured_settings)
    headers = {"Authorization": "Bearer test-session"}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/imports/arxiv/search",
                headers=headers,
                json={"query": "attention is all you need", "max_results": 2},
            )

    assert response.status_code == 200
    assert response.json() == {
        "query": "attention is all you need",
        "results": [
            {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "summary": "The original transformer paper.",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "abs_url": "https://arxiv.org/abs/1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            }
        ],
    }


@pytest.mark.asyncio
async def test_arxiv_import_api_downloads_and_ingests_pdf(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _seed_file_library(configured_settings)

    async def verify_session_token(_self, token: str):
        if token != "test-session":
            return None
        return ClerkVerifiedSessionToken(
            subject="user_123",
            session_id="sess_123",
            token_id="tok_123",
            expiration=None,
        )

    async def get_user_record(_self, clerk_user_id: str) -> ClerkUserRecord:
        assert clerk_user_id == "user_123"
        return ClerkUserRecord(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
        )

    async def fake_download_arxiv_pdf(
        _self,
        *,
        paper: ArxivPaperCandidate,
    ) -> DownloadedRemoteFile:
        pdf_path = tmp_path / "attention-is-all-you-need.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF")
        return DownloadedRemoteFile(
            local_path=pdf_path,
            filename="attention-is-all-you-need.pdf",
            media_type="application/pdf",
            source_url=paper.pdf_url,
        )

    async def fake_upload_original_file(_self, *, local_path: Path, purpose: str) -> str:
        assert local_path.name == "attention-is-all-you-need.pdf"
        assert purpose == "assistants"
        return "file_attention"

    async def fake_create_text_artifact_and_attach(
        _self,
        *,
        vector_store_id: str,
        filename: str,
        text_content: str,
        attributes,
    ) -> str:
        assert vector_store_id == "vs_alpha"
        assert filename.endswith(".arxiv_listing.md")
        assert "Attention Is All You Need" in text_content
        assert attributes["derived_kind"] == "arxiv_listing"
        return "artifact_attention"

    async def fake_attach_existing_file_to_vector_store(
        _self,
        *,
        vector_store_id: str,
        file_id: str,
        attributes,
    ) -> str:
        assert vector_store_id == "vs_alpha"
        assert file_id == "file_attention"
        assert attributes["derived_kind"] == "direct_file"
        return file_id

    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.verify_session_token",
        verify_session_token,
    )
    monkeypatch.setattr(
        "backend.clerk.ClerkAuthService.get_user_record",
        get_user_record,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.download_arxiv_pdf",
        fake_download_arxiv_pdf,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.upload_original_file",
        fake_upload_original_file,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.create_text_artifact_and_attach",
        fake_create_text_artifact_and_attach,
    )
    monkeypatch.setattr(
        "backend.file_library_gateway.OpenAIFileLibraryGateway.attach_existing_file_to_vector_store",
        fake_attach_existing_file_to_vector_store,
    )

    app = create_fastapi_app(configured_settings)
    headers = {"Authorization": "Bearer test-session"}
    paper = {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "summary": "The original transformer paper.",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "abs_url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    }

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/imports/arxiv",
                headers=headers,
                json={"paper": paper, "tag_ids": []},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file"]["display_title"] == "Attention Is All You Need"
    assert payload["file"]["original_filename"] == "attention-is-all-you-need.pdf"
    assert payload["file"]["derived_kinds"] == ["arxiv_listing"]

    database = DatabaseManager(configured_settings)
    await database.ensure_ready()
    async with database.session() as session:
        imported_file = await session.scalar(
            select(LibraryFile).where(LibraryFile.display_title == "Attention Is All You Need")
        )
        assert imported_file is not None
        imported_artifacts = (
            (await session.execute(select(DerivedArtifact).where(DerivedArtifact.file_id == imported_file.id)))
            .scalars()
            .all()
        )
        assert [artifact.kind for artifact in imported_artifacts] == ["arxiv_listing"]
        assert imported_artifacts[0].structured_payload == paper
    await database.close()


async def _seed_file_library(settings: AppSettings) -> None:
    database = DatabaseManager(settings)
    await database.ensure_ready()
    async with database.session() as session:
        app_user = AppUser(
            clerk_user_id="user_123",
            primary_email="owner@example.com",
            display_name="File Desk Owner",
            active=True,
            role="admin",
            last_seen_at=datetime.now(UTC),
        )
        session.add(app_user)
        await session.flush()

        file_library = FileLibrary(
            id="kb_alpha",
            user_id=app_user.id,
            title="Owner Library",
            description="Personal file desk",
            openai_vector_store_id="vs_alpha",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(file_library)
        await session.flush()

        tag = FileTag(
            id="tag_ops",
            file_library_id=file_library.id,
            name="Operations",
            slug="operations",
            color="#c46a32",
            created_at=datetime.now(UTC),
        )
        session.add(tag)

        file_record = LibraryFile(
            id="node_alpha",
            file_library_id=file_library.id,
            uploaded_by_user_id=app_user.id,
            display_title="Alpha Notes",
            original_filename="alpha-notes.txt",
            media_type="text/plain",
            source_kind="document",
            status="ready",
            byte_size=128,
            original_mime_type="text/plain",
            openai_original_file_id="file_alpha",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(file_record)
        await session.flush()

        session.add(
            FileTagLink(
                file_id=file_record.id,
                tag_id=tag.id,
            )
        )
        session.add(
            DerivedArtifact(
                id="artifact_alpha",
                file_id=file_record.id,
                kind="document_text",
                openai_file_id="artifact_file_alpha",
                text_content="Alpha notes explain how the file desk should work.",
                structured_payload=None,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_database_session_adapter_supports_async_get(configured_settings: AppSettings) -> None:
    database = DatabaseManager(configured_settings)
    await database.ensure_ready()
    async with database.session() as session:
        app_user = AppUser(
            clerk_user_id="user_async_get",
            primary_email="async-get@example.com",
            display_name="Async Get",
            active=True,
            role="admin",
            last_seen_at=datetime.now(UTC),
        )
        session.add(app_user)
        await session.flush()
        app_user_id = app_user.id
        await session.commit()

    async with database.session() as session:
        loaded_user = await session.get(AppUser, app_user_id)
        assert loaded_user is not None
        assert loaded_user.clerk_user_id == "user_async_get"
    await database.close()
