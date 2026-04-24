from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp.server.auth import AccessToken, RemoteAuthProvider, TokenVerifier

from apps.openai_vectorstore_mcp_app.backend.auth import (
    pop_clerk_user_record,
    push_clerk_user_record,
)
from apps.openai_vectorstore_mcp_app.backend.clerk import ClerkAuthService, ClerkUserRecord
from apps.openai_vectorstore_mcp_app.backend.db import DatabaseManager
from apps.openai_vectorstore_mcp_app.backend.knowledge_base_service import KnowledgeBaseService
from apps.openai_vectorstore_mcp_app.backend.models import (
    AppUser,
    DerivedArtifact,
    KnowledgeBase,
    KnowledgeNode,
    KnowledgeNodeTag,
    KnowledgeTag,
)
from apps.openai_vectorstore_mcp_app.backend.schemas import (
    DocumentLibraryQueryResult,
    DocumentLibraryStateResult,
    FileSearchCallSummary,
    KnowledgeAnswerCitation,
    KnowledgeBaseContext,
    KnowledgeChatResult,
    SearchHit,
)
from apps.openai_vectorstore_mcp_app.backend.server import (
    ASK_RESOURCE_URI,
    LIBRARY_RESOURCE_URI,
    RESOURCE_MIME_TYPE,
    create_http_app,
    create_server,
)
from apps.openai_vectorstore_mcp_app.backend.settings import AppSettings
from apps.openai_vectorstore_mcp_app.backend.upload_sessions import (
    KnowledgeBaseSessionService,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "apps/openai_vectorstore_mcp_app/ui"
LIBRARY_UI_DIST_PATH = UI_DIR / "dist/library.html"
ASK_UI_DIST_PATH = UI_DIR / "dist/ask.html"


class MockTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        if token == "inactive-token":
            return AccessToken(
                token=token,
                client_id="test-client",
                scopes=[],
                claims={"sub": "user_inactive"},
            )
        if token == "active-token":
            return AccessToken(
                token=token,
                client_id="test-client",
                scopes=[],
                claims={"sub": "user_active"},
            )
        return None


class FakeGateway:
    def __init__(self) -> None:
        self.last_search_filters: object | None = None
        self.updated_file_attributes: list[dict[str, object]] = []

    async def close(self) -> None:
        return None

    async def search_vector_store(
        self,
        *,
        vector_store_id: str,
        query: str,
        max_results: int,
        rewrite_query: bool,
        filters: object | None,
    ) -> list[SearchHit]:
        self.last_search_filters = filters
        del vector_store_id, max_results, rewrite_query
        node_ids = extract_node_ids(filters)
        if not node_ids:
            node_ids = ["doc_alpha", "doc_beta", "doc_gamma"]
        hits: list[SearchHit] = []
        for node_id in node_ids:
            hits.append(
                SearchHit(
                    node_id=node_id,
                    node_title=f"Title {node_id}",
                    original_filename=f"{node_id}.md",
                    derived_artifact_id=f"artifact_{node_id}",
                    openai_file_id=f"file_{node_id}",
                    original_openai_file_id=f"original_{node_id}",
                    media_type="text/markdown",
                    source_kind="document",
                    score=0.82,
                    text=f"{query} matched in {node_id}",
                    tags=["mock"],
                    attributes={"node_id": node_id},
                )
            )
        return hits

    async def update_vector_store_file_attributes(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, str | float | bool],
    ) -> None:
        self.updated_file_attributes.append(
            {
                "vector_store_id": vector_store_id,
                "file_id": file_id,
                "attributes": attributes,
            }
        )


class FakeQuestionAnswerer:
    def __init__(self) -> None:
        self.last_filters: object | None = None

    async def ask(
        self,
        *,
        knowledge_base_id: str,
        vector_store_id: str,
        question: str,
        context: KnowledgeBaseContext,
        conversation_id: str | None,
        filters: object | None,
    ) -> KnowledgeChatResult:
        self.last_filters = filters
        del knowledge_base_id, vector_store_id, conversation_id
        node_ids = extract_node_ids(filters)
        if not node_ids:
            node_ids = context.scoped_node_ids
        hits = [
            SearchHit(
                node_id=node_id,
                node_title=f"Title {node_id}",
                original_filename=f"{node_id}.md",
                derived_artifact_id=f"artifact_{node_id}",
                openai_file_id=f"file_{node_id}",
                original_openai_file_id=f"original_{node_id}",
                media_type="text/markdown",
                source_kind="document",
                score=0.91,
                text=f"Answer evidence for {node_id}",
                tags=["mock"],
                attributes={"node_id": node_id},
            )
            for node_id in node_ids
        ]
        return KnowledgeChatResult(
            knowledge_base_id="kb_test",
            question=question,
            answer="Grounded answer from the matching documents.",
            model="fake-model",
            include_web=False,
            conversation_id="conversation_test",
            context=context,
            search_calls=[
                FileSearchCallSummary(
                    id="search_call_1",
                    status="completed",
                    queries=[question],
                    results=hits,
                )
            ],
            web_search_calls=[],
            citations=[
                KnowledgeAnswerCitation(
                    source="knowledge_base",
                    label=f"Title {node_ids[0]}",
                    node_id=node_ids[0],
                    node_title=f"Title {node_ids[0]}",
                    original_filename=f"{node_ids[0]}.md",
                    quote="Answer evidence for the first matching document.",
                )
            ]
            if node_ids
            else [],
        )


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-clerk-secret")
    monkeypatch.setenv("CLERK_OAUTH_CLIENT_ID", "test-clerk-client-id")
    monkeypatch.setenv("CLERK_OAUTH_CLIENT_SECRET", "test-clerk-client-secret")
    monkeypatch.setenv("APP_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("CLERK_ISSUER_URL", "https://clerk.example.com")
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'document-library.db'}",
    )
    monkeypatch.setenv("MCP_REQUIRED_SCOPES", "")
    return AppSettings()


@pytest.fixture
def test_auth_provider(configured_settings: AppSettings) -> RemoteAuthProvider:
    verifier = MockTokenVerifier(
        base_url=configured_settings.normalized_app_base_url,
        resource_base_url=configured_settings.normalized_app_base_url,
    )
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[str(configured_settings.clerk_issuer_url)],
        base_url=configured_settings.normalized_app_base_url,
        resource_base_url=configured_settings.normalized_app_base_url,
    )


@pytest.fixture(scope="session")
def built_ui() -> tuple[Path, Path]:
    subprocess.run(["npm", "run", "build"], check=True, cwd=UI_DIR)
    assert LIBRARY_UI_DIST_PATH.is_file()
    assert ASK_UI_DIST_PATH.is_file()
    return LIBRARY_UI_DIST_PATH, ASK_UI_DIST_PATH


def test_settings_load_from_env(configured_settings: AppSettings) -> None:
    settings = configured_settings
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert settings.clerk_secret_key.get_secret_value() == "test-clerk-secret"
    assert settings.clerk_oauth_client_id == "test-clerk-client-id"
    assert settings.clerk_oauth_client_secret.get_secret_value() == "test-clerk-client-secret"
    assert settings.normalized_mcp_resource_server_url.endswith("/mcp")
    assert settings.mcp_required_scopes == []


@pytest.mark.asyncio
async def test_server_exposes_expected_tools_and_resources(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_user_record(
        self: ClerkAuthService,
        clerk_user_id: str,
    ) -> ClerkUserRecord:
        return ClerkUserRecord(
            clerk_user_id=clerk_user_id,
            display_name="Active User",
            primary_email="active@example.com",
            active=True,
            role="admin",
        )

    monkeypatch.setattr(
        "apps.openai_vectorstore_mcp_app.backend.auth.get_access_token",
        lambda: AccessToken(
            token="direct-test-token",
            client_id="test-client",
            scopes=[],
            claims={"sub": "user_active"},
        ),
    )
    monkeypatch.setattr(ClerkAuthService, "get_user_record", fake_get_user_record)
    server = create_server(configured_settings)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {
        "open_document_library",
        "open_document_ask",
    }
    assert tools["open_document_library"].meta == {
        "ui": {"resourceUri": LIBRARY_RESOURCE_URI}
    }
    assert tools["open_document_ask"].meta == {
        "ui": {"resourceUri": ASK_RESOURCE_URI}
    }


@pytest.mark.asyncio
async def test_server_exposes_built_document_resources(
    built_ui: tuple[Path, Path],
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = built_ui
    async def fake_get_user_record(
        self: ClerkAuthService,
        clerk_user_id: str,
    ) -> ClerkUserRecord:
        return ClerkUserRecord(
            clerk_user_id=clerk_user_id,
            display_name="Active User",
            primary_email="active@example.com",
            active=True,
            role="admin",
        )

    monkeypatch.setattr(
        "apps.openai_vectorstore_mcp_app.backend.auth.get_access_token",
        lambda: AccessToken(
            token="direct-test-token",
            client_id="test-client",
            scopes=[],
            claims={"sub": "user_active"},
        ),
    )
    monkeypatch.setattr(ClerkAuthService, "get_user_record", fake_get_user_record)
    server = create_server(configured_settings)
    resources = await server.list_resources()
    resource_by_uri = {str(resource.uri): resource for resource in resources}

    assert resource_by_uri[LIBRARY_RESOURCE_URI].mime_type == RESOURCE_MIME_TYPE
    assert resource_by_uri[ASK_RESOURCE_URI].mime_type == RESOURCE_MIME_TYPE

    library_contents = await server.read_resource(LIBRARY_RESOURCE_URI)
    ask_contents = await server.read_resource(ASK_RESOURCE_URI)
    library_html = read_resource_text(library_contents)
    ask_html = read_resource_text(ask_contents)

    assert "<title>Document Library</title>" in library_html
    assert "document-library-root" in library_html
    assert "<title>Document Ask</title>" in ask_html
    assert "document-ask-root" in ask_html


@pytest.mark.asyncio
async def test_http_auth_boundary_and_active_user_tool_listing(
    configured_settings: AppSettings,
    test_auth_provider: RemoteAuthProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_user_record(
        self: ClerkAuthService,
        clerk_user_id: str,
    ) -> ClerkUserRecord:
        if clerk_user_id == "user_inactive":
            return ClerkUserRecord(
                clerk_user_id=clerk_user_id,
                display_name="Inactive User",
                primary_email="inactive@example.com",
                active=False,
                role="viewer",
            )
        return ClerkUserRecord(
            clerk_user_id=clerk_user_id,
            display_name="Active User",
            primary_email="active@example.com",
            active=True,
            role="admin",
        )

    monkeypatch.setattr(ClerkAuthService, "get_user_record", fake_get_user_record)

    server = create_server(configured_settings, auth_provider=test_auth_provider)
    app = create_http_app(server)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            headers = {"accept": "application/json, text/event-stream"}

            unauthorized_response = await client.post(
                "/mcp",
                json=initialize_request(request_id=1),
                headers=headers,
            )
            assert unauthorized_response.status_code == 401
            assert (
                'resource_metadata="http://localhost:8000/.well-known/oauth-protected-resource/mcp"'
                in unauthorized_response.headers["www-authenticate"]
            )

            protected_resource_metadata = await client.get(
                "/.well-known/oauth-protected-resource/mcp"
            )
            assert protected_resource_metadata.status_code == 200
            assert protected_resource_metadata.json() == {
                "resource": configured_settings.normalized_mcp_resource_server_url,
                "authorization_servers": [str(configured_settings.clerk_issuer_url)],
                "scopes_supported": [],
                "bearer_methods_supported": ["header"],
            }

            initialize_response = await client.post(
                "/mcp",
                json=initialize_request(request_id=2),
                headers={**headers, "authorization": "Bearer active-token"},
            )
            assert initialize_response.status_code == 200
            initialize_payload = parse_sse_json(initialize_response.text)
            assert initialize_payload["result"]["serverInfo"]["name"] == configured_settings.app_name

            initialized_response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                headers={**headers, "authorization": "Bearer active-token"},
            )
            assert initialized_response.status_code == 202

            tool_list_response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
                headers={**headers, "authorization": "Bearer active-token"},
            )
            assert tool_list_response.status_code == 200
            tool_list_payload = parse_sse_json(tool_list_response.text)
            tool_names = {tool["name"] for tool in tool_list_payload["result"]["tools"]}
            assert tool_names == {"open_document_library", "open_document_ask"}

            inactive_response = await client.post(
                "/mcp",
                json=initialize_request(request_id=4),
                headers={**headers, "authorization": "Bearer inactive-token"},
            )
            assert inactive_response.status_code == 200
            inactive_payload = parse_sse_json(inactive_response.text)
            assert "error" in inactive_payload
            assert "result" not in inactive_payload


@pytest.mark.asyncio
async def test_document_library_state_filters_metadata(
    configured_settings: AppSettings,
) -> None:
    service, database, _, _ = make_test_service(configured_settings)
    seeded = await seed_library(database)

    with active_user_context():
        result = await service.get_document_library_state(
            tag_ids=[seeded["tag_research_id"]],
            tag_match_mode="all",
            filename_query="roadmap",
            created_from=datetime(2026, 4, 19, tzinfo=UTC).date(),
            created_to=datetime(2026, 4, 24, tzinfo=UTC).date(),
            detail_document_id=seeded["doc_alpha_id"],
        )

    assert isinstance(result, DocumentLibraryStateResult)
    library = result.document_library_state.library
    assert library is not None
    assert [document.title for document in library.documents] == ["Roadmap Draft"]
    assert result.document_detail is not None
    assert result.document_detail.title == "Roadmap Draft"
    assert library.filters.filename_query == "roadmap"
    assert library.filters.tag_ids == [seeded["tag_research_id"]]
    assert library.filters.matching_document_ids == [seeded["doc_alpha_id"]]


@pytest.mark.asyncio
async def test_update_document_library_and_search_uses_filtered_document_ids(
    configured_settings: AppSettings,
) -> None:
    service, database, gateway, _ = make_test_service(configured_settings)
    seeded = await seed_library(database)

    with active_user_context():
        created_tag = await service.update_document_library(
            action="create_tag",
            document_id=None,
            tag_ids=[],
            name="finance",
            color="#0f766e",
        )
        assert created_tag.tag is not None

        updated_document = await service.update_document_library(
            action="set_document_tags",
            document_id=seeded["doc_beta_id"],
            tag_ids=[created_tag.tag.id],
            name=None,
            color=None,
        )
        assert updated_document.document is not None
        assert [tag.name for tag in updated_document.document.tags] == ["finance"]

        search_result = await service.query_document_library(
            query="incident",
            mode="search",
            tag_ids=[created_tag.tag.id],
            tag_match_mode="all",
            filename_query="incident",
            created_from=None,
            created_to=None,
        )

    assert isinstance(search_result, DocumentLibraryQueryResult)
    assert search_result.search_result is not None
    assert gateway.last_search_filters is not None
    assert extract_node_ids(gateway.last_search_filters) == [seeded["doc_beta_id"]]
    assert search_result.search_result.hits[0].document_id == seeded["doc_beta_id"]


@pytest.mark.asyncio
async def test_query_document_library_ask_uses_same_filtered_scope(
    configured_settings: AppSettings,
) -> None:
    service, database, _, question_answerer = make_test_service(configured_settings)
    seeded = await seed_library(database)

    with active_user_context():
        result = await service.query_document_library(
            query="How should we simplify the MVP?",
            mode="ask",
            tag_ids=[seeded["tag_product_id"]],
            tag_match_mode="all",
            filename_query="roadmap",
            created_from=None,
            created_to=None,
        )

    assert result.ask_result is not None
    assert result.ask_result.answer == "Grounded answer from the matching documents."
    assert extract_node_ids(question_answerer.last_filters) == [seeded["doc_alpha_id"]]
    assert result.ask_result.citations[0].document_id == seeded["doc_alpha_id"]


def initialize_request(*, request_id: int) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }


def parse_sse_json(payload: str) -> dict[str, Any]:
    data_prefix = "data: "
    for line in payload.splitlines():
        if line.startswith(data_prefix):
            return json.loads(line[len(data_prefix) :])
    raise AssertionError(f"Could not find an SSE data payload in: {payload!r}")


def read_resource_text(result: object) -> str:
    contents = getattr(result, "contents")
    if not isinstance(contents, Sequence) or len(contents) != 1:
        raise AssertionError("Expected a single resource content item.")
    content = getattr(contents[0], "content")
    if isinstance(content, str):
        return content
    return str(content, encoding="utf-8")


@contextmanager
def active_user_context() -> Any:
    token = push_clerk_user_record(
        ClerkUserRecord(
            clerk_user_id="user_active",
            display_name="Active User",
            primary_email="active@example.com",
            active=True,
            role="admin",
        )
    )
    try:
        yield
    finally:
        pop_clerk_user_record(token)


def make_test_service(
    settings: AppSettings,
) -> tuple[KnowledgeBaseService, DatabaseManager, FakeGateway, FakeQuestionAnswerer]:
    database = DatabaseManager(settings)
    gateway = FakeGateway()
    question_answerer = FakeQuestionAnswerer()
    service = KnowledgeBaseService(
        settings=settings,
        database=database,
        clerk_auth=ClerkAuthService(settings),
        session_tokens=KnowledgeBaseSessionService(settings),
        openai_gateway=gateway,  # type: ignore[arg-type]
        question_answerer=question_answerer,  # type: ignore[arg-type]
    )
    return service, database, gateway, question_answerer


async def seed_library(database: DatabaseManager) -> dict[str, str]:
    await database.ensure_ready()
    async with database.session() as session:
        user = AppUser(
            clerk_user_id="user_active",
            primary_email="active@example.com",
            display_name="Active User",
            active=True,
            role="admin",
            last_seen_at=datetime(2026, 4, 24, tzinfo=UTC),
        )
        session.add(user)
        await session.flush()

        knowledge_base = KnowledgeBase(
            user_id=user.id,
            title="Active User's Document Library",
            description="Test library",
            openai_vector_store_id="vs_test",
            created_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 4, 24, 12, 0, tzinfo=UTC),
        )
        session.add(knowledge_base)
        await session.flush()

        research_tag = KnowledgeTag(
            knowledge_base_id=knowledge_base.id,
            name="research",
            slug="research",
            color="#0f766e",
            created_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        product_tag = KnowledgeTag(
            knowledge_base_id=knowledge_base.id,
            name="product",
            slug="product",
            color="#2563eb",
            created_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        ops_tag = KnowledgeTag(
            knowledge_base_id=knowledge_base.id,
            name="ops",
            slug="ops",
            color="#b45309",
            created_at=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        session.add_all([research_tag, product_tag, ops_tag])
        await session.flush()

        doc_alpha = KnowledgeNode(
            id="doc_alpha",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Roadmap Draft",
            original_filename="roadmap-draft.md",
            media_type="text/markdown",
            source_kind="document",
            status="ready",
            byte_size=1200,
            original_mime_type="text/markdown",
            openai_original_file_id="file_doc_alpha",
            created_at=datetime(2026, 4, 20, 9, 0, tzinfo=UTC),
            updated_at=datetime(2026, 4, 20, 9, 0, tzinfo=UTC),
        )
        doc_beta = KnowledgeNode(
            id="doc_beta",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Incident Review",
            original_filename="incident-review.txt",
            media_type="text/plain",
            source_kind="document",
            status="ready",
            byte_size=980,
            original_mime_type="text/plain",
            openai_original_file_id="file_doc_beta",
            created_at=datetime(2026, 4, 22, 15, 0, tzinfo=UTC),
            updated_at=datetime(2026, 4, 22, 15, 0, tzinfo=UTC),
        )
        doc_gamma = KnowledgeNode(
            id="doc_gamma",
            knowledge_base_id=knowledge_base.id,
            created_by_user_id=user.id,
            display_title="Ops Checklist",
            original_filename="ops-checklist.md",
            media_type="text/markdown",
            source_kind="document",
            status="ready",
            byte_size=860,
            original_mime_type="text/markdown",
            openai_original_file_id="file_doc_gamma",
            created_at=datetime(2026, 4, 24, 8, 0, tzinfo=UTC),
            updated_at=datetime(2026, 4, 24, 8, 0, tzinfo=UTC),
        )
        session.add_all([doc_alpha, doc_beta, doc_gamma])
        await session.flush()

        session.add_all(
            [
                KnowledgeNodeTag(node_id=doc_alpha.id, tag_id=research_tag.id),
                KnowledgeNodeTag(node_id=doc_alpha.id, tag_id=product_tag.id),
                KnowledgeNodeTag(node_id=doc_beta.id, tag_id=ops_tag.id),
                KnowledgeNodeTag(node_id=doc_gamma.id, tag_id=research_tag.id),
            ]
        )
        session.add_all(
            [
                DerivedArtifact(
                    node_id=doc_alpha.id,
                    kind="document_text",
                    openai_file_id="artifact_doc_alpha",
                    text_content="Roadmap notes about simplifying the MVP around tags and metadata.",
                    structured_payload=None,
                    created_at=datetime(2026, 4, 20, 9, 0, tzinfo=UTC),
                    updated_at=datetime(2026, 4, 20, 9, 0, tzinfo=UTC),
                ),
                DerivedArtifact(
                    node_id=doc_beta.id,
                    kind="document_text",
                    openai_file_id="artifact_doc_beta",
                    text_content="Incident review notes and filename-driven recovery tactics.",
                    structured_payload=None,
                    created_at=datetime(2026, 4, 22, 15, 0, tzinfo=UTC),
                    updated_at=datetime(2026, 4, 22, 15, 0, tzinfo=UTC),
                ),
                DerivedArtifact(
                    node_id=doc_gamma.id,
                    kind="document_text",
                    openai_file_id="artifact_doc_gamma",
                    text_content="Checklist for operational rollout and library maintenance.",
                    structured_payload=None,
                    created_at=datetime(2026, 4, 24, 8, 0, tzinfo=UTC),
                    updated_at=datetime(2026, 4, 24, 8, 0, tzinfo=UTC),
                ),
            ]
        )
        await session.commit()

    return {
        "doc_alpha_id": "doc_alpha",
        "doc_beta_id": "doc_beta",
        "doc_gamma_id": "doc_gamma",
        "tag_research_id": research_tag.id,
        "tag_product_id": product_tag.id,
        "tag_ops_id": ops_tag.id,
    }


def extract_node_ids(filters: object | None) -> list[str]:
    if filters is None:
        return []
    if not isinstance(filters, dict):
        return []

    filter_type = filters.get("type")
    if filter_type == "eq" and filters.get("key") == "node_id":
        value = filters.get("value")
        return [str(value)] if isinstance(value, str) else []

    child_filters = filters.get("filters")
    if not isinstance(child_filters, list):
        return []

    node_ids: list[str] = []
    for child in child_filters:
        if not isinstance(child, dict):
            continue
        node_ids.extend(extract_node_ids(child))
    return list(dict.fromkeys(node_ids))
