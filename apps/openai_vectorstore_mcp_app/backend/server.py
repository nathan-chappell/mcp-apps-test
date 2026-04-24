from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider
from fastmcp.server.auth.providers.clerk import ClerkProvider
from fastmcp.tools import ToolResult
from mcp.types import TextContent, ToolAnnotations
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from .auth import RequireActiveClerkUserMiddleware
from .clerk import ClerkAuthService
from .db import DatabaseManager
from .knowledge_base_service import KnowledgeBaseService
from .logging import configure_logging
from .openai_gateway import OpenAIKnowledgeBaseGateway
from .qa_agent import KnowledgeBaseQuestionAnswerer
from .schemas import (
    DocumentLibraryQueryResult,
    DocumentLibraryStateResult,
    DocumentQueryMode,
    UpdateDocumentLibraryAction,
    UpdateDocumentLibraryResult,
)
from .settings import AppSettings, get_settings
from .upload_sessions import KnowledgeBaseSessionService

logger = logging.getLogger(__name__)

RESOURCE_MIME_TYPE = "text/html;profile=mcp-app"
LIBRARY_RESOURCE_URI = "ui://document-library/index.html"
ASK_RESOURCE_URI = "ui://document-ask/index.html"
LIBRARY_UI_PATH = (
    Path(__file__).resolve().parent.parent / "ui" / "dist" / "library.html"
)
ASK_UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "dist" / "ask.html"


@dataclass(slots=True)
class ServerResources:
    database: DatabaseManager
    clerk_auth: ClerkAuthService
    gateway: OpenAIKnowledgeBaseGateway

    async def close(self) -> None:
        await self.gateway.close()
        await self.clerk_auth.close()
        await self.database.close()


def create_server(
    settings: AppSettings | None = None,
    *,
    auth_provider: AuthProvider | None = None,
) -> FastMCP:
    """Create the FastMCP server for the document-library app."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    database = DatabaseManager(resolved_settings)
    clerk_auth = ClerkAuthService(resolved_settings)
    session_tokens = KnowledgeBaseSessionService(resolved_settings)
    gateway = OpenAIKnowledgeBaseGateway(resolved_settings)
    question_answerer = KnowledgeBaseQuestionAnswerer(resolved_settings)
    knowledge_base_service = KnowledgeBaseService(
        settings=resolved_settings,
        database=database,
        clerk_auth=clerk_auth,
        session_tokens=session_tokens,
        openai_gateway=gateway,
        question_answerer=question_answerer,
    )

    resources = ServerResources(
        database=database,
        clerk_auth=clerk_auth,
        gateway=gateway,
    )
    resolved_auth_provider = auth_provider or _create_clerk_auth_provider(resolved_settings)
    server = FastMCP(
        name=resolved_settings.app_name,
        instructions=(
            "Use open_document_library to browse and manage the document library UI. "
            "Use open_document_ask to search the library or ask grounded questions. "
            "The helper tools get_document_library_state, query_document_library, and "
            "update_document_library are app-only."
        ),
        auth=resolved_auth_provider,
        middleware=[RequireActiveClerkUserMiddleware(clerk_auth)],
    )
    setattr(server, "_server_resources", resources)

    @server.tool(
        name="open_document_library",
        title="Open Document Library",
        description=(
            "Open the document library UI for tag filtering, filename/date filtering, "
            "metadata review, uploads, and tag management."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"ui": {"resourceUri": LIBRARY_RESOURCE_URI}},
    )
    async def open_document_library(
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        filename_query: Annotated[str | None, Field(min_length=1)] = None,
        created_from: date | None = None,
        created_to: date | None = None,
        detail_document_id: Annotated[str | None, Field(min_length=1)] = None,
    ) -> ToolResult:
        payload = await knowledge_base_service.get_document_library_state(
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            filename_query=filename_query,
            created_from=created_from,
            created_to=created_to,
            detail_document_id=detail_document_id,
        )
        return _tool_result(
            "Opened the document library.",
            payload,
            meta={"ui": {"resourceUri": LIBRARY_RESOURCE_URI}},
        )

    @server.tool(
        name="open_document_ask",
        title="Open Document Ask",
        description=(
            "Open the ask UI to search the filtered library or ask grounded questions "
            "against matching documents."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"ui": {"resourceUri": ASK_RESOURCE_URI}},
    )
    async def open_document_ask(
        query: Annotated[str | None, Field(min_length=1)] = None,
        mode: DocumentQueryMode = "search",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        filename_query: Annotated[str | None, Field(min_length=1)] = None,
        created_from: date | None = None,
        created_to: date | None = None,
    ) -> ToolResult:
        payload: DocumentLibraryQueryResult
        if query is None or not query.strip():
            state_payload = await knowledge_base_service.get_document_library_state(
                tag_ids=tag_ids or [],
                tag_match_mode=tag_match_mode,
                filename_query=filename_query,
                created_from=created_from,
                created_to=created_to,
                detail_document_id=None,
            )
            payload = DocumentLibraryQueryResult(
                mode=mode,
                document_library_state=state_payload.document_library_state,
            )
        else:
            payload = await knowledge_base_service.query_document_library(
                query=query,
                mode=mode,
                tag_ids=tag_ids or [],
                tag_match_mode=tag_match_mode,
                filename_query=filename_query,
                created_from=created_from,
                created_to=created_to,
            )
        return _tool_result(
            "Opened document search and ask.",
            payload,
            meta={"ui": {"resourceUri": ASK_RESOURCE_URI}},
        )

    @server.tool(
        name="get_document_library_state",
        title="Get Document Library State",
        description="Return the current filtered library state and optional document detail.",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def get_document_library_state(
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        filename_query: Annotated[str | None, Field(min_length=1)] = None,
        created_from: date | None = None,
        created_to: date | None = None,
        detail_document_id: Annotated[str | None, Field(min_length=1)] = None,
    ) -> DocumentLibraryStateResult:
        return await knowledge_base_service.get_document_library_state(
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            filename_query=filename_query,
            created_from=created_from,
            created_to=created_to,
            detail_document_id=detail_document_id,
        )

    @server.tool(
        name="query_document_library",
        title="Query Document Library",
        description=(
            "Run filtered search or grounded QA against the document library. "
            "This is intended for app-driven refreshes."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def query_document_library(
        query: Annotated[str, Field(min_length=1)],
        mode: DocumentQueryMode = "search",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        filename_query: Annotated[str | None, Field(min_length=1)] = None,
        created_from: date | None = None,
        created_to: date | None = None,
    ) -> DocumentLibraryQueryResult:
        return await knowledge_base_service.query_document_library(
            query=query,
            mode=mode,
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            filename_query=filename_query,
            created_from=created_from,
            created_to=created_to,
        )

    @server.tool(
        name="update_document_library",
        title="Update Document Library",
        description=(
            "Prepare uploads, create tags, or update the tags on a document. "
            "This tool is app-only."
        ),
        annotations=ToolAnnotations(
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def update_document_library(
        action: UpdateDocumentLibraryAction,
        document_id: Annotated[str | None, Field(min_length=1)] = None,
        tag_ids: list[str] | None = None,
        name: Annotated[str | None, Field(min_length=1)] = None,
        color: Annotated[str | None, Field(min_length=1)] = None,
    ) -> UpdateDocumentLibraryResult:
        return await knowledge_base_service.update_document_library(
            action=action,
            document_id=document_id,
            tag_ids=tag_ids or [],
            name=name,
            color=color,
        )

    @server.resource(
        LIBRARY_RESOURCE_URI,
        name="document_library_resource",
        title="Document Library",
        description="Single-file React UI for browsing documents, tags, and upload flows.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    def document_library_resource() -> str:
        return _load_ui_html(
            path=LIBRARY_UI_PATH,
            resource_uri=LIBRARY_RESOURCE_URI,
            title="Document Library Build Required",
        )

    @server.resource(
        ASK_RESOURCE_URI,
        name="document_ask_resource",
        title="Document Ask",
        description="Single-file React UI for filtered search and grounded QA.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    def document_ask_resource() -> str:
        return _load_ui_html(
            path=ASK_UI_PATH,
            resource_uri=ASK_RESOURCE_URI,
            title="Document Ask Build Required",
        )

    @server.custom_route("/api/uploads", methods=["POST"])
    async def upload_route(request: Request) -> Response:
        form = await request.form()
        upload_token = form.get("upload_token")
        if not isinstance(upload_token, str) or not upload_token:
            return PlainTextResponse("Missing upload_token.", status_code=400)

        claims = session_tokens.verify_upload_session(upload_token)
        if claims is None:
            return PlainTextResponse("Upload token expired or invalid.", status_code=403)

        uploaded_file = form.get("file")
        if not isinstance(uploaded_file, UploadFile):
            return PlainTextResponse("Missing file upload.", status_code=400)

        tag_ids = _parse_tag_ids(form)
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await uploaded_file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
        try:
            payload = await knowledge_base_service.ingest_upload(
                claims=claims,
                local_path=temp_path,
                filename=uploaded_file.filename or temp_path.name,
                declared_media_type=uploaded_file.content_type,
                tag_ids=tag_ids,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        return JSONResponse(payload.model_dump(mode="json"))

    @server.custom_route("/api/documents/{document_id}/content", methods=["GET"])
    async def document_download_route(request: Request) -> Response:
        token = request.query_params.get("token")
        document_id = request.path_params["document_id"]
        if token is None:
            return PlainTextResponse("Missing token.", status_code=400)
        claims = session_tokens.verify_node_download(token)
        if claims is None or claims.node_id != document_id:
            return PlainTextResponse("Invalid download token.", status_code=403)
        detail, payload = await knowledge_base_service.download_node_bytes(claims=claims)
        headers = {
            "content-disposition": f'inline; filename="{detail.original_filename}"',
        }
        return Response(
            payload,
            media_type=detail.original_mime_type or detail.media_type,
            headers=headers,
        )

    @server.custom_route("/", methods=["GET"])
    async def info_route(_request: Request) -> Response:
        return Response(
            _root_info_page(resolved_settings),
            media_type="text/html",
            headers={
                "cache-control": "no-cache, no-store, must-revalidate",
                "pragma": "no-cache",
                "expires": "0",
            },
        )

    logger.info(
        "mcp_server_ready name=%s tools=%s",
        resolved_settings.app_name,
        [
            "open_document_library",
            "open_document_ask",
            "get_document_library_state",
            "query_document_library",
            "update_document_library",
        ],
    )
    return server


def create_http_app(server: FastMCP) -> Starlette:
    app = server.http_app(
        path="/mcp",
        transport="streamable-http",
        stateless_http=True,
    )
    original_lifespan_context = app.router.lifespan_context
    resources = _get_server_resources(server)

    @asynccontextmanager
    async def combined_lifespan(starlette_app: Starlette) -> AsyncIterator[None]:
        try:
            async with original_lifespan_context(starlette_app):
                yield
        finally:
            await resources.close()

    app.router.lifespan_context = combined_lifespan
    return app


def _create_clerk_auth_provider(settings: AppSettings) -> AuthProvider:
    return ClerkProvider(
        domain=settings.clerk_domain,
        client_id=settings.clerk_oauth_client_id,
        client_secret=settings.clerk_oauth_client_secret.get_secret_value(),
        base_url=settings.normalized_app_base_url,
        resource_base_url=settings.normalized_app_base_url,
        issuer_url=settings.normalized_app_base_url,
        required_scopes=settings.mcp_required_scopes,
        require_authorization_consent="external",
    )


def _load_ui_html(*, path: Path, resource_uri: str, title: str) -> str:
    if not path.is_file():
        logger.warning(
            "mcp_app_resource_missing uri=%s path=%s",
            resource_uri,
            path,
        )
        return _build_required_page(title=title)

    html = path.read_text(encoding="utf-8")
    logger.info(
        "mcp_app_resource_ready uri=%s bytes=%s path=%s",
        resource_uri,
        len(html),
        path,
    )
    return html


def _build_required_page(*, title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
</head>
<body>
  <main style="font-family: sans-serif; padding: 24px;">
    <h1>{title}</h1>
    <p>Run <code>npm install</code> and <code>npm run build:watch</code> in <code>apps/openai_vectorstore_mcp_app/ui</code>, then reopen the tool.</p>
  </main>
</body>
</html>
"""


def _root_info_page(settings: AppSettings) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{settings.app_name}</title>
</head>
<body style="font-family: sans-serif; padding: 32px; line-height: 1.5;">
  <main style="max-width: 760px;">
    <h1>Document Library MCP Server</h1>
    <p>This server is intended to be used through a real MCP host or the MCP Inspector.</p>
    <p>Connect your host to <code>{settings.normalized_app_base_url}/mcp</code>. Unauthenticated MCP requests will return a standards-compliant <code>401</code> with protected-resource metadata so the client can discover Clerk OAuth.</p>
    <p>For local frontend preview work, use <code>npm run dev:mock</code> in <code>apps/openai_vectorstore_mcp_app/ui</code>.</p>
  </main>
</body>
</html>
"""


def _parse_tag_ids(form) -> list[str]:
    raw_values = [value for value in form.getlist("tag_ids") if isinstance(value, str)]
    if not raw_values:
        return []
    if len(raw_values) == 1:
        raw_value = raw_values[0].strip()
        if not raw_value:
            return []
        if raw_value.startswith("["):
            parsed = json.loads(raw_value)
            if not isinstance(parsed, list):
                raise ValueError("tag_ids must decode to a JSON array.")
            return [str(item) for item in parsed if str(item)]
        if "," in raw_value:
            return [part.strip() for part in raw_value.split(",") if part.strip()]
    return [value.strip() for value in raw_values if value.strip()]


def _tool_result(
    summary: str,
    payload: BaseModel,
    *,
    meta: dict[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=summary)],
        structured_content=payload.model_dump(mode="json"),
        meta=meta,
    )


def _get_server_resources(server: FastMCP) -> ServerResources:
    resources = getattr(server, "_server_resources", None)
    if not isinstance(resources, ServerResources):
        raise RuntimeError("FastMCP server was missing its resource bundle.")
    return resources
