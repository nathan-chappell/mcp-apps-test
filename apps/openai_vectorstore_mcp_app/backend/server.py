from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from .auth import ClerkTokenVerifier
from .clerk import ClerkAuthService
from .db import DatabaseManager
from .knowledge_base_service import KnowledgeBaseService
from .logging import configure_logging
from .openai_gateway import OpenAIKnowledgeBaseGateway
from .qa_agent import KnowledgeBaseQuestionAnswerer
from .settings import AppSettings, get_settings
from .upload_sessions import KnowledgeBaseSessionService

logger = logging.getLogger(__name__)

RESOURCE_MIME_TYPE = "text/html;profile=mcp-app"
DESK_RESOURCE_URI = "ui://knowledge-base-desk/index.html"
DESK_UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "dist" / "mcp-app.html"
DEV_HOST_INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "ui" / "host-dist" / "dev-host" / "index.html"
)
DEV_HOST_SANDBOX_PATH = (
    Path(__file__).resolve().parent.parent
    / "ui"
    / "host-dist"
    / "dev-host"
    / "sandbox.html"
)


def create_server(settings: AppSettings | None = None) -> FastMCP:
    """Create the FastMCP server for the knowledge-base desk."""

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

    @asynccontextmanager
    async def server_lifespan(_: FastMCP[None]) -> AsyncIterator[None]:
        try:
            yield None
        finally:
            await gateway.close()
            await clerk_auth.close()
            await database.close()

    server = FastMCP(
        name=resolved_settings.app_name,
        instructions=(
            "Use query_knowledge_base to open the knowledge-base desk UI or run a grounded "
            "knowledge-base query. Use get_knowledge_base_info for the current graph state "
            "and optional node detail. Use update_knowledge_base, run_knowledge_base_command, "
            "and confirm_knowledge_base_command for app-driven graph mutations, uploads, and "
            "destructive confirmations."
        ),
        log_level=resolved_settings.log_level,
        stateless_http=True,
        lifespan=server_lifespan,
        auth=AuthSettings(
            issuer_url=resolved_settings.clerk_issuer_url,
            resource_server_url=resolved_settings.app_base_url,
            required_scopes=resolved_settings.mcp_required_scopes,
        ),
        token_verifier=ClerkTokenVerifier(clerk_auth),
    )

    @server.tool(
        name="query_knowledge_base",
        title="Query Knowledge Base",
        description=(
            "Open the knowledge-base desk UI and optionally run QA, raw file search, or "
            "branching search over the current graph and tag scope."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"ui": {"resourceUri": DESK_RESOURCE_URI}},
    )
    async def query_knowledge_base(
        query: Annotated[str | None, Field(min_length=1)] = None,
        mode: Literal["qa", "file_search", "branch_search"] = "qa",
        selected_node_id: Annotated[str | None, Field(min_length=1)] = None,
        graph_selection_mode: Literal["self", "children", "descendants"] = "self",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        media_types: list[str] | None = None,
        include_web: bool = False,
        rewrite_query: bool = True,
        branch_factor: Annotated[int, Field(ge=1, le=6)] = 3,
        depth: Annotated[int, Field(ge=1, le=4)] = 2,
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
    ) -> CallToolResult:
        payload = await knowledge_base_service.query_knowledge_base(
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            media_types=media_types or [],
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
            query=query,
            mode=mode,
        )
        summary = {
            "knowledge_base": "Opened the knowledge-base desk.",
            "qa": "Answered a knowledge-base question.",
            "file_search": "Ran knowledge-base file search.",
            "branch_search": "Ran knowledge-base branching search.",
        }[payload.kind]
        return _tool_result(
            summary,
            payload,
            meta={"ui": {"resourceUri": DESK_RESOURCE_URI}},
        )

    @server.tool(
        name="get_knowledge_base_info",
        title="Get Knowledge Base Info",
        description=(
            "Return the current knowledge-base graph state and optionally the full detail for "
            "one node."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_knowledge_base_info(
        selected_node_id: Annotated[str | None, Field(min_length=1)] = None,
        graph_selection_mode: Literal["self", "children", "descendants"] = "self",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        media_types: list[str] | None = None,
        include_web: bool = False,
        rewrite_query: bool = True,
        branch_factor: Annotated[int, Field(ge=1, le=6)] = 3,
        depth: Annotated[int, Field(ge=1, le=4)] = 2,
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
        detail_node_id: Annotated[str | None, Field(min_length=1)] = None,
    ) -> CallToolResult:
        payload = await knowledge_base_service.get_knowledge_base_info(
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            media_types=media_types or [],
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
            detail_node_id=detail_node_id,
        )
        return _tool_result("Loaded knowledge-base info.", payload)

    @server.tool(
        name="update_knowledge_base",
        title="Update Knowledge Base",
        description=(
            "Run app-driven typed graph mutations such as upload preparation, node rename, "
            "tag changes, and low-level edge updates."
        ),
        annotations=ToolAnnotations(
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def update_knowledge_base(
        action: Literal[
            "prepare_upload",
            "rename_node",
            "create_tag",
            "set_node_tags",
            "upsert_edge",
            "delete_edge",
            "delete_node",
        ],
        node_id: Annotated[str | None, Field(min_length=1)] = None,
        edge_id: Annotated[str | None, Field(min_length=1)] = None,
        from_node_id: Annotated[str | None, Field(min_length=1)] = None,
        to_node_id: Annotated[str | None, Field(min_length=1)] = None,
        tag_ids: list[str] | None = None,
        title: Annotated[str | None, Field(min_length=1)] = None,
        name: Annotated[str | None, Field(min_length=1)] = None,
        color: Annotated[str | None, Field(min_length=1)] = None,
        label: Annotated[str | None, Field(min_length=1)] = None,
    ) -> CallToolResult:
        payload = await knowledge_base_service.update_knowledge_base(
            action=action,
            node_id=node_id,
            edge_id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            tag_ids=tag_ids or [],
            title=title,
            name=name,
            color=color,
            label=label,
        )
        return _tool_result(f"Completed knowledge-base action {action}.", payload)

    @server.tool(
        name="run_knowledge_base_command",
        title="Run Knowledge Base Command",
        description=(
            "Interpret a natural-language command for one graph mutation such as renaming a "
            "node, creating a tag, connecting nodes, or requesting deletion."
        ),
        annotations=ToolAnnotations(
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def run_knowledge_base_command(
        command: Annotated[str, Field(min_length=1)],
        selected_node_id: Annotated[str | None, Field(min_length=1)] = None,
        graph_selection_mode: Literal["self", "children", "descendants"] = "self",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        media_types: list[str] | None = None,
        include_web: bool = False,
        rewrite_query: bool = True,
        branch_factor: Annotated[int, Field(ge=1, le=6)] = 3,
        depth: Annotated[int, Field(ge=1, le=4)] = 2,
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
    ) -> CallToolResult:
        payload = await knowledge_base_service.run_command(
            raw_command=command,
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            media_types=media_types or [],
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        return _tool_result("Processed the knowledge-base command.", payload)

    @server.tool(
        name="confirm_knowledge_base_command",
        title="Confirm Knowledge Base Command",
        description="Confirm a pending destructive knowledge-base command token.",
        annotations=ToolAnnotations(
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        meta={"ui": {"visibility": ["app"]}},
    )
    async def confirm_knowledge_base_command(
        token: Annotated[str, Field(min_length=1)],
        selected_node_id: Annotated[str | None, Field(min_length=1)] = None,
        graph_selection_mode: Literal["self", "children", "descendants"] = "self",
        tag_ids: list[str] | None = None,
        tag_match_mode: Literal["all", "any"] = "all",
        media_types: list[str] | None = None,
        include_web: bool = False,
        rewrite_query: bool = True,
        branch_factor: Annotated[int, Field(ge=1, le=6)] = 3,
        depth: Annotated[int, Field(ge=1, le=4)] = 2,
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
    ) -> CallToolResult:
        payload = await knowledge_base_service.confirm_command(
            token=token,
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids or [],
            tag_match_mode=tag_match_mode,
            media_types=media_types or [],
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        return _tool_result("Confirmed the knowledge-base command.", payload)

    @server.resource(
        DESK_RESOURCE_URI,
        name="knowledge_base_desk_resource",
        title="Knowledge Base Desk",
        description=(
            "Single-file React UI for graph navigation, uploads, filtering, search, branching "
            "search, and agent-driven graph mutations."
        ),
        mime_type=RESOURCE_MIME_TYPE,
    )
    def knowledge_base_desk_resource() -> str:
        if not DESK_UI_PATH.is_file():
            logger.warning(
                "mcp_app_resource_missing uri=%s path=%s",
                DESK_RESOURCE_URI,
                DESK_UI_PATH,
            )
            return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Knowledge Base Desk Build Required</title>
</head>
<body>
  <main style="font-family: sans-serif; padding: 24px;">
    <h1>Knowledge Base Desk UI not built yet</h1>
    <p>Run <code>npm install</code> and <code>npm run build:watch</code> in <code>apps/openai_vectorstore_mcp_app/ui</code>, then reopen the tool.</p>
  </main>
</body>
</html>
"""

        html = DESK_UI_PATH.read_text(encoding="utf-8")
        logger.info(
            "mcp_app_resource_ready uri=%s bytes=%s path=%s",
            DESK_RESOURCE_URI,
            len(html),
            DESK_UI_PATH,
        )
        return html

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

    @server.custom_route("/api/nodes/{node_id}/content", methods=["GET"])
    async def node_download_route(request: Request) -> Response:
        token = request.query_params.get("token")
        node_id = request.path_params["node_id"]
        if token is None:
            return PlainTextResponse("Missing token.", status_code=400)
        claims = session_tokens.verify_node_download(token)
        if claims is None or claims.node_id != node_id:
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

    @server.custom_route("/api/dev-auth-config", methods=["GET"])
    async def dev_auth_config_route(_request: Request) -> Response:
        return JSONResponse(
            {
                "clerk_publishable_key": resolved_settings.clerk_publishable_key,
                "app_name": resolved_settings.app_name,
            }
        )

    @server.custom_route("/", methods=["GET"])
    async def dev_host_root_route(_request: Request) -> Response:
        html = _load_dev_host_html(
            DEV_HOST_INDEX_PATH,
            label="dev_host_index",
            title="Knowledge Base Dev Host Build Required",
        )
        return Response(
            html,
            media_type="text/html",
            headers=_build_dev_host_headers(),
        )

    @server.custom_route("/index.html", methods=["GET"])
    async def dev_host_index_html_route(_request: Request) -> Response:
        html = _load_dev_host_html(
            DEV_HOST_INDEX_PATH,
            label="dev_host_index",
            title="Knowledge Base Dev Host Build Required",
        )
        return Response(
            html,
            media_type="text/html",
            headers=_build_dev_host_headers(),
        )

    @server.custom_route("/sandbox", methods=["GET"])
    async def dev_host_sandbox_route(request: Request) -> Response:
        csp_header: str | None = None
        raw_csp = request.query_params.get("csp")
        if raw_csp is not None:
            try:
                parsed_csp = json.loads(raw_csp)
            except json.JSONDecodeError:
                logger.warning("dev_host_sandbox_invalid_csp_json")
            else:
                if isinstance(parsed_csp, dict):
                    csp_header = _build_dev_host_sandbox_csp(parsed_csp)
                else:
                    logger.warning(
                        "dev_host_sandbox_invalid_csp_payload payload_type=%s",
                        type(parsed_csp).__name__,
                    )

        html = _load_dev_host_html(
            DEV_HOST_SANDBOX_PATH,
            label="dev_host_sandbox",
            title="Knowledge Base Sandbox Build Required",
        )
        return Response(
            html,
            media_type="text/html",
            headers=_build_dev_host_headers(content_security_policy=csp_header),
        )

    logger.info(
        "mcp_server_ready name=%s tools=%s",
        resolved_settings.app_name,
        [
            "query_knowledge_base",
            "get_knowledge_base_info",
            "update_knowledge_base",
            "run_knowledge_base_command",
            "confirm_knowledge_base_command",
        ],
    )
    return server


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


def _load_dev_host_html(path: Path, *, label: str, title: str) -> str:
    if not path.is_file():
        logger.warning("dev_host_asset_missing label=%s path=%s", label, path)
        return _render_dev_host_build_required_page(title=title, missing_path=path)

    html = path.read_text(encoding="utf-8")
    logger.info("dev_host_asset_ready label=%s bytes=%s path=%s", label, len(html), path)
    return html


def _render_dev_host_build_required_page(*, title: str, missing_path: Path) -> str:
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
    <p>The backend could not find the built dev-host asset at <code>{missing_path}</code>.</p>
    <p>Run <code>npm install</code> and <code>npm run build:watch</code> in <code>apps/openai_vectorstore_mcp_app/ui</code>, then reload this page.</p>
  </main>
</body>
</html>
"""


def _build_dev_host_headers(
    *, content_security_policy: str | None = None
) -> dict[str, str]:
    headers = {
        "cache-control": "no-cache, no-store, must-revalidate",
        "pragma": "no-cache",
        "expires": "0",
    }
    if content_security_policy is not None:
        headers["content-security-policy"] = content_security_policy
    return headers


def _build_dev_host_sandbox_csp(csp: dict[str, object]) -> str:
    resource_domains = " ".join(_sanitize_csp_domains(csp.get("resourceDomains")))
    connect_domains = " ".join(_sanitize_csp_domains(csp.get("connectDomains")))
    frame_domains = " ".join(_sanitize_csp_domains(csp.get("frameDomains")))
    base_uri_domains = " ".join(_sanitize_csp_domains(csp.get("baseUriDomains")))

    directives = [
        "default-src 'self' 'unsafe-inline'",
        f"script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: data: {resource_domains}".strip(),
        f"style-src 'self' 'unsafe-inline' blob: data: {resource_domains}".strip(),
        f"img-src 'self' data: blob: {resource_domains}".strip(),
        f"font-src 'self' data: blob: {resource_domains}".strip(),
        f"media-src 'self' data: blob: {resource_domains}".strip(),
        f"connect-src 'self' {connect_domains}".strip(),
        f"worker-src 'self' blob: {resource_domains}".strip(),
        f"frame-src {frame_domains}" if frame_domains else "frame-src 'none'",
        "object-src 'none'",
        f"base-uri {base_uri_domains}" if base_uri_domains else "base-uri 'none'",
    ]
    return "; ".join(directives)


def _sanitize_csp_domains(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    return [
        value
        for value in raw_value
        if isinstance(value, str) and not any(char in value for char in ";\r\n'\" ")
    ]


def _tool_result(
    summary: str,
    payload: BaseModel,
    *,
    meta: dict[str, object] | None = None,
) -> CallToolResult:
    return CallToolResult(
        _meta=meta,
        content=[TextContent(type="text", text=summary)],
        structuredContent=payload.model_dump(mode="json"),
    )
