from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import re
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp.server.dependencies import get_access_token
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .auth import clerk_user_id_from_access_token, get_current_clerk_user_record
from .clerk import ClerkAuthService, ClerkUserRecord
from .db import DatabaseManager
from .models import (
    AppUser,
    DerivedArtifact,
    KnowledgeBase,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeNodeTag,
    KnowledgeTag,
)
from .openai_gateway import (
    OpenAIKnowledgeBaseGateway,
    build_filter_groups,
    build_searchable_attributes,
    guess_media_type,
)
from .qa_agent import KnowledgeBaseQuestionAnswerer
from .schemas import (
    BranchSearchNode,
    DocumentAskResult,
    DocumentCitation,
    DocumentDetail,
    DocumentFilters,
    DocumentLibraryCapabilities,
    DocumentLibraryQueryResult,
    DocumentLibraryState,
    DocumentLibraryStateResult,
    DocumentLibrarySummary,
    DocumentLibraryViewState,
    DocumentQueryMode,
    DocumentSearchHit,
    DocumentSearchResult,
    DocumentSummary,
    DocumentUploadFinalizeResult,
    CommandParserKind,
    DerivedArtifactSummary,
    GraphSelectionMode,
    KnowledgeAccessState,
    KnowledgeAnswerCitation,
    KnowledgeBaseCapabilities,
    KnowledgeBaseContext,
    KnowledgeBaseDeskState,
    KnowledgeBaseState,
    KnowledgeBaseSummary,
    KnowledgeBranchSearchResult,
    KnowledgeChatResult,
    KnowledgeBaseCommandResult,
    KnowledgeEdgeSummary,
    KnowledgeFileSearchResult,
    KnowledgeInfoResult,
    KnowledgeNodeDetail,
    KnowledgeNodeSummary,
    KnowledgeQueryMode,
    KnowledgeQueryResult,
    KnowledgeTagSummary,
    PendingCommandResult,
    SearchHit,
    TagMatchMode,
    UpdateDocumentLibraryAction,
    UpdateDocumentLibraryResult,
    UpdateKnowledgeBaseAction,
    UpdateKnowledgeBaseResult,
    UserSummary,
)
from .settings import AppSettings
from .upload_sessions import (
    KnowledgeBaseSessionService,
    NodeDownloadClaims,
    UploadSessionClaims,
)

if TYPE_CHECKING:
    from .command_agent import CommandExecutionResult, KnowledgeBaseCommandAgent

TEXT_EXTENSIONS = {
    ".c",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".html",
    ".htm",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".markdown",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".scss",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(slots=True)
class ResolvedUser:
    app_user: AppUser
    summary: UserSummary


class KnowledgeBaseService:
    """Owns knowledge-base state, graph persistence, and ingestion orchestration."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        database: DatabaseManager,
        clerk_auth: ClerkAuthService,
        session_tokens: KnowledgeBaseSessionService,
        openai_gateway: OpenAIKnowledgeBaseGateway,
        question_answerer: KnowledgeBaseQuestionAnswerer,
    ) -> None:
        self._settings = settings
        self._database = database
        self._clerk_auth = clerk_auth
        self._session_tokens = session_tokens
        self._openai_gateway = openai_gateway
        self._question_answerer = question_answerer
        self._command_agent: KnowledgeBaseCommandAgent | None = None

    async def get_document_library_state(
        self,
        *,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        filename_query: str | None,
        created_from: date | None,
        created_to: date | None,
        detail_document_id: str | None,
    ) -> DocumentLibraryStateResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            filters = await self._document_filters_for_request(
                session,
                knowledge_base=knowledge_base,
                tag_ids=tag_ids,
                tag_match_mode=tag_match_mode,
                filename_query=filename_query,
                created_from=created_from,
                created_to=created_to,
            )
            document_library_state = await self._document_library_view_state(
                session,
                resolved_user=resolved_user,
                knowledge_base=knowledge_base,
                filters=filters,
            )
            document_detail: DocumentDetail | None = None
            if (
                detail_document_id is not None
                and detail_document_id in set(filters.matching_document_ids)
                and resolved_user.summary.active
            ):
                node = await self._node_for_user(
                    session,
                    resolved_user=resolved_user,
                    node_id=detail_document_id,
                )
                detail = await self._node_detail(
                    session,
                    knowledge_base=knowledge_base,
                    node=node,
                    clerk_user_id=resolved_user.summary.clerk_user_id,
                )
                document_detail = self._document_detail_from_node_detail(detail)
            return DocumentLibraryStateResult(
                document_library_state=document_library_state,
                document_detail=document_detail,
            )

    async def query_document_library(
        self,
        *,
        query: str,
        mode: DocumentQueryMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        filename_query: str | None,
        created_from: date | None,
        created_to: date | None,
    ) -> DocumentLibraryQueryResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required.")

        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            filters = await self._document_filters_for_request(
                session,
                knowledge_base=knowledge_base,
                tag_ids=tag_ids,
                tag_match_mode=tag_match_mode,
                filename_query=filename_query,
                created_from=created_from,
                created_to=created_to,
            )
            empty_state = await self._document_library_view_state(
                session,
                resolved_user=resolved_user,
                knowledge_base=knowledge_base,
                filters=filters,
            )

            if not filters.matching_document_ids:
                if mode == "ask":
                    return DocumentLibraryQueryResult(
                        mode=mode,
                        document_library_state=empty_state,
                        ask_result=DocumentAskResult(
                            query=normalized_query,
                            answer="No documents match the current filters yet.",
                            model=self._settings.openai_agent_model,
                            conversation_id=knowledge_base.openai_conversation_id or "",
                            citations=[],
                            hits=[],
                        ),
                    )
                return DocumentLibraryQueryResult(
                    mode=mode,
                    document_library_state=empty_state,
                    search_result=DocumentSearchResult(
                        query=normalized_query,
                        hits=[],
                        total_hits=0,
                    ),
                )

            if knowledge_base.openai_vector_store_id is None:
                if mode == "ask":
                    return DocumentLibraryQueryResult(
                        mode=mode,
                        document_library_state=empty_state,
                        ask_result=DocumentAskResult(
                            query=normalized_query,
                            answer="There are no indexed documents in this library yet.",
                            model=self._settings.openai_agent_model,
                            conversation_id=knowledge_base.openai_conversation_id or "",
                            citations=[],
                            hits=[],
                        ),
                    )
                return DocumentLibraryQueryResult(
                    mode=mode,
                    document_library_state=empty_state,
                    search_result=DocumentSearchResult(
                        query=normalized_query,
                        hits=[],
                        total_hits=0,
                    ),
                )

            vector_store_filters = self._vector_store_filters_for_documents(
                knowledge_base=knowledge_base,
                document_ids=filters.matching_document_ids,
            )

            if mode == "search":
                hits = await self._openai_gateway.search_vector_store(
                    vector_store_id=knowledge_base.openai_vector_store_id,
                    query=normalized_query,
                    max_results=self._settings.openai_file_search_max_results,
                    rewrite_query=True,
                    filters=vector_store_filters,
                )
                refreshed_state = await self._document_library_view_state(
                    session,
                    resolved_user=resolved_user,
                    knowledge_base=knowledge_base,
                    filters=filters,
                )
                return DocumentLibraryQueryResult(
                    mode=mode,
                    document_library_state=refreshed_state,
                    search_result=DocumentSearchResult(
                        query=normalized_query,
                        hits=[self._document_search_hit(hit) for hit in hits],
                        total_hits=len(hits),
                    ),
                )

            ask_context = KnowledgeBaseContext(
                tag_ids=filters.tag_ids,
                selected_tag_names=filters.selected_tag_names,
                tag_match_mode=filters.tag_match_mode,
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=self._settings.openai_file_search_max_results,
                visible_node_ids=filters.matching_document_ids,
                scoped_node_ids=filters.matching_document_ids,
            )
            chat_result = await self._question_answerer.ask(
                knowledge_base_id=knowledge_base.id,
                vector_store_id=knowledge_base.openai_vector_store_id,
                question=normalized_query,
                context=ask_context,
                conversation_id=knowledge_base.openai_conversation_id,
                filters=vector_store_filters,
            )
            knowledge_base.openai_conversation_id = chat_result.conversation_id
            knowledge_base.updated_at = _utcnow()
            await session.commit()

            refreshed_state = await self._document_library_view_state(
                session,
                resolved_user=resolved_user,
                knowledge_base=knowledge_base,
                filters=filters,
            )
            return DocumentLibraryQueryResult(
                mode=mode,
                document_library_state=refreshed_state,
                ask_result=DocumentAskResult(
                    query=normalized_query,
                    answer=chat_result.answer,
                    model=chat_result.model,
                    conversation_id=chat_result.conversation_id,
                    citations=[
                        self._document_citation(citation)
                        for citation in chat_result.citations
                    ],
                    hits=self._document_hits_from_chat_result(chat_result),
                ),
            )

    async def update_document_library(
        self,
        *,
        action: UpdateDocumentLibraryAction,
        document_id: str | None,
        tag_ids: list[str],
        name: str | None,
        color: str | None,
    ) -> UpdateDocumentLibraryResult:
        if action == "prepare_upload":
            upload_session = await self.issue_upload_session()
            return UpdateDocumentLibraryResult(
                action=action,
                upload_session=upload_session,
            )

        if action == "create_tag":
            if name is None or not name.strip():
                raise ValueError("A tag name is required.")
            tag = await self.create_tag(
                name=name.strip(),
                color=color.strip() if color else None,
            )
            return UpdateDocumentLibraryResult(
                action=action,
                tag=tag,
            )

        if document_id is None:
            raise ValueError("document_id is required for set_document_tags.")
        updated = await self.set_node_tags(node_id=document_id, tag_ids=tag_ids)
        return UpdateDocumentLibraryResult(
            action=action,
            document=self._document_summary_from_node_summary(updated),
        )

    async def get_knowledge_base_info(
        self,
        *,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
        detail_node_id: str | None,
    ) -> KnowledgeInfoResult:
        desk_state = await self.get_knowledge_base_state(
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            media_types=media_types,
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        node_detail: KnowledgeNodeDetail | None = None
        if detail_node_id is not None and desk_state.access.status == "active":
            node_detail = await self.get_node_detail(node_id=detail_node_id)
        return KnowledgeInfoResult(
            knowledge_base_state=desk_state,
            node_detail=node_detail,
        )

    async def get_knowledge_base_state(
        self,
        *,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> KnowledgeBaseDeskState:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            return await self._build_knowledge_base_state(
                session,
                resolved_user=resolved_user,
                knowledge_base=knowledge_base,
                selected_node_id=selected_node_id,
                graph_selection_mode=graph_selection_mode,
                tag_ids=tag_ids,
                tag_match_mode=tag_match_mode,
                media_types=media_types,
                include_web=include_web,
                rewrite_query=rewrite_query,
                branch_factor=branch_factor,
                depth=depth,
                max_results=max_results,
            )

    async def query_knowledge_base(
        self,
        *,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
        query: str | None,
        mode: KnowledgeQueryMode,
    ) -> KnowledgeQueryResult:
        desk_state = await self.get_knowledge_base_state(
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            media_types=media_types,
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        knowledge_base_state = desk_state.knowledge_base
        if knowledge_base_state is None or query is None or not query.strip():
            return KnowledgeQueryResult(
                kind="knowledge_base",
                knowledge_base_state=desk_state,
            )

        context = knowledge_base_state.context
        normalized_query = query.strip()

        if mode == "file_search":
            file_search_result = await self.knowledge_base_file_search(
                query=normalized_query,
                context=context,
            )
            refreshed_state = await self.get_knowledge_base_state(
                selected_node_id=context.selected_node_id,
                graph_selection_mode=context.graph_selection_mode,
                tag_ids=context.tag_ids,
                tag_match_mode=context.tag_match_mode,
                media_types=context.media_types,
                include_web=context.include_web,
                rewrite_query=context.rewrite_query,
                branch_factor=context.branch_factor,
                depth=context.depth,
                max_results=context.max_results,
            )
            return KnowledgeQueryResult(
                kind="file_search",
                knowledge_base_state=refreshed_state,
                file_search_result=file_search_result,
            )

        if mode == "branch_search":
            branch_search_result = await self.knowledge_base_branch_search(
                query=normalized_query,
                context=context,
            )
            refreshed_state = await self.get_knowledge_base_state(
                selected_node_id=context.selected_node_id,
                graph_selection_mode=context.graph_selection_mode,
                tag_ids=context.tag_ids,
                tag_match_mode=context.tag_match_mode,
                media_types=context.media_types,
                include_web=context.include_web,
                rewrite_query=context.rewrite_query,
                branch_factor=context.branch_factor,
                depth=context.depth,
                max_results=context.max_results,
            )
            return KnowledgeQueryResult(
                kind="branch_search",
                knowledge_base_state=refreshed_state,
                branch_search_result=branch_search_result,
            )

        chat_result = await self.knowledge_base_chat(
            question=normalized_query,
            context=context,
        )
        refreshed_state = await self.get_knowledge_base_state(
            selected_node_id=context.selected_node_id,
            graph_selection_mode=context.graph_selection_mode,
            tag_ids=context.tag_ids,
            tag_match_mode=context.tag_match_mode,
            media_types=context.media_types,
            include_web=context.include_web,
            rewrite_query=context.rewrite_query,
            branch_factor=context.branch_factor,
            depth=context.depth,
            max_results=context.max_results,
        )
        return KnowledgeQueryResult(
            kind="qa",
            knowledge_base_state=refreshed_state,
            chat_result=chat_result,
        )

    async def update_knowledge_base(
        self,
        *,
        action: UpdateKnowledgeBaseAction,
        node_id: str | None,
        edge_id: str | None,
        from_node_id: str | None,
        to_node_id: str | None,
        tag_ids: list[str],
        title: str | None,
        name: str | None,
        color: str | None,
        label: str | None,
    ) -> UpdateKnowledgeBaseResult:
        if action == "prepare_upload":
            upload_session = await self.issue_upload_session()
            return UpdateKnowledgeBaseResult(
                action=action,
                upload_session=upload_session,
            )

        if action == "rename_node":
            if node_id is None:
                raise ValueError("node_id is required for rename_node.")
            if title is None or not title.strip():
                raise ValueError("A new node title is required.")
            node = await self.rename_node(node_id=node_id, new_title=title.strip())
            state = await self.get_knowledge_base_state(
                selected_node_id=node.id,
                graph_selection_mode="self",
                tag_ids=[],
                tag_match_mode="all",
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=8,
            )
            return UpdateKnowledgeBaseResult(
                action=action,
                knowledge_base_state=state,
                node=node,
            )

        if action == "create_tag":
            if name is None or not name.strip():
                raise ValueError("A tag name is required.")
            tag = await self.create_tag(name=name.strip(), color=color.strip() if color else None)
            state = await self.get_knowledge_base_state(
                selected_node_id=None,
                graph_selection_mode="self",
                tag_ids=[],
                tag_match_mode="all",
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=8,
            )
            return UpdateKnowledgeBaseResult(
                action=action,
                knowledge_base_state=state,
                tag=tag,
            )

        if action == "set_node_tags":
            if node_id is None:
                raise ValueError("node_id is required for set_node_tags.")
            node = await self.set_node_tags(node_id=node_id, tag_ids=tag_ids)
            state = await self.get_knowledge_base_state(
                selected_node_id=node.id,
                graph_selection_mode="self",
                tag_ids=[],
                tag_match_mode="all",
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=8,
            )
            return UpdateKnowledgeBaseResult(
                action=action,
                knowledge_base_state=state,
                node=node,
            )

        if action == "upsert_edge":
            if from_node_id is None or to_node_id is None:
                raise ValueError("from_node_id and to_node_id are required for upsert_edge.")
            if label is None or not label.strip():
                raise ValueError("A label is required for upsert_edge.")
            edge = await self.upsert_edge(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                label=label.strip(),
            )
            state = await self.get_knowledge_base_state(
                selected_node_id=edge.from_node_id,
                graph_selection_mode="children",
                tag_ids=[],
                tag_match_mode="all",
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=8,
            )
            return UpdateKnowledgeBaseResult(
                action=action,
                knowledge_base_state=state,
                edge=edge,
            )

        if action == "delete_edge":
            if edge_id is None:
                raise ValueError("edge_id is required for delete_edge.")
            deleted_edge_id = await self.delete_edge(edge_id=edge_id)
            state = await self.get_knowledge_base_state(
                selected_node_id=None,
                graph_selection_mode="self",
                tag_ids=[],
                tag_match_mode="all",
                media_types=[],
                include_web=False,
                rewrite_query=True,
                branch_factor=3,
                depth=2,
                max_results=8,
            )
            return UpdateKnowledgeBaseResult(
                action=action,
                knowledge_base_state=state,
                deleted_edge_id=deleted_edge_id,
            )

        if node_id is None:
            raise ValueError("node_id is required for delete_node.")
        deleted_node_id = await self.delete_node(node_id=node_id)
        state = await self.get_knowledge_base_state(
            selected_node_id=None,
            graph_selection_mode="self",
            tag_ids=[],
            tag_match_mode="all",
            media_types=[],
            include_web=False,
            rewrite_query=True,
            branch_factor=3,
            depth=2,
            max_results=8,
        )
        return UpdateKnowledgeBaseResult(
            action=action,
            knowledge_base_state=state,
            deleted_node_id=deleted_node_id,
        )

    async def run_command(
        self,
        *,
        raw_command: str,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> KnowledgeBaseCommandResult:
        execution = await self._get_command_agent().run_command(
            raw_command=raw_command,
            selected_node_id=selected_node_id,
        )
        next_selected_node_id = (
            None if execution.action == "delete_node" else execution.node_id or selected_node_id
        )
        state = await self.get_knowledge_base_state(
            selected_node_id=next_selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            media_types=media_types,
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        node = (
            await self.get_node_summary(execution.node_id)
            if execution.node_id is not None and execution.action != "delete_node"
            else None
        )
        edge = (
            await self.get_edge_summary(execution.edge_id)
            if execution.edge_id is not None
            else None
        )
        tag = (
            await self.get_tag_summary(execution.tag_id)
            if execution.tag_id is not None
            else None
        )
        return KnowledgeBaseCommandResult(
            status=execution.status,
            message=execution.message,
            action=execution.action,
            parser=execution.parser,
            raw_command=raw_command,
            knowledge_base_state=state,
            pending_confirmation=execution.pending_confirmation,
            node=node,
            edge=edge,
            tag=tag,
        )

    async def confirm_command(
        self,
        *,
        token: str,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> KnowledgeBaseCommandResult:
        claims = self._session_tokens.verify_command_confirmation(token)
        if claims is None:
            state = await self.get_knowledge_base_state(
                selected_node_id=selected_node_id,
                graph_selection_mode=graph_selection_mode,
                tag_ids=tag_ids,
                tag_match_mode=tag_match_mode,
                media_types=media_types,
                include_web=include_web,
                rewrite_query=rewrite_query,
                branch_factor=branch_factor,
                depth=depth,
                max_results=max_results,
            )
            return KnowledgeBaseCommandResult(
                status="rejected",
                message="This confirmation token is invalid or expired.",
                action=None,
                parser="manual",
                raw_command="confirm",
                knowledge_base_state=state,
            )

        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            if resolved_user.summary.clerk_user_id != claims.clerk_user_id:
                raise PermissionError(
                    "This confirmation token was issued for a different user."
                )
            if knowledge_base.id != claims.knowledge_base_id:
                raise PermissionError(
                    "This confirmation token does not match the active knowledge base."
                )

        if claims.action != "delete_node":
            raise ValueError(f"Unsupported confirmation action: {claims.action}")
        node_id = claims.payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Pending delete-node confirmation payload is invalid.")

        deleted_node_id = await self.delete_node(node_id=node_id)
        state = await self.get_knowledge_base_state(
            selected_node_id=None if selected_node_id == deleted_node_id else selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            media_types=media_types,
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        return KnowledgeBaseCommandResult(
            status="executed",
            message="Deleted the node and its connected edges.",
            action="delete_node",
            parser="manual",
            raw_command="confirm",
            knowledge_base_state=state,
        )

    async def get_node_detail(self, *, node_id: str) -> KnowledgeNodeDetail:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            return await self._node_detail(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )

    async def get_node_summary(self, node_id: str) -> KnowledgeNodeSummary | None:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            try:
                node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            except PermissionError:
                return None
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            return await self._node_summary(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )

    async def get_edge_summary(self, edge_id: str) -> KnowledgeEdgeSummary | None:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            edge = next((item for item in knowledge_base.edges if item.id == edge_id), None)
            if edge is None:
                return None
            return self._edge_summary(edge)

    async def get_tag_summary(self, tag_id: str) -> KnowledgeTagSummary | None:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            tag = next((item for item in knowledge_base.tags if item.id == tag_id), None)
            if tag is None:
                return None
            node_count = sum(1 for node in knowledge_base.nodes if any(link.tag_id == tag.id for link in node.tag_links))
            return self._tag_summary(tag, node_count=node_count)

    async def issue_upload_session(self):
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            return self._session_tokens.issue_upload_session(
                clerk_user_id=resolved_user.summary.clerk_user_id,
                knowledge_base_id=knowledge_base.id,
            )

    async def ingest_upload(
        self,
        *,
        claims: UploadSessionClaims,
        local_path: Path,
        filename: str,
        declared_media_type: str | None,
        tag_ids: list[str],
    ) -> DocumentUploadFinalizeResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            app_user = await self._user_by_clerk_id(session, claims.clerk_user_id)
            if app_user is None:
                raise PermissionError("Upload token does not map to a known user.")
            if not app_user.active:
                raise PermissionError("User is pending manual activation.")
            resolved_user = ResolvedUser(
                app_user=app_user,
                summary=UserSummary(
                    clerk_user_id=app_user.clerk_user_id,
                    display_name=app_user.display_name or app_user.clerk_user_id,
                    primary_email=app_user.primary_email,
                    active=app_user.active,
                    role=app_user.role,
                ),
            )
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            if knowledge_base.id != claims.knowledge_base_id:
                raise PermissionError("Upload token does not match the active knowledge base.")
            await self._ensure_vector_store(session, knowledge_base, resolved_user)

            tag_records = await self._knowledge_tags_by_ids(
                session,
                knowledge_base_id=knowledge_base.id,
                tag_ids=tag_ids,
            )
            media_type = guess_media_type(local_path, declared_media_type)
            source_kind = classify_source_kind(local_path=local_path, media_type=media_type)
            display_title = await self._unique_display_title(
                session,
                knowledge_base_id=knowledge_base.id,
                base_title=Path(filename).stem or filename,
            )
            node = KnowledgeNode(
                knowledge_base_id=knowledge_base.id,
                created_by_user_id=app_user.id,
                display_title=display_title,
                original_filename=filename,
                media_type=media_type,
                source_kind=source_kind,
                status="processing",
                byte_size=local_path.stat().st_size,
                original_mime_type=media_type,
                updated_at=_utcnow(),
            )
            knowledge_base.updated_at = _utcnow()
            session.add(node)
            await session.flush()
            node.tag_links = [KnowledgeNodeTag(node_id=node.id, tag_id=tag.id) for tag in tag_records]

            try:
                original_file_id = await self._openai_gateway.upload_original_file(
                    local_path=local_path,
                    purpose=self._openai_gateway.choose_original_file_purpose(
                        source_kind=source_kind
                    ),
                )
                node.openai_original_file_id = original_file_id

                tag_names = [tag.name for tag in tag_records]
                tag_slugs = [tag.slug for tag in tag_records]

                derived_text = extract_text_document(local_path=local_path, media_type=media_type)
                if source_kind == "image":
                    image_payload = await self._openai_gateway.describe_image(
                        openai_file_id=original_file_id
                    )
                    derived_text = render_image_description(image_payload)
                    await self._store_derived_artifact(
                        session=session,
                        knowledge_base=knowledge_base,
                        node=node,
                        kind="image_description",
                        text_content=derived_text,
                        structured_payload=image_payload.model_dump(mode="json"),
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                elif source_kind == "audio":
                    derived_text, payload = await self._openai_gateway.transcribe_audio(
                        local_path=local_path
                    )
                    await self._store_derived_artifact(
                        session=session,
                        knowledge_base=knowledge_base,
                        node=node,
                        kind="audio_transcript",
                        text_content=derived_text,
                        structured_payload=payload,
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                elif source_kind == "video":
                    derived_text, payload = await self._openai_gateway.transcribe_video(
                        local_path=local_path
                    )
                    await self._store_derived_artifact(
                        session=session,
                        knowledge_base=knowledge_base,
                        node=node,
                        kind="video_transcript",
                        text_content=derived_text,
                        structured_payload=payload,
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                elif derived_text is not None:
                    await self._store_derived_artifact(
                        session=session,
                        knowledge_base=knowledge_base,
                        node=node,
                        kind="document_text",
                        text_content=derived_text,
                        structured_payload=None,
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                else:
                    await self._openai_gateway.attach_existing_file_to_vector_store(
                        vector_store_id=knowledge_base.openai_vector_store_id or "",
                        file_id=original_file_id,
                        attributes=build_searchable_attributes(
                            knowledge_base_id=knowledge_base.id,
                            node_id=node.id,
                            node_title=node.display_title,
                            derived_artifact_id=None,
                            source_kind=source_kind,
                            media_type=media_type,
                            derived_kind="direct_file",
                            original_openai_file_id=original_file_id,
                            original_filename=node.original_filename,
                            tag_names=tag_names,
                            tag_slugs=tag_slugs,
                        ),
                    )

                node.status = "ready"
                node.error_message = None
                node.updated_at = _utcnow()
                knowledge_base.updated_at = _utcnow()
                await session.commit()
                await session.refresh(node)
            except Exception as exc:
                node.status = "failed"
                node.error_message = str(exc)
                node.updated_at = _utcnow()
                knowledge_base.updated_at = _utcnow()
                await session.commit()
                raise

            node_summary = await self._node_summary(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )
            return DocumentUploadFinalizeResult(
                document=self._document_summary_from_node_summary(node_summary)
            )

    async def download_node_bytes(
        self,
        *,
        claims: NodeDownloadClaims,
    ) -> tuple[KnowledgeNodeDetail, bytes]:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            app_user = await self._user_by_clerk_id(session, claims.clerk_user_id)
            if app_user is None:
                raise PermissionError("Download token does not map to a known user.")
            resolved_user = ResolvedUser(
                app_user=app_user,
                summary=UserSummary(
                    clerk_user_id=app_user.clerk_user_id,
                    display_name=app_user.display_name or app_user.clerk_user_id,
                    primary_email=app_user.primary_email,
                    active=app_user.active,
                    role=app_user.role,
                ),
            )
            node = await self._node_for_user(
                session,
                resolved_user=resolved_user,
                node_id=claims.node_id,
            )
            if node.openai_original_file_id is None:
                raise FileNotFoundError("The requested node has no stored original file.")
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            detail = await self._node_detail(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )
            payload = await self._openai_gateway.read_file_bytes(
                file_id=node.openai_original_file_id
            )
            return detail, payload

    async def knowledge_base_file_search(
        self,
        *,
        query: str,
        context: KnowledgeBaseContext,
    ) -> KnowledgeFileSearchResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            adjusted_context = await self._normalized_context(
                session,
                knowledge_base=knowledge_base,
                selected_node_id=context.selected_node_id,
                graph_selection_mode=context.graph_selection_mode,
                tag_ids=context.tag_ids,
                tag_match_mode=context.tag_match_mode,
                media_types=context.media_types,
                include_web=context.include_web,
                rewrite_query=context.rewrite_query,
                branch_factor=context.branch_factor,
                depth=context.depth,
                max_results=context.max_results,
            )
            if not adjusted_context.scoped_node_ids:
                return KnowledgeFileSearchResult(
                    knowledge_base_id=knowledge_base.id,
                    query=query,
                    context=adjusted_context,
                    hits=[],
                    total_hits=0,
                )
            if knowledge_base.openai_vector_store_id is None:
                return KnowledgeFileSearchResult(
                    knowledge_base_id=knowledge_base.id,
                    query=query,
                    context=adjusted_context,
                    hits=[],
                    total_hits=0,
                )

            filters = await self._filters_for_context(
                session,
                knowledge_base=knowledge_base,
                context=adjusted_context,
            )
            hits = await self._openai_gateway.search_vector_store(
                vector_store_id=knowledge_base.openai_vector_store_id,
                query=query,
                max_results=context.max_results,
                rewrite_query=context.rewrite_query,
                filters=filters,
            )
            return KnowledgeFileSearchResult(
                knowledge_base_id=knowledge_base.id,
                query=query,
                context=adjusted_context,
                hits=hits,
                total_hits=len(hits),
            )

    async def knowledge_base_branch_search(
        self,
        *,
        query: str,
        context: KnowledgeBaseContext,
    ) -> KnowledgeBranchSearchResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            adjusted_context = await self._normalized_context(
                session,
                knowledge_base=knowledge_base,
                selected_node_id=context.selected_node_id,
                graph_selection_mode=context.graph_selection_mode,
                tag_ids=context.tag_ids,
                tag_match_mode=context.tag_match_mode,
                media_types=context.media_types,
                include_web=context.include_web,
                rewrite_query=context.rewrite_query,
                branch_factor=context.branch_factor,
                depth=context.depth,
                max_results=context.max_results,
            )
            if not adjusted_context.scoped_node_ids or knowledge_base.openai_vector_store_id is None:
                return KnowledgeBranchSearchResult(
                    knowledge_base_id=knowledge_base.id,
                    seed_query=query,
                    context=adjusted_context,
                    nodes=[],
                    merged_hits=[],
                )
            filters = await self._filters_for_context(
                session,
                knowledge_base=knowledge_base,
                context=adjusted_context,
            )

            queue: list[tuple[str, str | None, int, str, str | None]] = [
                ("node_1", None, 0, query, None)
            ]
            nodes: list[BranchSearchNode] = []
            merged_hits: dict[str, tuple[SearchHit, float, int]] = {}
            next_id = 2

            while queue:
                node_id, parent_id, branch_depth, node_query, rationale = queue.pop(0)
                hits = await self._openai_gateway.search_vector_store(
                    vector_store_id=knowledge_base.openai_vector_store_id,
                    query=node_query,
                    max_results=adjusted_context.max_results,
                    rewrite_query=adjusted_context.rewrite_query,
                    filters=filters,
                )
                branch_node = BranchSearchNode(
                    id=node_id,
                    parent_id=parent_id,
                    depth=branch_depth,
                    query=node_query,
                    rationale=rationale,
                    hits=hits,
                )
                nodes.append(branch_node)

                for hit in hits:
                    hit_key = f"{hit.openai_file_id}:{hit.text[:160]}"
                    existing = merged_hits.get(hit_key)
                    if existing is None:
                        merged_hits[hit_key] = (hit, hit.score, 1)
                    else:
                        merged_hits[hit_key] = (
                            existing[0],
                            max(existing[1], hit.score),
                            existing[2] + 1,
                        )

                if branch_depth + 1 >= adjusted_context.depth:
                    continue
                expansion = await self._openai_gateway.expand_branch_queries(
                    query=node_query,
                    branch_factor=adjusted_context.branch_factor,
                    tag_names=adjusted_context.selected_tag_names,
                    hit_snippets=[hit.text for hit in hits],
                )
                child_queries = expansion.queries[: adjusted_context.branch_factor]
                child_ids: list[str] = []
                for child_query in child_queries:
                    child_id = f"node_{next_id}"
                    next_id += 1
                    child_ids.append(child_id)
                    queue.append(
                        (
                            child_id,
                            node_id,
                            branch_depth + 1,
                            child_query,
                            expansion.rationale,
                        )
                    )
                branch_node.children = child_ids

            ranked_hits = sorted(
                merged_hits.values(),
                key=lambda item: (item[1] + (item[2] - 1) * 0.05),
                reverse=True,
            )
            return KnowledgeBranchSearchResult(
                knowledge_base_id=knowledge_base.id,
                seed_query=query,
                context=adjusted_context,
                nodes=nodes,
                merged_hits=[item[0] for item in ranked_hits[: adjusted_context.max_results]],
            )

    async def knowledge_base_chat(
        self,
        *,
        question: str,
        context: KnowledgeBaseContext,
    ) -> KnowledgeChatResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            adjusted_context = await self._normalized_context(
                session,
                knowledge_base=knowledge_base,
                selected_node_id=context.selected_node_id,
                graph_selection_mode=context.graph_selection_mode,
                tag_ids=context.tag_ids,
                tag_match_mode=context.tag_match_mode,
                media_types=context.media_types,
                include_web=context.include_web,
                rewrite_query=context.rewrite_query,
                branch_factor=context.branch_factor,
                depth=context.depth,
                max_results=context.max_results,
            )
            if not adjusted_context.scoped_node_ids:
                return KnowledgeChatResult(
                    knowledge_base_id=knowledge_base.id,
                    question=question,
                    answer="No documents match the current node and tag filters yet.",
                    model=self._settings.openai_agent_model,
                    include_web=adjusted_context.include_web,
                    conversation_id=knowledge_base.openai_conversation_id or "",
                    context=adjusted_context,
                    citations=[],
                )
            if knowledge_base.openai_vector_store_id is None:
                return KnowledgeChatResult(
                    knowledge_base_id=knowledge_base.id,
                    question=question,
                    answer="There are no indexed documents in the current knowledge-base scope yet.",
                    model=self._settings.openai_agent_model,
                    include_web=adjusted_context.include_web,
                    conversation_id=knowledge_base.openai_conversation_id or "",
                    context=adjusted_context,
                    citations=[],
                )

            filters = await self._filters_for_context(
                session,
                knowledge_base=knowledge_base,
                context=adjusted_context,
            )
            chat_result = await self._question_answerer.ask(
                knowledge_base_id=knowledge_base.id,
                vector_store_id=knowledge_base.openai_vector_store_id,
                question=question,
                context=adjusted_context,
                conversation_id=knowledge_base.openai_conversation_id,
                filters=filters,
            )
            knowledge_base.openai_conversation_id = chat_result.conversation_id
            knowledge_base.updated_at = _utcnow()
            await session.commit()
            return chat_result

    async def rename_node(self, *, node_id: str, new_title: str) -> KnowledgeNodeSummary:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            node.display_title = await self._unique_display_title(
                session,
                knowledge_base_id=knowledge_base.id,
                base_title=new_title,
                exclude_node_id=node.id,
            )
            node.updated_at = _utcnow()
            knowledge_base.updated_at = _utcnow()
            await session.flush()
            await self._sync_node_vector_store_attributes(
                session,
                knowledge_base=knowledge_base,
                node=node,
            )
            await session.commit()
            await session.refresh(node)
            return await self._node_summary(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )

    async def create_tag(self, *, name: str, color: str | None) -> KnowledgeTagSummary:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            existing = next((tag for tag in knowledge_base.tags if tag.name.lower() == name.lower()), None)
            if existing is not None:
                return self._tag_summary(existing, node_count=self._tag_node_count(existing))
            tag = KnowledgeTag(
                knowledge_base_id=knowledge_base.id,
                name=name.strip(),
                slug=await self._unique_tag_slug(session, knowledge_base.id, slugify(name)),
                color=color.strip() if color else None,
            )
            knowledge_base.updated_at = _utcnow()
            session.add(tag)
            await session.commit()
            await session.refresh(tag)
            return self._tag_summary(tag, node_count=0)

    async def set_node_tags(self, *, node_id: str, tag_ids: list[str]) -> KnowledgeNodeSummary:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            tag_records = await self._knowledge_tags_by_ids(
                session,
                knowledge_base_id=knowledge_base.id,
                tag_ids=tag_ids,
            )
            node.tag_links = [KnowledgeNodeTag(node_id=node.id, tag_id=tag.id) for tag in tag_records]
            node.updated_at = _utcnow()
            knowledge_base.updated_at = _utcnow()
            await session.flush()
            await self._sync_node_vector_store_attributes(
                session,
                knowledge_base=knowledge_base,
                node=node,
            )
            await session.commit()
            await session.refresh(node)
            return await self._node_summary(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )

    async def upsert_edge(
        self,
        *,
        from_node_id: str,
        to_node_id: str,
        label: str,
    ) -> KnowledgeEdgeSummary:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            from_node = await self._node_for_user(
                session,
                resolved_user=resolved_user,
                node_id=from_node_id,
            )
            to_node = await self._node_for_user(
                session,
                resolved_user=resolved_user,
                node_id=to_node_id,
            )
            if from_node.id == to_node.id:
                raise ValueError("Self-referential edges are not supported.")
            edge = next(
                (
                    item
                    for item in knowledge_base.edges
                    if item.from_node_id == from_node.id and item.to_node_id == to_node.id
                ),
                None,
            )
            if edge is None:
                edge = KnowledgeEdge(
                    knowledge_base_id=knowledge_base.id,
                    from_node_id=from_node.id,
                    to_node_id=to_node.id,
                    label=label,
                    updated_at=_utcnow(),
                )
                session.add(edge)
            else:
                edge.label = label
                edge.updated_at = _utcnow()
            knowledge_base.updated_at = _utcnow()
            await session.commit()
            await session.refresh(edge)
            edge.from_node = from_node
            edge.to_node = to_node
            return self._edge_summary(edge)

    async def delete_edge(self, *, edge_id: str) -> str:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            edge = next((item for item in knowledge_base.edges if item.id == edge_id), None)
            if edge is None:
                raise PermissionError("Edge not found or not owned by the current user.")
            knowledge_base.updated_at = _utcnow()
            await session.delete(edge)
            await session.commit()
            return edge_id

    async def delete_node(self, *, node_id: str) -> str:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            file_ids = {
                file_id
                for file_id in [
                    node.openai_original_file_id,
                    *[
                        artifact.openai_file_id
                        for artifact in node.derived_artifacts
                        if artifact.openai_file_id is not None
                    ],
                ]
                if file_id is not None
            }
            for file_id in file_ids:
                await self._openai_gateway.delete_file(file_id=file_id)
            knowledge_base.updated_at = _utcnow()
            await session.delete(node)
            await session.commit()
            return node_id

    async def rename_node_from_command(
        self,
        *,
        node_title: str | None,
        selected_node_id: str | None,
        new_title: str,
        parser: CommandParserKind,
    ) -> CommandExecutionResult:
        node = await self._resolve_node_reference(
            node_title=node_title,
            selected_node_id=selected_node_id,
        )
        updated = await self.rename_node(node_id=node.id, new_title=new_title)
        from .command_agent import CommandExecutionResult

        return CommandExecutionResult(
            status="executed",
            action="rename_node",
            message=f"Renamed node '{node.display_title}' to '{updated.display_title}'.",
            parser=parser,
            node_id=updated.id,
        )

    async def create_tag_from_command(
        self,
        *,
        name: str,
        color: str | None,
        parser: CommandParserKind,
    ) -> CommandExecutionResult:
        tag = await self.create_tag(name=name, color=color)
        from .command_agent import CommandExecutionResult

        return CommandExecutionResult(
            status="executed",
            action="create_tag",
            message=f"Created tag '{tag.name}'.",
            parser=parser,
            tag_id=tag.id,
        )

    async def set_node_tags_from_command(
        self,
        *,
        node_title: str | None,
        selected_node_id: str | None,
        tag_names: list[str],
        parser: CommandParserKind,
    ) -> CommandExecutionResult:
        node = await self._resolve_node_reference(
            node_title=node_title,
            selected_node_id=selected_node_id,
        )
        tag_ids = await self._ensure_tag_ids_by_names(tag_names)
        updated = await self.set_node_tags(node_id=node.id, tag_ids=tag_ids)
        from .command_agent import CommandExecutionResult

        return CommandExecutionResult(
            status="executed",
            action="set_node_tags",
            message=f"Updated tags on '{updated.display_title}'.",
            parser=parser,
            node_id=updated.id,
        )

    async def upsert_edge_from_command(
        self,
        *,
        from_node_title: str | None,
        to_node_title: str,
        label: str,
        selected_node_id: str | None,
        parser: CommandParserKind,
    ) -> CommandExecutionResult:
        from_node = await self._resolve_node_reference(
            node_title=from_node_title,
            selected_node_id=selected_node_id,
        )
        to_node = await self._resolve_node_reference(
            node_title=to_node_title,
            selected_node_id=None,
        )
        edge = await self.upsert_edge(
            from_node_id=from_node.id,
            to_node_id=to_node.id,
            label=label,
        )
        from .command_agent import CommandExecutionResult

        return CommandExecutionResult(
            status="executed",
            action="upsert_edge",
            message=(
                f"Connected '{edge.from_node_title}' to '{edge.to_node_title}' with label "
                f"'{edge.label}'."
            ),
            parser=parser,
            edge_id=edge.id,
            node_id=edge.from_node_id,
        )

    async def delete_node_from_command(
        self,
        *,
        node_title: str | None,
        selected_node_id: str | None,
        parser: CommandParserKind,
    ) -> CommandExecutionResult:
        node = await self._resolve_node_reference(
            node_title=node_title,
            selected_node_id=selected_node_id,
        )
        pending = await self._issue_delete_node_confirmation(node_id=node.id)
        from .command_agent import CommandExecutionResult

        return CommandExecutionResult(
            status="pending_confirmation",
            action="delete_node",
            message=f"Confirm deletion of '{node.display_title}' to continue.",
            parser=parser,
            node_id=node.id,
            pending_confirmation=pending,
        )

    async def _issue_delete_node_confirmation(self, *, node_id: str) -> PendingCommandResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            node = await self._node_for_user(session, resolved_user=resolved_user, node_id=node_id)
            return self._session_tokens.issue_command_confirmation(
                clerk_user_id=resolved_user.summary.clerk_user_id,
                knowledge_base_id=knowledge_base.id,
                action="delete_node",
                payload={"node_id": node.id},
                prompt=f"Delete '{node.display_title}' and remove its edges from the graph?",
                summary=f"Delete node '{node.display_title}'",
            )

    async def _ensure_tag_ids_by_names(self, tag_names: list[str]) -> list[str]:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            tag_records = await self._ensure_tags_by_names(
                session,
                knowledge_base=knowledge_base,
                tag_names=tag_names,
            )
            await session.commit()
            return [tag.id for tag in tag_records]

    async def _resolve_node_reference(
        self,
        *,
        node_title: str | None,
        selected_node_id: str | None,
    ) -> KnowledgeNode:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            knowledge_base = await self._knowledge_base_for_user(
                session,
                resolved_user=resolved_user,
            )
            if node_title is None:
                if selected_node_id is None:
                    raise ValueError("No node is selected.")
                return await self._node_for_user(
                    session,
                    resolved_user=resolved_user,
                    node_id=selected_node_id,
                )

            lower_title = node_title.strip().lower()
            node = next(
                (
                    item
                    for item in knowledge_base.nodes
                    if item.display_title.lower() == lower_title
                    or item.original_filename.lower() == lower_title
                ),
                None,
            )
            if node is None:
                raise ValueError(f"Could not find a node named '{node_title}'.")
            return node

    async def _store_derived_artifact(
        self,
        *,
        session,
        knowledge_base: KnowledgeBase,
        node: KnowledgeNode,
        kind: str,
        text_content: str,
        structured_payload,
        tag_names: list[str],
        tag_slugs: list[str],
    ) -> None:
        derived = DerivedArtifact(
            node_id=node.id,
            kind=kind,
            text_content=text_content,
            structured_payload=structured_payload,
            updated_at=_utcnow(),
        )
        session.add(derived)
        await session.flush()
        derived.openai_file_id = await self._openai_gateway.create_text_artifact_and_attach(
            vector_store_id=knowledge_base.openai_vector_store_id or "",
            filename=f"{node.original_filename}.{kind}.md",
            text_content=text_content,
            attributes=build_searchable_attributes(
                knowledge_base_id=knowledge_base.id,
                node_id=node.id,
                node_title=node.display_title,
                derived_artifact_id=derived.id,
                source_kind=node.source_kind,
                media_type=node.media_type,
                derived_kind=kind,
                original_openai_file_id=node.openai_original_file_id,
                original_filename=node.original_filename,
                tag_names=tag_names,
                tag_slugs=tag_slugs,
            ),
        )
        derived.updated_at = _utcnow()

    async def _document_filters_for_request(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        filename_query: str | None,
        created_from: date | None,
        created_to: date | None,
    ) -> DocumentFilters:
        if created_from is not None and created_to is not None and created_from > created_to:
            raise ValueError("created_from must be on or before created_to.")

        tag_records = await self._knowledge_tags_by_ids(
            session,
            knowledge_base_id=knowledge_base.id,
            tag_ids=tag_ids,
        )
        normalized_filename_query = (
            filename_query.strip().lower() if filename_query and filename_query.strip() else None
        )
        selected_tag_ids = [tag.id for tag in tag_records]
        selected_tag_names = [tag.name for tag in tag_records]
        required_tag_ids = set(selected_tag_ids)

        matching_document_ids: list[str] = []
        for node in sorted(
            knowledge_base.nodes,
            key=lambda item: (item.created_at, item.updated_at),
            reverse=True,
        ):
            node_tag_ids = {link.tag_id for link in node.tag_links}
            if required_tag_ids:
                if tag_match_mode == "all" and not required_tag_ids.issubset(node_tag_ids):
                    continue
                if tag_match_mode == "any" and not required_tag_ids.intersection(node_tag_ids):
                    continue

            if normalized_filename_query is not None and normalized_filename_query not in (
                node.original_filename.lower()
            ):
                continue

            created_date = (
                node.created_at.astimezone(UTC).date()
                if node.created_at.tzinfo is not None
                else node.created_at.date()
            )
            if created_from is not None and created_date < created_from:
                continue
            if created_to is not None and created_date > created_to:
                continue
            matching_document_ids.append(node.id)

        return DocumentFilters(
            tag_ids=selected_tag_ids,
            selected_tag_names=selected_tag_names,
            tag_match_mode=tag_match_mode,
            filename_query=normalized_filename_query,
            created_from=created_from,
            created_to=created_to,
            matching_document_ids=matching_document_ids,
        )

    async def _document_library_view_state(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        knowledge_base: KnowledgeBase,
        filters: DocumentFilters,
    ) -> DocumentLibraryViewState:
        library_state: DocumentLibraryState | None = None
        if resolved_user.summary.active:
            library_state = await self._document_library_state(
                session,
                knowledge_base=knowledge_base,
                filters=filters,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )

        access = KnowledgeAccessState(
            status="active" if resolved_user.summary.active else "pending_access",
            message=(
                "Access active. You can upload documents, manage tags, and search the library."
                if resolved_user.summary.active
                else "Signed in successfully. Access is pending manual activation in Clerk."
            ),
            user=resolved_user.summary,
        )
        return DocumentLibraryViewState(
            access=access,
            library=library_state,
            capabilities=DocumentLibraryCapabilities(
                upload_url=f"{self._settings.normalized_app_base_url}/api/uploads",
                upload_token_ttl_seconds=self._settings.upload_session_max_age_seconds,
                supports_video_audio_extraction=True,
                accepted_hint=(
                    "Upload text-like files directly, plus images, audio, and video. "
                    "Filter the library by tags, filename, and created date."
                ),
            ),
        )

    async def _document_library_state(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        filters: DocumentFilters,
        clerk_user_id: str,
    ) -> DocumentLibraryState:
        tag_counts = {tag.id: 0 for tag in knowledge_base.tags}
        for node in knowledge_base.nodes:
            for link in node.tag_links:
                tag_counts[link.tag_id] = tag_counts.get(link.tag_id, 0) + 1

        matching_document_ids = set(filters.matching_document_ids)
        documents: list[DocumentSummary] = []
        for node in sorted(
            knowledge_base.nodes,
            key=lambda item: (item.created_at, item.updated_at),
            reverse=True,
        ):
            if node.id not in matching_document_ids:
                continue
            node_summary = await self._node_summary(
                session,
                knowledge_base=knowledge_base,
                node=node,
                clerk_user_id=clerk_user_id,
            )
            documents.append(self._document_summary_from_node_summary(node_summary))

        tags = [
            self._tag_summary(tag, node_count=tag_counts.get(tag.id, 0))
            for tag in sorted(knowledge_base.tags, key=lambda item: item.name.lower())
        ]
        return DocumentLibraryState(
            library=DocumentLibrarySummary(
                id=knowledge_base.id,
                title=knowledge_base.title,
                description=knowledge_base.description,
                created_at=knowledge_base.created_at,
                updated_at=knowledge_base.updated_at,
                document_count=len(knowledge_base.nodes),
                filtered_document_count=len(documents),
                tag_count=len(tags),
                vector_store_ready=knowledge_base.openai_vector_store_id is not None,
            ),
            tags=tags,
            documents=documents,
            filters=filters,
        )

    def _vector_store_filters_for_documents(
        self,
        *,
        knowledge_base: KnowledgeBase,
        document_ids: list[str],
    ):
        all_document_ids = {node.id for node in knowledge_base.nodes}
        scoped_document_ids = [] if set(document_ids) == all_document_ids else document_ids
        return build_filter_groups(
            node_ids=scoped_document_ids,
            media_types=[],
            tag_slugs=[],
            tag_match_mode="all",
        )

    @staticmethod
    def _document_summary_from_node_summary(node: KnowledgeNodeSummary) -> DocumentSummary:
        return DocumentSummary(
            id=node.id,
            title=node.display_title,
            original_filename=node.original_filename,
            media_type=node.media_type,
            source_kind=node.source_kind,
            status=node.status,
            byte_size=node.byte_size,
            error_message=node.error_message,
            created_at=node.created_at,
            updated_at=node.updated_at,
            tags=node.tags,
            derived_kinds=node.derived_kinds,
            openai_original_file_id=node.openai_original_file_id,
            download_url=node.download_url,
        )

    @classmethod
    def _document_detail_from_node_detail(cls, node: KnowledgeNodeDetail) -> DocumentDetail:
        return DocumentDetail(
            **cls._document_summary_from_node_summary(node).model_dump(mode="python"),
            original_mime_type=node.original_mime_type,
            derived_artifacts=node.derived_artifacts,
        )

    @staticmethod
    def _document_search_hit(hit: SearchHit) -> DocumentSearchHit:
        return DocumentSearchHit(
            document_id=hit.node_id,
            document_title=hit.node_title,
            original_filename=hit.original_filename,
            derived_artifact_id=hit.derived_artifact_id,
            openai_file_id=hit.openai_file_id,
            original_openai_file_id=hit.original_openai_file_id,
            media_type=hit.media_type,
            source_kind=hit.source_kind,
            score=hit.score,
            text=hit.text,
            tags=hit.tags,
        )

    @staticmethod
    def _document_hits_from_chat_result(chat_result: KnowledgeChatResult) -> list[DocumentSearchHit]:
        hits_by_key: dict[str, DocumentSearchHit] = {}
        for search_call in chat_result.search_calls:
            for hit in search_call.results:
                converted = KnowledgeBaseService._document_search_hit(hit)
                key = f"{converted.document_id}:{converted.openai_file_id}:{converted.text[:160]}"
                hits_by_key.setdefault(key, converted)
        return list(hits_by_key.values())

    @staticmethod
    def _document_citation(citation: KnowledgeAnswerCitation) -> DocumentCitation:
        return DocumentCitation(
            label=citation.label,
            document_id=citation.node_id,
            document_title=citation.node_title,
            original_filename=citation.original_filename,
            quote=citation.quote,
            url=citation.url,
            source="web" if citation.source == "web" else "document_library",
        )

    async def _resolve_request_user(self, session) -> ResolvedUser:
        clerk_record = get_current_clerk_user_record()
        if clerk_record is None:
            access_token = get_access_token()
            if access_token is None:
                raise PermissionError("Authentication is required.")
            clerk_record = await self._clerk_auth.get_user_record(
                clerk_user_id_from_access_token(access_token)
            )
        app_user = await self._upsert_clerk_user(session, clerk_record)
        return ResolvedUser(
            app_user=app_user,
            summary=UserSummary(
                clerk_user_id=clerk_record.clerk_user_id,
                display_name=clerk_record.display_name,
                primary_email=clerk_record.primary_email,
                active=clerk_record.active,
                role=clerk_record.role,
            ),
        )

    async def _upsert_clerk_user(
        self,
        session,
        clerk_record: ClerkUserRecord,
    ) -> AppUser:
        existing = await self._user_by_clerk_id(session, clerk_record.clerk_user_id)
        now = _utcnow()
        if existing is None:
            existing = AppUser(
                clerk_user_id=clerk_record.clerk_user_id,
                primary_email=clerk_record.primary_email,
                display_name=clerk_record.display_name,
                active=clerk_record.active,
                role=clerk_record.role,
                last_seen_at=now,
            )
            session.add(existing)
        else:
            existing.primary_email = clerk_record.primary_email
            existing.display_name = clerk_record.display_name
            existing.active = clerk_record.active
            existing.role = clerk_record.role
            existing.last_seen_at = now
        await session.commit()
        await session.refresh(existing)
        return existing

    async def _user_by_clerk_id(self, session, clerk_user_id: str) -> AppUser | None:
        return await session.scalar(
            select(AppUser).where(AppUser.clerk_user_id == clerk_user_id)
        )

    def _require_active(self, resolved_user: ResolvedUser) -> None:
        if resolved_user.summary.active:
            return
        raise PermissionError(
            "Your account is signed in but is still pending manual activation."
        )

    async def _knowledge_base_for_user(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
    ) -> KnowledgeBase:
        knowledge_base = await session.scalar(
            select(KnowledgeBase)
            .where(KnowledgeBase.user_id == resolved_user.app_user.id)
            .options(
                selectinload(KnowledgeBase.tags).selectinload(KnowledgeTag.node_links),
                selectinload(KnowledgeBase.nodes)
                .selectinload(KnowledgeNode.derived_artifacts),
                selectinload(KnowledgeBase.nodes)
                .selectinload(KnowledgeNode.tag_links)
                .selectinload(KnowledgeNodeTag.tag),
                selectinload(KnowledgeBase.edges).selectinload(KnowledgeEdge.from_node),
                selectinload(KnowledgeBase.edges).selectinload(KnowledgeEdge.to_node),
            )
        )
        if knowledge_base is not None:
            return knowledge_base

        knowledge_base = KnowledgeBase(
            user_id=resolved_user.app_user.id,
            title=build_knowledge_base_title(resolved_user.summary.display_name),
            description="Personal document library",
            updated_at=_utcnow(),
        )
        session.add(knowledge_base)
        await session.commit()
        return await self._knowledge_base_for_user(session, resolved_user=resolved_user)

    async def _ensure_vector_store(
        self,
        session,
        knowledge_base: KnowledgeBase,
        resolved_user: ResolvedUser,
    ) -> None:
        if knowledge_base.openai_vector_store_id is not None:
            return
        vector_store_id = await self._openai_gateway.create_vector_store(
            name=knowledge_base.title,
            description=knowledge_base.description,
            metadata={"owner": resolved_user.summary.clerk_user_id},
        )
        knowledge_base.openai_vector_store_id = vector_store_id
        knowledge_base.updated_at = _utcnow()
        await session.flush()

    async def _node_for_user(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        node_id: str,
    ) -> KnowledgeNode:
        node = await session.scalar(
            select(KnowledgeNode)
            .join(KnowledgeBase, KnowledgeBase.id == KnowledgeNode.knowledge_base_id)
            .where(
                KnowledgeNode.id == node_id,
                KnowledgeBase.user_id == resolved_user.app_user.id,
            )
            .options(
                selectinload(KnowledgeNode.derived_artifacts),
                selectinload(KnowledgeNode.tag_links)
                .selectinload(KnowledgeNodeTag.tag)
                .selectinload(KnowledgeTag.node_links),
                selectinload(KnowledgeNode.outgoing_edges).selectinload(KnowledgeEdge.to_node),
                selectinload(KnowledgeNode.incoming_edges).selectinload(KnowledgeEdge.from_node),
            )
        )
        if node is None:
            raise PermissionError("Node not found or not owned by the current user.")
        return node

    async def _knowledge_tags_by_ids(
        self,
        session,
        *,
        knowledge_base_id: str,
        tag_ids: list[str],
    ) -> list[KnowledgeTag]:
        if not tag_ids:
            return []
        records = (
            (
                await session.execute(
                    select(KnowledgeTag).where(
                        KnowledgeTag.knowledge_base_id == knowledge_base_id,
                        KnowledgeTag.id.in_(tag_ids),
                    )
                )
            )
            .scalars()
            .all()
        )
        if len(records) != len(set(tag_ids)):
            raise ValueError("One or more tag IDs are invalid for this knowledge base.")
        return sorted(records, key=lambda tag: tag.name.lower())

    async def _ensure_tags_by_names(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        tag_names: list[str],
    ) -> list[KnowledgeTag]:
        existing_by_name = {tag.name.lower(): tag for tag in knowledge_base.tags}
        records: list[KnowledgeTag] = []
        for raw_name in tag_names:
            normalized_name = raw_name.strip()
            if not normalized_name:
                continue
            existing = existing_by_name.get(normalized_name.lower())
            if existing is not None:
                records.append(existing)
                continue
            tag = KnowledgeTag(
                knowledge_base_id=knowledge_base.id,
                name=normalized_name,
                slug=await self._unique_tag_slug(session, knowledge_base.id, slugify(normalized_name)),
                color=None,
            )
            session.add(tag)
            await session.flush()
            knowledge_base.tags.append(tag)
            existing_by_name[normalized_name.lower()] = tag
            records.append(tag)
        knowledge_base.updated_at = _utcnow()
        return sorted(records, key=lambda tag: tag.name.lower())

    async def _build_knowledge_base_state(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        knowledge_base: KnowledgeBase,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> KnowledgeBaseDeskState:
        selected_context = await self._normalized_context(
            session,
            knowledge_base=knowledge_base,
            selected_node_id=selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            media_types=media_types,
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        selected_state = None
        if resolved_user.summary.active:
            selected_state = await self._knowledge_base_state(
                session,
                knowledge_base=knowledge_base,
                context=selected_context,
                clerk_user_id=resolved_user.summary.clerk_user_id,
            )
        access = KnowledgeAccessState(
            status="active" if resolved_user.summary.active else "pending_access",
            message=(
                "Access active. You can upload documents, edit graph structure, and run retrieval."
                if resolved_user.summary.active
                else "Signed in successfully. Access is pending manual activation in Clerk."
            ),
            user=resolved_user.summary,
        )
        return KnowledgeBaseDeskState(
            access=access,
            knowledge_base=selected_state,
            capabilities=KnowledgeBaseCapabilities(
                upload_url=f"{self._settings.normalized_app_base_url}/api/uploads",
                upload_token_ttl_seconds=self._settings.upload_session_max_age_seconds,
                confirmation_token_ttl_seconds=self._settings.command_confirmation_max_age_seconds,
                supports_video_audio_extraction=True,
                accepted_hint=(
                    "Upload text-like files directly, plus images, audio, and video. "
                    "Each uploaded document becomes one graph node in your knowledge base."
                ),
            ),
        )

    async def _knowledge_base_state(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        context: KnowledgeBaseContext,
        clerk_user_id: str,
    ) -> KnowledgeBaseState:
        tag_counts = {tag.id: 0 for tag in knowledge_base.tags}
        node_summaries: list[KnowledgeNodeSummary] = []
        sorted_nodes = sorted(
            knowledge_base.nodes,
            key=lambda node: (node.updated_at, node.created_at),
            reverse=True,
        )
        for node in sorted_nodes:
            for link in node.tag_links:
                tag_counts[link.tag_id] = tag_counts.get(link.tag_id, 0) + 1
            node_summaries.append(
                await self._node_summary(
                    session,
                    knowledge_base=knowledge_base,
                    node=node,
                    clerk_user_id=clerk_user_id,
                )
            )
        tags = [
            self._tag_summary(tag, node_count=tag_counts.get(tag.id, 0))
            for tag in sorted(knowledge_base.tags, key=lambda tag: tag.name.lower())
        ]
        edges = [
            self._edge_summary(edge)
            for edge in sorted(
                knowledge_base.edges,
                key=lambda edge: (
                    edge.from_node.display_title.lower(),
                    edge.to_node.display_title.lower(),
                    edge.label.lower(),
                ),
            )
        ]
        return KnowledgeBaseState(
            knowledge_base=self._knowledge_base_summary(knowledge_base),
            tags=tags,
            nodes=node_summaries,
            edges=edges,
            context=context,
        )

    async def _normalized_context(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> KnowledgeBaseContext:
        normalized_selected_node_id = (
            selected_node_id
            if selected_node_id in {node.id for node in knowledge_base.nodes}
            else None
        )
        tag_records = await self._knowledge_tags_by_ids(
            session,
            knowledge_base_id=knowledge_base.id,
            tag_ids=tag_ids,
        )
        selected_tag_names = [tag.name for tag in tag_records]
        visible_node_ids = sorted(
            self._node_ids_for_tag_scope(
                knowledge_base=knowledge_base,
                tag_ids=[tag.id for tag in tag_records],
                tag_match_mode=tag_match_mode,
            )
        )
        graph_node_ids = sorted(
            self._node_ids_for_graph_scope(
                knowledge_base=knowledge_base,
                selected_node_id=normalized_selected_node_id,
                graph_selection_mode=graph_selection_mode,
            )
        )
        scoped_node_ids = sorted(set(visible_node_ids).intersection(graph_node_ids))
        return KnowledgeBaseContext(
            selected_node_id=normalized_selected_node_id,
            graph_selection_mode=graph_selection_mode,
            tag_ids=[tag.id for tag in tag_records],
            selected_tag_names=selected_tag_names,
            tag_match_mode=tag_match_mode,
            media_types=sorted(dict.fromkeys(media_types)),
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
            visible_node_ids=visible_node_ids,
            scoped_node_ids=scoped_node_ids,
        )

    def _node_ids_for_tag_scope(
        self,
        *,
        knowledge_base: KnowledgeBase,
        tag_ids: list[str],
        tag_match_mode: TagMatchMode,
    ) -> set[str]:
        all_node_ids = {node.id for node in knowledge_base.nodes}
        if not tag_ids:
            return all_node_ids
        node_tag_ids = {
            node.id: {link.tag_id for link in node.tag_links}
            for node in knowledge_base.nodes
        }
        if tag_match_mode == "all":
            return {
                node_id
                for node_id, tags in node_tag_ids.items()
                if set(tag_ids).issubset(tags)
            }
        return {
            node_id
            for node_id, tags in node_tag_ids.items()
            if set(tag_ids).intersection(tags)
        }

    def _node_ids_for_graph_scope(
        self,
        *,
        knowledge_base: KnowledgeBase,
        selected_node_id: str | None,
        graph_selection_mode: GraphSelectionMode,
    ) -> set[str]:
        all_node_ids = {node.id for node in knowledge_base.nodes}
        if selected_node_id is None:
            return all_node_ids
        adjacency = self._adjacency_map(knowledge_base)
        if graph_selection_mode == "self":
            return {selected_node_id}
        if graph_selection_mode == "children":
            return {selected_node_id, *adjacency.get(selected_node_id, set())}
        descendants = self._descendants_from(adjacency=adjacency, start_node_id=selected_node_id)
        descendants.add(selected_node_id)
        return descendants

    @staticmethod
    def _adjacency_map(knowledge_base: KnowledgeBase) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {node.id: set() for node in knowledge_base.nodes}
        for edge in knowledge_base.edges:
            adjacency.setdefault(edge.from_node_id, set()).add(edge.to_node_id)
        return adjacency

    @staticmethod
    def _descendants_from(
        *,
        adjacency: dict[str, set[str]],
        start_node_id: str,
    ) -> set[str]:
        visited: set[str] = set()
        queue: list[str] = [start_node_id]
        while queue:
            current = queue.pop(0)
            for child_id in adjacency.get(current, set()):
                if child_id in visited:
                    continue
                visited.add(child_id)
                queue.append(child_id)
        return visited

    async def _filters_for_context(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        context: KnowledgeBaseContext,
    ):
        tag_records = await self._knowledge_tags_by_ids(
            session,
            knowledge_base_id=knowledge_base.id,
            tag_ids=context.tag_ids,
        )
        all_node_ids = {node.id for node in knowledge_base.nodes}
        scoped_node_ids = (
            []
            if set(context.scoped_node_ids) == all_node_ids
            else context.scoped_node_ids
        )
        return build_filter_groups(
            node_ids=scoped_node_ids,
            media_types=context.media_types,
            tag_slugs=[tag.slug for tag in tag_records],
            tag_match_mode=context.tag_match_mode,
        )

    async def _sync_node_vector_store_attributes(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        node: KnowledgeNode,
    ) -> None:
        if knowledge_base.openai_vector_store_id is None:
            return
        if node.status != "ready":
            return
        tag_records = await self._knowledge_tags_by_ids(
            session,
            knowledge_base_id=knowledge_base.id,
            tag_ids=[link.tag_id for link in node.tag_links],
        )
        tag_names = [tag.name for tag in tag_records]
        tag_slugs = [tag.slug for tag in tag_records]
        files_to_update: list[tuple[str, str | None, str]] = []
        if node.openai_original_file_id is not None:
            files_to_update.append((node.openai_original_file_id, None, "direct_file"))
        for artifact in node.derived_artifacts:
            if artifact.openai_file_id is None:
                continue
            files_to_update.append((artifact.openai_file_id, artifact.id, artifact.kind))

        for file_id, derived_artifact_id, derived_kind in files_to_update:
            await self._openai_gateway.update_vector_store_file_attributes(
                vector_store_id=knowledge_base.openai_vector_store_id,
                file_id=file_id,
                attributes=build_searchable_attributes(
                    knowledge_base_id=knowledge_base.id,
                    node_id=node.id,
                    node_title=node.display_title,
                    derived_artifact_id=derived_artifact_id,
                    source_kind=node.source_kind,
                    media_type=node.media_type,
                    derived_kind=derived_kind,
                    original_openai_file_id=node.openai_original_file_id,
                    original_filename=node.original_filename,
                    tag_names=tag_names,
                    tag_slugs=tag_slugs,
                ),
            )

    async def _unique_display_title(
        self,
        session,
        *,
        knowledge_base_id: str,
        base_title: str,
        exclude_node_id: str | None = None,
    ) -> str:
        candidate = base_title.strip() or "Untitled document"
        suffix = 2
        while True:
            existing = await session.scalar(
                select(KnowledgeNode.id).where(
                    KnowledgeNode.knowledge_base_id == knowledge_base_id,
                    func.lower(KnowledgeNode.display_title) == candidate.lower(),
                )
            )
            if existing is None or existing == exclude_node_id:
                return candidate
            candidate = f"{base_title.strip() or 'Untitled document'} ({suffix})"
            suffix += 1

    async def _unique_tag_slug(
        self,
        session,
        knowledge_base_id: str,
        base_slug: str,
    ) -> str:
        normalized_base_slug = base_slug or "tag"
        candidate = normalized_base_slug
        suffix = 2
        while True:
            existing = await session.scalar(
                select(KnowledgeTag.id).where(
                    KnowledgeTag.knowledge_base_id == knowledge_base_id,
                    KnowledgeTag.slug == candidate,
                )
            )
            if existing is None:
                return candidate
            candidate = f"{normalized_base_slug}-{suffix}"
            suffix += 1

    @staticmethod
    def _knowledge_base_summary(knowledge_base: KnowledgeBase) -> KnowledgeBaseSummary:
        return KnowledgeBaseSummary(
            id=knowledge_base.id,
            title=knowledge_base.title,
            description=knowledge_base.description,
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
            node_count=len(knowledge_base.nodes),
            tag_count=len(knowledge_base.tags),
            edge_count=len(knowledge_base.edges),
            vector_store_ready=knowledge_base.openai_vector_store_id is not None,
        )

    @staticmethod
    def _tag_summary(tag: KnowledgeTag, *, node_count: int) -> KnowledgeTagSummary:
        return KnowledgeTagSummary(
            id=tag.id,
            name=tag.name,
            slug=tag.slug,
            color=tag.color,
            node_count=node_count,
        )

    @staticmethod
    def _edge_summary(edge: KnowledgeEdge) -> KnowledgeEdgeSummary:
        return KnowledgeEdgeSummary(
            id=edge.id,
            from_node_id=edge.from_node_id,
            to_node_id=edge.to_node_id,
            from_node_title=edge.from_node.display_title,
            to_node_title=edge.to_node.display_title,
            label=edge.label,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
        )

    async def _node_summary(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        node: KnowledgeNode,
        clerk_user_id: str,
    ) -> KnowledgeNodeSummary:
        return KnowledgeNodeSummary(
            id=node.id,
            display_title=node.display_title,
            original_filename=node.original_filename,
            media_type=node.media_type,
            source_kind=node.source_kind,  # type: ignore[arg-type]
            status=node.status,  # type: ignore[arg-type]
            byte_size=node.byte_size,
            error_message=node.error_message,
            created_at=node.created_at,
            updated_at=node.updated_at,
            tags=[
                self._tag_summary(link.tag, node_count=self._tag_node_count(link.tag))
                for link in sorted(node.tag_links, key=lambda link: link.tag.name.lower())
            ],
            derived_kinds=sorted(artifact.kind for artifact in node.derived_artifacts),
            openai_original_file_id=node.openai_original_file_id,
            download_url=self._session_tokens.issue_node_download_url(
                clerk_user_id=clerk_user_id,
                node_id=node.id,
            )
            if node.openai_original_file_id
            else None,
            outgoing_edge_count=len(node.outgoing_edges),
            incoming_edge_count=len(node.incoming_edges),
        )

    async def _node_detail(
        self,
        session,
        *,
        knowledge_base: KnowledgeBase,
        node: KnowledgeNode,
        clerk_user_id: str,
    ) -> KnowledgeNodeDetail:
        summary = await self._node_summary(
            session,
            knowledge_base=knowledge_base,
            node=node,
            clerk_user_id=clerk_user_id,
        )
        return KnowledgeNodeDetail(
            **summary.model_dump(mode="python"),
            original_mime_type=node.original_mime_type,
            derived_artifacts=[
                DerivedArtifactSummary(
                    id=artifact.id,
                    kind=artifact.kind,
                    openai_file_id=artifact.openai_file_id,
                    text_content=artifact.text_content,
                    structured_payload=artifact.structured_payload,
                    created_at=artifact.created_at,
                    updated_at=artifact.updated_at,
                )
                for artifact in sorted(
                    node.derived_artifacts,
                    key=lambda artifact: artifact.created_at,
                )
            ],
            outgoing_edges=[
                self._edge_summary(edge)
                for edge in sorted(
                    node.outgoing_edges,
                    key=lambda edge: (edge.to_node.display_title.lower(), edge.label.lower()),
                )
            ],
            incoming_edges=[
                self._edge_summary(edge)
                for edge in sorted(
                    node.incoming_edges,
                    key=lambda edge: (edge.from_node.display_title.lower(), edge.label.lower()),
                )
            ],
        )

    @staticmethod
    def _tag_node_count(tag: KnowledgeTag) -> int:
        return len(tag.node_links)

    def _get_command_agent(self) -> KnowledgeBaseCommandAgent:
        if self._command_agent is None:
            from .command_agent import KnowledgeBaseCommandAgent

            self._command_agent = KnowledgeBaseCommandAgent(self._settings, self)
        return self._command_agent


def build_knowledge_base_title(display_name: str) -> str:
    normalized = display_name.strip() or "User"
    if normalized.endswith("s"):
        return f"{normalized}' Document Library"
    return f"{normalized}'s Document Library"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if normalized:
        return normalized[:72]
    return "tag"


def classify_source_kind(*, local_path: Path, media_type: str) -> str:
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("audio/"):
        return "audio"
    if media_type.startswith("video/"):
        return "video"
    if media_type.startswith("text/") or local_path.suffix.lower() in TEXT_EXTENSIONS:
        return "document"
    if media_type in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }:
        return "document"
    return "document"


def extract_text_document(*, local_path: Path, media_type: str) -> str | None:
    suffix = local_path.suffix.lower()
    if not (media_type.startswith("text/") or suffix in TEXT_EXTENSIONS):
        return None
    raw_bytes = local_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return normalized or None
    return None


def render_image_description(payload) -> str:
    lines = [payload.summary, "", payload.detailed_description]
    if payload.visible_text:
        lines.append("")
        lines.append("Visible text:")
        lines.extend(f"- {item}" for item in payload.visible_text)
    if payload.keywords:
        lines.append("")
        lines.append(f"Keywords: {', '.join(payload.keywords)}")
    return "\n".join(line for line in lines if line is not None).strip()


def _utcnow() -> datetime:
    return datetime.now(UTC)
