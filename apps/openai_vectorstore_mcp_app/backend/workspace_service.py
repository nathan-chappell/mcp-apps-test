from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .auth import get_current_clerk_access_token
from .clerk import ClerkAuthService, ClerkUserRecord
from .db import DatabaseManager
from .models import AppUser, Asset, AssetTag, DerivedArtifact, Workspace, WorkspaceTag
from .openai_gateway import (
    OpenAIWorkspaceGateway,
    build_filter_groups,
    build_searchable_attributes,
    guess_media_type,
)
from .qa_agent import WorkspaceQuestionAnswerer
from .schemas import (
    CreateWorkspaceResult,
    CreateWorkspaceTagResult,
    DeleteAssetResult,
    DeskAccessState,
    DeskCapabilities,
    DerivedArtifactSummary,
    SearchHit,
    UpdateWorkspaceAction,
    UpdateAssetTagsResult,
    UpdateWorkspaceResult,
    UploadFinalizeResult,
    UserSummary,
    WorkspaceAssetDetail,
    WorkspaceAssetSummary,
    WorkspaceBranchSearchResult,
    WorkspaceContext,
    WorkspaceDeskState,
    WorkspaceFileSearchResult,
    WorkspaceInfoResult,
    WorkspaceQueryMode,
    WorkspaceQueryResult,
    WorkspaceState,
    WorkspaceSummary,
    WorkspaceTagSummary,
)
from .settings import AppSettings
from .upload_sessions import AssetDownloadClaims, DeskSessionService, UploadSessionClaims

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


class WorkspaceService:
    """Owns desk state, workspace persistence, and ingestion orchestration."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        database: DatabaseManager,
        clerk_auth: ClerkAuthService,
        session_tokens: DeskSessionService,
        openai_gateway: OpenAIWorkspaceGateway,
        question_answerer: WorkspaceQuestionAnswerer,
    ) -> None:
        self._settings = settings
        self._database = database
        self._clerk_auth = clerk_auth
        self._session_tokens = session_tokens
        self._openai_gateway = openai_gateway
        self._question_answerer = question_answerer

    async def open_desk(self, *, selected_workspace_id: str | None) -> WorkspaceDeskState:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            return await self._build_desk_state(
                session,
                resolved_user=resolved_user,
                selected_workspace_id=selected_workspace_id,
            )

    async def get_desk_state(
        self,
        *,
        selected_workspace_id: str | None,
    ) -> WorkspaceDeskState:
        return await self.open_desk(selected_workspace_id=selected_workspace_id)

    async def get_workspace_info(
        self,
        *,
        selected_workspace_id: str | None,
        asset_id: str | None,
    ) -> WorkspaceInfoResult:
        desk_state = await self.get_desk_state(selected_workspace_id=selected_workspace_id)
        asset_detail: WorkspaceAssetDetail | None = None
        if asset_id is not None and desk_state.access.status == "active":
            asset_detail = await self.get_asset_detail(asset_id=asset_id)
        return WorkspaceInfoResult(desk_state=desk_state, asset_detail=asset_detail)

    async def query_workspace(
        self,
        *,
        selected_workspace_id: str | None,
        query: str | None,
        mode: WorkspaceQueryMode,
        asset_ids: list[str],
        tag_ids: list[str],
        media_types: list[str],
        include_web: bool,
        rewrite_query: bool,
        branch_factor: int,
        depth: int,
        max_results: int,
    ) -> WorkspaceQueryResult:
        desk_state = await self.get_desk_state(selected_workspace_id=selected_workspace_id)
        selected_workspace = desk_state.selected_workspace
        if selected_workspace is None or query is None or not query.strip():
            return WorkspaceQueryResult(kind="desk", desk_state=desk_state)

        context = WorkspaceContext(
            workspace_id=selected_workspace.workspace.id,
            asset_ids=asset_ids,
            tag_ids=tag_ids,
            media_types=media_types,
            available_tag_names=[tag.name for tag in selected_workspace.tags],
            include_web=include_web,
            rewrite_query=rewrite_query,
            branch_factor=branch_factor,
            depth=depth,
            max_results=max_results,
        )
        normalized_query = query.strip()

        if mode == "file_search":
            file_search_result = await self.workspace_file_search(
                query=normalized_query,
                context=context,
            )
            refreshed_desk_state = await self.get_desk_state(
                selected_workspace_id=file_search_result.workspace_id
            )
            return WorkspaceQueryResult(
                kind="file_search",
                desk_state=refreshed_desk_state,
                file_search_result=file_search_result,
            )

        if mode == "branch_search":
            branch_search_result = await self.workspace_branch_search(
                query=normalized_query,
                context=context,
            )
            refreshed_desk_state = await self.get_desk_state(
                selected_workspace_id=branch_search_result.workspace_id
            )
            return WorkspaceQueryResult(
                kind="branch_search",
                desk_state=refreshed_desk_state,
                branch_search_result=branch_search_result,
            )

        chat_result = await self.workspace_chat(
            question=normalized_query,
            context=context,
        )
        refreshed_desk_state = await self.get_desk_state(
            selected_workspace_id=chat_result.workspace_id
        )
        return WorkspaceQueryResult(
            kind="qa",
            desk_state=refreshed_desk_state,
            chat_result=chat_result,
        )

    async def update_workspace(
        self,
        *,
        action: UpdateWorkspaceAction,
        workspace_id: str | None,
        asset_id: str | None,
        tag_ids: list[str],
        title: str | None,
        description: str | None,
        name: str | None,
        color: str | None,
    ) -> UpdateWorkspaceResult:
        if action == "create_workspace":
            if title is None or not title.strip():
                raise ValueError("A workspace title is required.")
            result = await self.create_workspace(
                title=title.strip(),
                description=description.strip() if description else None,
            )
            return UpdateWorkspaceResult(
                action=action,
                desk_state=result.desk_state,
                workspace=result.workspace,
            )

        if action == "create_tag":
            if workspace_id is None:
                raise ValueError("workspace_id is required for create_tag.")
            if name is None or not name.strip():
                raise ValueError("A tag name is required.")
            result = await self.create_workspace_tag(
                workspace_id=workspace_id,
                name=name.strip(),
                color=color.strip() if color else None,
            )
            return UpdateWorkspaceResult(
                action=action,
                desk_state=result.desk_state,
                tag=result.tag,
            )

        if action == "set_asset_tags":
            if asset_id is None:
                raise ValueError("asset_id is required for set_asset_tags.")
            result = await self.update_asset_tags(asset_id=asset_id, tag_ids=tag_ids)
            return UpdateWorkspaceResult(
                action=action,
                desk_state=result.desk_state,
                asset=result.asset,
            )

        if action == "delete_asset":
            if asset_id is None:
                raise ValueError("asset_id is required for delete_asset.")
            result = await self.delete_asset(asset_id=asset_id)
            return UpdateWorkspaceResult(
                action=action,
                desk_state=result.desk_state,
                asset_id=result.asset_id,
                deleted=result.deleted,
            )

        if workspace_id is None:
            raise ValueError("workspace_id is required for prepare_upload.")
        upload_session = await self.issue_upload_session(workspace_id=workspace_id)
        return UpdateWorkspaceResult(
            action=action,
            upload_session=upload_session,
        )

    async def create_workspace(
        self,
        *,
        title: str,
        description: str | None,
    ) -> CreateWorkspaceResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)

            vector_store_id = await self._openai_gateway.create_vector_store(
                name=title,
                description=description,
                metadata={"owner": resolved_user.summary.clerk_user_id},
            )
            workspace = Workspace(
                user_id=resolved_user.app_user.id,
                title=title,
                description=description,
                openai_vector_store_id=vector_store_id,
                updated_at=_utcnow(),
            )
            session.add(workspace)
            await session.commit()
            await session.refresh(workspace)

            desk_state = await self._build_desk_state(
                session,
                resolved_user=resolved_user,
                selected_workspace_id=workspace.id,
            )
            return CreateWorkspaceResult(
                workspace=self._workspace_summary(workspace, asset_count=0, tag_count=0),
                desk_state=desk_state,
            )

    async def create_workspace_tag(
        self,
        *,
        workspace_id: str,
        name: str,
        color: str | None,
    ) -> CreateWorkspaceTagResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=workspace_id,
            )
            slug = slugify(name)
            tag = WorkspaceTag(
                workspace_id=workspace.id,
                name=name.strip(),
                slug=slug,
                color=color.strip() if color else None,
            )
            workspace.updated_at = _utcnow()
            session.add(tag)
            await session.commit()
            await session.refresh(tag)
            desk_state = await self._build_desk_state(
                session,
                resolved_user=resolved_user,
                selected_workspace_id=workspace.id,
            )
            return CreateWorkspaceTagResult(
                tag=WorkspaceTagSummary(
                    id=tag.id,
                    name=tag.name,
                    slug=tag.slug,
                    color=tag.color,
                    asset_count=0,
                ),
                desk_state=desk_state,
            )

    async def update_asset_tags(
        self,
        *,
        asset_id: str,
        tag_ids: list[str],
    ) -> UpdateAssetTagsResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            asset = await self._asset_for_user(session, resolved_user=resolved_user, asset_id=asset_id)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=asset.workspace_id,
            )
            tag_records = await self._workspace_tags_by_ids(
                session,
                workspace_id=workspace.id,
                tag_ids=tag_ids,
            )
            asset.tag_links = [AssetTag(asset_id=asset.id, tag_id=tag.id) for tag in tag_records]
            workspace.updated_at = _utcnow()
            await session.commit()
            await session.refresh(asset)
            desk_state = await self._build_desk_state(
                session,
                resolved_user=resolved_user,
                selected_workspace_id=workspace.id,
            )
            asset_summary = await self._asset_summary(session, asset, resolved_user.summary.clerk_user_id)
            return UpdateAssetTagsResult(asset=asset_summary, desk_state=desk_state)

    async def delete_asset(self, *, asset_id: str) -> DeleteAssetResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            asset = await self._asset_for_user(session, resolved_user=resolved_user, asset_id=asset_id)
            workspace_id = asset.workspace_id
            file_ids = {
                file_id
                for file_id in [
                    asset.openai_original_file_id,
                    *[
                        artifact.openai_file_id
                        for artifact in asset.derived_artifacts
                        if artifact.openai_file_id is not None
                    ],
                ]
                if file_id is not None
            }
            for file_id in file_ids:
                await self._openai_gateway.delete_file(file_id=file_id)
            await session.delete(asset)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=workspace_id,
            )
            workspace.updated_at = _utcnow()
            await session.commit()
            desk_state = await self._build_desk_state(
                session,
                resolved_user=resolved_user,
                selected_workspace_id=workspace_id,
            )
            return DeleteAssetResult(asset_id=asset_id, deleted=True, desk_state=desk_state)

    async def get_asset_detail(self, *, asset_id: str) -> WorkspaceAssetDetail:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            asset = await self._asset_for_user(session, resolved_user=resolved_user, asset_id=asset_id)
            return await self._asset_detail(
                session,
                asset,
                resolved_user.summary.clerk_user_id,
            )

    async def issue_upload_session(self, *, workspace_id: str):
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=workspace_id,
            )
            return self._session_tokens.issue_upload_session(
                clerk_user_id=resolved_user.summary.clerk_user_id,
                workspace_id=workspace_id,
            )

    async def ingest_upload(
        self,
        *,
        claims: UploadSessionClaims,
        local_path: Path,
        filename: str,
        declared_media_type: str | None,
        tag_ids: list[str],
    ) -> UploadFinalizeResult:
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
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=claims.workspace_id,
            )
            tag_records = await self._workspace_tags_by_ids(
                session,
                workspace_id=workspace.id,
                tag_ids=tag_ids,
            )
            media_type = guess_media_type(local_path, declared_media_type)
            source_kind = classify_source_kind(local_path=local_path, media_type=media_type)
            asset = Asset(
                workspace_id=workspace.id,
                created_by_user_id=app_user.id,
                filename=filename,
                media_type=media_type,
                source_kind=source_kind,
                status="processing",
                byte_size=local_path.stat().st_size,
                original_mime_type=media_type,
                updated_at=_utcnow(),
            )
            workspace.updated_at = _utcnow()
            session.add(asset)
            await session.flush()
            asset.tag_links = [AssetTag(asset_id=asset.id, tag_id=tag.id) for tag in tag_records]

            try:
                original_file_id = await self._openai_gateway.upload_original_file(
                    local_path=local_path,
                    purpose=self._openai_gateway.choose_original_file_purpose(
                        source_kind=source_kind
                    ),
                )
                asset.openai_original_file_id = original_file_id

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
                        workspace=workspace,
                        asset=asset,
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
                        workspace=workspace,
                        asset=asset,
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
                        workspace=workspace,
                        asset=asset,
                        kind="video_transcript",
                        text_content=derived_text,
                        structured_payload=payload,
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                elif derived_text is not None:
                    await self._store_derived_artifact(
                        session=session,
                        workspace=workspace,
                        asset=asset,
                        kind="document_text",
                        text_content=derived_text,
                        structured_payload=None,
                        tag_names=tag_names,
                        tag_slugs=tag_slugs,
                    )
                else:
                    await self._openai_gateway.attach_existing_file_to_vector_store(
                        vector_store_id=workspace.openai_vector_store_id,
                        file_id=original_file_id,
                        attributes=build_searchable_attributes(
                            workspace_id=workspace.id,
                            asset_id=asset.id,
                            derived_artifact_id=None,
                            source_kind=source_kind,
                            media_type=media_type,
                            derived_kind="direct_file",
                            original_openai_file_id=original_file_id,
                            original_filename=asset.filename,
                            tag_names=tag_names,
                            tag_slugs=tag_slugs,
                        ),
                    )

                asset.status = "ready"
                asset.error_message = None
                asset.updated_at = _utcnow()
                workspace.updated_at = _utcnow()
                await session.commit()
                await session.refresh(asset)
            except Exception as exc:
                asset.status = "failed"
                asset.error_message = str(exc)
                asset.updated_at = _utcnow()
                workspace.updated_at = _utcnow()
                await session.commit()
                raise

            asset_summary = await self._asset_summary(
                session,
                asset,
                resolved_user.summary.clerk_user_id,
            )
            return UploadFinalizeResult(asset=asset_summary)

    async def download_asset_bytes(
        self,
        *,
        claims: AssetDownloadClaims,
    ) -> tuple[WorkspaceAssetDetail, bytes]:
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
            asset = await self._asset_for_user(
                session,
                resolved_user=resolved_user,
                asset_id=claims.asset_id,
            )
            if asset.openai_original_file_id is None:
                raise FileNotFoundError("The requested asset has no stored original file.")
            detail = await self._asset_detail(
                session,
                asset,
                resolved_user.summary.clerk_user_id,
            )
            payload = await self._openai_gateway.read_file_bytes(
                file_id=asset.openai_original_file_id
            )
            return detail, payload

    async def workspace_file_search(
        self,
        *,
        query: str,
        context: WorkspaceContext,
    ) -> WorkspaceFileSearchResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=context.workspace_id,
            )
            filters, selected_names = await self._filters_for_context(
                session,
                workspace=workspace,
                context=context,
            )
            adjusted_context = context.model_copy(
                update={"selected_tag_names": selected_names}
            )
            hits = await self._openai_gateway.search_vector_store(
                vector_store_id=workspace.openai_vector_store_id,
                query=query,
                max_results=context.max_results,
                rewrite_query=context.rewrite_query,
                filters=filters,
            )
            return WorkspaceFileSearchResult(
                workspace_id=workspace.id,
                query=query,
                context=adjusted_context,
                hits=hits,
                total_hits=len(hits),
            )

    async def workspace_branch_search(
        self,
        *,
        query: str,
        context: WorkspaceContext,
    ) -> WorkspaceBranchSearchResult:
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=context.workspace_id,
            )
            filters, selected_names = await self._filters_for_context(
                session,
                workspace=workspace,
                context=context,
            )
            adjusted_context = context.model_copy(
                update={"selected_tag_names": selected_names}
            )

            queue: list[tuple[str, str | None, int, str, str | None]] = [
                ("node_1", None, 0, query, None)
            ]
            nodes: list[dict[str, object]] = []
            merged_hits: dict[str, tuple[SearchHit, float, int]] = {}
            next_id = 2

            while queue:
                node_id, parent_id, depth, node_query, rationale = queue.pop(0)
                hits = await self._openai_gateway.search_vector_store(
                    vector_store_id=workspace.openai_vector_store_id,
                    query=node_query,
                    max_results=context.max_results,
                    rewrite_query=context.rewrite_query,
                    filters=filters,
                )
                node_record: dict[str, object] = {
                    "id": node_id,
                    "parent_id": parent_id,
                    "depth": depth,
                    "query": node_query,
                    "rationale": rationale,
                    "hits": hits,
                    "children": [],
                }
                nodes.append(node_record)

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

                if depth + 1 >= context.depth:
                    continue
                expansion = await self._openai_gateway.expand_branch_queries(
                    query=node_query,
                    branch_factor=context.branch_factor,
                    tag_names=selected_names or context.available_tag_names,
                    hit_snippets=[hit.text for hit in hits],
                )
                child_queries = expansion.queries[: context.branch_factor]
                child_ids: list[str] = []
                for child_query in child_queries:
                    child_id = f"node_{next_id}"
                    next_id += 1
                    child_ids.append(child_id)
                    queue.append(
                        (
                            child_id,
                            node_id,
                            depth + 1,
                            child_query,
                            expansion.rationale,
                        )
                    )
                node_record["children"] = child_ids

            ranked_hits = sorted(
                merged_hits.values(),
                key=lambda item: (item[1] + (item[2] - 1) * 0.05),
                reverse=True,
            )
            return WorkspaceBranchSearchResult(
                workspace_id=workspace.id,
                seed_query=query,
                context=adjusted_context,
                nodes=[
                    {
                        "id": node["id"],
                        "parent_id": node["parent_id"],
                        "depth": node["depth"],
                        "query": node["query"],
                        "rationale": node["rationale"],
                        "hits": node["hits"],
                        "children": node["children"],
                    }
                    for node in nodes
                ],
                merged_hits=[item[0] for item in ranked_hits[: context.max_results]],
            )

    async def workspace_chat(self, *, question: str, context: WorkspaceContext):
        await self._database.ensure_ready()
        async with self._database.session() as session:
            resolved_user = await self._resolve_request_user(session)
            self._require_active(resolved_user)
            workspace = await self._workspace_for_user(
                session,
                resolved_user=resolved_user,
                workspace_id=context.workspace_id,
            )
            filters, selected_names = await self._filters_for_context(
                session,
                workspace=workspace,
                context=context,
            )
            adjusted_context = context.model_copy(
                update={"selected_tag_names": selected_names}
            )
            chat_result = await self._question_answerer.ask(
                workspace_id=workspace.id,
                vector_store_id=workspace.openai_vector_store_id,
                question=question,
                context=adjusted_context,
                conversation_id=workspace.openai_conversation_id,
                filters=filters,
            )
            workspace.openai_conversation_id = chat_result.conversation_id
            workspace.updated_at = _utcnow()
            await session.commit()
            return chat_result

    async def _store_derived_artifact(
        self,
        *,
        session,
        workspace: Workspace,
        asset: Asset,
        kind: str,
        text_content: str,
        structured_payload,
        tag_names: list[str],
        tag_slugs: list[str],
    ) -> None:
        derived = DerivedArtifact(
            asset_id=asset.id,
            kind=kind,
            text_content=text_content,
            structured_payload=structured_payload,
            updated_at=_utcnow(),
        )
        session.add(derived)
        await session.flush()
        derived.openai_file_id = await self._openai_gateway.create_text_artifact_and_attach(
            vector_store_id=workspace.openai_vector_store_id,
            filename=f"{asset.filename}.{kind}.md",
            text_content=text_content,
            attributes=build_searchable_attributes(
                workspace_id=workspace.id,
                asset_id=asset.id,
                derived_artifact_id=derived.id,
                source_kind=asset.source_kind,
                media_type=asset.media_type,
                derived_kind=kind,
                original_openai_file_id=asset.openai_original_file_id,
                original_filename=asset.filename,
                tag_names=tag_names,
                tag_slugs=tag_slugs,
            ),
        )
        derived.updated_at = _utcnow()

    async def _resolve_request_user(self, session) -> ResolvedUser:
        token = get_current_clerk_access_token()
        if token is None:
            app_user = await self._ensure_local_dev_user(session)
            return ResolvedUser(
                app_user=app_user,
                summary=UserSummary(
                    clerk_user_id=app_user.clerk_user_id,
                    display_name=app_user.display_name or "Local Developer",
                    primary_email=app_user.primary_email,
                    active=app_user.active,
                    role=app_user.role,
                ),
            )

        clerk_record = await self._clerk_auth.get_user_record(token.subject)
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

    async def _ensure_local_dev_user(self, session) -> AppUser:
        existing = await self._user_by_clerk_id(session, "local-dev")
        now = _utcnow()
        if existing is None:
            existing = AppUser(
                clerk_user_id="local-dev",
                primary_email="local-dev@example.com",
                display_name="Local Developer",
                active=True,
                role="admin",
                last_seen_at=now,
            )
            session.add(existing)
        else:
            existing.display_name = "Local Developer"
            existing.active = True
            existing.role = "admin"
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

    async def _build_desk_state(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        selected_workspace_id: str | None,
    ) -> WorkspaceDeskState:
        workspaces = (
            (
                await session.execute(
                    select(Workspace)
                    .where(Workspace.user_id == resolved_user.app_user.id)
                    .options(
                        selectinload(Workspace.tags),
                        selectinload(Workspace.assets).selectinload(Asset.derived_artifacts),
                        selectinload(Workspace.assets)
                        .selectinload(Asset.tag_links)
                        .selectinload(AssetTag.tag),
                    )
                    .order_by(Workspace.updated_at.desc(), Workspace.created_at.desc())
                )
            )
            .scalars()
            .unique()
            .all()
        )
        workspace_summaries = [
            self._workspace_summary(
                workspace,
                asset_count=len(workspace.assets),
                tag_count=len(workspace.tags),
            )
            for workspace in workspaces
        ]
        selected_workspace = None
        selected_id = selected_workspace_id
        if selected_id is None and workspaces:
            selected_id = workspaces[0].id
        if selected_id is not None:
            selected_workspace = next(
                (workspace for workspace in workspaces if workspace.id == selected_id),
                None,
            )
        selected_state = None
        if selected_workspace is not None and resolved_user.summary.active:
            selected_state = await self._workspace_state(
                session,
                selected_workspace,
                resolved_user.summary.clerk_user_id,
            )
        access = DeskAccessState(
            status="active" if resolved_user.summary.active else "pending_access",
            message=(
                "Access active. You can create workspaces, upload files, and run retrieval."
                if resolved_user.summary.active
                else "Signed in successfully. Access is pending manual activation in Clerk."
            ),
            user=resolved_user.summary,
        )
        return WorkspaceDeskState(
            access=access,
            workspaces=workspace_summaries,
            selected_workspace_id=selected_state.workspace.id if selected_state else selected_id,
            selected_workspace=selected_state,
            capabilities=DeskCapabilities(
                upload_url=f"{self._settings.normalized_app_base_url}/api/uploads",
                upload_token_ttl_seconds=self._settings.upload_session_max_age_seconds,
                supports_video_audio_extraction=True,
                accepted_hint=(
                    "Upload text-like files directly, plus images, audio, and video. "
                    "Non-text documents can still be attached to the vector store when OpenAI supports them."
                ),
            ),
        )

    async def _workspace_state(
        self,
        session,
        workspace: Workspace,
        clerk_user_id: str,
    ) -> WorkspaceState:
        tag_counts = {
            tag.id: 0
            for tag in workspace.tags
        }
        asset_summaries: list[WorkspaceAssetSummary] = []
        sorted_assets = sorted(
            workspace.assets,
            key=lambda asset: (asset.updated_at, asset.created_at),
            reverse=True,
        )
        for asset in sorted_assets:
            for link in asset.tag_links:
                tag_counts[link.tag_id] = tag_counts.get(link.tag_id, 0) + 1
            asset_summaries.append(
                await self._asset_summary(session, asset, clerk_user_id)
            )
        tags = [
            WorkspaceTagSummary(
                id=tag.id,
                name=tag.name,
                slug=tag.slug,
                color=tag.color,
                asset_count=tag_counts.get(tag.id, 0),
            )
            for tag in sorted(workspace.tags, key=lambda tag: tag.name.lower())
        ]
        return WorkspaceState(
            workspace=self._workspace_summary(
                workspace,
                asset_count=len(workspace.assets),
                tag_count=len(workspace.tags),
            ),
            tags=tags,
            assets=asset_summaries,
            context=WorkspaceContext(
                workspace_id=workspace.id,
                available_tag_names=[tag.name for tag in tags],
            ),
        )

    async def _workspace_tags_by_ids(
        self,
        session,
        *,
        workspace_id: str,
        tag_ids: list[str],
    ) -> list[WorkspaceTag]:
        if not tag_ids:
            return []
        records = (
            (
                await session.execute(
                    select(WorkspaceTag).where(
                        WorkspaceTag.workspace_id == workspace_id,
                        WorkspaceTag.id.in_(tag_ids),
                    )
                )
            )
            .scalars()
            .all()
        )
        if len(records) != len(set(tag_ids)):
            raise ValueError("One or more tag IDs are invalid for this workspace.")
        return sorted(records, key=lambda tag: tag.name.lower())

    async def _workspace_for_user(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        workspace_id: str,
    ) -> Workspace:
        workspace = await session.scalar(
            select(Workspace)
            .where(
                Workspace.id == workspace_id,
                Workspace.user_id == resolved_user.app_user.id,
            )
            .options(
                selectinload(Workspace.tags),
                selectinload(Workspace.assets).selectinload(Asset.derived_artifacts),
                selectinload(Workspace.assets)
                .selectinload(Asset.tag_links)
                .selectinload(AssetTag.tag),
            )
        )
        if workspace is None:
            raise PermissionError("Workspace not found or not owned by the current user.")
        return workspace

    async def _asset_for_user(
        self,
        session,
        *,
        resolved_user: ResolvedUser,
        asset_id: str,
    ) -> Asset:
        asset = await session.scalar(
            select(Asset)
            .join(Workspace, Workspace.id == Asset.workspace_id)
            .where(
                Asset.id == asset_id,
                Workspace.user_id == resolved_user.app_user.id,
            )
            .options(
                selectinload(Asset.derived_artifacts),
                selectinload(Asset.tag_links).selectinload(AssetTag.tag),
            )
        )
        if asset is None:
            raise PermissionError("Asset not found or not owned by the current user.")
        return asset

    async def _asset_summary(
        self,
        session,
        asset: Asset,
        clerk_user_id: str,
    ) -> WorkspaceAssetSummary:
        return WorkspaceAssetSummary(
            id=asset.id,
            filename=asset.filename,
            media_type=asset.media_type,
            source_kind=asset.source_kind,  # type: ignore[arg-type]
            status=asset.status,  # type: ignore[arg-type]
            byte_size=asset.byte_size,
            error_message=asset.error_message,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
            tags=[
                WorkspaceTagSummary(
                    id=link.tag.id,
                    name=link.tag.name,
                    slug=link.tag.slug,
                    color=link.tag.color,
                )
                for link in sorted(asset.tag_links, key=lambda link: link.tag.name.lower())
            ],
            derived_kinds=sorted(artifact.kind for artifact in asset.derived_artifacts),
            openai_original_file_id=asset.openai_original_file_id,
            download_url=self._session_tokens.issue_asset_download_url(
                clerk_user_id=clerk_user_id,
                asset_id=asset.id,
            )
            if asset.openai_original_file_id
            else None,
        )

    async def _asset_detail(
        self,
        session,
        asset: Asset,
        clerk_user_id: str,
    ) -> WorkspaceAssetDetail:
        summary = await self._asset_summary(session, asset, clerk_user_id)
        return WorkspaceAssetDetail(
            **summary.model_dump(mode="python"),
            original_mime_type=asset.original_mime_type,
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
                    asset.derived_artifacts,
                    key=lambda artifact: artifact.created_at,
                )
            ],
        )

    async def _filters_for_context(
        self,
        session,
        *,
        workspace: Workspace,
        context: WorkspaceContext,
    ) -> tuple[object, list[str]]:
        tag_records = await self._workspace_tags_by_ids(
            session,
            workspace_id=workspace.id,
            tag_ids=context.tag_ids,
        )
        tag_slugs = [tag.slug for tag in tag_records]
        tag_names = [tag.name for tag in tag_records]
        filters = build_filter_groups(
            asset_ids=context.asset_ids,
            media_types=context.media_types,
            tag_slugs=tag_slugs,
        )
        return filters, tag_names

    @staticmethod
    def _workspace_summary(
        workspace: Workspace,
        *,
        asset_count: int,
        tag_count: int,
    ) -> WorkspaceSummary:
        return WorkspaceSummary(
            id=workspace.id,
            title=workspace.title,
            description=workspace.description,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            asset_count=asset_count,
            tag_count=tag_count,
        )


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
