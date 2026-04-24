from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeAlias

from openai.types.responses import ResponseFileSearchToolCall, ResponseFunctionWebSearch
from openai.types.vector_store_search_response import VectorStoreSearchResponse
from pydantic import BaseModel, Field

StructuredPayload: TypeAlias = dict[str, Any] | list[Any] | None
OpenAIAttributeValue: TypeAlias = str | float | bool
OpenAIAttributes: TypeAlias = dict[str, OpenAIAttributeValue]
NodeStatus: TypeAlias = Literal["processing", "ready", "failed"]
SourceKind: TypeAlias = Literal["document", "audio", "image", "video", "other"]
KnowledgeQueryMode: TypeAlias = Literal["qa", "file_search", "branch_search"]
KnowledgeQueryKind: TypeAlias = Literal["knowledge_base", "qa", "file_search", "branch_search"]
KnowledgeAccessStatus: TypeAlias = Literal["active", "pending_access"]
UpdateKnowledgeBaseAction: TypeAlias = Literal[
    "prepare_upload",
    "rename_node",
    "create_tag",
    "set_node_tags",
    "upsert_edge",
    "delete_edge",
    "delete_node",
]
GraphSelectionMode: TypeAlias = Literal["self", "children", "descendants"]
TagMatchMode: TypeAlias = Literal["all", "any"]
KnowledgeCommandStatus: TypeAlias = Literal["executed", "pending_confirmation", "rejected"]
CommandParserKind: TypeAlias = Literal["agent", "fallback", "manual"]


def _read_text_from_search_result(search_result: VectorStoreSearchResponse) -> str:
    return "\n".join(
        content.text for content in search_result.content if content.type == "text"
    ).strip()


def _extract_tags(attributes: OpenAIAttributes | None) -> list[str]:
    if attributes is None:
        return []
    raw_tag_names = attributes.get("tag_names")
    if not isinstance(raw_tag_names, str) or not raw_tag_names:
        return []
    return [part for part in raw_tag_names.split(",") if part]


def _string_attribute(
    attributes: OpenAIAttributes | None,
    key: str,
) -> str | None:
    value = (attributes or {}).get(key)
    return value if isinstance(value, str) else None


class UserSummary(BaseModel):
    clerk_user_id: str
    display_name: str
    primary_email: str | None = None
    active: bool
    role: str | None = None


class KnowledgeAccessState(BaseModel):
    status: KnowledgeAccessStatus
    message: str
    user: UserSummary


class KnowledgeTagSummary(BaseModel):
    id: str
    name: str
    slug: str
    color: str | None = None
    node_count: int = 0


class DerivedArtifactSummary(BaseModel):
    id: str
    kind: str
    openai_file_id: str | None = None
    text_content: str
    structured_payload: StructuredPayload = None
    created_at: datetime
    updated_at: datetime


class KnowledgeEdgeSummary(BaseModel):
    id: str
    from_node_id: str
    to_node_id: str
    from_node_title: str
    to_node_title: str
    label: str
    created_at: datetime
    updated_at: datetime


class KnowledgeNodeSummary(BaseModel):
    id: str
    display_title: str
    original_filename: str
    media_type: str
    source_kind: SourceKind
    status: NodeStatus
    byte_size: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    tags: list[KnowledgeTagSummary] = Field(default_factory=list)
    derived_kinds: list[str] = Field(default_factory=list)
    openai_original_file_id: str | None = None
    download_url: str | None = None
    outgoing_edge_count: int = 0
    incoming_edge_count: int = 0


class KnowledgeNodeDetail(KnowledgeNodeSummary):
    original_mime_type: str | None = None
    derived_artifacts: list[DerivedArtifactSummary] = Field(default_factory=list)
    outgoing_edges: list[KnowledgeEdgeSummary] = Field(default_factory=list)
    incoming_edges: list[KnowledgeEdgeSummary] = Field(default_factory=list)


class KnowledgeBaseSummary(BaseModel):
    id: str
    title: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime
    node_count: int = 0
    tag_count: int = 0
    edge_count: int = 0
    vector_store_ready: bool = False


class KnowledgeBaseContext(BaseModel):
    selected_node_id: str | None = None
    graph_selection_mode: GraphSelectionMode = "self"
    tag_ids: list[str] = Field(default_factory=list)
    selected_tag_names: list[str] = Field(default_factory=list)
    tag_match_mode: TagMatchMode = "all"
    media_types: list[str] = Field(default_factory=list)
    include_web: bool = False
    rewrite_query: bool = True
    branch_factor: int = Field(default=3, ge=1, le=6)
    depth: int = Field(default=2, ge=1, le=4)
    max_results: int = Field(default=8, ge=1, le=20)
    visible_node_ids: list[str] = Field(default_factory=list)
    scoped_node_ids: list[str] = Field(default_factory=list)


class KnowledgeBaseState(BaseModel):
    knowledge_base: KnowledgeBaseSummary
    tags: list[KnowledgeTagSummary] = Field(default_factory=list)
    nodes: list[KnowledgeNodeSummary] = Field(default_factory=list)
    edges: list[KnowledgeEdgeSummary] = Field(default_factory=list)
    context: KnowledgeBaseContext


class KnowledgeBaseCapabilities(BaseModel):
    upload_url: str
    upload_token_ttl_seconds: int
    confirmation_token_ttl_seconds: int
    supports_video_audio_extraction: bool
    accepted_hint: str


class KnowledgeBaseDeskState(BaseModel):
    access: KnowledgeAccessState
    knowledge_base: KnowledgeBaseState | None = None
    capabilities: KnowledgeBaseCapabilities


class SearchHit(BaseModel):
    node_id: str
    node_title: str
    original_filename: str
    derived_artifact_id: str | None = None
    openai_file_id: str
    original_openai_file_id: str | None = None
    media_type: str
    source_kind: str
    score: float
    text: str
    tags: list[str] = Field(default_factory=list)
    attributes: OpenAIAttributes | None = None

    @classmethod
    def from_openai(cls, search_result: VectorStoreSearchResponse) -> SearchHit:
        attributes = search_result.attributes
        node_title = _string_attribute(attributes, "node_title") or search_result.filename
        original_filename = (
            _string_attribute(attributes, "original_filename") or search_result.filename
        )
        return cls(
            node_id=str((attributes or {}).get("node_id") or ""),
            node_title=node_title,
            original_filename=original_filename,
            derived_artifact_id=_string_attribute(attributes, "derived_artifact_id"),
            openai_file_id=search_result.file_id,
            original_openai_file_id=_string_attribute(
                attributes,
                "original_openai_file_id",
            ),
            media_type=_string_attribute(attributes, "media_type")
            or "application/octet-stream",
            source_kind=_string_attribute(attributes, "source_kind") or "other",
            score=search_result.score,
            text=_read_text_from_search_result(search_result),
            tags=_extract_tags(attributes),
            attributes=attributes,
        )


class FileSearchCallSummary(BaseModel):
    id: str
    status: str
    queries: list[str]
    results: list[SearchHit] = Field(default_factory=list)

    @classmethod
    def from_openai(
        cls,
        tool_call: ResponseFileSearchToolCall,
    ) -> FileSearchCallSummary:
        return cls(
            id=tool_call.id,
            status=tool_call.status,
            queries=list(tool_call.queries),
            results=[
                SearchHit(
                    node_id=str((result.attributes or {}).get("node_id") or ""),
                    node_title=_string_attribute(result.attributes, "node_title")
                    or result.filename
                    or "",
                    original_filename=_string_attribute(
                        result.attributes,
                        "original_filename",
                    )
                    or result.filename
                    or "",
                    derived_artifact_id=_string_attribute(
                        result.attributes,
                        "derived_artifact_id",
                    ),
                    openai_file_id=result.file_id or "",
                    original_openai_file_id=_string_attribute(
                        result.attributes,
                        "original_openai_file_id",
                    ),
                    media_type=_string_attribute(result.attributes, "media_type")
                    or "application/octet-stream",
                    source_kind=_string_attribute(result.attributes, "source_kind")
                    or "other",
                    score=result.score or 0.0,
                    text=result.text or "",
                    tags=_extract_tags(result.attributes),
                    attributes=result.attributes,
                )
                for result in (tool_call.results or [])
            ],
        )


class WebSearchCallSummary(BaseModel):
    id: str
    status: str
    query: str
    sources: list[str] = Field(default_factory=list)

    @classmethod
    def from_openai(
        cls,
        tool_call: ResponseFunctionWebSearch,
    ) -> WebSearchCallSummary:
        action = tool_call.action
        query = getattr(action, "query", "")
        sources: list[str] = []
        raw_sources = getattr(action, "sources", None)
        if raw_sources is not None:
            for source in raw_sources:
                source_url = getattr(source, "url", None)
                if isinstance(source_url, str) and source_url:
                    sources.append(source_url)
        if getattr(action, "url", None):
            sources.append(str(action.url))
        return cls(
            id=tool_call.id,
            status=tool_call.status,
            query=query,
            sources=list(dict.fromkeys(sources)),
        )


class KnowledgeFileSearchResult(BaseModel):
    knowledge_base_id: str
    query: str
    context: KnowledgeBaseContext
    hits: list[SearchHit]
    total_hits: int


class BranchSearchNode(BaseModel):
    id: str
    parent_id: str | None = None
    depth: int
    query: str
    rationale: str | None = None
    hits: list[SearchHit] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


class KnowledgeBranchSearchResult(BaseModel):
    knowledge_base_id: str
    seed_query: str
    context: KnowledgeBaseContext
    nodes: list[BranchSearchNode]
    merged_hits: list[SearchHit]


class KnowledgeAnswerCitation(BaseModel):
    source: Literal["knowledge_base", "web"]
    label: str
    node_id: str | None = None
    node_title: str | None = None
    original_filename: str | None = None
    url: str | None = None
    quote: str | None = None


class KnowledgeChatResult(BaseModel):
    knowledge_base_id: str
    question: str
    answer: str
    model: str
    include_web: bool
    conversation_id: str
    context: KnowledgeBaseContext
    search_calls: list[FileSearchCallSummary] = Field(default_factory=list)
    web_search_calls: list[WebSearchCallSummary] = Field(default_factory=list)
    citations: list[KnowledgeAnswerCitation] = Field(default_factory=list)


class KnowledgeQueryResult(BaseModel):
    kind: KnowledgeQueryKind
    knowledge_base_state: KnowledgeBaseDeskState
    file_search_result: KnowledgeFileSearchResult | None = None
    branch_search_result: KnowledgeBranchSearchResult | None = None
    chat_result: KnowledgeChatResult | None = None


class KnowledgeInfoResult(BaseModel):
    knowledge_base_state: KnowledgeBaseDeskState
    node_detail: KnowledgeNodeDetail | None = None


class UploadSessionResult(BaseModel):
    upload_url: str
    upload_token: str
    expires_at: int


class UploadFinalizeResult(BaseModel):
    node: KnowledgeNodeSummary


class PendingCommandResult(BaseModel):
    token: str
    prompt: str
    summary: str
    expires_at: int


class UpdateKnowledgeBaseResult(BaseModel):
    action: UpdateKnowledgeBaseAction
    knowledge_base_state: KnowledgeBaseDeskState | None = None
    node: KnowledgeNodeSummary | None = None
    edge: KnowledgeEdgeSummary | None = None
    tag: KnowledgeTagSummary | None = None
    deleted_node_id: str | None = None
    deleted_edge_id: str | None = None
    upload_session: UploadSessionResult | None = None


class KnowledgeBaseCommandResult(BaseModel):
    status: KnowledgeCommandStatus
    message: str
    action: str | None = None
    parser: CommandParserKind
    raw_command: str
    knowledge_base_state: KnowledgeBaseDeskState
    pending_confirmation: PendingCommandResult | None = None
    node: KnowledgeNodeSummary | None = None
    edge: KnowledgeEdgeSummary | None = None
    tag: KnowledgeTagSummary | None = None


class BranchExpansion(BaseModel):
    rationale: str | None = None
    queries: list[str] = Field(default_factory=list)


class ImageDescriptionPayload(BaseModel):
    summary: str
    detailed_description: str
    visible_text: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
