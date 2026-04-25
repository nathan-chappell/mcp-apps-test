from __future__ import annotations

from collections.abc import AsyncIterator
import logging
from time import perf_counter
from typing import Any, cast

import httpx
from agents import Agent, Runner
from agents.items import ModelResponse
from agents.lifecycle import RunHooksBase
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.tool import Tool
from chatkit.agents import AgentContext as ChatKitAgentContext
from chatkit.agents import ThreadItemConverter, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ChatKitReq, ThreadMetadata, ThreadStreamEvent, UserMessageItem
from fastapi import HTTPException, status
from openai.types.responses.response_input_item_param import Message, ResponseInputItemParam
from openai.types.shared import Reasoning
from pydantic import TypeAdapter

from .chat_metadata import (
    AppChatMetadata,
    ChatMetadataPatch,
    ChatRequestMetadata,
    merge_chat_metadata,
    parse_chat_metadata,
    parse_chat_request_metadata,
)
from .chat_store import FileDeskChatContext, FileDeskChatStore
from .chat_usage import accumulate_usage
from .file_library_service import FileLibraryService
from .openai_tracing import (
    OpenAITraceRefs,
    build_openai_trace_refs,
    clear_active_openai_tool_trace_refs,
    latest_openai_trace_refs,
    set_active_openai_tool_trace_refs,
)
from .settings import AppSettings

logger = logging.getLogger("chatkit.server")

MODEL_ALIASES = {
    "default": "gpt-5.4-mini",
    "lightweight": "gpt-5.4-nano",
    "balanced": "gpt-5.4-mini",
    "powerful": "gpt-5.4",
}
DEFAULT_MODEL = MODEL_ALIASES["balanced"]
MAX_AGENT_TURNS = 20


class _ChatRunHooks(RunHooksBase[Any, Any]):
    def __init__(
        self,
        *,
        thread_id: str,
        model: str,
        trace_refs: OpenAITraceRefs,
    ) -> None:
        self._thread_id = thread_id
        self._model = model
        self._latest_trace_refs = trace_refs

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: ModelResponse,
    ) -> None:
        del context, agent
        self._latest_trace_refs = build_openai_trace_refs(
            response_id=response.response_id,
            conversation_id=self._latest_trace_refs.conversation_id,
        )
        logger.info(
            "chat_model_response_received thread_id=%s model=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s",
            self._thread_id,
            self._model,
            self._latest_trace_refs.response_id,
            self._latest_trace_refs.response_log_url,
            self._latest_trace_refs.conversation_id,
            self._latest_trace_refs.conversation_log_url,
        )

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Tool,
    ) -> None:
        del agent
        tool_name = _tool_name(tool=tool, context=context)
        tool_call_id = getattr(context, "tool_call_id", None)
        tool_arguments = getattr(context, "tool_arguments", None)
        set_active_openai_tool_trace_refs(self._latest_trace_refs)
        logger.info(
            "chat_mcp_tool_started thread_id=%s tool=%s tool_call_id=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s arguments=%s",
            self._thread_id,
            tool_name,
            tool_call_id,
            self._latest_trace_refs.response_id,
            self._latest_trace_refs.response_log_url,
            self._latest_trace_refs.conversation_id,
            self._latest_trace_refs.conversation_log_url,
            _truncate_for_log(tool_arguments),
        )

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Tool,
        result: str,
    ) -> None:
        del agent
        tool_name = _tool_name(tool=tool, context=context)
        tool_call_id = getattr(context, "tool_call_id", None)
        clear_active_openai_tool_trace_refs()
        logger.info(
            "chat_mcp_tool_completed thread_id=%s tool=%s tool_call_id=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s result=%s",
            self._thread_id,
            tool_name,
            tool_call_id,
            self._latest_trace_refs.response_id,
            self._latest_trace_refs.response_log_url,
            self._latest_trace_refs.conversation_id,
            self._latest_trace_refs.conversation_log_url,
            _truncate_for_log(result),
        )


class FileDeskChatKitServer(ChatKitServer[FileDeskChatContext]):
    def __init__(
        self,
        *,
        settings: AppSettings,
        store: FileDeskChatStore,
        file_library: FileLibraryService,
    ) -> None:
        super().__init__(store=store)
        self._settings = settings
        self._file_library = file_library
        self._converter = ThreadItemConverter()

    async def build_request_context(
        self,
        raw_request: bytes | str,
        *,
        clerk_user_id: str,
        user_email: str | None,
        display_name: str,
        bearer_token: str,
        request_app: Any,
    ) -> FileDeskChatContext:
        parsed_request = TypeAdapter(ChatKitReq).validate_json(raw_request)
        try:
            request_metadata = parse_chat_request_metadata(parsed_request.metadata)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        return FileDeskChatContext(
            clerk_user_id=clerk_user_id,
            user_email=user_email,
            display_name=display_name,
            bearer_token=bearer_token,
            selected_file_ids=list(request_metadata.get("selected_file_ids", [])),
            thread_origin=_normalize_origin(request_metadata),
            request_app=request_app,
        )

    def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: FileDeskChatContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        return self._respond(thread=thread, input_user_message=input_user_message, context=context)

    async def _respond(
        self,
        *,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: FileDeskChatContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        started_at = perf_counter()
        typed_metadata = parse_chat_metadata(thread.metadata)
        requested_model = self._resolve_requested_model(input_user_message=input_user_message)
        prior_trace_refs = build_openai_trace_refs(
            response_id=typed_metadata.get("openai_previous_response_id"),
            conversation_id=typed_metadata.get("openai_conversation_id"),
        )
        if thread.title is None and input_user_message is not None:
            resolved_title = _title_from_user_message(input_user_message)
            if resolved_title is not None:
                thread.title = resolved_title

        agent_input, input_mode = await self._agent_input_for_turn(
            thread=thread,
            input_user_message=input_user_message,
            context=context,
            typed_metadata=typed_metadata,
        )
        selected_file_context = await self._selected_file_context_items(context=context)
        if selected_file_context:
            agent_input = selected_file_context + agent_input

        logger.info(
            "chat_turn_started thread_id=%s model=%s input_mode=%s previous_response_id=%s previous_response_log_url=%s conversation_id=%s conversation_log_url=%s selected_file_ids=%s",
            thread.id,
            requested_model,
            input_mode,
            prior_trace_refs.response_id,
            prior_trace_refs.response_log_url,
            prior_trace_refs.conversation_id,
            prior_trace_refs.conversation_log_url,
            len(context.selected_file_ids),
        )

        agent_context = ChatKitAgentContext[FileDeskChatContext](
            thread=thread,
            store=self.store,
            request_context=context,
        )
        mcp_server = self._build_mcp_server(context)
        run_hooks = _ChatRunHooks(
            thread_id=thread.id,
            model=requested_model,
            trace_refs=prior_trace_refs,
        )
        try:
            async with mcp_server:
                agent = Agent[ChatKitAgentContext[FileDeskChatContext]](
                    name="file_desk_agent",
                    model=requested_model,
                    model_settings=_model_settings_override_for_model(requested_model) or ModelSettings(),
                    mcp_servers=[mcp_server],
                    instructions=self._agent_instructions,
                )
                result = Runner.run_streamed(
                    agent,
                    agent_input,
                    context=agent_context,
                    max_turns=MAX_AGENT_TURNS,
                    hooks=run_hooks,
                    previous_response_id=prior_trace_refs.response_id,
                    conversation_id=prior_trace_refs.conversation_id,
                )
                async for event in stream_agent_response(agent_context, result):
                    yield event
        finally:
            clear_active_openai_tool_trace_refs()

        usage = typed_metadata.get("usage")
        for response in result.raw_responses:
            usage = accumulate_usage(usage, response.usage, model=requested_model)

        latest_trace_refs = latest_openai_trace_refs(list(result.raw_responses))
        if latest_trace_refs.response_id is None and result.last_response_id:
            latest_trace_refs = build_openai_trace_refs(
                response_id=result.last_response_id,
                conversation_id=latest_trace_refs.conversation_id or prior_trace_refs.conversation_id,
            )
        if latest_trace_refs.conversation_id is None and prior_trace_refs.conversation_id is not None:
            latest_trace_refs = build_openai_trace_refs(
                response_id=latest_trace_refs.response_id,
                conversation_id=prior_trace_refs.conversation_id,
            )

        patch: ChatMetadataPatch = {}
        if thread.title:
            patch["title"] = thread.title
        if usage is not None:
            patch["usage"] = usage
        if latest_trace_refs.response_id is not None:
            patch["openai_previous_response_id"] = latest_trace_refs.response_id
        if latest_trace_refs.conversation_id is not None:
            patch["openai_conversation_id"] = latest_trace_refs.conversation_id
        merged_metadata = self._apply_metadata_patch(thread, patch=patch)
        merged_usage = merged_metadata.get("usage")
        logger.info(
            "chat_turn_completed thread_id=%s model=%s input_mode=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s total_input_tokens=%s total_output_tokens=%s total_cost_usd=%s duration_ms=%.1f",
            thread.id,
            requested_model,
            input_mode,
            latest_trace_refs.response_id,
            latest_trace_refs.response_log_url,
            latest_trace_refs.conversation_id,
            latest_trace_refs.conversation_log_url,
            merged_usage.get("input_tokens") if merged_usage is not None else None,
            merged_usage.get("output_tokens") if merged_usage is not None else None,
            merged_usage.get("cost_usd") if merged_usage is not None else None,
            (perf_counter() - started_at) * 1000,
        )

    def _apply_metadata_patch(
        self,
        thread: ThreadMetadata,
        *,
        patch: ChatMetadataPatch,
    ) -> AppChatMetadata:
        merged_metadata = merge_chat_metadata(parse_chat_metadata(thread.metadata), patch)
        thread.metadata = dict(merged_metadata)
        return merged_metadata

    def _build_mcp_server(self, context: FileDeskChatContext) -> MCPServerStreamableHttp:
        def httpx_client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=context.request_app),
                base_url=self._settings.normalized_app_base_url,
                follow_redirects=True,
                headers=headers,
                timeout=timeout,
                auth=auth,
            )

        return MCPServerStreamableHttp(
            params={
                "url": f"{self._settings.normalized_app_base_url}/mcp/",
                "headers": {
                    "Authorization": f"Bearer {context.bearer_token}",
                },
                "httpx_client_factory": httpx_client_factory,
            },
            name="file_desk_mcp",
        )

    def _resolve_requested_model(self, *, input_user_message: UserMessageItem | None) -> str:
        requested_model = None
        if input_user_message is not None:
            requested_model = input_user_message.inference_options.model
        if requested_model is None:
            return DEFAULT_MODEL
        normalized = requested_model.strip()
        if not normalized:
            return DEFAULT_MODEL
        return MODEL_ALIASES.get(normalized, normalized)

    async def _agent_input_for_turn(
        self,
        *,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: FileDeskChatContext,
        typed_metadata: AppChatMetadata,
    ) -> tuple[list[ResponseInputItemParam], str]:
        if input_user_message is not None and self._uses_server_managed_conversation(typed_metadata):
            return (
                cast(
                    list[ResponseInputItemParam],
                    await self._converter.to_agent_input(input_user_message),
                ),
                "delta",
            )

        history = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=100,
            order="asc",
            context=context,
        )
        return (
            cast(
                list[ResponseInputItemParam],
                await self._converter.to_agent_input(history.data),
            ),
            "full",
        )

    async def _selected_file_context_items(
        self,
        *,
        context: FileDeskChatContext,
    ) -> list[ResponseInputItemParam]:
        if not context.selected_file_ids:
            return []

        file_lines: list[str] = []
        for file_id in context.selected_file_ids[:8]:
            try:
                detail = await self._file_library.get_file_detail(
                    clerk_user_id=context.clerk_user_id,
                    file_id=file_id,
                )
            except Exception:
                continue
            file_lines.append(
                f"- {detail.display_title} ({detail.id}, {detail.media_type}, tags: "
                f"{', '.join(tag.name for tag in detail.tags) or 'none'})"
            )
        if not file_lines:
            return []

        return [
            cast(
                ResponseInputItemParam,
                Message(
                    role="user",
                    type="message",
                    content=[
                        {
                            "type": "input_text",
                            "text": (
                                "The user currently has these files selected in the file explorer. "
                                "Use them as the first place to look before widening the search.\n"
                                + "\n".join(file_lines)
                            ),
                        }
                    ],
                ),
            )
        ]

    @staticmethod
    def _uses_server_managed_conversation(typed_metadata: AppChatMetadata) -> bool:
        return bool(typed_metadata.get("openai_conversation_id") or typed_metadata.get("openai_previous_response_id"))

    @staticmethod
    async def _agent_instructions(
        _context,
        _agent,
    ) -> str:
        return (
            "You are the file desk assistant for a personal document library. "
            "Use the MCP tools to list files, inspect file details, read extracted text, and "
            "run semantic search over the user's uploaded files. Start with the user's selected "
            "files when that context is available. Be concise, grounded in retrieved content, and "
            "say when you cannot find supporting evidence in the library."
        )


def _normalize_origin(request_metadata: ChatRequestMetadata) -> str | None:
    origin = request_metadata.get("origin")
    if isinstance(origin, str) and origin.strip():
        return origin.strip()
    return None


def _title_from_user_message(item: UserMessageItem) -> str | None:
    text_parts = [
        part.text.strip()
        for part in item.content
        if getattr(part, "type", None) == "text" and isinstance(part.text, str)
    ]
    combined = " ".join(part for part in text_parts if part).strip()
    if not combined:
        return None
    if len(combined) <= 72:
        return combined
    return combined[:69].rstrip() + "..."


def _model_settings_override_for_model(model: str | None) -> ModelSettings | None:
    if not isinstance(model, str) or not model.startswith("gpt-5.4"):
        return None
    return ModelSettings(reasoning=Reasoning(effort="low", summary="auto"))


def _tool_name(*, tool: Tool, context: object) -> str:
    context_tool_name = getattr(context, "tool_name", None)
    if isinstance(context_tool_name, str) and context_tool_name.strip():
        return context_tool_name.strip()
    tool_name = getattr(tool, "name", None)
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return type(tool).__name__


def _truncate_for_log(value: object, *, max_chars: int = 280) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\n", "\\n").strip()
    if not text:
        return "-"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
