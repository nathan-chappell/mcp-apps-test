from __future__ import annotations

import logging
from time import perf_counter

from agents import (
    Agent,
    FileSearchTool,
    Runner,
    WebSearchTool,
    set_default_openai_api,
    set_default_openai_key,
)
from agents.items import ToolCallItem
from agents.memory import OpenAIConversationsSession
from openai.types.responses import ResponseFileSearchToolCall, ResponseFunctionWebSearch
from openai.types.shared_params.comparison_filter import ComparisonFilter
from openai.types.shared_params.compound_filter import CompoundFilter

from .schemas import (
    FileSearchCallSummary,
    KnowledgeAnswerCitation,
    KnowledgeBaseContext,
    KnowledgeChatResult,
    WebSearchCallSummary,
)
from .settings import AppSettings


class KnowledgeBaseQuestionAnswerer:
    """Agents SDK chat layer for the knowledge-base desk."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        set_default_openai_key(
            settings.openai_api_key.get_secret_value(),
            use_for_tracing=False,
        )
        set_default_openai_api("responses")

    async def ask(
        self,
        *,
        knowledge_base_id: str,
        vector_store_id: str,
        question: str,
        context: KnowledgeBaseContext,
        conversation_id: str | None,
        filters: ComparisonFilter | CompoundFilter | None,
    ) -> KnowledgeChatResult:
        started_at = perf_counter()
        selected_tags = ", ".join(context.selected_tag_names) or "none"
        tools: list[FileSearchTool | WebSearchTool] = [
            FileSearchTool(
                vector_store_ids=[vector_store_id],
                max_num_results=context.max_results,
                include_search_results=True,
                filters=filters,
            )
        ]
        if context.include_web:
            tools.append(WebSearchTool())

        agent = Agent(
            name="knowledge-base-assistant",
            model=self._settings.openai_agent_model,
            instructions=(
                "You are the assistant for a personal document knowledge base. "
                "Always search the indexed knowledge-base documents before answering. "
                "Use only information grounded in retrieved knowledge-base content unless "
                "web search is explicitly available and necessary. "
                "When web information is used, clearly separate it from knowledge-base facts. "
                f"Selected tags: {selected_tags}. "
                "Reference node titles or original filenames when they help the user navigate back "
                "to the underlying material."
            ),
            tools=tools,
        )

        session = OpenAIConversationsSession(conversation_id=conversation_id)
        result = await Runner.run(agent, question, session=session)

        search_calls: list[FileSearchCallSummary] = []
        web_search_calls: list[WebSearchCallSummary] = []
        for item in result.new_items:
            if not isinstance(item, ToolCallItem):
                continue
            raw_item = item.raw_item
            if isinstance(raw_item, ResponseFileSearchToolCall):
                search_calls.append(FileSearchCallSummary.from_openai(raw_item))
            elif isinstance(raw_item, ResponseFunctionWebSearch):
                web_search_calls.append(WebSearchCallSummary.from_openai(raw_item))

        citations = _build_citations(
            search_calls=search_calls,
            web_search_calls=web_search_calls,
        )
        conversation_identifier = session.session_id
        answer = str(result.final_output).strip()
        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "knowledge_base_agent_answered knowledge_base_id=%s conversation_id=%s web=%s search_calls=%s web_calls=%s duration_ms=%.1f",
            knowledge_base_id,
            conversation_identifier,
            context.include_web,
            len(search_calls),
            len(web_search_calls),
            duration_ms,
        )

        return KnowledgeChatResult(
            knowledge_base_id=knowledge_base_id,
            question=question,
            answer=answer,
            model=self._settings.openai_agent_model,
            include_web=context.include_web,
            conversation_id=conversation_identifier,
            context=context,
            search_calls=search_calls,
            web_search_calls=web_search_calls,
            citations=citations,
        )


def _build_citations(
    *,
    search_calls: list[FileSearchCallSummary],
    web_search_calls: list[WebSearchCallSummary],
) -> list[KnowledgeAnswerCitation]:
    citations: list[KnowledgeAnswerCitation] = []
    seen_nodes: set[str] = set()
    for search_call in search_calls:
        for result in search_call.results:
            if not result.node_id or result.node_id in seen_nodes:
                continue
            seen_nodes.add(result.node_id)
            citations.append(
                KnowledgeAnswerCitation(
                    source="knowledge_base",
                    label=result.node_title,
                    node_id=result.node_id,
                    node_title=result.node_title,
                    original_filename=result.original_filename,
                    quote=result.text[:280].strip(),
                )
            )
            if len(citations) >= 6:
                return citations

    seen_urls: set[str] = set()
    for web_call in web_search_calls:
        for source_url in web_call.sources:
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)
            citations.append(
                KnowledgeAnswerCitation(
                    source="web",
                    label=source_url,
                    url=source_url,
                )
            )
            if len(citations) >= 6:
                return citations
    return citations
