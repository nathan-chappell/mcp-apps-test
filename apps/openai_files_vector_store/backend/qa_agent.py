from __future__ import annotations

import logging
from time import perf_counter

from agents import (
    Agent,
    FileSearchTool,
    Runner,
    set_default_openai_api,
    set_default_openai_key,
)
from agents.items import ToolCallItem
from openai.types.responses import ResponseFileSearchToolCall

from .schemas import AskVectorStoreResult, FileSearchCallSummary
from .settings import AppSettings


class VectorStoreQuestionAnswerer:
    """Question-answering layer backed by openai-agents and hosted file search."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        set_default_openai_key(
            settings.openai_api_key.get_secret_value(), use_for_tracing=False
        )
        set_default_openai_api("responses")

    async def ask(
        self,
        *,
        vector_store_id: str,
        question: str,
        max_num_results: int | None,
    ) -> AskVectorStoreResult:
        started_at = perf_counter()
        resolved_max_results = (
            max_num_results or self._settings.openai_file_search_max_results
        )

        agent = Agent(
            name="vector-store-question-answerer",
            model=self._settings.openai_agent_model,
            instructions=(
                "You are a retrieval assistant for a single OpenAI vector store. "
                "Always call the file_search tool before answering. "
                "Answer only from retrieved content. "
                "If the retrieved content does not answer the question, say that clearly. "
                "Mention filenames when they help ground the answer."
            ),
            tools=[
                FileSearchTool(
                    vector_store_ids=[vector_store_id],
                    max_num_results=resolved_max_results,
                    include_search_results=True,
                )
            ],
        )

        result = await Runner.run(agent, question)
        search_calls: list[FileSearchCallSummary] = []
        for item in result.new_items:
            if not isinstance(item, ToolCallItem):
                continue
            raw_item = item.raw_item
            if isinstance(raw_item, ResponseFileSearchToolCall):
                search_calls.append(FileSearchCallSummary.from_openai(raw_item))

        answer = str(result.final_output).strip()
        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_agents_vector_store_ask vector_store_id=%s model=%s search_calls=%s duration_ms=%.1f",
            vector_store_id,
            self._settings.openai_agent_model,
            len(search_calls),
            duration_ms,
        )

        return AskVectorStoreResult(
            vector_store_id=vector_store_id,
            question=question,
            answer=answer,
            model=self._settings.openai_agent_model,
            search_calls=search_calls,
        )
