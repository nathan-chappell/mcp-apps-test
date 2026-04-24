from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import TYPE_CHECKING

from agents import Agent, RunContextWrapper, Runner, function_tool, set_default_openai_api, set_default_openai_key

from .schemas import CommandParserKind, KnowledgeCommandStatus, PendingCommandResult
from .settings import AppSettings

if TYPE_CHECKING:
    from .knowledge_base_service import KnowledgeBaseService


@dataclass(slots=True)
class CommandExecutionResult:
    status: KnowledgeCommandStatus
    action: str | None
    message: str
    parser: CommandParserKind
    node_id: str | None = None
    edge_id: str | None = None
    tag_id: str | None = None
    pending_confirmation: PendingCommandResult | None = None


@dataclass(slots=True)
class CommandAgentContext:
    selected_node_id: str | None
    result: CommandExecutionResult | None = None


class KnowledgeBaseCommandAgent:
    """Interpret natural-language graph mutation commands into service calls."""

    def __init__(self, settings: AppSettings, service: KnowledgeBaseService) -> None:
        self._settings = settings
        self._service = service
        self._logger = logging.getLogger(__name__)
        set_default_openai_key(
            settings.openai_api_key.get_secret_value(),
            use_for_tracing=False,
        )
        set_default_openai_api("responses")

    async def run_command(
        self,
        *,
        raw_command: str,
        selected_node_id: str | None,
    ) -> CommandExecutionResult:
        normalized_command = raw_command.strip()
        if not normalized_command:
            return CommandExecutionResult(
                status="rejected",
                action=None,
                message="Enter a command to modify the knowledge base.",
                parser="manual",
            )

        if self._use_local_fallback():
            return await self._run_fallback(
                raw_command=normalized_command,
                selected_node_id=selected_node_id,
            )

        command_context = CommandAgentContext(selected_node_id=selected_node_id)

        @function_tool
        async def rename_node(
            ctx: RunContextWrapper[CommandAgentContext],
            new_title: str,
            node_title: str | None = None,
        ) -> str:
            """Rename a node. Omit node_title when the user refers to the selected node."""

            ctx.context.result = await self._service.rename_node_from_command(
                node_title=node_title,
                selected_node_id=ctx.context.selected_node_id,
                new_title=new_title,
                parser="agent",
            )
            return ctx.context.result.message

        @function_tool
        async def create_tag(
            ctx: RunContextWrapper[CommandAgentContext],
            name: str,
            color: str | None = None,
        ) -> str:
            """Create a tag if it does not already exist."""

            ctx.context.result = await self._service.create_tag_from_command(
                name=name,
                color=color,
                parser="agent",
            )
            return ctx.context.result.message

        @function_tool
        async def set_node_tags(
            ctx: RunContextWrapper[CommandAgentContext],
            tag_names: list[str],
            node_title: str | None = None,
        ) -> str:
            """Replace a node's tags with the provided tag names."""

            ctx.context.result = await self._service.set_node_tags_from_command(
                node_title=node_title,
                selected_node_id=ctx.context.selected_node_id,
                tag_names=tag_names,
                parser="agent",
            )
            return ctx.context.result.message

        @function_tool
        async def add_edge(
            ctx: RunContextWrapper[CommandAgentContext],
            to_node_title: str,
            label: str,
            from_node_title: str | None = None,
        ) -> str:
            """Create or update a labeled directed edge. Omit from_node_title for the selected node."""

            ctx.context.result = await self._service.upsert_edge_from_command(
                from_node_title=from_node_title,
                to_node_title=to_node_title,
                label=label,
                selected_node_id=ctx.context.selected_node_id,
                parser="agent",
            )
            return ctx.context.result.message

        @function_tool
        async def delete_node(
            ctx: RunContextWrapper[CommandAgentContext],
            node_title: str | None = None,
        ) -> str:
            """Delete a node. Omit node_title when the user refers to the selected node."""

            ctx.context.result = await self._service.delete_node_from_command(
                node_title=node_title,
                selected_node_id=ctx.context.selected_node_id,
                parser="agent",
            )
            return ctx.context.result.message

        @function_tool
        async def reject_command(
            ctx: RunContextWrapper[CommandAgentContext],
            reason: str,
        ) -> str:
            """Reject the command when it is too ambiguous or unsupported."""

            ctx.context.result = CommandExecutionResult(
                status="rejected",
                action=None,
                message=reason,
                parser="agent",
            )
            return reason

        agent = Agent(
            name="knowledge-base-command-agent",
            model=self._settings.openai_branching_model,
            instructions=(
                "You convert one user command into exactly one knowledge-base graph mutation. "
                "Always call a tool. If the user refers to the selected node, omit node_title or "
                "from_node_title so the tool can resolve the current selection. Prefer add_edge for "
                "graph relationships, rename_node for display-title changes, create_tag for tag "
                "creation, set_node_tags for changing a node's tags, and delete_node for removals. "
                "If the request is ambiguous or unsupported, call reject_command with a short reason."
            ),
            tools=[rename_node, create_tag, set_node_tags, add_edge, delete_node, reject_command],
        )

        try:
            await Runner.run(agent, normalized_command, context=command_context, max_turns=4)
            if command_context.result is not None:
                return command_context.result
        except Exception:
            self._logger.warning(
                "knowledge_base_command_agent_fallback command=%s",
                normalized_command,
                exc_info=True,
            )

        return await self._run_fallback(
            raw_command=normalized_command,
            selected_node_id=selected_node_id,
        )

    def _use_local_fallback(self) -> bool:
        api_key = self._settings.openai_api_key.get_secret_value().strip()
        return not api_key or api_key.startswith("test-")

    async def _run_fallback(
        self,
        *,
        raw_command: str,
        selected_node_id: str | None,
    ) -> CommandExecutionResult:
        lowered = raw_command.strip().lower()

        rename_selected_match = re.match(
            r"^(?:rename|change)\s+(?:the\s+)?selected node(?:'s)?(?:\s+name)?\s+to\s+(.+)$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if rename_selected_match:
            return await self._service.rename_node_from_command(
                node_title=None,
                selected_node_id=selected_node_id,
                new_title=_strip_quotes(rename_selected_match.group(1)),
                parser="fallback",
            )

        rename_named_match = re.match(
            r"^(?:rename|change)\s+node\s+(.+?)\s+(?:to|name to)\s+(.+)$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if rename_named_match:
            return await self._service.rename_node_from_command(
                node_title=_strip_quotes(rename_named_match.group(1)),
                selected_node_id=selected_node_id,
                new_title=_strip_quotes(rename_named_match.group(2)),
                parser="fallback",
            )

        add_edge_match = re.match(
            r"^add (?:an )?edge from (.+?) to (.+?)(?:\s+labeled\s+(.+))?$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if add_edge_match:
            return await self._service.upsert_edge_from_command(
                from_node_title=_strip_quotes(add_edge_match.group(1)),
                to_node_title=_strip_quotes(add_edge_match.group(2)),
                label=_strip_quotes(add_edge_match.group(3) or "related"),
                selected_node_id=selected_node_id,
                parser="fallback",
            )

        add_edge_from_selected_match = re.match(
            r"^add (?:an )?edge from (?:the\s+)?selected node to (.+?)(?:\s+labeled\s+(.+))?$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if add_edge_from_selected_match:
            return await self._service.upsert_edge_from_command(
                from_node_title=None,
                to_node_title=_strip_quotes(add_edge_from_selected_match.group(1)),
                label=_strip_quotes(add_edge_from_selected_match.group(2) or "related"),
                selected_node_id=selected_node_id,
                parser="fallback",
            )

        delete_selected_match = re.match(
            r"^delete (?:the\s+)?selected node$",
            lowered,
        )
        if delete_selected_match:
            return await self._service.delete_node_from_command(
                node_title=None,
                selected_node_id=selected_node_id,
                parser="fallback",
            )

        delete_named_match = re.match(
            r"^delete node (.+)$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if delete_named_match:
            return await self._service.delete_node_from_command(
                node_title=_strip_quotes(delete_named_match.group(1)),
                selected_node_id=selected_node_id,
                parser="fallback",
            )

        create_tag_match = re.match(
            r"^(?:create|add)\s+tag\s+(.+)$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if create_tag_match:
            return await self._service.create_tag_from_command(
                name=_strip_quotes(create_tag_match.group(1)),
                color=None,
                parser="fallback",
            )

        tag_selected_match = re.match(
            r"^(?:set|add)\s+tags?\s+(.+?)\s+to\s+(?:the\s+)?selected node$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if tag_selected_match:
            return await self._service.set_node_tags_from_command(
                node_title=None,
                selected_node_id=selected_node_id,
                tag_names=_split_names(tag_selected_match.group(1)),
                parser="fallback",
            )

        tag_named_match = re.match(
            r"^(?:set|add)\s+tags?\s+(.+?)\s+to\s+node\s+(.+)$",
            raw_command,
            flags=re.IGNORECASE,
        )
        if tag_named_match:
            return await self._service.set_node_tags_from_command(
                node_title=_strip_quotes(tag_named_match.group(2)),
                selected_node_id=selected_node_id,
                tag_names=_split_names(tag_named_match.group(1)),
                parser="fallback",
            )

        return CommandExecutionResult(
            status="rejected",
            action=None,
            message=(
                "I couldn't map that command to a supported mutation yet. Try a command like "
                "'rename the selected node to X', 'add an edge from A to B labeled cites', "
                "'create tag research', or 'delete node Y'."
            ),
            parser="fallback",
        )


def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def _split_names(value: str) -> list[str]:
    normalized = value.replace(" and ", ",")
    return [
        _strip_quotes(part)
        for part in (item.strip() for item in normalized.split(","))
        if part
    ]
