---
name: debug-openai-conversations
description: Use when the user wants to debug an OpenAI response or conversation by ID, inspect response and conversation items from the OpenAI SDK, correlate them with local app logs, or troubleshoot missing tool calls, duplicated history, or broken conversation state in this repo.
---

# Debug OpenAI Conversations

Inspect OpenAI `resp_*` and `conv_*` objects directly from the repo's `.venv` so Codex can analyze what happened in a chat run and connect it back to local app behavior.

## When to use this

- The user gives a `resp_*` or `conv_*` ID and wants to understand what happened.
- A chat turn used the wrong context, repeated prior history, or seems to have lost conversation state.
- A model call should have produced tool calls but did not, or tool calls look malformed.
- You need to compare OpenAI-side traces with local logs in `tmp/logs.txt`.

## Quick start

Run the bundled inspector first.

```bash
./.venv/bin/python codex_skills/debug-openai-conversations/scripts/inspect_openai_trace.py --response-id resp_...
```

Or:

```bash
./.venv/bin/python codex_skills/debug-openai-conversations/scripts/inspect_openai_trace.py --conversation-id conv_...
```

You can pass both. If only a response ID is given, the script will retrieve the response and then inspect the linked conversation when one is present.

Useful flags:

- `--limit 100` to inspect more conversation items
- `--order asc` to read the conversation chronologically
- `--pretty` for human-readable JSON

## Workflow

1. Run the script with the provided response or conversation ID.
2. Read the normalized JSON instead of the raw SDK object.
3. Summarize the turn or conversation chronologically:
   - response IDs and conversation ID
   - assistant text
   - tool calls
   - errors or incomplete status
   - recent conversation items
4. Correlate the OpenAI trace with local app logs:
   - `tmp/logs.txt`
   - chat turn logs in `backend/chatkit_server.py`
   - MCP tool logs in `backend/mcp_app.py`
5. Call out likely causes, especially:
   - missing or stale `previous_response_id`
   - missing or stale `openai_conversation_id`
   - duplicate history replay
   - tool call arguments that do not match MCP tool expectations
   - OpenAI response objects that have no tool calls when the prompt clearly should have produced them

## Output interpretation

- `response.log_url` and `conversation.log_url` are dashboard links that should be clickable in terminal output.
- `response.tool_calls` gives a compact tool-centric summary of the response output.
- `conversation.items` is intentionally normalized to a compact chronology rather than the full raw payload.

## Notes

- Prefer the bundled script over ad-hoc Python snippets so repeated debugging stays consistent.
- Keep quotes short when relaying response content back to the user.
- If the script fails because `OPENAI_API_KEY` is missing, report that directly.
