# OpenAI Files + Vector Store MCP App

This is the first conceptual MCP app in the repo, built backend-first so we can use the same VS Code chat surface to develop and test it.

## Current scope

- Python FastMCP backend under [`backend/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/backend)
- Live OpenAI Files + Vector Stores integration
- `openai-agents`-backed `ask_vector_store` tool for grounded Q&A
- Workspace-local VS Code MCP configuration in [`.vscode/mcp.json`](/home/uphill/programming/mcp_apps_test/.vscode/mcp.json)
- Deferred UI work under [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui)

## Available tools

- `upload_file`
- `list_files`
- `create_vector_store`
- `list_vector_stores`
- `attach_files_to_vector_store`
- `get_vector_store_status`
- `search_vector_store`
- `ask_vector_store`

## Local workflow

1. Put your key in `.env` or export `OPENAI_API_KEY`.
2. Open VS Code Agent mode and enable the `openai-files-vector-store` MCP server from the tools picker.
3. Use the chat UI to run a flow like:
   - create a vector store
   - upload or attach a file
   - inspect ingestion with `get_vector_store_status`
   - run `search_vector_store`
   - run `ask_vector_store`

## Testing

- Automated tests live in [`tests/integration/test_openai_files_vector_store.py`](/home/uphill/programming/mcp_apps_test/tests/integration/test_openai_files_vector_store.py).
- The live OpenAI test skips cleanly when `OPENAI_API_KEY` is not set.

## Next phase

Phase 2 will add the inline MCP App operator console on top of this backend. The repo-local `mcp-apps` skills are staged under [`.agents/skills/`](/home/uphill/programming/mcp_apps_test/.agents/skills) for that next step.
