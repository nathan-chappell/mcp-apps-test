# OpenAI Files + Vector Store MCP App

This is the first conceptual MCP app in the repo, built backend-first so we can use the same VS Code chat surface to develop and test it.

## Current scope

- Python FastMCP backend under [`backend/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/backend)
- Live OpenAI Files + Vector Stores integration
- `openai-agents`-backed `ask_vector_store` tool for grounded Q&A
- Mantine-based MCP App UI under [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui)
- Workspace-local VS Code MCP configuration in [`.vscode/mcp.json`](/home/uphill/programming/mcp_apps_test/.vscode/mcp.json)

## Available tools

- `open_vector_store_console`
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
2. In [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui), run `npm install` once and then `npm run build:watch` while working on the app.
3. Open VS Code Agent mode and enable the `openai-files-vector-store` MCP server from the tools picker.
4. Use the chat UI to run a flow like:
   - open `open_vector_store_console`
   - create a vector store
   - upload or attach a file
   - inspect ingestion with `get_vector_store_status`
   - run `search_vector_store`
   - run `ask_vector_store`

## UI development

- The app ships as a single `dist/mcp-app.html` resource served from the Python MCP server.
- Local browser iteration is supported with mock data via `npm run dev`.
- The first UI slice is intentionally retrieval-focused: list vector stores, inspect one store, run raw search, and run grounded Q&A.

## Testing

- Automated tests live in [`tests/integration/test_openai_files_vector_store.py`](/home/uphill/programming/mcp_apps_test/tests/integration/test_openai_files_vector_store.py).
- Frontend verification lives in `npm run typecheck` and `npm run build` under [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui).
- The live OpenAI test skips cleanly when `OPENAI_API_KEY` is not set.

## Next phase

Phase 2 can add write-side console flows like create-store and file attachment from the app UI. The repo-local `mcp-apps` skills are staged under [`.agents/skills/`](/home/uphill/programming/mcp_apps_test/.agents/skills) for that next step.
