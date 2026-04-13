# OpenAI Files + Vector Store MCP App

This is the first conceptual MCP app in the repo, built backend-first so we can exercise it through the local host harness and integration tests.

## Current scope

- Python FastMCP backend under [`backend/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/backend)
- Live OpenAI Files + Vector Stores integration
- `openai-agents`-backed `ask_vector_store` tool for grounded Q&A
- Mantine-based MCP App UI under [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui)

## Persistence model

This app follows the repo default: local-first and near-stateless.

- OpenAI is the source of truth for uploaded files, vector stores, and retrieval state.
- The MCP server reads local files on demand for upload-oriented tool calls, but it does not mirror them into app-owned storage.
- The app does not provide app-owned custody or retrieval of user-provided files.
- MCP/session state stays in memory; the app does not keep durable recent-store history, aliases, or saved searches.

If a future version needs durable app-owned state, it should document the exception first and define the new source of truth explicitly.

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
- `update_vector_store_file_attributes`
- `delete_file`

## Local workflow

1. Put your key in `.env` or export `OPENAI_API_KEY`.
2. In [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui), run `npm install` once.
3. For the full local MCP App loop, run `npm run dev` in [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui) and open `http://127.0.0.1:8080/`.
4. For host-level testing with the upstream `modelcontextprotocol/ext-apps` harness instead, run `npm run build:watch` and `npm run dev:server` in [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui), then point that host at `http://127.0.0.1:8000/mcp`.
5. Use either the repo-local test host or the upstream host to run a flow like:
   - open `open_vector_store_console`
   - create a vector store
   - upload or attach a file
   - inspect ingestion with `get_vector_store_status`
   - run `search_vector_store`
   - run `ask_vector_store`

## UI development

- The app ships as a single `dist/mcp-app.html` resource served from the Python MCP server.
- `npm run dev` now uses a repo-local test host adapted from `modelcontextprotocol/ext-apps/examples/basic-host`, so you can exercise the real MCP transport and iframe bridge from the browser.
- Local browser iteration is still supported with mock data via `npm run dev:mock`.
- The current UI focuses on retrieval and file management: inspect one store, scope search/ask to one attached file when needed, edit attachment metadata, and delete files globally.
- Workspace-local VS Code and Codex MCP registration is intentionally omitted so opening the repo does not auto-start the dev server.

## Testing

- Automated tests live in [`tests/integration/test_openai_files_vector_store.py`](/home/uphill/programming/mcp_apps_test/tests/integration/test_openai_files_vector_store.py).
- Frontend verification lives in `npm run typecheck` and `npm run build` under [`ui/`](/home/uphill/programming/mcp_apps_test/apps/openai_files_vector_store/ui).
- The live OpenAI test skips cleanly when `OPENAI_API_KEY` is not set.

## Next phase

Phase 2 can add write-side console flows like create-store and file attachment from the app UI. The repo-local `mcp-apps` skills are staged under [`.agents/skills/`](/home/uphill/programming/mcp_apps_test/.agents/skills) for that next step.
