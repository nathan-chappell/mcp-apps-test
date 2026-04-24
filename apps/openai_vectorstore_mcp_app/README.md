# OpenAI Knowledge Base Desk

This package contains the single MCP server product in the repo: a per-user knowledge-base graph with an MCP Apps UI.

## Scope

- FastMCP backend in [`backend/`](backend)
- React/Mantine MCP Apps UI in [`ui/`](ui)
- OpenAI vector stores and hosted file search for retrieval
- Agents SDK chat with knowledge-base-scoped conversation memory
- Directed graph nodes and labeled edges for uploaded documents
- Tags, raw file search, branching search, upload ingestion, and command-driven graph mutations

## Tool Surface

Model-visible tools:

- `query_knowledge_base`
- `get_knowledge_base_info`

App-only tools:

- `update_knowledge_base`
- `run_knowledge_base_command`
- `confirm_knowledge_base_command`

`query_knowledge_base` is the only UI-opening tool. It opens the desk and optionally runs QA, file search, or branch search in the current knowledge-base scope. `get_knowledge_base_info` returns graph state without opening UI. The app-only tools handle typed mutations, natural-language graph commands, upload preparation, and destructive confirmations.

## Runtime Model

- Clerk handles sign-in, with access gated by `private_metadata.active` and `private_metadata.role`.
- The app DB stores users, one knowledge base per user, graph nodes, graph edges, tags, and derived artifacts.
- OpenAI stores original files, derived searchable files, vector stores, and conversation memory.
- The MCP Apps UI is the primary user experience for graph navigation, ingestion, retrieval, and chat.

## Local Workflow

1. Fill in `.env` from the repo root example.
2. Run `npm install` in [`ui/`](ui).
3. Run `npm run build:watch` in [`ui/`](ui).
4. Start the server with `./.venv/bin/openai-vectorstore-mcp-http`.
5. Connect an MCP Apps host to `http://localhost:8000/mcp`.
6. If you use the bundled local dev host at `http://localhost:8000/`, also set `CLERK_PUBLISHABLE_KEY` so the host can complete Clerk sign-in before connecting. Use `localhost` for Clerk-backed local sign-in.

## Verification

- Backend integration tests: [`tests/integration/test_openai_vectorstore_mcp_app.py`](../../tests/integration/test_openai_vectorstore_mcp_app.py)
- Frontend checks in [`ui/`](ui): `npm run typecheck`, `npm run build`
