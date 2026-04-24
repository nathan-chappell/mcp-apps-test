# OpenAI Document Library MCP Server

This repo now centers on a single Python MCP server plus two MCP app UIs for a simpler document-library workflow.

## What It Does

- Protects `/mcp` with FastMCP and Clerk-backed OAuth discovery.
- Requires authenticated users to also have `private_metadata.active == true` in Clerk.
- Stores one per-user document library with tags, uploaded files, derived searchable text, and OpenAI vector-store backing.
- Exposes two UI entry tools:
  - `open_document_library`
  - `open_document_ask`

## Main Pieces

- Backend MCP server: [`apps/openai_vectorstore_mcp_app/backend`](apps/openai_vectorstore_mcp_app/backend)
- MCP app UI: [`apps/openai_vectorstore_mcp_app/ui`](apps/openai_vectorstore_mcp_app/ui)
- Integration tests: [`tests/integration/test_openai_vectorstore_mcp_app.py`](tests/integration/test_openai_vectorstore_mcp_app.py)

## Local Development

1. Create `.env` values from [`.env.example`](.env.example).
2. Install Python deps into the repo venv.
3. In [`apps/openai_vectorstore_mcp_app/ui`](apps/openai_vectorstore_mcp_app/ui), run `npm install`.
4. Run `npm run build:watch` in the UI directory.
5. Start the HTTP server with `./.venv/bin/openai-vectorstore-mcp-http`.
6. Point an MCP-compatible host or Inspector at `http://localhost:8000/mcp`.
7. For standalone UI work, run `npm run dev:mock` in the UI directory.

## Auth Notes

- Unauthenticated `POST /mcp` requests return `401` with protected-resource metadata.
- FastMCP handles the MCP/OAuth resource-server flow.
- Clerk remains the upstream identity system, and the app separately checks Clerk user metadata for active access.

## Verification

- Backend tests: `./.venv/bin/pytest tests/integration/test_openai_vectorstore_mcp_app.py`
- Lint: `./.venv/bin/ruff check apps/openai_vectorstore_mcp_app tests`
- UI typecheck: `npm run typecheck`
- UI build: `npm run build`
