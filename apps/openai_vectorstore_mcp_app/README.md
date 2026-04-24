# OpenAI Document Library App

This package contains the repo's MCP product: a per-user document library with two MCP app entry points.

## Scope

- FastMCP backend in [`backend/`](backend)
- React MCP app UI in [`ui/`](ui)
- OpenAI vector stores and hosted file search for retrieval
- Agents SDK grounded QA for the ask flow
- Tags, filename filtering, created-date filtering, uploads, and document metadata review

## Tool Surface

Model-visible tools:

- `open_document_library`
- `open_document_ask`

App-only helper tools:

- `get_document_library_state`
- `query_document_library`
- `update_document_library`

`open_document_library` is the management view for uploads, tags, filters, and document detail. `open_document_ask` is the search/ask view that uses the same filter model.

## Runtime Model

- FastMCP provides the MCP HTTP transport plus OAuth resource-server behavior.
- Clerk is the upstream identity provider; active access still depends on Clerk `private_metadata.active`.
- The app DB stores users, one document library per user, documents, tags, and derived artifacts.
- OpenAI stores original files, derived searchable files, vector stores, and ask-flow conversation memory.

## Local Workflow

1. Fill in `.env` from the repo root example.
2. Run `npm install` in [`ui/`](ui).
3. Run `npm run build:watch` in [`ui/`](ui).
4. Start the server with `./.venv/bin/openai-vectorstore-mcp-http`.
5. Connect an MCP host to `http://localhost:8000/mcp`.
6. Use `npm run dev:mock` in [`ui/`](ui) for standalone mock UI work.

## Verification

- Backend integration tests: [`tests/integration/test_openai_vectorstore_mcp_app.py`](../../tests/integration/test_openai_vectorstore_mcp_app.py)
- Frontend checks in [`ui/`](ui): `npm run typecheck`, `npm run build`
