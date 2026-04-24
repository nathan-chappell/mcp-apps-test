# OpenAI File Desk App

This package contains the file-desk application that backs both the MCP experience and the companion web app.

## Scope

- FastMCP server for file search, file detail, file text reading, and deletion
- Prefab-powered MCP file-library UI for upload and file management
- FastAPI web app for Clerk-authenticated file browsing, upload, delete, download, and ChatKit
- Shared `FileLibraryService` for all active file-library behavior
- OpenAI Files plus vector stores for storage-backed retrieval and derived searchable artifacts

## Runtime Model

- The MCP server is the primary product surface.
- The React web app mirrors the same file library and adds a standalone Clerk + ChatKit experience.
- ChatKit intentionally reaches the file library through the mounted `/mcp` app over ASGI transport.
- The database stores one file library per user, uploaded files, tags, derived artifacts, and chat threads.

## Backend Layout

- [`backend/bootstrap.py`](backend/bootstrap.py): shared service construction
- [`backend/mcp_app.py`](backend/mcp_app.py): MCP auth, tools, and Prefab UI
- [`backend/web_app.py`](backend/web_app.py): FastAPI routes, SPA hosting, `/api/chatkit`, and mounted `/mcp`
- [`backend/file_library_service.py`](backend/file_library_service.py): canonical file-library domain logic

## Local Workflow

1. Fill in `.env` from the repo root example.
2. Run `npm install` in [`ui/`](ui).
3. Run `npm run build:watch` in [`ui/`](ui).
4. Start the HTTP app with `./.venv/bin/openai-vectorstore-mcp-http`.
5. Open `http://localhost:8000/` for the web companion app.
6. Connect an MCP host to `http://localhost:8000/mcp`.

## Verification

- Backend integration test: [`tests/integration/test_openai_vectorstore_mcp_app.py`](../../tests/integration/test_openai_vectorstore_mcp_app.py)
- Frontend checks in [`ui/`](ui): `npm run typecheck`, `npm run build`
