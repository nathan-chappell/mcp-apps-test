# OpenAI Files Vector Store UI

This subproject contains the phase-1 MCP App UI for the `openai-files-vector-store` server.

## Commands

- `npm install`
- `npm run dev`
- `npm run build:watch`
- `npm run typecheck`
- `npm run build`

## Development modes

- `npm run dev` starts a standalone Vite app with mock vector-store data so the UI can be iterated outside an MCP host.
- `npm run build:watch` keeps `dist/mcp-app.html` fresh for VS Code and the Python MCP server.

## Current scope

- Browse recent vector stores
- Inspect one store's status and attached files
- Run direct vector store search
- Run `ask_vector_store` grounded retrieval

Phase 1 intentionally excludes browser-side uploads and file attachment flows.
