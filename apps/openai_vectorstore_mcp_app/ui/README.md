# Document Library UI

This subproject contains the two MCP app bundles for the document-library backend.

## Commands

- `npm install`
- `npm run dev`
- `npm run dev:mock`
- `npm run build:watch`
- `npm run typecheck`
- `npm run build`

## Development Modes

- `npm run dev` starts the UI watch builds plus the Python HTTP server.
- `npm run dev:mock` runs standalone mock pages on `http://localhost:5174/`.
- `npm run build:watch` keeps both MCP app bundles fresh:
  - `dist/library.html`
  - `dist/ask.html`

## Current Scope

- `Document Library`: tag filtering, filename filtering, created-date filtering, upload, tag creation, and document detail
- `Document Ask`: the same filters plus grounded search and ask

For real authenticated behavior, use an MCP host against the backend server at `http://localhost:8000/mcp`. The local Vite mode is only for standalone UI development with mock data.
