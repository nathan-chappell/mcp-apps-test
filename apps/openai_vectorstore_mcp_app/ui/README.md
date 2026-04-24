# Knowledge Base Desk UI

This subproject contains the MCP Apps UI for the knowledge-base graph server.

## Commands

- `npm install`
- `npm run dev`
- `npm run dev:mock`
- `npm run build:watch`
- `npm run host:build`
- `npm run typecheck`
- `npm run build`

## Development modes

- `npm run dev` starts the real local MCP App loop:
  - the UI single-file build watcher for `dist/mcp-app.html`
  - the backend-served dev-host asset watchers for `host-dist/dev-host/index.html` and `host-dist/dev-host/sandbox.html`
  - the Python FastMCP server over streamable HTTP at `http://localhost:8000/mcp`
  - the host UI at `http://localhost:8000/`
- `npm run dev:mock` keeps a standalone Vite mode with mock knowledge-base graph data on `http://localhost:5174/`.
- `npm run build:watch` keeps both `dist/mcp-app.html` and the backend-served dev-host assets fresh.
- `npm run host:build` builds only the dev-host assets into `host-dist/` without starting any long-running processes.

If Clerk shows a local deployment or allowed-origin warning, make sure you open the bundled host on `http://localhost:8000/` and keep Clerk configured for `localhost`.

## Current scope

- View the per-user document graph rendered with Mermaid
- Filter visible nodes by tags and retrieval by graph scope, tags, and media type
- Upload and inspect documents, images, audio, and video
- Run raw file search, branching search, and chat against the current scope
- Execute graph mutations from the command bar, with confirmation for destructive deletes
