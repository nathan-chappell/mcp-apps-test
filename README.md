# OpenAI Workspace Desk MCP Server

This repo now centers on a single Python MCP server with an MCP Apps UI for a practical workspace RAG demo.

## What It Does

- Authenticates users with Clerk and gates access with `private_metadata.active` plus `private_metadata.role`.
- Lets active users create workspaces, define reusable tags, and upload documents, images, audio, and video from the MCP Apps UI.
- Uses OpenAI Files plus vector stores as the main retrieval layer.
- Stores derived searchable text for transcripts and image descriptions while preserving links back to the original uploaded asset.
- Exposes a single UI-entry query tool plus workspace info and mutation helpers, all backed by the same UI state.

## Main Pieces

- Backend MCP server: [`apps/openai_vectorstore_mcp_app/backend`](apps/openai_vectorstore_mcp_app/backend)
- MCP Apps UI: [`apps/openai_vectorstore_mcp_app/ui`](apps/openai_vectorstore_mcp_app/ui)
- Integration tests: [`tests/integration/test_openai_vectorstore_mcp_app.py`](tests/integration/test_openai_vectorstore_mcp_app.py)

## Local Development

1. Create `.env` values from [`.env.example`](.env.example).
2. Install Python dependencies into the repo venv.
3. In [`apps/openai_vectorstore_mcp_app/ui`](apps/openai_vectorstore_mcp_app/ui), run `npm install`.
4. Run `npm run build:watch` to keep both the MCP app bundle and the backend-served dev-host assets fresh.
5. Run the MCP server over HTTP with `./.venv/bin/openai-vectorstore-mcp-http`.
6. Point an MCP Apps-compatible host at `http://localhost:8000/mcp`.
7. For the bundled local dev host, open `http://localhost:8000/`.
8. Set `CLERK_PUBLISHABLE_KEY` so the local dev host can sign in and attach a Clerk session token to MCP requests. Use `localhost` for Clerk-backed local sign-in.

## Notes

- The server bootstraps its SQL schema directly at runtime; there is no migration layer anymore.
- The only supported product surface is the MCP server plus MCP Apps UI resource, not a standalone web app.
