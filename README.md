# MCP Apps Test

This repo is for experimenting with MCP apps and servers with a bias toward thin, local-first integrations.

## Repo defaults

Unless an app documents an exception, use these defaults:

- `config + secrets`: load from local environment or local secret handling
- `ephemeral MCP/session state`: keep in memory only
- `remote resource state`: treat the upstream service as the source of truth
- `user-owned local files`: read on demand for tool calls; do not mirror or retain them in app storage
- `convenience state`: omit durable recent-item history, aliases, saved searches, and similar UX state by default

Remote HTTP support does not by itself justify adding a database, bucket, or other durable app-owned store. Start near-stateless and add persistence only when an app has a concrete need that cannot be handled with request-scoped or in-memory state.

## Persistence policy

The repo-wide persistence policy lives in [docs/persistence-policy.md](/home/uphill/programming/py/mcp-apps-test/docs/persistence-policy.md).

Use that document whenever a new MCP app needs to decide:

- what state, if any, the app should own
- whether the app is still local-first or needs remote multi-user durability
- which system is the source of truth for files, jobs, or credentials

## Current app

The current OpenAI Files + Vector Store app follows this default model:

- OpenAI is the system of record for uploaded files, vector stores, and retrieval state
- the MCP server is a thin orchestration layer over OpenAI APIs
- local files remain user-owned inputs rather than app-owned stored artifacts

See [apps/openai_files_vector_store/README.md](/home/uphill/programming/py/mcp-apps-test/apps/openai_files_vector_store/README.md) for app-specific details.
