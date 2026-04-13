# Persistence Policy

## Default stance

All MCP apps in this repo should start **local-first** and **near-stateless**.

That means:

- upstream services own their own durable resources
- MCP servers keep request and session state in memory
- user files stay user-owned unless an app explicitly needs to store them
- durable convenience state is absent by default

For the current vector-store app, OpenAI is the source of truth for files, vector stores, attachment metadata, and retrieval state.

## Default storage model

| State category | Default owner | Default storage |
| --- | --- | --- |
| Config and secrets | Local runtime | `.env`, environment variables, or local secret handling |
| MCP/session state | MCP server process | In memory only |
| Remote resources | Upstream service | Service-managed storage |
| User local files | User | Local filesystem, read on demand |
| Convenience UX state | Nobody by default | Not persisted |

## What this means in practice

By default, apps in this repo should not add:

- SQLite or other local databases
- object storage or buckets for mirrored uploads
- app-owned file custody or retrieval surfaces for user inputs
- durable recent-item history, aliases, saved searches, or cached console state
- generic persistence abstractions added "just in case"

Remote HTTP support is allowed, but it does not change the default policy by itself.

## Exception checklist

If an app needs durable state, document the answers to these questions before adding it:

1. What exact state is being persisted?
2. Why is in-memory or request-scoped state insufficient?
3. Which system is the source of truth for that state?
4. Is the app still local-only, or is it becoming remote and multi-user?
5. How are identity, credentials, and secrets bound and protected?

## Testing expectations

When an app follows the default policy:

- integration tests should pass after server restart without requiring app-local history
- normal runs should not create repo-owned persistence files

When an app makes an exception and adds durable state:

- add app-specific integration tests for restart behavior
- verify identity binding and secret handling
- verify missing-state and recovery behavior
