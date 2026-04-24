import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";

import type {
  DocumentLibraryQueryResult,
  DocumentLibraryStateResult,
  DocumentUploadFinalizeResult,
  GetDocumentLibraryStateArguments,
  QueryDocumentLibraryArguments,
  ToolResultName,
  UpdateDocumentLibraryArguments,
  UpdateDocumentLibraryResult,
} from "./types";
import { getStructuredContent } from "./types";

export interface UploadDocumentInput {
  tag_ids: string[];
  file: File;
}

export interface DocumentLibraryBridge {
  readonly mode: "host" | "mock";
  readonly hostContext?: McpUiHostContext;
  readonly initial_library_state: DocumentLibraryStateResult | null;
  readonly initial_query_result: DocumentLibraryQueryResult | null;
  get_document_library_state(
    args: GetDocumentLibraryStateArguments,
  ): Promise<DocumentLibraryStateResult>;
  query_document_library(
    args: QueryDocumentLibraryArguments,
  ): Promise<DocumentLibraryQueryResult>;
  update_document_library(
    args: UpdateDocumentLibraryArguments,
  ): Promise<UpdateDocumentLibraryResult>;
  upload_document(input: UploadDocumentInput): Promise<DocumentUploadFinalizeResult>;
}

async function callStructuredTool<T, TArgs extends object = Record<string, unknown>>(
  app: App,
  name: ToolResultName,
  args: TArgs,
): Promise<T> {
  const result = await app.callServerTool({
    name,
    arguments: args as Record<string, unknown>,
  });
  return getStructuredContent<T>(result);
}

export function createLibraryHostBridge(
  app: App,
  initialState: DocumentLibraryStateResult,
  hostContext?: McpUiHostContext,
): DocumentLibraryBridge {
  return {
    mode: "host",
    hostContext,
    initial_library_state: initialState,
    initial_query_result: null,
    get_document_library_state(args) {
      return callStructuredTool<
        DocumentLibraryStateResult,
        GetDocumentLibraryStateArguments
      >(app, "get_document_library_state", args);
    },
    query_document_library(args) {
      return callStructuredTool<
        DocumentLibraryQueryResult,
        QueryDocumentLibraryArguments
      >(app, "query_document_library", args);
    },
    update_document_library(args) {
      return callStructuredTool<
        UpdateDocumentLibraryResult,
        UpdateDocumentLibraryArguments
      >(app, "update_document_library", args);
    },
    async upload_document(input) {
      const updateResult = await callStructuredTool<
        UpdateDocumentLibraryResult,
        UpdateDocumentLibraryArguments
      >(app, "update_document_library", {
        action: "prepare_upload",
      });
      const uploadSession = updateResult.upload_session;
      if (!uploadSession) {
        throw new Error("Document upload preparation did not return an upload session.");
      }

      const form = new FormData();
      form.set("upload_token", uploadSession.upload_token);
      form.set("file", input.file);
      for (const tagId of input.tag_ids) {
        form.append("tag_ids", tagId);
      }
      const response = await fetch(uploadSession.upload_url, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return (await response.json()) as DocumentUploadFinalizeResult;
    },
  };
}

export function createAskHostBridge(
  app: App,
  initialResult: DocumentLibraryQueryResult,
  hostContext?: McpUiHostContext,
): DocumentLibraryBridge {
  return {
    mode: "host",
    hostContext,
    initial_library_state: null,
    initial_query_result: initialResult,
    get_document_library_state(args) {
      return callStructuredTool<
        DocumentLibraryStateResult,
        GetDocumentLibraryStateArguments
      >(app, "get_document_library_state", args);
    },
    query_document_library(args) {
      return callStructuredTool<
        DocumentLibraryQueryResult,
        QueryDocumentLibraryArguments
      >(app, "query_document_library", args);
    },
    update_document_library(args) {
      return callStructuredTool<
        UpdateDocumentLibraryResult,
        UpdateDocumentLibraryArguments
      >(app, "update_document_library", args);
    },
    async upload_document(input) {
      const updateResult = await callStructuredTool<
        UpdateDocumentLibraryResult,
        UpdateDocumentLibraryArguments
      >(app, "update_document_library", {
        action: "prepare_upload",
      });
      const uploadSession = updateResult.upload_session;
      if (!uploadSession) {
        throw new Error("Document upload preparation did not return an upload session.");
      }

      const form = new FormData();
      form.set("upload_token", uploadSession.upload_token);
      form.set("file", input.file);
      for (const tagId of input.tag_ids) {
        form.append("tag_ids", tagId);
      }
      const response = await fetch(uploadSession.upload_url, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return (await response.json()) as DocumentUploadFinalizeResult;
    },
  };
}
