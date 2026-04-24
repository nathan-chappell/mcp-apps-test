import type {
  App,
  McpUiDisplayMode,
  McpUiHostContext,
} from "@modelcontextprotocol/ext-apps";

import type {
  ConfirmKnowledgeBaseCommandArguments,
  KnowledgeBaseCommandResult,
  KnowledgeBaseDeskState,
  KnowledgeBaseInfoArguments,
  KnowledgeInfoResult,
  KnowledgeBaseQueryArguments,
  KnowledgeQueryResult,
  RunKnowledgeBaseCommandArguments,
  ToolResultName,
  UpdateKnowledgeBaseArguments,
  UpdateKnowledgeBaseResult,
  UploadFinalizeResult,
} from "./types";
import { getStructuredContent } from "./types";

export interface UploadNodeInput {
  tag_ids: string[];
  file: File;
}

export interface KnowledgeBaseBridge {
  readonly mode: "host" | "mock";
  readonly hostContext?: McpUiHostContext;
  readonly initial_state: KnowledgeBaseDeskState;
  readonly initial_query_result: KnowledgeQueryResult;
  get_knowledge_base_info(
    args: KnowledgeBaseInfoArguments,
  ): Promise<KnowledgeInfoResult>;
  query_knowledge_base(
    args: KnowledgeBaseQueryArguments,
  ): Promise<KnowledgeQueryResult>;
  update_knowledge_base(
    args: UpdateKnowledgeBaseArguments,
  ): Promise<UpdateKnowledgeBaseResult>;
  run_knowledge_base_command(
    args: RunKnowledgeBaseCommandArguments,
  ): Promise<KnowledgeBaseCommandResult>;
  confirm_knowledge_base_command(
    args: ConfirmKnowledgeBaseCommandArguments,
  ): Promise<KnowledgeBaseCommandResult>;
  upload_node(input: UploadNodeInput): Promise<UploadFinalizeResult>;
}

export interface KnowledgeBaseHostControls {
  readonly mode: "host" | "mock";
  readonly supportedDisplayModes: McpUiDisplayMode[];
  readonly supportsModelContext: boolean;
  update_model_context(markdown: string): Promise<void>;
  request_display_mode(mode: McpUiDisplayMode): Promise<McpUiDisplayMode | null>;
}

async function callStructuredTool<T, TArgs extends object = Record<string, unknown>>(
  app: App,
  name: ToolResultName,
  args: TArgs,
): Promise<T> {
  const result = await app.callServerTool({
    name,
    arguments: args as unknown as Record<string, unknown>,
  });
  return getStructuredContent<T>(result);
}

export function createHostBridge(
  app: App,
  initialQueryResult: KnowledgeQueryResult,
  hostContext?: McpUiHostContext,
): KnowledgeBaseBridge {
  return {
    mode: "host",
    hostContext,
    initial_state: initialQueryResult.knowledge_base_state,
    initial_query_result: initialQueryResult,
    get_knowledge_base_info(args) {
      return callStructuredTool<KnowledgeInfoResult, KnowledgeBaseInfoArguments>(
        app,
        "get_knowledge_base_info",
        args,
      );
    },
    query_knowledge_base(args) {
      return callStructuredTool<KnowledgeQueryResult, KnowledgeBaseQueryArguments>(
        app,
        "query_knowledge_base",
        args,
      );
    },
    update_knowledge_base(args) {
      return callStructuredTool<
        UpdateKnowledgeBaseResult,
        UpdateKnowledgeBaseArguments
      >(app, "update_knowledge_base", args);
    },
    run_knowledge_base_command(args) {
      return callStructuredTool<
        KnowledgeBaseCommandResult,
        RunKnowledgeBaseCommandArguments
      >(app, "run_knowledge_base_command", args);
    },
    confirm_knowledge_base_command(args) {
      return callStructuredTool<
        KnowledgeBaseCommandResult,
        ConfirmKnowledgeBaseCommandArguments
      >(app, "confirm_knowledge_base_command", args);
    },
    async upload_node(input) {
      const updateResult = await callStructuredTool<
        UpdateKnowledgeBaseResult,
        UpdateKnowledgeBaseArguments
      >(app, "update_knowledge_base", {
        action: "prepare_upload",
      });
      const uploadSession = updateResult.upload_session;
      if (!uploadSession) {
        throw new Error(
          "Knowledge-base upload preparation did not return an upload session.",
        );
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
      return (await response.json()) as UploadFinalizeResult;
    },
  };
}

export function createHostControls(app: App): KnowledgeBaseHostControls {
  return {
    mode: "host",
    get supportedDisplayModes() {
      return app.getHostContext()?.availableDisplayModes ?? [];
    },
    get supportsModelContext() {
      return Boolean(app.getHostCapabilities()?.updateModelContext);
    },
    async update_model_context(markdown) {
      if (!app.getHostCapabilities()?.updateModelContext) {
        return;
      }
      await app.updateModelContext({
        content: [{ type: "text", text: markdown }],
      });
    },
    async request_display_mode(mode) {
      const supportedModes = app.getHostContext()?.availableDisplayModes ?? [];
      if (!supportedModes.includes(mode)) {
        return null;
      }
      const result = await app.requestDisplayMode({ mode });
      return result.mode;
    },
  };
}
