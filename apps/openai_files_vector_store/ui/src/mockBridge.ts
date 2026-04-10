import type { McpUiHostContext } from "@modelcontextprotocol/ext-apps";

import type {
  AskVectorStoreArguments,
  AskVectorStoreResult,
  AttributeValue,
  DeleteFileArguments,
  DeleteFileResult,
  GetVectorStoreStatusArguments,
  OpenVectorStoreConsoleResult,
  SearchHit,
  SearchVectorStoreArguments,
  SearchVectorStoreResult,
  UpdateVectorStoreFileAttributesArguments,
  VectorStoreFileSummary,
  VectorStoreListResult,
  VectorStoreStatusResult,
} from "./types";
import type { VectorStoreConsoleBridge } from "./bridge";

const RESERVED_FILE_ID_ATTRIBUTE = "openai_file_id";
const RESERVED_FILENAME_ATTRIBUTE = "filename";

type MockDocument = {
  vector_store_id: string;
  file_id: string;
  filename: string;
  text: string;
};

const HOST_CONTEXT: McpUiHostContext = {
  theme: "light",
  locale: "en-US",
  displayMode: "inline",
  availableDisplayModes: ["inline", "fullscreen"],
  safeAreaInsets: { top: 0, right: 0, bottom: 0, left: 0 },
};

const INITIAL_VECTOR_STORE_LIST: VectorStoreListResult = {
  total_returned: 2,
  vector_stores: [
    {
      id: "vs_demo_ops",
      name: "Operations Runbooks",
      status: "completed",
      created_at: 1_744_281_600,
      last_active_at: 1_744_296_400,
      usage_bytes: 13_824,
      expires_at: null,
      metadata: { owner: "platform", dataset: "runbooks" },
      file_counts: {
        completed: 2,
        failed: 0,
        in_progress: 0,
        cancelled: 0,
        total: 2,
      },
    },
    {
      id: "vs_demo_agent",
      name: "Agent Guidance Notes",
      status: "completed",
      created_at: 1_744_288_800,
      last_active_at: 1_744_297_300,
      usage_bytes: 9_216,
      expires_at: null,
      metadata: { owner: "assistant", dataset: "guidance" },
      file_counts: {
        completed: 1,
        failed: 0,
        in_progress: 0,
        cancelled: 0,
        total: 1,
      },
    },
  ],
};

const INITIAL_STATUS_BY_ID: Record<string, VectorStoreStatusResult> = {
  vs_demo_ops: {
    vector_store: INITIAL_VECTOR_STORE_LIST.vector_stores[0],
    files: [
      {
        id: "file_ops_alpha",
        created_at: 1_744_281_620,
        status: "completed",
        usage_bytes: 6_912,
        vector_store_id: "vs_demo_ops",
        attributes: {
          source: "ops-alpha",
          owner: "platform",
          [RESERVED_FILE_ID_ATTRIBUTE]: "file_ops_alpha",
          [RESERVED_FILENAME_ATTRIBUTE]: "incident-runbook.md",
        },
        last_error: null,
      },
      {
        id: "file_ops_beta",
        created_at: 1_744_281_780,
        status: "completed",
        usage_bytes: 6_912,
        vector_store_id: "vs_demo_ops",
        attributes: {
          source: "ops-beta",
          tier: 2,
          [RESERVED_FILE_ID_ATTRIBUTE]: "file_ops_beta",
          [RESERVED_FILENAME_ATTRIBUTE]: "search-tuning.md",
        },
        last_error: null,
      },
    ],
    batch: null,
    batch_files: [],
  },
  vs_demo_agent: {
    vector_store: INITIAL_VECTOR_STORE_LIST.vector_stores[1],
    files: [
      {
        id: "file_agent_notes",
        created_at: 1_744_288_820,
        status: "completed",
        usage_bytes: 9_216,
        vector_store_id: "vs_demo_agent",
        attributes: {
          source: "agent-notes",
          [RESERVED_FILE_ID_ATTRIBUTE]: "file_agent_notes",
          [RESERVED_FILENAME_ATTRIBUTE]: "agent-capabilities.md",
        },
        last_error: null,
      },
    ],
    batch: null,
    batch_files: [],
  },
};

const INITIAL_DOCUMENTS: MockDocument[] = [
  {
    vector_store_id: "vs_demo_ops",
    file_id: "file_ops_alpha",
    filename: "incident-runbook.md",
    text: "Escalate pager alerts to the platform rotation and verify vector store ingestion before retrying agent retrieval.",
  },
  {
    vector_store_id: "vs_demo_ops",
    file_id: "file_ops_beta",
    filename: "search-tuning.md",
    text: "Use exact markers for smoke tests and prefer raw search before ask_vector_store when validating retrieval quality.",
  },
  {
    vector_store_id: "vs_demo_agent",
    file_id: "file_agent_notes",
    filename: "agent-capabilities.md",
    text: "The VS Code agent can use OpenAI files and vector stores as additional retrieval context through the MCP console.",
  },
];

function wait(durationMs: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, durationMs));
}

function cloneVectorStoreList(source: VectorStoreListResult): VectorStoreListResult {
  return {
    total_returned: source.total_returned,
    vector_stores: source.vector_stores.map((store) => ({
      ...store,
      metadata: store.metadata ? { ...store.metadata } : null,
      file_counts: { ...store.file_counts },
    })),
  };
}

function cloneStatusById(
  source: Record<string, VectorStoreStatusResult>,
): Record<string, VectorStoreStatusResult> {
  return Object.fromEntries(
    Object.entries(source).map(([vectorStoreId, status]) => [
      vectorStoreId,
      {
        vector_store: {
          ...status.vector_store,
          metadata: status.vector_store.metadata
            ? { ...status.vector_store.metadata }
            : null,
          file_counts: { ...status.vector_store.file_counts },
        },
        files: status.files.map((file) => ({
          ...file,
          attributes: file.attributes ? { ...file.attributes } : null,
        })),
        batch: status.batch
          ? {
              ...status.batch,
              file_counts: { ...status.batch.file_counts },
            }
          : null,
        batch_files: status.batch_files.map((file) => ({
          ...file,
          attributes: file.attributes ? { ...file.attributes } : null,
        })),
      },
    ]),
  );
}

function cloneDocuments(source: MockDocument[]): MockDocument[] {
  return source.map((document) => ({ ...document }));
}

function scoreHit(query: string, text: string): number {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return 0;
  }
  const normalizedText = text.toLowerCase();
  if (normalizedText.includes(normalizedQuery)) {
    return 0.97;
  }
  const queryTerms = normalizedQuery.split(/\s+/);
  const matches = queryTerms.filter((term) => normalizedText.includes(term)).length;
  return matches === 0 ? 0 : matches / queryTerms.length;
}

function buildSearchHits(
  documents: MockDocument[],
  args: SearchVectorStoreArguments | AskVectorStoreArguments,
  maxNumResults: number,
): SearchHit[] {
  return documents
    .filter((document) => document.vector_store_id === args.vector_store_id)
    .filter((document) => (args.file_id ? document.file_id === args.file_id : true))
    .filter((document) => (args.filename ? document.filename === args.filename : true))
    .map((document) => ({
      file_id: document.file_id,
      filename: document.filename,
      score: scoreHit("query" in args ? args.query : args.question, document.text),
      text: document.text,
      attributes: null,
    }))
    .filter((hit) => hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, maxNumResults);
}

function updateVectorStoreCounts(state: {
  vectorStoreList: VectorStoreListResult;
  statusById: Record<string, VectorStoreStatusResult>;
}): void {
  for (const vectorStore of state.vectorStoreList.vector_stores) {
    const status = state.statusById[vectorStore.id];
    const total = status?.files.length ?? 0;
    vectorStore.file_counts = {
      completed: total,
      failed: 0,
      in_progress: 0,
      cancelled: 0,
      total,
    };
    if (status) {
      status.vector_store.file_counts = { ...vectorStore.file_counts };
    }
  }
}

function asReservedAttributes(
  file: VectorStoreFileSummary,
  attributes: Record<string, AttributeValue> | undefined,
): Record<string, AttributeValue> {
  const nextAttributes = { ...(attributes ?? {}) };
  nextAttributes[RESERVED_FILE_ID_ATTRIBUTE] = file.id;
  nextAttributes[RESERVED_FILENAME_ATTRIBUTE] =
    typeof file.attributes?.[RESERVED_FILENAME_ATTRIBUTE] === "string"
      ? file.attributes[RESERVED_FILENAME_ATTRIBUTE]
      : file.id;
  return nextAttributes;
}

export function createMockBridge(): VectorStoreConsoleBridge {
  const state = {
    vectorStoreList: cloneVectorStoreList(INITIAL_VECTOR_STORE_LIST),
    statusById: cloneStatusById(INITIAL_STATUS_BY_ID),
    documents: cloneDocuments(INITIAL_DOCUMENTS),
  };
  updateVectorStoreCounts(state);

  const initialVectorStoreId = state.vectorStoreList.vector_stores[0]?.id ?? null;
  const initialState: OpenVectorStoreConsoleResult = {
    vector_store_list: state.vectorStoreList,
    selected_vector_store_id: initialVectorStoreId,
    selected_vector_store_status:
      initialVectorStoreId === null ? null : state.statusById[initialVectorStoreId],
    search_panel: {
      query: "",
      max_num_results: 5,
      rewrite_query: false,
      scope: "vector_store",
      file_id: null,
      filename: null,
    },
    ask_panel: {
      question: "",
      max_num_results: 5,
      scope: "vector_store",
      file_id: null,
      filename: null,
    },
  };

  return {
    mode: "mock",
    hostContext: HOST_CONTEXT,
    initial_state: initialState,
    async list_vector_stores() {
      await wait(120);
      updateVectorStoreCounts(state);
      return cloneVectorStoreList(state.vectorStoreList);
    },
    async get_vector_store_status(args: GetVectorStoreStatusArguments) {
      await wait(160);
      const status = state.statusById[args.vector_store_id];
      if (!status) {
        throw new Error(`Unknown mock vector store: ${args.vector_store_id}`);
      }
      return cloneStatusById({ [args.vector_store_id]: status })[args.vector_store_id];
    },
    async search_vector_store(args: SearchVectorStoreArguments): Promise<SearchVectorStoreResult> {
      await wait(180);
      const hits = buildSearchHits(state.documents, args, args.max_num_results ?? 5);
      return {
        vector_store_id: args.vector_store_id,
        query: args.query,
        file_id: args.file_id ?? null,
        filename: args.filename ?? null,
        hits,
        total_hits: hits.length,
      };
    },
    async ask_vector_store(args: AskVectorStoreArguments): Promise<AskVectorStoreResult> {
      await wait(240);
      const hits = buildSearchHits(state.documents, args, args.max_num_results ?? 5);
      const answer =
        hits.length > 0
          ? `Based on the indexed content, ${hits[0].text}`
          : "I could not find grounded support for that question in the selected vector store.";

      return {
        vector_store_id: args.vector_store_id,
        question: args.question,
        file_id: args.file_id ?? null,
        filename: args.filename ?? null,
        answer,
        model: "mock-file-search-agent",
        search_calls: [
          {
            id: "mock-search-call-1",
            status: "completed",
            queries: [args.question],
            results: hits,
          },
        ],
      };
    },
    async update_vector_store_file_attributes(
      args: UpdateVectorStoreFileAttributesArguments,
    ): Promise<VectorStoreFileSummary> {
      await wait(140);
      const status = state.statusById[args.vector_store_id];
      const file = status?.files.find((candidate) => candidate.id === args.file_id);
      if (!status || !file) {
        throw new Error(`Unknown mock vector store file: ${args.file_id}`);
      }

      file.attributes = asReservedAttributes(file, args.attributes);
      return {
        ...file,
        attributes: file.attributes ? { ...file.attributes } : null,
      };
    },
    async delete_file(args: DeleteFileArguments): Promise<DeleteFileResult> {
      await wait(140);
      for (const status of Object.values(state.statusById)) {
        status.files = status.files.filter((file) => file.id !== args.file_id);
        status.batch_files = status.batch_files.filter((file) => file.id !== args.file_id);
      }
      state.documents = state.documents.filter((document) => document.file_id !== args.file_id);
      updateVectorStoreCounts(state);
      return {
        file_id: args.file_id,
        deleted: true,
      };
    },
  };
}
