import type { McpUiHostContext } from "@modelcontextprotocol/ext-apps";

import type {
  DerivedArtifactSummary,
  DocumentAskResult,
  DocumentCitation,
  DocumentDetail,
  DocumentFilters,
  DocumentLibraryQueryResult,
  DocumentLibraryState,
  DocumentLibraryStateResult,
  DocumentLibraryViewState,
  DocumentSearchHit,
  DocumentSearchResult,
  DocumentSummary,
  DocumentUploadFinalizeResult,
  GetDocumentLibraryStateArguments,
  KnowledgeTagSummary,
  QueryDocumentLibraryArguments,
  UpdateDocumentLibraryArguments,
  UpdateDocumentLibraryResult,
} from "./types";
import type { DocumentLibraryBridge, UploadDocumentInput } from "./bridge";

const HOST_CONTEXT: McpUiHostContext = {
  theme: "light",
  locale: "en-US",
  displayMode: "inline",
  availableDisplayModes: ["inline", "fullscreen"],
  safeAreaInsets: { top: 0, right: 0, bottom: 0, left: 0 },
};

type MockStore = {
  library_id: string;
  title: string;
  description: string;
  created_at: string;
  updated_at: string;
  conversation_id: string;
  tags: Map<string, KnowledgeTagSummary>;
  documents: Map<string, DocumentDetail>;
};

function nowIso(): string {
  return new Date().toISOString();
}

function uniqueId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function summarizeDocument(document: DocumentDetail): DocumentSummary {
  const {
    original_mime_type: _originalMimeType,
    derived_artifacts: _derivedArtifacts,
    ...summary
  } = document;
  return summary;
}

function buildTag(
  id: string,
  name: string,
  color: string | null,
): KnowledgeTagSummary {
  return {
    id,
    name,
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    color,
    node_count: 0,
  };
}

function buildArtifact(
  id: string,
  kind: string,
  text: string,
  structuredPayload: unknown = null,
): DerivedArtifactSummary {
  const timestamp = nowIso();
  return {
    id,
    kind,
    openai_file_id: id,
    text_content: text,
    structured_payload: structuredPayload,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function seededStore(): MockStore {
  const createdAt = nowIso();
  const research = buildTag("tag_research", "research", "#0f766e");
  const ops = buildTag("tag_ops", "ops", "#b45309");
  const product = buildTag("tag_product", "product", "#2563eb");

  const documents = new Map<string, DocumentDetail>();
  documents.set("doc_playbook", {
    id: "doc_playbook",
    title: "MVP Search Playbook",
    original_filename: "mvp-search-playbook.md",
    media_type: "text/markdown",
    source_kind: "document",
    status: "ready",
    byte_size: 3200,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [research, product],
    derived_kinds: ["document_text"],
    openai_original_file_id: "file_playbook",
    download_url: "#",
    original_mime_type: "text/markdown",
    derived_artifacts: [
      buildArtifact(
        "artifact_playbook",
        "document_text",
        "This playbook argues for a simpler MVP: tag filtering first, then filename and created date filtering, with grounded search and ask layered on top.",
      ),
    ],
  });
  documents.set("doc_incident", {
    id: "doc_incident",
    title: "Incident Notes",
    original_filename: "incident-notes.txt",
    media_type: "text/plain",
    source_kind: "document",
    status: "ready",
    byte_size: 1800,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [ops],
    derived_kinds: ["document_text"],
    openai_original_file_id: "file_incident",
    download_url: "#",
    original_mime_type: "text/plain",
    derived_artifacts: [
      buildArtifact(
        "artifact_incident",
        "document_text",
        "When retrieval quality drops, inspect upload freshness, confirm the expected tags, and narrow by filename before broadening the search.",
      ),
    ],
  });
  documents.set("doc_interviews", {
    id: "doc_interviews",
    title: "Customer Interview Themes",
    original_filename: "customer-interviews.md",
    media_type: "text/markdown",
    source_kind: "document",
    status: "ready",
    byte_size: 4100,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [research],
    derived_kinds: ["document_text"],
    openai_original_file_id: "file_interviews",
    download_url: "#",
    original_mime_type: "text/markdown",
    derived_artifacts: [
      buildArtifact(
        "artifact_interviews",
        "document_text",
        "Teams repeatedly asked for simpler browsing, direct metadata filters, and a clear separation between document management and question answering.",
      ),
    ],
  });

  return {
    library_id: "library_demo",
    title: "Mock Document Library",
    description: "Standalone mock data for UI development.",
    created_at: createdAt,
    updated_at: createdAt,
    conversation_id: "mock-conversation",
    tags: new Map([
      [research.id, research],
      [ops.id, ops],
      [product.id, product],
    ]),
    documents,
  };
}

const store = seededStore();

function documentText(document: DocumentDetail): string {
  return document.derived_artifacts.map((artifact) => artifact.text_content).join("\n");
}

function recountTags(inputStore: MockStore): KnowledgeTagSummary[] {
  const counts = new Map<string, number>();
  for (const document of inputStore.documents.values()) {
    for (const tag of document.tags) {
      counts.set(tag.id, (counts.get(tag.id) ?? 0) + 1);
    }
  }

  return [...inputStore.tags.values()]
    .map((tag) => ({
      ...tag,
      node_count: counts.get(tag.id) ?? 0,
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function normalizeFilters(
  args: Partial<GetDocumentLibraryStateArguments>,
): Omit<DocumentFilters, "selected_tag_names" | "matching_document_ids"> {
  return {
    tag_ids: [...new Set(args.tag_ids ?? [])],
    tag_match_mode: args.tag_match_mode ?? "all",
    filename_query: args.filename_query?.trim() || null,
    created_from: args.created_from ?? null,
    created_to: args.created_to ?? null,
  };
}

function filterDocuments(
  inputStore: MockStore,
  args: Partial<GetDocumentLibraryStateArguments>,
): { filters: DocumentFilters; documents: DocumentDetail[]; tags: KnowledgeTagSummary[] } {
  const normalized = normalizeFilters(args);
  const tags = recountTags(inputStore);
  const tagNameById = new Map(tags.map((tag) => [tag.id, tag.name]));
  const selectedTagIds = new Set(normalized.tag_ids);
  const normalizedFilenameQuery = normalized.filename_query?.toLowerCase() ?? null;

  const documents = [...inputStore.documents.values()]
    .filter((document) => {
      if (selectedTagIds.size > 0) {
        const documentTagIds = new Set(document.tags.map((tag) => tag.id));
        if (normalized.tag_match_mode === "all") {
          if ([...selectedTagIds].some((tagId) => !documentTagIds.has(tagId))) {
            return false;
          }
        } else if ([...selectedTagIds].every((tagId) => !documentTagIds.has(tagId))) {
          return false;
        }
      }

      if (
        normalizedFilenameQuery &&
        !document.original_filename.toLowerCase().includes(normalizedFilenameQuery)
      ) {
        return false;
      }

      const createdDate = document.created_at.slice(0, 10);
      if (normalized.created_from && createdDate < normalized.created_from) {
        return false;
      }
      if (normalized.created_to && createdDate > normalized.created_to) {
        return false;
      }
      return true;
    })
    .sort((left, right) => right.created_at.localeCompare(left.created_at));

  const filters: DocumentFilters = {
    ...normalized,
    selected_tag_names: normalized.tag_ids
      .map((tagId) => tagNameById.get(tagId))
      .filter((value): value is string => Boolean(value)),
    matching_document_ids: documents.map((document) => document.id),
  };

  return { filters, documents, tags };
}

function buildViewState(
  inputStore: MockStore,
  args: Partial<GetDocumentLibraryStateArguments>,
): DocumentLibraryViewState {
  const { filters, documents, tags } = filterDocuments(inputStore, args);
  const state: DocumentLibraryState = {
    library: {
      id: inputStore.library_id,
      title: inputStore.title,
      description: inputStore.description,
      created_at: inputStore.created_at,
      updated_at: inputStore.updated_at,
      document_count: inputStore.documents.size,
      filtered_document_count: documents.length,
      tag_count: tags.length,
      vector_store_ready: true,
    },
    tags,
    documents: documents.map(summarizeDocument),
    filters,
  };

  return {
    access: {
      status: "active",
      message: "Mock access active.",
      user: {
        clerk_user_id: "mock-user",
        display_name: "Mock User",
        primary_email: "mock@example.com",
        active: true,
        role: "admin",
      },
    },
    library: state,
    capabilities: {
      upload_url: "mock://uploads",
      upload_token_ttl_seconds: 900,
      supports_video_audio_extraction: true,
      accepted_hint:
        "Upload text-like files directly, plus images, audio, and video.",
    },
  };
}

function buildStateResult(
  inputStore: MockStore,
  args: Partial<GetDocumentLibraryStateArguments>,
): DocumentLibraryStateResult {
  const viewState = buildViewState(inputStore, args);
  const detailDocumentId = args.detail_document_id ?? null;
  const detail =
    detailDocumentId && inputStore.documents.has(detailDocumentId)
      ? inputStore.documents.get(detailDocumentId) ?? null
      : null;
  return {
    document_library_state: viewState,
    document_detail: detail,
  };
}

function searchDocuments(
  inputStore: MockStore,
  args: QueryDocumentLibraryArguments,
): DocumentSearchResult {
  const { documents } = filterDocuments(inputStore, args);
  const query = args.query.trim().toLowerCase();
  const hits: DocumentSearchHit[] = [];
  for (const document of documents) {
    const haystack = [document.title, document.original_filename, documentText(document)]
      .join("\n")
      .toLowerCase();
    const queryTerms = query.split(/\s+/).filter(Boolean);
    const matches = queryTerms.filter((term) => haystack.includes(term)).length;
    if (matches === 0 && !haystack.includes(query)) {
      continue;
    }
    const score =
      haystack.includes(query) && queryTerms.length > 0
        ? 0.95
        : Math.max(0.45, matches / Math.max(queryTerms.length, 1));
    hits.push({
      document_id: document.id,
      document_title: document.title,
      original_filename: document.original_filename,
      derived_artifact_id: document.derived_artifacts[0]?.id ?? null,
      openai_file_id: document.derived_artifacts[0]?.openai_file_id ?? document.id,
      original_openai_file_id: document.openai_original_file_id,
      media_type: document.media_type,
      source_kind: document.source_kind,
      score,
      text: documentText(document).slice(0, 360),
      tags: document.tags.map((tag) => tag.name),
    });
  }
  hits.sort((left, right) => right.score - left.score);

  return {
    query: args.query,
    hits: hits.slice(0, 8),
    total_hits: Math.min(hits.length, 8),
  };
}

function answerFromHits(
  inputStore: MockStore,
  args: QueryDocumentLibraryArguments,
): DocumentAskResult {
  const searchResult = searchDocuments(inputStore, args);
  if (searchResult.hits.length === 0) {
    return {
      query: args.query,
      answer: "No documents match the current filters yet.",
      model: "mock-model",
      conversation_id: store.conversation_id,
      citations: [],
      hits: [],
    };
  }

  const citations: DocumentCitation[] = searchResult.hits.slice(0, 3).map((hit) => ({
    label: hit.document_title,
    document_id: hit.document_id,
    document_title: hit.document_title,
    original_filename: hit.original_filename,
    quote: hit.text,
    url: null,
    source: "document_library",
  }));
  const answer = `Top matching documents point to the same theme: ${searchResult.hits
    .slice(0, 3)
    .map((hit) => hit.document_title)
    .join(", ")}. The library is easiest to work with when tags, filename search, and created-date filters narrow the corpus before asking a grounded question.`;

  return {
    query: args.query,
    answer,
    model: "mock-model",
    conversation_id: store.conversation_id,
    citations,
    hits: searchResult.hits,
  };
}

function buildQueryResult(
  inputStore: MockStore,
  args: QueryDocumentLibraryArguments,
): DocumentLibraryQueryResult {
  const viewState = buildViewState(inputStore, args);
  if (args.mode === "ask") {
    return {
      mode: "ask",
      document_library_state: viewState,
      search_result: null,
      ask_result: answerFromHits(inputStore, args),
    };
  }
  return {
    mode: "search",
    document_library_state: viewState,
    search_result: searchDocuments(inputStore, args),
    ask_result: null,
  };
}

async function uploadDocument(
  inputStore: MockStore,
  input: UploadDocumentInput,
): Promise<DocumentUploadFinalizeResult> {
  const createdAt = nowIso();
  const text =
    input.file.type.startsWith("text/") || /\.(md|txt|json|ts|tsx|js|py)$/i.test(input.file.name)
      ? (await input.file.text()).slice(0, 8000)
      : `Uploaded ${input.file.name}.`;
  const tags = recountTags(inputStore).filter((tag) => input.tag_ids.includes(tag.id));
  const document: DocumentDetail = {
    id: uniqueId("doc"),
    title: input.file.name.replace(/\.[^.]+$/, "") || "Uploaded document",
    original_filename: input.file.name,
    media_type: input.file.type || "application/octet-stream",
    source_kind: input.file.type.startsWith("image/")
      ? "image"
      : input.file.type.startsWith("audio/")
        ? "audio"
        : input.file.type.startsWith("video/")
          ? "video"
          : "document",
    status: "ready",
    byte_size: input.file.size,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags,
    derived_kinds: ["document_text"],
    openai_original_file_id: uniqueId("file"),
    download_url: "#",
    original_mime_type: input.file.type || null,
    derived_artifacts: [buildArtifact(uniqueId("artifact"), "document_text", text)],
  };

  inputStore.documents.set(document.id, document);
  inputStore.updated_at = createdAt;
  return { document: summarizeDocument(document) };
}

export function createMockBridge(
  hostContext: McpUiHostContext | undefined = HOST_CONTEXT,
): DocumentLibraryBridge {
  const initialLibraryState = buildStateResult(store, {});
  const initialQueryResult: DocumentLibraryQueryResult = {
    mode: "search",
    document_library_state: initialLibraryState.document_library_state,
    search_result: null,
    ask_result: null,
  };

  return {
    mode: "mock",
    hostContext,
    initial_library_state: initialLibraryState,
    initial_query_result: initialQueryResult,
    async get_document_library_state(args) {
      return buildStateResult(store, args);
    },
    async query_document_library(args) {
      return buildQueryResult(store, {
        ...args,
        mode: args.mode ?? "search",
      });
    },
    async update_document_library(args) {
      if (args.action === "prepare_upload") {
        return {
          action: args.action,
          upload_session: {
            upload_url: "mock://uploads",
            upload_token: uniqueId("upload"),
            expires_at: Math.floor(Date.now() / 1000) + 900,
          },
          tag: null,
          document: null,
        };
      }

      if (args.action === "create_tag") {
        const name = args.name?.trim();
        if (!name) {
          throw new Error("A tag name is required.");
        }
        const existing = [...store.tags.values()].find(
          (tag) => tag.name.toLowerCase() === name.toLowerCase(),
        );
        if (existing) {
          return {
            action: args.action,
            upload_session: null,
            tag: existing,
            document: null,
          };
        }

        const tag = buildTag(
          uniqueId("tag"),
          name,
          args.color?.trim() || "#4f46e5",
        );
        store.tags.set(tag.id, tag);
        store.updated_at = nowIso();
        return {
          action: args.action,
          upload_session: null,
          tag,
          document: null,
        };
      }

      if (!args.document_id) {
        throw new Error("document_id is required for set_document_tags.");
      }
      const document = store.documents.get(args.document_id);
      if (!document) {
        throw new Error("Document not found.");
      }
      document.tags = recountTags(store).filter((tag) => (args.tag_ids ?? []).includes(tag.id));
      document.updated_at = nowIso();
      store.documents.set(document.id, document);
      store.updated_at = document.updated_at;
      return {
        action: args.action,
        upload_session: null,
        tag: null,
        document: summarizeDocument(document),
      };
    },
    async upload_document(input) {
      return uploadDocument(store, input);
    },
  };
}
