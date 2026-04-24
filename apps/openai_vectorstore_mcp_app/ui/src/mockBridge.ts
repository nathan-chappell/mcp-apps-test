import type { McpUiHostContext } from "@modelcontextprotocol/ext-apps";

import type {
  BranchSearchNode,
  ConfirmKnowledgeBaseCommandArguments,
  DerivedArtifactSummary,
  FileSearchCallSummary,
  KnowledgeAnswerCitation,
  KnowledgeBaseCommandResult,
  KnowledgeBaseContext,
  KnowledgeBaseDeskState,
  KnowledgeBaseInfoArguments,
  KnowledgeBaseQueryArguments,
  KnowledgeBaseState,
  KnowledgeBranchSearchResult,
  KnowledgeChatResult,
  KnowledgeEdgeSummary,
  KnowledgeFileSearchResult,
  KnowledgeInfoResult,
  KnowledgeNodeDetail,
  KnowledgeNodeSummary,
  KnowledgeQueryResult,
  KnowledgeTagSummary,
  PendingCommandResult,
  RunKnowledgeBaseCommandArguments,
  SearchHit,
  UpdateKnowledgeBaseArguments,
  UpdateKnowledgeBaseResult,
  UploadFinalizeResult,
  UploadSessionResult,
  WebSearchCallSummary,
} from "./types";
import type {
  KnowledgeBaseBridge,
  KnowledgeBaseHostControls,
  UploadNodeInput,
} from "./bridge";

const HOST_CONTEXT: McpUiHostContext = {
  theme: "light",
  locale: "en-US",
  displayMode: "inline",
  availableDisplayModes: ["inline", "fullscreen"],
  safeAreaInsets: { top: 0, right: 0, bottom: 0, left: 0 },
};

type MockStore = {
  knowledge_base_id: string;
  title: string;
  description: string;
  created_at: string;
  updated_at: string;
  conversation_id: string;
  tags: Map<string, KnowledgeTagSummary>;
  nodes: Map<string, KnowledgeNodeDetail>;
  edges: Map<string, KnowledgeEdgeSummary>;
  pending_confirmations: Map<string, { action: "delete_node"; node_id: string }>;
};

type SelectionArgs = {
  selected_node_id: string | null;
  graph_selection_mode: "self" | "children" | "descendants";
  tag_ids: string[];
  tag_match_mode: "all" | "any";
  media_types: string[];
  include_web: boolean;
  rewrite_query: boolean;
  branch_factor: number;
  depth: number;
  max_results: number;
};

function nowIso(): string {
  return new Date().toISOString();
}

function uniqueId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function formatUploadSession(): UploadSessionResult {
  return {
    upload_url: "mock://uploads",
    upload_token: `mock-upload-${Math.random().toString(36).slice(2, 10)}`,
    expires_at: Math.floor(Date.now() / 1000) + 900,
  };
}

function scoreText(query: string, text: string): number {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return 0;
  }
  const normalizedText = text.toLowerCase();
  if (normalizedText.includes(normalizedQuery)) {
    return 0.96;
  }
  const terms = normalizedQuery.split(/\s+/).filter(Boolean);
  if (terms.length === 0) {
    return 0;
  }
  const hits = terms.filter((term) => normalizedText.includes(term)).length;
  return hits === 0 ? 0 : hits / terms.length;
}

function detailToSummary(node: KnowledgeNodeDetail): KnowledgeNodeSummary {
  const {
    original_mime_type: _originalMimeType,
    derived_artifacts: _derivedArtifacts,
    outgoing_edges: _outgoingEdges,
    incoming_edges: _incomingEdges,
    ...summary
  } = node;
  return summary;
}

function nodeSearchText(node: KnowledgeNodeDetail): string {
  const artifactText = node.derived_artifacts
    .map((artifact) => artifact.text_content)
    .join("\n");
  return [node.display_title, node.original_filename, artifactText]
    .filter(Boolean)
    .join("\n");
}

function seededStore(): MockStore {
  const createdAt = nowIso();
  const knowledgeBaseId = "kb_demo";

  const tagResearch: KnowledgeTagSummary = {
    id: "tag_research",
    name: "research",
    slug: "research",
    color: "#0f766e",
    node_count: 0,
  };
  const tagOps: KnowledgeTagSummary = {
    id: "tag_ops",
    name: "ops",
    slug: "ops",
    color: "#b45309",
    node_count: 0,
  };
  const tagCustomer: KnowledgeTagSummary = {
    id: "tag_customer",
    name: "customer",
    slug: "customer",
    color: "#2563eb",
    node_count: 0,
  };

  const primer: KnowledgeNodeDetail = {
    id: "node_primer",
    display_title: "Retrieval Primer",
    original_filename: "retrieval-primer.md",
    media_type: "text/markdown",
    source_kind: "document",
    status: "ready",
    byte_size: 2320,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [tagResearch],
    derived_kinds: ["document_text"],
    openai_original_file_id: "file_primer",
    download_url: "#",
    outgoing_edge_count: 0,
    incoming_edge_count: 0,
    original_mime_type: "text/markdown",
    derived_artifacts: [
      {
        id: "artifact_primer",
        kind: "document_text",
        openai_file_id: "file_primer_search",
        text_content:
          "Primer on retrieval quality. Discusses node-level filtering, graph scope, vector search recall, and using tags to create orthogonal slices across a corpus.",
        structured_payload: null,
        created_at: createdAt,
        updated_at: createdAt,
      },
    ],
    outgoing_edges: [],
    incoming_edges: [],
  };

  const runbook: KnowledgeNodeDetail = {
    id: "node_runbook",
    display_title: "Incident Runbook",
    original_filename: "incident-runbook.md",
    media_type: "text/markdown",
    source_kind: "document",
    status: "ready",
    byte_size: 1890,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [tagOps],
    derived_kinds: ["document_text"],
    openai_original_file_id: "file_runbook",
    download_url: "#",
    outgoing_edge_count: 0,
    incoming_edge_count: 0,
    original_mime_type: "text/markdown",
    derived_artifacts: [
      {
        id: "artifact_runbook",
        kind: "document_text",
        openai_file_id: "file_runbook_search",
        text_content:
          "Escalation runbook for search outages. Check ingestion status, confirm vector store processing, and narrow retrieval with node scope before broadening to the full knowledge base.",
        structured_payload: null,
        created_at: createdAt,
        updated_at: createdAt,
      },
    ],
    outgoing_edges: [],
    incoming_edges: [],
  };

  const whiteboard: KnowledgeNodeDetail = {
    id: "node_whiteboard",
    display_title: "Support Whiteboard",
    original_filename: "support-whiteboard.jpg",
    media_type: "image/jpeg",
    source_kind: "image",
    status: "ready",
    byte_size: 88432,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [tagCustomer],
    derived_kinds: ["image_description"],
    openai_original_file_id: "file_whiteboard",
    download_url: "#",
    outgoing_edge_count: 0,
    incoming_edge_count: 0,
    original_mime_type: "image/jpeg",
    derived_artifacts: [
      {
        id: "artifact_whiteboard",
        kind: "image_description",
        openai_file_id: "file_whiteboard_search",
        text_content:
          "Whiteboard photo with notes about support triage, customer escalation paths, tag filters, and improving graph-based retrieval flows.",
        structured_payload: {
          summary: "A photographed whiteboard used during support planning.",
          keywords: ["support", "triage", "graph", "tag filters"],
        },
        created_at: createdAt,
        updated_at: createdAt,
      },
    ],
    outgoing_edges: [],
    incoming_edges: [],
  };

  const meeting: KnowledgeNodeDetail = {
    id: "node_meeting",
    display_title: "Roadmap Sync",
    original_filename: "roadmap-sync.wav",
    media_type: "audio/wav",
    source_kind: "audio",
    status: "ready",
    byte_size: 2140000,
    error_message: null,
    created_at: createdAt,
    updated_at: createdAt,
    tags: [tagResearch, tagOps],
    derived_kinds: ["audio_transcript"],
    openai_original_file_id: "file_meeting",
    download_url: "#",
    outgoing_edge_count: 0,
    incoming_edge_count: 0,
    original_mime_type: "audio/wav",
    derived_artifacts: [
      {
        id: "artifact_meeting",
        kind: "audio_transcript",
        openai_file_id: "file_meeting_search",
        text_content:
          "[Speaker 1] The knowledge base should be a graph, not a folder tree.\n[Speaker 2] Tag filters need to remain independent from edges so search scope can combine both dimensions cleanly.",
        structured_payload: {
          duration: 74,
          segments: [
            {
              speaker: "Speaker 1",
              text: "The knowledge base should be a graph, not a folder tree.",
            },
            {
              speaker: "Speaker 2",
              text: "Tag filters need to remain independent from edges so search scope can combine both dimensions cleanly.",
            },
          ],
        },
        created_at: createdAt,
        updated_at: createdAt,
      },
    ],
    outgoing_edges: [],
    incoming_edges: [],
  };

  const edges: KnowledgeEdgeSummary[] = [
    {
      id: "edge_primer_runbook",
      from_node_id: primer.id,
      to_node_id: runbook.id,
      from_node_title: primer.display_title,
      to_node_title: runbook.display_title,
      label: "grounds",
      created_at: createdAt,
      updated_at: createdAt,
    },
    {
      id: "edge_primer_whiteboard",
      from_node_id: primer.id,
      to_node_id: whiteboard.id,
      from_node_title: primer.display_title,
      to_node_title: whiteboard.display_title,
      label: "illustrated by",
      created_at: createdAt,
      updated_at: createdAt,
    },
    {
      id: "edge_runbook_meeting",
      from_node_id: runbook.id,
      to_node_id: meeting.id,
      from_node_title: runbook.display_title,
      to_node_title: meeting.display_title,
      label: "discussed in",
      created_at: createdAt,
      updated_at: createdAt,
    },
  ];

  const tags = new Map<string, KnowledgeTagSummary>(
    [tagResearch, tagOps, tagCustomer].map((tag) => [tag.id, tag]),
  );
  const nodes = new Map<string, KnowledgeNodeDetail>(
    [primer, runbook, whiteboard, meeting].map((node) => [node.id, node]),
  );
  const edgeMap = new Map<string, KnowledgeEdgeSummary>(
    edges.map((edge) => [edge.id, edge]),
  );

  const store: MockStore = {
    knowledge_base_id: knowledgeBaseId,
    title: "Local Developer's Knowledge Base",
    description: "Mock graph for standalone development.",
    created_at: createdAt,
    updated_at: createdAt,
    conversation_id: "mock-conversation-1",
    tags,
    nodes,
    edges: edgeMap,
    pending_confirmations: new Map(),
  };
  recomputeGraphRelationships(store);
  return store;
}

function sortedNodes(store: MockStore): KnowledgeNodeDetail[] {
  return [...store.nodes.values()].sort((left, right) =>
    right.updated_at.localeCompare(left.updated_at),
  );
}

function sortedTags(store: MockStore): KnowledgeTagSummary[] {
  return [...store.tags.values()].sort((left, right) =>
    left.name.localeCompare(right.name),
  );
}

function sortedEdges(store: MockStore): KnowledgeEdgeSummary[] {
  return [...store.edges.values()].sort((left, right) =>
    `${left.from_node_title}:${left.to_node_title}:${left.label}`.localeCompare(
      `${right.from_node_title}:${right.to_node_title}:${right.label}`,
    ),
  );
}

function defaultSelectionArgs(): SelectionArgs {
  return {
    selected_node_id: null,
    graph_selection_mode: "self",
    tag_ids: [],
    tag_match_mode: "all",
    media_types: [],
    include_web: false,
    rewrite_query: true,
    branch_factor: 3,
    depth: 2,
    max_results: 8,
  };
}

function normalizeSelectionArgs(
  input: Partial<SelectionArgs>,
  store: MockStore,
): SelectionArgs {
  const defaults = defaultSelectionArgs();
  const nodeIds = new Set(store.nodes.keys());
  const tagIds = new Set(store.tags.keys());
  return {
    selected_node_id:
      input.selected_node_id && nodeIds.has(input.selected_node_id)
        ? input.selected_node_id
        : null,
    graph_selection_mode: input.graph_selection_mode ?? defaults.graph_selection_mode,
    tag_ids: [...new Set(input.tag_ids ?? [])].filter((tagId) => tagIds.has(tagId)),
    tag_match_mode: input.tag_match_mode ?? defaults.tag_match_mode,
    media_types: [...new Set(input.media_types ?? [])].filter(Boolean),
    include_web: input.include_web ?? defaults.include_web,
    rewrite_query: input.rewrite_query ?? defaults.rewrite_query,
    branch_factor: input.branch_factor ?? defaults.branch_factor,
    depth: input.depth ?? defaults.depth,
    max_results: input.max_results ?? defaults.max_results,
  };
}

function computeTagVisibleNodeIds(
  store: MockStore,
  tagIds: string[],
  tagMatchMode: "all" | "any",
): string[] {
  const nodes = sortedNodes(store);
  if (tagIds.length === 0) {
    return nodes.map((node) => node.id);
  }
  const selected = new Set(tagIds);
  return nodes
    .filter((node) => {
      const nodeTagIds = new Set(node.tags.map((tag) => tag.id));
      if (tagMatchMode === "all") {
        return [...selected].every((tagId) => nodeTagIds.has(tagId));
      }
      return [...selected].some((tagId) => nodeTagIds.has(tagId));
    })
    .map((node) => node.id);
}

function computeGraphScopeNodeIds(
  store: MockStore,
  selectedNodeId: string | null,
  graphSelectionMode: "self" | "children" | "descendants",
): string[] {
  const allNodeIds = sortedNodes(store).map((node) => node.id);
  if (!selectedNodeId) {
    return allNodeIds;
  }
  if (graphSelectionMode === "self") {
    return [selectedNodeId];
  }
  const adjacency = new Map<string, string[]>();
  for (const edge of store.edges.values()) {
    const bucket = adjacency.get(edge.from_node_id) ?? [];
    bucket.push(edge.to_node_id);
    adjacency.set(edge.from_node_id, bucket);
  }
  if (graphSelectionMode === "children") {
    return [
      selectedNodeId,
      ...(adjacency.get(selectedNodeId) ?? []).sort((left, right) =>
        left.localeCompare(right),
      ),
    ];
  }
  const visited = new Set<string>([selectedNodeId]);
  const queue = [selectedNodeId];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) {
      continue;
    }
    for (const childId of adjacency.get(current) ?? []) {
      if (visited.has(childId)) {
        continue;
      }
      visited.add(childId);
      queue.push(childId);
    }
  }
  return allNodeIds.filter((nodeId) => visited.has(nodeId));
}

function buildContext(
  store: MockStore,
  rawArgs: Partial<SelectionArgs>,
): KnowledgeBaseContext {
  const args = normalizeSelectionArgs(rawArgs, store);
  const selectedTags = args.tag_ids
    .map((tagId) => store.tags.get(tagId))
    .filter((tag): tag is KnowledgeTagSummary => tag !== undefined);
  const visible_node_ids = computeTagVisibleNodeIds(
    store,
    args.tag_ids,
    args.tag_match_mode,
  );
  const graphScopeIds = computeGraphScopeNodeIds(
    store,
    args.selected_node_id,
    args.graph_selection_mode,
  );
  const visibleNodeSet = new Set(visible_node_ids);
  const scoped_node_ids = graphScopeIds.filter((nodeId) => visibleNodeSet.has(nodeId));
  return {
    selected_node_id: args.selected_node_id,
    graph_selection_mode: args.graph_selection_mode,
    tag_ids: args.tag_ids,
    selected_tag_names: selectedTags.map((tag) => tag.name),
    tag_match_mode: args.tag_match_mode,
    media_types: args.media_types,
    include_web: args.include_web,
    rewrite_query: args.rewrite_query,
    branch_factor: args.branch_factor,
    depth: args.depth,
    max_results: args.max_results,
    visible_node_ids,
    scoped_node_ids,
  };
}

function knowledgeBaseState(
  store: MockStore,
  context: KnowledgeBaseContext,
): KnowledgeBaseState {
  const nodes = sortedNodes(store).map((node) => detailToSummary(node));
  const tagCounts = new Map<string, number>();
  for (const node of store.nodes.values()) {
    for (const tag of node.tags) {
      tagCounts.set(tag.id, (tagCounts.get(tag.id) ?? 0) + 1);
    }
  }
  return {
    knowledge_base: {
      id: store.knowledge_base_id,
      title: store.title,
      description: store.description,
      created_at: store.created_at,
      updated_at: store.updated_at,
      node_count: store.nodes.size,
      tag_count: store.tags.size,
      edge_count: store.edges.size,
      vector_store_ready: store.nodes.size > 0,
    },
    tags: sortedTags(store).map((tag) => ({
      ...tag,
      node_count: tagCounts.get(tag.id) ?? 0,
    })),
    nodes,
    edges: sortedEdges(store),
    context,
  };
}

function deskState(
  store: MockStore,
  args: Partial<SelectionArgs>,
): KnowledgeBaseDeskState {
  const context = buildContext(store, args);
  return {
    access: {
      status: "active",
      message:
        "Access active. You can upload documents, edit the graph, and run scoped retrieval.",
      user: {
        clerk_user_id: "local-dev",
        display_name: "Local Developer",
        primary_email: "local-dev@example.com",
        active: true,
        role: "admin",
      },
    },
    knowledge_base: knowledgeBaseState(store, context),
    capabilities: {
      upload_url: "mock://uploads",
      upload_token_ttl_seconds: 900,
      confirmation_token_ttl_seconds: 900,
      supports_video_audio_extraction: true,
      accepted_hint:
        "Upload text-like files directly, plus images, audio, and video. Each upload becomes one graph node.",
    },
  };
}

function infoResult(
  store: MockStore,
  args: KnowledgeBaseInfoArguments,
): KnowledgeInfoResult {
  const state = deskState(store, args);
  const detailNodeId = args.detail_node_id ?? null;
  return {
    knowledge_base_state: state,
    node_detail: detailNodeId ? cloneNode(store.nodes.get(detailNodeId) ?? null) : null,
  };
}

function cloneNode(node: KnowledgeNodeDetail | null): KnowledgeNodeDetail | null {
  return node ? structuredClone(node) : null;
}

function searchHits(
  store: MockStore,
  query: string,
  context: KnowledgeBaseContext,
): SearchHit[] {
  const scopedNodeIds = new Set(context.scoped_node_ids);
  const mediaTypeSet = new Set(context.media_types);
  const candidates = sortedNodes(store).filter((node) => scopedNodeIds.has(node.id));
  return candidates
    .filter((node) =>
      mediaTypeSet.size === 0 ? true : mediaTypeSet.has(node.media_type),
    )
    .map((node) => {
      const text = nodeSearchText(node);
      const score = scoreText(query, text);
      return {
        node_id: node.id,
        node_title: node.display_title,
        original_filename: node.original_filename,
        derived_artifact_id: node.derived_artifacts[0]?.id ?? null,
        openai_file_id:
          node.derived_artifacts[0]?.openai_file_id ??
          node.openai_original_file_id ??
          node.id,
        original_openai_file_id: node.openai_original_file_id,
        media_type: node.media_type,
        source_kind: node.source_kind,
        score,
        text,
        tags: node.tags.map((tag) => tag.name),
        attributes: {
          node_id: node.id,
          node_title: node.display_title,
          media_type: node.media_type,
        },
      };
    })
    .filter((hit) => hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, context.max_results);
}

function branchQueries(query: string, context: KnowledgeBaseContext): string[] {
  const tagQueries = context.selected_tag_names.map((tagName) => `${query} ${tagName}`);
  return [
    query,
    `${query} summary`,
    `${query} examples`,
    ...tagQueries,
  ]
    .filter(Boolean)
    .slice(0, Math.max(1, context.branch_factor));
}

function queryResult(
  store: MockStore,
  args: KnowledgeBaseQueryArguments,
): KnowledgeQueryResult {
  const state = deskState(store, args);
  const context = state.knowledge_base?.context;
  const query = args.query?.trim();
  if (!context || !query) {
    return {
      kind: "knowledge_base",
      knowledge_base_state: state,
      file_search_result: null,
      branch_search_result: null,
      chat_result: null,
    };
  }

  if (args.mode === "file_search") {
    return {
      kind: "file_search",
      knowledge_base_state: state,
      file_search_result: {
        knowledge_base_id: store.knowledge_base_id,
        query,
        context,
        hits: searchHits(store, query, context),
        total_hits: searchHits(store, query, context).length,
      },
      branch_search_result: null,
      chat_result: null,
    };
  }

  if (args.mode === "branch_search") {
    const seeds = branchQueries(query, context).slice(0, context.branch_factor);
    const nodes: BranchSearchNode[] = seeds.map((seed, index) => ({
      id: `branch_${index + 1}`,
      parent_id: index === 0 ? null : "branch_1",
      depth: index === 0 ? 0 : 1,
      query: seed,
      rationale:
        index === 0 ? "Initial query." : "Diversified branch based on the current scope.",
      hits: searchHits(store, seed, context),
      children: [],
    }));
    const merged = new Map<string, SearchHit>();
    for (const node of nodes) {
      for (const hit of node.hits) {
        if (!merged.has(hit.node_id)) {
          merged.set(hit.node_id, hit);
        }
      }
    }
    const branch_search_result: KnowledgeBranchSearchResult = {
      knowledge_base_id: store.knowledge_base_id,
      seed_query: query,
      context,
      nodes,
      merged_hits: [...merged.values()].slice(0, context.max_results),
    };
    return {
      kind: "branch_search",
      knowledge_base_state: state,
      file_search_result: null,
      branch_search_result,
      chat_result: null,
    };
  }

  const hits = searchHits(store, query, context).slice(0, 4);
  const search_calls: FileSearchCallSummary[] = [
    {
      id: "mock-search-1",
      status: "completed",
      queries: [query],
      results: hits,
    },
  ];
  const citations: KnowledgeAnswerCitation[] = hits.map((hit) => ({
    source: "knowledge_base",
    label: hit.node_title,
    node_id: hit.node_id,
    node_title: hit.node_title,
    original_filename: hit.original_filename,
    url: null,
    quote: hit.text.slice(0, 220),
  }));
  const web_search_calls: WebSearchCallSummary[] = context.include_web
    ? [
        {
          id: "mock-web-1",
          status: "skipped",
          query,
          sources: [],
        },
      ]
    : [];
  const answer =
    hits.length === 0
      ? "No documents matched the current graph and tag scope."
      : `Top matching nodes: ${hits.map((hit) => hit.node_title).join(", ")}. The current scope narrows retrieval before answering, so these results reflect both the selected graph region and any active tag filters.`;
  const chat_result: KnowledgeChatResult = {
    knowledge_base_id: store.knowledge_base_id,
    question: query,
    answer,
    model: "mock-gpt",
    include_web: context.include_web,
    conversation_id: store.conversation_id,
    context,
    search_calls,
    web_search_calls,
    citations,
  };
  return {
    kind: "qa",
    knowledge_base_state: state,
    file_search_result: null,
    branch_search_result: null,
    chat_result,
  };
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 72) || "tag";
}

function uniqueNodeTitle(store: MockStore, baseTitle: string, excludeId?: string): string {
  const normalizedBase = baseTitle.trim() || "Untitled document";
  let candidate = normalizedBase;
  let index = 2;
  while (true) {
    const exists = [...store.nodes.values()].find(
      (node) =>
        node.id !== excludeId &&
        node.display_title.toLowerCase() === candidate.toLowerCase(),
    );
    if (!exists) {
      return candidate;
    }
    candidate = `${normalizedBase} (${index})`;
    index += 1;
  }
}

function ensureTag(
  store: MockStore,
  name: string,
  color: string | null = null,
): KnowledgeTagSummary {
  const existing = [...store.tags.values()].find(
    (tag) => tag.name.toLowerCase() === name.toLowerCase(),
  );
  if (existing) {
    return existing;
  }
  const baseSlug = slugify(name);
  let candidateSlug = baseSlug;
  let suffix = 2;
  while ([...store.tags.values()].some((tag) => tag.slug === candidateSlug)) {
    candidateSlug = `${baseSlug}-${suffix}`;
    suffix += 1;
  }
  const tag: KnowledgeTagSummary = {
    id: uniqueId("tag"),
    name,
    slug: candidateSlug,
    color,
    node_count: 0,
  };
  store.tags.set(tag.id, tag);
  store.updated_at = nowIso();
  return tag;
}

function recomputeGraphRelationships(store: MockStore): void {
  for (const node of store.nodes.values()) {
    node.outgoing_edges = [];
    node.incoming_edges = [];
  }
  for (const edge of store.edges.values()) {
    const fromNode = store.nodes.get(edge.from_node_id);
    const toNode = store.nodes.get(edge.to_node_id);
    if (!fromNode || !toNode) {
      continue;
    }
    edge.from_node_title = fromNode.display_title;
    edge.to_node_title = toNode.display_title;
    fromNode.outgoing_edges.push(edge);
    toNode.incoming_edges.push(edge);
  }
  for (const node of store.nodes.values()) {
    node.outgoing_edges.sort((left, right) =>
      `${left.to_node_title}:${left.label}`.localeCompare(
        `${right.to_node_title}:${right.label}`,
      ),
    );
    node.incoming_edges.sort((left, right) =>
      `${left.from_node_title}:${left.label}`.localeCompare(
        `${right.from_node_title}:${right.label}`,
      ),
    );
    node.outgoing_edge_count = node.outgoing_edges.length;
    node.incoming_edge_count = node.incoming_edges.length;
  }
}

function setNodeTags(
  store: MockStore,
  nodeId: string,
  tagIds: string[],
): KnowledgeNodeDetail {
  const node = store.nodes.get(nodeId);
  if (!node) {
    throw new Error("Node not found.");
  }
  node.tags = tagIds
    .map((tagId) => store.tags.get(tagId))
    .filter((tag): tag is KnowledgeTagSummary => tag !== undefined)
    .sort((left, right) => left.name.localeCompare(right.name));
  node.updated_at = nowIso();
  store.updated_at = node.updated_at;
  return node;
}

function renameNode(
  store: MockStore,
  nodeId: string,
  title: string,
): KnowledgeNodeDetail {
  const node = store.nodes.get(nodeId);
  if (!node) {
    throw new Error("Node not found.");
  }
  node.display_title = uniqueNodeTitle(store, title, node.id);
  node.updated_at = nowIso();
  store.updated_at = node.updated_at;
  recomputeGraphRelationships(store);
  return node;
}

function upsertEdge(
  store: MockStore,
  fromNodeId: string,
  toNodeId: string,
  label: string,
): KnowledgeEdgeSummary {
  if (fromNodeId === toNodeId) {
    throw new Error("Self-referential edges are not supported.");
  }
  const fromNode = store.nodes.get(fromNodeId);
  const toNode = store.nodes.get(toNodeId);
  if (!fromNode || !toNode) {
    throw new Error("Edge nodes were not found.");
  }
  const existing = [...store.edges.values()].find(
    (edge) => edge.from_node_id === fromNodeId && edge.to_node_id === toNodeId,
  );
  const updatedAt = nowIso();
  const edge =
    existing ??
    {
      id: uniqueId("edge"),
      from_node_id: fromNodeId,
      to_node_id: toNodeId,
      from_node_title: fromNode.display_title,
      to_node_title: toNode.display_title,
      label,
      created_at: updatedAt,
      updated_at: updatedAt,
    };
  edge.label = label;
  edge.from_node_title = fromNode.display_title;
  edge.to_node_title = toNode.display_title;
  edge.updated_at = updatedAt;
  store.edges.set(edge.id, edge);
  store.updated_at = updatedAt;
  recomputeGraphRelationships(store);
  return edge;
}

function deleteEdge(store: MockStore, edgeId: string): string {
  if (!store.edges.delete(edgeId)) {
    throw new Error("Edge not found.");
  }
  store.updated_at = nowIso();
  recomputeGraphRelationships(store);
  return edgeId;
}

function deleteNode(store: MockStore, nodeId: string): string {
  if (!store.nodes.has(nodeId)) {
    throw new Error("Node not found.");
  }
  store.nodes.delete(nodeId);
  for (const edge of [...store.edges.values()]) {
    if (edge.from_node_id === nodeId || edge.to_node_id === nodeId) {
      store.edges.delete(edge.id);
    }
  }
  store.updated_at = nowIso();
  recomputeGraphRelationships(store);
  return nodeId;
}

function resolveNodeByReference(
  store: MockStore,
  nodeTitle: string | null | undefined,
  selectedNodeId?: string,
): KnowledgeNodeDetail {
  if (!nodeTitle) {
    if (!selectedNodeId) {
      throw new Error("No node is selected.");
    }
    const selected = store.nodes.get(selectedNodeId);
    if (!selected) {
      throw new Error("Selected node was not found.");
    }
    return selected;
  }
  const lowered = nodeTitle.trim().toLowerCase();
  const node = [...store.nodes.values()].find(
    (candidate) =>
      candidate.display_title.toLowerCase() === lowered ||
      candidate.original_filename.toLowerCase() === lowered,
  );
  if (!node) {
    throw new Error(`Could not find a node named '${nodeTitle}'.`);
  }
  return node;
}

function pendingDelete(
  store: MockStore,
  node: KnowledgeNodeDetail,
): PendingCommandResult {
  const token = uniqueId("confirm");
  store.pending_confirmations.set(token, {
    action: "delete_node",
    node_id: node.id,
  });
  return {
    token,
    prompt: `Delete '${node.display_title}' and remove its edges from the graph?`,
    summary: `Delete node '${node.display_title}'`,
    expires_at: Math.floor(Date.now() / 1000) + 900,
  };
}

function stripQuotes(value: string): string {
  const stripped = value.trim();
  if (
    stripped.length >= 2 &&
    stripped[0] === stripped[stripped.length - 1] &&
    [`"`, `'`].includes(stripped[0] ?? "")
  ) {
    return stripped.slice(1, -1).trim();
  }
  return stripped;
}

function splitNames(value: string): string[] {
  return value
    .replace(/\sand\s/gi, ",")
    .split(",")
    .map((part) => stripQuotes(part))
    .filter(Boolean);
}

function runCommand(
  store: MockStore,
  args: RunKnowledgeBaseCommandArguments,
): KnowledgeBaseCommandResult {
  const rawCommand = args.command.trim();
  const baseState = deskState(store, args);
  const rejected = (message: string): KnowledgeBaseCommandResult => ({
    status: "rejected",
    message,
    action: null,
    parser: "fallback",
    raw_command: rawCommand,
    knowledge_base_state: baseState,
    pending_confirmation: null,
    node: null,
    edge: null,
    tag: null,
  });

  if (!rawCommand) {
    return rejected("Enter a command to modify the knowledge base.");
  }

  try {
    const renameSelectedMatch = rawCommand.match(
      /^(?:rename|change)\s+(?:the\s+)?selected node(?:'s)?(?:\s+name)?\s+to\s+(.+)$/i,
    );
    if (renameSelectedMatch) {
      const node = resolveNodeByReference(store, null, args.selected_node_id);
      const updated = renameNode(store, node.id, stripQuotes(renameSelectedMatch[1] ?? ""));
      return {
        status: "executed",
        message: `Renamed node to '${updated.display_title}'.`,
        action: "rename_node",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: updated.id,
        }),
        pending_confirmation: null,
        node: detailToSummary(updated),
        edge: null,
        tag: null,
      };
    }

    const renameNamedMatch = rawCommand.match(
      /^(?:rename|change)\s+node\s+(.+?)\s+(?:to|name to)\s+(.+)$/i,
    );
    if (renameNamedMatch) {
      const node = resolveNodeByReference(
        store,
        stripQuotes(renameNamedMatch[1] ?? ""),
        args.selected_node_id,
      );
      const updated = renameNode(store, node.id, stripQuotes(renameNamedMatch[2] ?? ""));
      return {
        status: "executed",
        message: `Renamed node to '${updated.display_title}'.`,
        action: "rename_node",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: updated.id,
        }),
        pending_confirmation: null,
        node: detailToSummary(updated),
        edge: null,
        tag: null,
      };
    }

    const createTagMatch = rawCommand.match(/^(?:create|add)\s+tag\s+(.+)$/i);
    if (createTagMatch) {
      const tag = ensureTag(store, stripQuotes(createTagMatch[1] ?? ""));
      return {
        status: "executed",
        message: `Created tag '${tag.name}'.`,
        action: "create_tag",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, args),
        pending_confirmation: null,
        node: null,
        edge: null,
        tag,
      };
    }

    const tagSelectedMatch = rawCommand.match(
      /^(?:set|add)\s+tags?\s+(.+?)\s+to\s+(?:the\s+)?selected node$/i,
    );
    if (tagSelectedMatch) {
      const node = resolveNodeByReference(store, null, args.selected_node_id);
      const tags = splitNames(tagSelectedMatch[1] ?? "").map((name) =>
        ensureTag(store, name),
      );
      const updated = setNodeTags(
        store,
        node.id,
        tags.map((tag) => tag.id),
      );
      return {
        status: "executed",
        message: `Updated tags on '${updated.display_title}'.`,
        action: "set_node_tags",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: updated.id,
        }),
        pending_confirmation: null,
        node: detailToSummary(updated),
        edge: null,
        tag: null,
      };
    }

    const tagNamedMatch = rawCommand.match(
      /^(?:set|add)\s+tags?\s+(.+?)\s+to\s+node\s+(.+)$/i,
    );
    if (tagNamedMatch) {
      const node = resolveNodeByReference(
        store,
        stripQuotes(tagNamedMatch[2] ?? ""),
        args.selected_node_id,
      );
      const tags = splitNames(tagNamedMatch[1] ?? "").map((name) =>
        ensureTag(store, name),
      );
      const updated = setNodeTags(
        store,
        node.id,
        tags.map((tag) => tag.id),
      );
      return {
        status: "executed",
        message: `Updated tags on '${updated.display_title}'.`,
        action: "set_node_tags",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: updated.id,
        }),
        pending_confirmation: null,
        node: detailToSummary(updated),
        edge: null,
        tag: null,
      };
    }

    const addEdgeMatch = rawCommand.match(
      /^add (?:an )?edge from (.+?) to (.+?)(?:\s+labeled\s+(.+))?$/i,
    );
    if (addEdgeMatch) {
      const fromNode = resolveNodeByReference(
        store,
        stripQuotes(addEdgeMatch[1] ?? ""),
        args.selected_node_id,
      );
      const toNode = resolveNodeByReference(
        store,
        stripQuotes(addEdgeMatch[2] ?? ""),
        undefined,
      );
      const edge = upsertEdge(
        store,
        fromNode.id,
        toNode.id,
        stripQuotes(addEdgeMatch[3] ?? "related"),
      );
      return {
        status: "executed",
        message: `Connected '${edge.from_node_title}' to '${edge.to_node_title}'.`,
        action: "upsert_edge",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: fromNode.id,
          graph_selection_mode: "children",
        }),
        pending_confirmation: null,
        node: detailToSummary(fromNode),
        edge,
        tag: null,
      };
    }

    const addEdgeFromSelectedMatch = rawCommand.match(
      /^add (?:an )?edge from (?:the\s+)?selected node to (.+?)(?:\s+labeled\s+(.+))?$/i,
    );
    if (addEdgeFromSelectedMatch) {
      const fromNode = resolveNodeByReference(store, null, args.selected_node_id);
      const toNode = resolveNodeByReference(
        store,
        stripQuotes(addEdgeFromSelectedMatch[1] ?? ""),
        undefined,
      );
      const edge = upsertEdge(
        store,
        fromNode.id,
        toNode.id,
        stripQuotes(addEdgeFromSelectedMatch[2] ?? "related"),
      );
      return {
        status: "executed",
        message: `Connected '${edge.from_node_title}' to '${edge.to_node_title}'.`,
        action: "upsert_edge",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: deskState(store, {
          ...args,
          selected_node_id: fromNode.id,
          graph_selection_mode: "children",
        }),
        pending_confirmation: null,
        node: detailToSummary(fromNode),
        edge,
        tag: null,
      };
    }

    if (/^delete (?:the\s+)?selected node$/i.test(rawCommand)) {
      const node = resolveNodeByReference(store, null, args.selected_node_id);
      const pending = pendingDelete(store, node);
      return {
        status: "pending_confirmation",
        message: `Confirm deletion of '${node.display_title}' to continue.`,
        action: "delete_node",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: baseState,
        pending_confirmation: pending,
        node: detailToSummary(node),
        edge: null,
        tag: null,
      };
    }

    const deleteNamedMatch = rawCommand.match(/^delete node (.+)$/i);
    if (deleteNamedMatch) {
      const node = resolveNodeByReference(
        store,
        stripQuotes(deleteNamedMatch[1] ?? ""),
        args.selected_node_id,
      );
      const pending = pendingDelete(store, node);
      return {
        status: "pending_confirmation",
        message: `Confirm deletion of '${node.display_title}' to continue.`,
        action: "delete_node",
        parser: "fallback",
        raw_command: rawCommand,
        knowledge_base_state: baseState,
        pending_confirmation: pending,
        node: detailToSummary(node),
        edge: null,
        tag: null,
      };
    }
  } catch (error) {
    return rejected(error instanceof Error ? error.message : String(error));
  }

  return rejected(
    "I couldn't map that command yet. Try renaming a node, adding a labeled edge, creating a tag, or deleting a node.",
  );
}

function confirmCommand(
  store: MockStore,
  args: ConfirmKnowledgeBaseCommandArguments,
): KnowledgeBaseCommandResult {
  const pending = store.pending_confirmations.get(args.token);
  if (!pending) {
    return {
      status: "rejected",
      message: "This confirmation token is invalid or expired.",
      action: null,
      parser: "manual",
      raw_command: "confirm",
      knowledge_base_state: deskState(store, args),
      pending_confirmation: null,
      node: null,
      edge: null,
      tag: null,
    };
  }
  store.pending_confirmations.delete(args.token);
  const deletedNodeId = deleteNode(store, pending.node_id);
  return {
    status: "executed",
    message: "Deleted the node and its connected edges.",
    action: "delete_node",
    parser: "manual",
    raw_command: "confirm",
    knowledge_base_state: deskState(store, {
      ...args,
      selected_node_id:
        args.selected_node_id === deletedNodeId ? undefined : args.selected_node_id,
    }),
    pending_confirmation: null,
    node: null,
    edge: null,
    tag: null,
  };
}

function updateKnowledgeBase(
  store: MockStore,
  args: UpdateKnowledgeBaseArguments,
): UpdateKnowledgeBaseResult {
  if (args.action === "prepare_upload") {
    return {
      action: args.action,
      knowledge_base_state: null,
      node: null,
      edge: null,
      tag: null,
      deleted_node_id: null,
      deleted_edge_id: null,
      upload_session: formatUploadSession(),
    };
  }

  if (args.action === "rename_node") {
    if (!args.node_id || !args.title) {
      throw new Error("rename_node requires node_id and title.");
    }
    const node = renameNode(store, args.node_id, args.title);
    return {
      action: args.action,
      knowledge_base_state: deskState(store, {
        selected_node_id: node.id,
      }),
      node: detailToSummary(node),
      edge: null,
      tag: null,
      deleted_node_id: null,
      deleted_edge_id: null,
      upload_session: null,
    };
  }

  if (args.action === "create_tag") {
    if (!args.name) {
      throw new Error("create_tag requires name.");
    }
    const tag = ensureTag(store, args.name, args.color ?? null);
    return {
      action: args.action,
      knowledge_base_state: deskState(store, {}),
      node: null,
      edge: null,
      tag,
      deleted_node_id: null,
      deleted_edge_id: null,
      upload_session: null,
    };
  }

  if (args.action === "set_node_tags") {
    if (!args.node_id) {
      throw new Error("set_node_tags requires node_id.");
    }
    const node = setNodeTags(store, args.node_id, args.tag_ids ?? []);
    return {
      action: args.action,
      knowledge_base_state: deskState(store, {
        selected_node_id: node.id,
      }),
      node: detailToSummary(node),
      edge: null,
      tag: null,
      deleted_node_id: null,
      deleted_edge_id: null,
      upload_session: null,
    };
  }

  if (args.action === "upsert_edge") {
    if (!args.from_node_id || !args.to_node_id || !args.label) {
      throw new Error("upsert_edge requires from_node_id, to_node_id, and label.");
    }
    const edge = upsertEdge(store, args.from_node_id, args.to_node_id, args.label);
    return {
      action: args.action,
      knowledge_base_state: deskState(store, {
        selected_node_id: edge.from_node_id,
        graph_selection_mode: "children",
      }),
      node: null,
      edge,
      tag: null,
      deleted_node_id: null,
      deleted_edge_id: null,
      upload_session: null,
    };
  }

  if (args.action === "delete_edge") {
    if (!args.edge_id) {
      throw new Error("delete_edge requires edge_id.");
    }
    const deletedEdgeId = deleteEdge(store, args.edge_id);
    return {
      action: args.action,
      knowledge_base_state: deskState(store, {}),
      node: null,
      edge: null,
      tag: null,
      deleted_node_id: null,
      deleted_edge_id: deletedEdgeId,
      upload_session: null,
    };
  }

  if (!args.node_id) {
    throw new Error("delete_node requires node_id.");
  }
  const deletedNodeId = deleteNode(store, args.node_id);
  return {
    action: args.action,
    knowledge_base_state: deskState(store, {}),
    node: null,
    edge: null,
    tag: null,
    deleted_node_id: deletedNodeId,
    deleted_edge_id: null,
    upload_session: null,
  };
}

async function uploadNode(
  store: MockStore,
  input: UploadNodeInput,
): Promise<UploadFinalizeResult> {
  const createdAt = nowIso();
  const fileText = input.file.type.startsWith("text/")
    ? await input.file.text()
    : `Uploaded ${input.file.name} (${input.file.type || "application/octet-stream"}).`;
  const nodeId = uniqueId("node");
  const title = uniqueNodeTitle(store, input.file.name.replace(/\.[^.]+$/, ""));
  const derivedKind = input.file.type.startsWith("image/")
    ? "image_description"
    : input.file.type.startsWith("audio/")
      ? "audio_transcript"
      : input.file.type.startsWith("video/")
        ? "video_transcript"
        : "document_text";
  const tags = (input.tag_ids ?? [])
    .map((tagId) => store.tags.get(tagId))
    .filter((tag): tag is KnowledgeTagSummary => tag !== undefined)
    .sort((left, right) => left.name.localeCompare(right.name));
  const objectUrl = URL.createObjectURL(input.file);
  const node: KnowledgeNodeDetail = {
    id: nodeId,
    display_title: title,
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
    derived_kinds: [derivedKind],
    openai_original_file_id: uniqueId("file"),
    download_url: objectUrl,
    outgoing_edge_count: 0,
    incoming_edge_count: 0,
    original_mime_type: input.file.type || null,
    derived_artifacts: [
      {
        id: uniqueId("artifact"),
        kind: derivedKind,
        openai_file_id: uniqueId("file"),
        text_content: fileText || `Uploaded ${input.file.name}.`,
        structured_payload: null,
        created_at: createdAt,
        updated_at: createdAt,
      },
    ],
    outgoing_edges: [],
    incoming_edges: [],
  };
  store.nodes.set(node.id, node);
  store.updated_at = createdAt;
  recomputeGraphRelationships(store);
  return { node: detailToSummary(node) };
}

export function createMockBridge(
  initialHostContext: McpUiHostContext = HOST_CONTEXT,
): KnowledgeBaseBridge {
  const store = seededStore();
  const initialQueryResult = queryResult(store, {});

  return {
    mode: "mock",
    hostContext: initialHostContext,
    initial_state: initialQueryResult.knowledge_base_state,
    initial_query_result: initialQueryResult,
    async get_knowledge_base_info(args) {
      return infoResult(store, args);
    },
    async query_knowledge_base(args) {
      return queryResult(store, args);
    },
    async update_knowledge_base(args) {
      return updateKnowledgeBase(store, args);
    },
    async run_knowledge_base_command(args) {
      return runCommand(store, args);
    },
    async confirm_knowledge_base_command(args) {
      return confirmCommand(store, args);
    },
    async upload_node(input) {
      return uploadNode(store, input);
    },
  };
}

export function createMockHostControls(): KnowledgeBaseHostControls {
  return {
    mode: "mock",
    supportedDisplayModes: ["inline", "fullscreen"],
    supportsModelContext: true,
    async update_model_context(_markdown) {},
    async request_display_mode(mode) {
      return mode;
    },
  };
}
