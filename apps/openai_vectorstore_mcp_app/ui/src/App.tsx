import {
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
  type McpUiDisplayMode,
  type McpUiHostContext,
} from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Code,
  Divider,
  FileInput,
  Group,
  Loader,
  MantineProvider,
  MultiSelect,
  NumberInput,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Tabs,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import mermaid from "mermaid";
import {
  startTransition,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from "react";

import {
  createHostBridge,
  createHostControls,
  type KnowledgeBaseBridge,
  type KnowledgeBaseHostControls,
} from "./bridge";
import { createMockBridge, createMockHostControls } from "./mockBridge";
import { appCssVariablesResolver, appTheme } from "./theme";
import type {
  KnowledgeBaseCommandResult,
  KnowledgeBaseDeskState,
  KnowledgeBaseState,
  KnowledgeBranchSearchResult,
  KnowledgeChatResult,
  KnowledgeEdgeSummary,
  KnowledgeFileSearchResult,
  KnowledgeNodeDetail,
  KnowledgeNodeSummary,
  KnowledgeQueryMode,
  KnowledgeQueryResult,
  PendingCommandResult,
  SearchHit,
} from "./types";
import { isKnowledgeQueryResult } from "./types";

mermaid.initialize({
  startOnLoad: false,
  securityLevel: "loose",
  theme: "base",
  flowchart: {
    htmlLabels: true,
    curve: "basis",
  },
  themeVariables: {
    primaryColor: "#e6f4ea",
    primaryTextColor: "#18303d",
    primaryBorderColor: "#2d4a52",
    lineColor: "#496670",
    secondaryColor: "#fff6dd",
    tertiaryColor: "#f7f3e8",
  },
});

declare global {
  interface Window {
    kbNodeClick?: (graphId: string) => void;
  }
}

const IMPLEMENTATION = {
  name: "Knowledge Base Desk",
  version: "2.0.0",
};

type MainTab = "search" | "branch" | "chat";

type SelectionState = {
  selectedNodeId: string | null;
  graphSelectionMode: "self" | "children" | "descendants";
  selectedTagIds: string[];
  tagMatchMode: "all" | "any";
  selectedMediaTypes: string[];
  includeWeb: boolean;
  rewriteQuery: boolean;
  branchFactor: number;
  depth: number;
  maxResults: number;
};

function isStandaloneMode(): boolean {
  const params = new URLSearchParams(window.location.search);
  return window.parent === window || params.get("mock") === "1";
}

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}

function excerpt(value: string, limit = 260): string {
  return value.length > limit ? `${value.slice(0, limit - 3)}...` : value;
}

function escapeMermaidLabel(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function graphNodeId(nodeId: string): string {
  return `node_${nodeId}`;
}

function nextSelectionMode(
  current: SelectionState["graphSelectionMode"],
): SelectionState["graphSelectionMode"] {
  if (current === "self") {
    return "children";
  }
  if (current === "children") {
    return "descendants";
  }
  return "self";
}

function statusColor(status: KnowledgeNodeSummary["status"]): string {
  switch (status) {
    case "ready":
      return "teal";
    case "processing":
      return "yellow";
    case "failed":
      return "red";
  }
}

function selectionSummary(
  knowledgeBase: KnowledgeBaseState | null,
  selectedNode: KnowledgeNodeSummary | null,
): string {
  if (!knowledgeBase) {
    return "No knowledge base loaded yet.";
  }
  if (!selectedNode) {
    return `Browsing the full graph. ${knowledgeBase.context.scoped_node_ids.length} visible node(s) are in retrieval scope.`;
  }
  const mode = knowledgeBase.context.graph_selection_mode;
  if (mode === "self") {
    return `Scoped to '${selectedNode.display_title}' only.`;
  }
  if (mode === "children") {
    return `Scoped to '${selectedNode.display_title}' and its direct outgoing neighbors.`;
  }
  return `Scoped to '${selectedNode.display_title}' and all reachable descendants.`;
}

function resolveColorScheme(hostContext?: McpUiHostContext): "light" | "dark" {
  return hostContext?.theme === "dark" ? "dark" : "light";
}

function extractSafeAreaStyle(hostContext?: McpUiHostContext): CSSProperties {
  return {
    paddingTop: hostContext?.safeAreaInsets?.top,
    paddingRight: hostContext?.safeAreaInsets?.right,
    paddingBottom: hostContext?.safeAreaInsets?.bottom,
    paddingLeft: hostContext?.safeAreaInsets?.left,
  };
}

function buildSelectionState(
  state: KnowledgeBaseDeskState,
): SelectionState {
  const context = state.knowledge_base?.context;
  return {
    selectedNodeId: context?.selected_node_id ?? null,
    graphSelectionMode: context?.graph_selection_mode ?? "self",
    selectedTagIds: context?.tag_ids ?? [],
    tagMatchMode: context?.tag_match_mode ?? "all",
    selectedMediaTypes: context?.media_types ?? [],
    includeWeb: context?.include_web ?? false,
    rewriteQuery: context?.rewrite_query ?? true,
    branchFactor: context?.branch_factor ?? 3,
    depth: context?.depth ?? 2,
    maxResults: context?.max_results ?? 8,
  };
}

function normalizeInteger(value: string | number, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.round(value);
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function tagsVisibleIds(
  knowledgeBase: KnowledgeBaseState | null,
  tagIds: string[],
  tagMatchMode: "all" | "any",
): Set<string> {
  if (!knowledgeBase || tagIds.length === 0) {
    return new Set(knowledgeBase?.nodes.map((node) => node.id) ?? []);
  }
  const selected = new Set(tagIds);
  return new Set(
    knowledgeBase.nodes
      .filter((node) => {
        const nodeTagIds = new Set(node.tags.map((tag) => tag.id));
        if (tagMatchMode === "all") {
          return [...selected].every((tagId) => nodeTagIds.has(tagId));
        }
        return [...selected].some((tagId) => nodeTagIds.has(tagId));
      })
      .map((node) => node.id),
  );
}

function buildModelContextMarkdown(input: {
  deskState: KnowledgeBaseDeskState;
  selectionState: SelectionState;
  fileSearchResult: KnowledgeFileSearchResult | null;
  branchSearchResult: KnowledgeBranchSearchResult | null;
  chatResult: KnowledgeChatResult | null;
  commandResult: KnowledgeBaseCommandResult | null;
}): string {
  const knowledgeBase = input.deskState.knowledge_base;
  const selectedNode =
    knowledgeBase?.nodes.find(
      (node) => node.id === input.selectionState.selectedNodeId,
    ) ?? null;
  const recentHits =
    input.fileSearchResult?.hits.map((hit) => hit.node_title) ??
    input.branchSearchResult?.merged_hits.map((hit) => hit.node_title) ??
    [];
  const citations = input.chatResult?.citations.map((citation) => citation.label) ?? [];
  return [
    "---",
    `tool: "query_knowledge_base"`,
    `knowledge-base-id: "${knowledgeBase?.knowledge_base.id ?? "none"}"`,
    `selected-node-id: "${selectedNode?.id ?? "none"}"`,
    `selected-node-title: "${selectedNode?.display_title ?? "none"}"`,
    `graph-selection-mode: "${input.selectionState.graphSelectionMode}"`,
    `selected-tag-count: ${input.selectionState.selectedTagIds.length}`,
    `selected-media-type-count: ${input.selectionState.selectedMediaTypes.length}`,
    `include-web: ${input.selectionState.includeWeb}`,
    `rewrite-query: ${input.selectionState.rewriteQuery}`,
    `branch-factor: ${input.selectionState.branchFactor}`,
    `depth: ${input.selectionState.depth}`,
    `max-results: ${input.selectionState.maxResults}`,
    `last-command: "${input.commandResult?.action ?? "none"}"`,
    "---",
    "",
    `Knowledge base: ${knowledgeBase?.knowledge_base.title ?? "none"}`,
    `Selected tags: ${
      knowledgeBase?.tags
        .filter((tag) => input.selectionState.selectedTagIds.includes(tag.id))
        .map((tag) => tag.name)
        .join(", ") || "none"
    }`,
    `Visible scoped nodes: ${knowledgeBase?.context.scoped_node_ids.length ?? 0}`,
    `Recent retrieval hits: ${recentHits.join(", ") || "none"}`,
    `Recent chat citations: ${citations.join(", ") || "none"}`,
  ].join("\n");
}

function buildMermaidGraph(args: {
  nodes: KnowledgeNodeSummary[];
  edges: KnowledgeEdgeSummary[];
  selectedNodeId: string | null;
  scopedNodeIds: string[];
}): string {
  const scopedSet = new Set(args.scopedNodeIds);
  const normalIds: string[] = [];
  const scopedIds: string[] = [];

  const lines = [
    "flowchart LR",
    "classDef normal fill:#f7f3e8,stroke:#496670,color:#18303d,stroke-width:1.5px;",
    "classDef scoped fill:#e7f3ea,stroke:#217a5e,color:#18303d,stroke-width:2px;",
    "classDef selected fill:#0f766e,stroke:#0b4f4a,color:#ffffff,stroke-width:3px;",
  ];

  for (const node of args.nodes) {
    const subtitle = escapeMermaidLabel(node.original_filename);
    lines.push(
      `${graphNodeId(node.id)}["${escapeMermaidLabel(
        node.display_title,
      )}<br/><small>${subtitle}</small>"]`,
    );
    lines.push(`click ${graphNodeId(node.id)} kbNodeClick "Cycle node scope"`);
    if (node.id === args.selectedNodeId) {
      continue;
    }
    if (scopedSet.has(node.id)) {
      scopedIds.push(graphNodeId(node.id));
    } else {
      normalIds.push(graphNodeId(node.id));
    }
  }

  for (const edge of args.edges) {
    lines.push(
      `${graphNodeId(edge.from_node_id)} -->|${escapeMermaidLabel(
        edge.label,
      )}| ${graphNodeId(edge.to_node_id)}`,
    );
  }

  if (normalIds.length > 0) {
    lines.push(`class ${normalIds.join(",")} normal`);
  }
  if (scopedIds.length > 0) {
    lines.push(`class ${scopedIds.join(",")} scoped`);
  }
  if (args.selectedNodeId) {
    lines.push(`class ${graphNodeId(args.selectedNodeId)} selected`);
  }

  return lines.join("\n");
}

function GraphCard(props: {
  nodes: KnowledgeNodeSummary[];
  edges: KnowledgeEdgeSummary[];
  selectedNodeId: string | null;
  scopedNodeIds: string[];
  onNodeClick: (nodeId: string) => void;
}): ReactElement {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const renderId = useId().replace(/:/g, "_");
  const [renderError, setRenderError] = useState<string | null>(null);
  const graphSource = useMemo(
    () =>
      buildMermaidGraph({
        nodes: props.nodes,
        edges: props.edges,
        selectedNodeId: props.selectedNodeId,
        scopedNodeIds: props.scopedNodeIds,
      }),
    [props.edges, props.nodes, props.scopedNodeIds, props.selectedNodeId],
  );

  useEffect(() => {
    window.kbNodeClick = (mermaidNodeId: string) => {
      const nodeId = mermaidNodeId.replace(/^node_/, "");
      props.onNodeClick(nodeId);
    };
    return () => {
      window.kbNodeClick = undefined;
    };
  }, [props]);

  useEffect(() => {
    let cancelled = false;

    if (!containerRef.current) {
      return;
    }
    if (props.nodes.length === 0) {
      containerRef.current.innerHTML = "";
      setRenderError(null);
      return;
    }

    const renderGraph = async (): Promise<void> => {
      try {
        const { svg, bindFunctions } = await mermaid.render(
          `graph_${renderId}`,
          graphSource,
        );
        if (cancelled || !containerRef.current) {
          return;
        }
        containerRef.current.innerHTML = svg;
        bindFunctions?.(containerRef.current);
        setRenderError(null);
      } catch (error) {
        if (!cancelled) {
          setRenderError(
            error instanceof Error ? error.message : "Mermaid render failed.",
          );
        }
      }
    };

    void renderGraph();
    return () => {
      cancelled = true;
    };
  }, [graphSource, props.nodes.length, renderId]);

  return (
    <Card className="kb-card graph-card" shadow="sm" radius="lg" withBorder>
      <Group justify="space-between" align="flex-start">
        <div>
          <Text className="section-kicker">Graph</Text>
          <Title order={3}>Visible Knowledge Graph</Title>
        </div>
        <Badge variant="light" color="teal">
          {props.scopedNodeIds.length} in scope
        </Badge>
      </Group>
      <Text c="dimmed" size="sm">
        Click a node to cycle through self, direct children, and all descendants.
      </Text>
      {renderError ? (
        <Alert color="red" title="Graph rendering failed">
          {renderError}
        </Alert>
      ) : null}
      {props.nodes.length === 0 ? (
        <EmptyState
          title="No visible nodes"
          body="Upload a document or loosen the active tag filters to populate the graph."
        />
      ) : (
        <div ref={containerRef} className="graph-svg" />
      )}
    </Card>
  );
}

function EmptyState(props: { title: string; body: string }): ReactElement {
  return (
    <div className="empty-state">
      <Title order={4}>{props.title}</Title>
      <Text c="dimmed">{props.body}</Text>
    </div>
  );
}

function ResultHitList(props: {
  hits: SearchHit[];
  onFocusNode: (nodeId: string) => void;
}): ReactElement {
  if (props.hits.length === 0) {
    return (
      <EmptyState
        title="No matching hits"
        body="Try broadening the graph scope, clearing tag filters, or changing the query."
      />
    );
  }
  return (
    <Stack gap="sm">
      {props.hits.map((hit) => (
        <Card key={`${hit.node_id}:${hit.derived_artifact_id ?? "node"}`} className="result-card" withBorder>
          <Group justify="space-between" align="flex-start">
            <div>
              <Button
                variant="subtle"
                size="compact-sm"
                className="inline-button"
                onClick={() => props.onFocusNode(hit.node_id)}
              >
                {hit.node_title}
              </Button>
              <Text c="dimmed" size="sm">
                {hit.original_filename}
              </Text>
            </div>
            <Badge variant="light" color="teal">
              {Math.round(hit.score * 100)}%
            </Badge>
          </Group>
          <Text size="sm">{excerpt(hit.text)}</Text>
          <Group gap="xs">
            {hit.tags.map((tag) => (
              <Badge key={`${hit.node_id}:${tag}`} variant="dot" color="blue">
                {tag}
              </Badge>
            ))}
          </Group>
        </Card>
      ))}
    </Stack>
  );
}

function HostedKnowledgeBaseDeskApp(): ReactElement {
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [initialQueryResult, setInitialQueryResult] =
    useState<KnowledgeQueryResult | null>(null);

  const { app, error, isConnected } = useApp({
    appInfo: IMPLEMENTATION,
    capabilities: {
      availableDisplayModes: ["inline", "fullscreen"],
    },
    onAppCreated: (createdApp) => {
      createdApp.ontoolresult = async (result) => {
        if (isKnowledgeQueryResult(result.structuredContent)) {
          setInitialQueryResult(result.structuredContent);
        }
      };
      createdApp.onhostcontextchanged = (params) => {
        setHostContext((previous) => ({ ...previous, ...params }));
      };
      createdApp.onteardown = async () => {
        setInitialQueryResult(null);
        return {};
      };
      createdApp.onerror = console.error;
    },
  });

  useEffect(() => {
    if (app) {
      setHostContext(app.getHostContext());
    }
  }, [app]);

  const bridge = useMemo(() => {
    if (!app || !initialQueryResult) {
      return null;
    }
    return createHostBridge(app, initialQueryResult, hostContext);
  }, [app, hostContext, initialQueryResult]);

  const hostControls = useMemo(() => {
    if (!app) {
      return null;
    }
    return createHostControls(app);
  }, [app]);

  if (error) {
    return (
      <MantineProvider
        theme={appTheme}
        cssVariablesResolver={appCssVariablesResolver}
        forceColorScheme={resolveColorScheme(hostContext)}
      >
        <CenteredState
          title="Unable to connect to the MCP host"
          body={String(error)}
        />
      </MantineProvider>
    );
  }

  if (!isConnected || !bridge || !hostControls) {
    return (
      <MantineProvider
        theme={appTheme}
        cssVariablesResolver={appCssVariablesResolver}
        forceColorScheme={resolveColorScheme(hostContext)}
      >
        <CenteredState
          title="Waiting for the MCP host"
          body="Open the app through query_knowledge_base to load the initial graph state."
          loading
        />
      </MantineProvider>
    );
  }

  return (
    <KnowledgeBaseDesk
      bridge={bridge}
      hostControls={hostControls}
      hostContext={hostContext}
    />
  );
}

function StandaloneKnowledgeBaseDeskApp(): ReactElement {
  const [hostContext] = useState<McpUiHostContext | undefined>({
    ...createMockBridge().hostContext,
    displayMode: "inline",
  });
  const bridge = useMemo(() => createMockBridge(hostContext), [hostContext]);
  const hostControls = useMemo(() => createMockHostControls(), []);

  return (
    <KnowledgeBaseDesk
      bridge={bridge}
      hostControls={hostControls}
      hostContext={hostContext}
    />
  );
}

function CenteredState(props: {
  title: string;
  body: string;
  loading?: boolean;
}): ReactElement {
  return (
    <div className="centered-shell">
      <Card className="kb-card centered-card" shadow="sm" radius="lg" withBorder>
        <Stack align="center">
          {props.loading ? <Loader color="teal" /> : null}
          <Title order={2}>{props.title}</Title>
          <Text ta="center" c="dimmed">
            {props.body}
          </Text>
        </Stack>
      </Card>
    </div>
  );
}

function KnowledgeBaseDesk(props: {
  bridge: KnowledgeBaseBridge;
  hostControls: KnowledgeBaseHostControls;
  hostContext?: McpUiHostContext;
}): ReactElement {
  const initialDeskState = props.bridge.initial_state;
  const [deskState, setDeskState] = useState<KnowledgeBaseDeskState>(initialDeskState);
  const [selectionState, setSelectionState] = useState<SelectionState>(
    buildSelectionState(initialDeskState),
  );
  const [queryText, setQueryText] = useState("");
  const [commandText, setCommandText] = useState("");
  const [activeTab, setActiveTab] = useState<MainTab>("search");
  const [fileSearchResult, setFileSearchResult] = useState<KnowledgeFileSearchResult | null>(
    props.bridge.initial_query_result.file_search_result,
  );
  const [branchSearchResult, setBranchSearchResult] =
    useState<KnowledgeBranchSearchResult | null>(
      props.bridge.initial_query_result.branch_search_result,
    );
  const [chatResult, setChatResult] = useState<KnowledgeChatResult | null>(
    props.bridge.initial_query_result.chat_result,
  );
  const [nodeDetail, setNodeDetail] = useState<KnowledgeNodeDetail | null>(null);
  const [commandResult, setCommandResult] =
    useState<KnowledgeBaseCommandResult | null>(null);
  const [pendingConfirmation, setPendingConfirmation] =
    useState<PendingCommandResult | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadTagIds, setUploadTagIds] = useState<string[]>(
    initialDeskState.knowledge_base?.context.tag_ids ?? [],
  );
  const [busyState, setBusyState] = useState<
    "idle" | "refresh" | "query" | "command" | "upload"
  >("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [requestedDisplayMode, setRequestedDisplayMode] =
    useState<McpUiDisplayMode | null>(null);

  const knowledgeBase = deskState.knowledge_base;
  const selectedNode =
    knowledgeBase?.nodes.find((node) => node.id === selectionState.selectedNodeId) ??
    null;
  const visibleNodeIds = new Set(knowledgeBase?.context.visible_node_ids ?? []);
  const scopedNodeIds = knowledgeBase?.context.scoped_node_ids ?? [];
  const visibleNodes =
    knowledgeBase?.nodes.filter((node) => visibleNodeIds.has(node.id)) ?? [];
  const visibleEdges =
    knowledgeBase?.edges.filter(
      (edge) =>
        visibleNodeIds.has(edge.from_node_id) && visibleNodeIds.has(edge.to_node_id),
    ) ?? [];

  const modelContextMarkdown = useMemo(
    () =>
      buildModelContextMarkdown({
        deskState,
        selectionState,
        fileSearchResult,
        branchSearchResult,
        chatResult,
        commandResult,
      }),
    [
      branchSearchResult,
      chatResult,
      commandResult,
      deskState,
      fileSearchResult,
      selectionState,
    ],
  );

  useEffect(() => {
    applyDocumentTheme(props.hostContext?.theme ?? "light");
    if (props.hostContext?.styles?.variables) {
      applyHostStyleVariables(props.hostContext.styles.variables);
    }
    if (props.hostContext?.styles?.css?.fonts) {
      applyHostFonts(props.hostContext.styles.css.fonts);
    }
  }, [props.hostContext]);

  useEffect(() => {
    if (props.hostContext?.displayMode) {
      setRequestedDisplayMode(null);
    }
  }, [props.hostContext?.displayMode]);

  useEffect(() => {
    if (!props.hostControls.supportsModelContext) {
      return;
    }
    void props.hostControls.update_model_context(modelContextMarkdown);
  }, [modelContextMarkdown, props.hostControls]);

  useEffect(() => {
    if (!selectionState.selectedNodeId) {
      setNodeDetail(null);
      return;
    }
    if (nodeDetail?.id === selectionState.selectedNodeId) {
      return;
    }
    void refreshNodeDetail(selectionState.selectedNodeId);
  }, [nodeDetail?.id, selectionState.selectedNodeId]);

  function applyDeskState(nextDeskState: KnowledgeBaseDeskState): void {
    startTransition(() => {
      setDeskState(nextDeskState);
      setSelectionState(buildSelectionState(nextDeskState));
      if (!nextDeskState.knowledge_base?.context.selected_node_id) {
        setNodeDetail(null);
      }
    });
  }

  function buildArgs(
    overrides: Partial<SelectionState> = {},
  ): {
    selected_node_id?: string;
    graph_selection_mode: SelectionState["graphSelectionMode"];
    tag_ids: string[];
    tag_match_mode: SelectionState["tagMatchMode"];
    media_types: string[];
    include_web: boolean;
    rewrite_query: boolean;
    branch_factor: number;
    depth: number;
    max_results: number;
  } {
    const nextSelectedNodeId =
      overrides.selectedNodeId !== undefined
        ? overrides.selectedNodeId
        : selectionState.selectedNodeId;
    return {
      selected_node_id: nextSelectedNodeId ?? undefined,
      graph_selection_mode:
        overrides.graphSelectionMode ?? selectionState.graphSelectionMode,
      tag_ids: overrides.selectedTagIds ?? selectionState.selectedTagIds,
      tag_match_mode: overrides.tagMatchMode ?? selectionState.tagMatchMode,
      media_types: overrides.selectedMediaTypes ?? selectionState.selectedMediaTypes,
      include_web: overrides.includeWeb ?? selectionState.includeWeb,
      rewrite_query: overrides.rewriteQuery ?? selectionState.rewriteQuery,
      branch_factor: overrides.branchFactor ?? selectionState.branchFactor,
      depth: overrides.depth ?? selectionState.depth,
      max_results: overrides.maxResults ?? selectionState.maxResults,
    };
  }

  async function refreshDesk(
    overrides: Partial<SelectionState> = {},
  ): Promise<void> {
    setBusyState("refresh");
    setErrorMessage(null);
    try {
      const requestArgs = buildArgs(overrides);
      const payload = await props.bridge.get_knowledge_base_info({
        ...requestArgs,
        detail_node_id: requestArgs.selected_node_id,
      });
      applyDeskState(payload.knowledge_base_state);
      setNodeDetail(payload.node_detail);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyState("idle");
    }
  }

  async function refreshNodeDetail(nodeId: string): Promise<void> {
    try {
      const payload = await props.bridge.get_knowledge_base_info({
        ...buildArgs(),
        detail_node_id: nodeId,
      });
      applyDeskState(payload.knowledge_base_state);
      setNodeDetail(payload.node_detail);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function runQuery(mode: KnowledgeQueryMode): Promise<void> {
    setBusyState("query");
    setErrorMessage(null);
    try {
      const payload = await props.bridge.query_knowledge_base({
        ...buildArgs(),
        query: queryText,
        mode,
      });
      applyDeskState(payload.knowledge_base_state);
      if (payload.file_search_result) {
        setFileSearchResult(payload.file_search_result);
      }
      if (payload.branch_search_result) {
        setBranchSearchResult(payload.branch_search_result);
      }
      if (payload.chat_result) {
        setChatResult(payload.chat_result);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyState("idle");
    }
  }

  async function executeCommand(): Promise<void> {
    setBusyState("command");
    setErrorMessage(null);
    try {
      const payload = await props.bridge.run_knowledge_base_command({
        command: commandText,
        ...buildArgs(),
      });
      applyDeskState(payload.knowledge_base_state);
      setCommandResult(payload);
      setPendingConfirmation(payload.pending_confirmation);
      const nextSelectedNodeId =
        payload.knowledge_base_state.knowledge_base?.context.selected_node_id ?? null;
      if (nextSelectedNodeId) {
        await refreshNodeDetail(nextSelectedNodeId);
      } else {
        setNodeDetail(null);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyState("idle");
    }
  }

  async function confirmCommand(): Promise<void> {
    if (!pendingConfirmation) {
      return;
    }
    setBusyState("command");
    setErrorMessage(null);
    try {
      const payload = await props.bridge.confirm_knowledge_base_command({
        token: pendingConfirmation.token,
        ...buildArgs(),
      });
      applyDeskState(payload.knowledge_base_state);
      setCommandResult(payload);
      setPendingConfirmation(null);
      setNodeDetail(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyState("idle");
    }
  }

  async function uploadCurrentFile(): Promise<void> {
    if (!uploadFile) {
      return;
    }
    setBusyState("upload");
    setErrorMessage(null);
    try {
      const payload = await props.bridge.upload_node({
        file: uploadFile,
        tag_ids: uploadTagIds,
      });
      setUploadFile(null);
      setCommandResult(null);
      await refreshDesk({
        selectedNodeId: payload.node.id,
        graphSelectionMode: "self",
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyState("idle");
    }
  }

  function handleNodeClick(nodeId: string): void {
    const nextSelection =
      selectionState.selectedNodeId === nodeId
        ? {
            selectedNodeId: nodeId,
            graphSelectionMode: nextSelectionMode(
              selectionState.graphSelectionMode,
            ),
          }
        : {
            selectedNodeId: nodeId,
            graphSelectionMode: "self" as const,
          };
    void refreshDesk(nextSelection);
  }

  function focusNode(nodeId: string): void {
    void refreshDesk({
      selectedNodeId: nodeId,
      graphSelectionMode: "self",
    });
  }

  function updateTagFilter(nextTagIds: string[]): void {
    const visibleIds = tagsVisibleIds(
      knowledgeBase,
      nextTagIds,
      selectionState.tagMatchMode,
    );
    const shouldClearSelection =
      selectionState.selectedNodeId !== null &&
      !visibleIds.has(selectionState.selectedNodeId);
    void refreshDesk({
      selectedTagIds: nextTagIds,
      selectedNodeId: shouldClearSelection ? null : selectionState.selectedNodeId,
      graphSelectionMode: shouldClearSelection
        ? "self"
        : selectionState.graphSelectionMode,
    });
  }

  function updateTagMatchMode(nextTagMatchMode: "all" | "any"): void {
    const visibleIds = tagsVisibleIds(
      knowledgeBase,
      selectionState.selectedTagIds,
      nextTagMatchMode,
    );
    const shouldClearSelection =
      selectionState.selectedNodeId !== null &&
      !visibleIds.has(selectionState.selectedNodeId);
    void refreshDesk({
      tagMatchMode: nextTagMatchMode,
      selectedNodeId: shouldClearSelection ? null : selectionState.selectedNodeId,
      graphSelectionMode: shouldClearSelection
        ? "self"
        : selectionState.graphSelectionMode,
    });
  }

  const hostDisplayMode = props.hostContext?.displayMode ?? "inline";

  return (
    <MantineProvider
      theme={appTheme}
      cssVariablesResolver={appCssVariablesResolver}
      forceColorScheme={resolveColorScheme(props.hostContext)}
    >
      <div className="kb-app-shell" style={extractSafeAreaStyle(props.hostContext)}>
        <header className="kb-header">
          <div>
            <Text className="section-kicker">MCP Knowledge Base Desk</Text>
            <Title order={1}>Graph-Scoped Retrieval</Title>
            <Text c="dimmed">
              {knowledgeBase?.knowledge_base.title ?? "Knowledge Base"} with{" "}
              {knowledgeBase?.knowledge_base.node_count ?? 0} node(s),{" "}
              {knowledgeBase?.knowledge_base.edge_count ?? 0} edge(s), and{" "}
              {knowledgeBase?.knowledge_base.tag_count ?? 0} tag(s).
            </Text>
          </div>
          <Group>
            <Badge color={deskState.access.status === "active" ? "teal" : "yellow"} variant="light">
              {deskState.access.status === "active" ? "Access active" : "Pending access"}
            </Badge>
            <Badge variant="outline" color="blue">
              {hostDisplayMode}
            </Badge>
            {props.hostControls.supportedDisplayModes.includes("fullscreen") ? (
              <Button
                variant="light"
                onClick={async () => {
                  setRequestedDisplayMode("fullscreen");
                  await props.hostControls.request_display_mode("fullscreen");
                }}
                loading={requestedDisplayMode === "fullscreen"}
              >
                Fullscreen
              </Button>
            ) : null}
          </Group>
        </header>

        {errorMessage ? (
          <Alert color="red" title="Something needs attention">
            {errorMessage}
          </Alert>
        ) : null}

        {commandResult ? (
          <Alert
            color={commandResult.status === "rejected" ? "red" : "teal"}
            title="Command result"
          >
            {commandResult.message}
          </Alert>
        ) : null}

        {pendingConfirmation ? (
          <Alert color="yellow" title="Confirm destructive command">
            <Stack gap="sm">
              <Text>{pendingConfirmation.prompt}</Text>
              <Group>
                <Button
                  color="red"
                  onClick={() => void confirmCommand()}
                  loading={busyState === "command"}
                >
                  Confirm delete
                </Button>
                <Button
                  variant="default"
                  onClick={() => setPendingConfirmation(null)}
                >
                  Cancel
                </Button>
              </Group>
            </Stack>
          </Alert>
        ) : null}

        <div className="desk-layout">
          <div className="desk-main-column">
            <GraphCard
              nodes={visibleNodes}
              edges={visibleEdges}
              selectedNodeId={selectionState.selectedNodeId}
              scopedNodeIds={scopedNodeIds}
              onNodeClick={handleNodeClick}
            />

            <Card className="kb-card" shadow="sm" radius="lg" withBorder>
              <Stack gap="md">
                <Group justify="space-between" align="flex-start">
                  <div>
                    <Text className="section-kicker">Retrieval</Text>
                    <Title order={3}>Scoped Search and Chat</Title>
                  </div>
                  {busyState === "query" ? <Loader size="sm" color="teal" /> : null}
                </Group>
                <Textarea
                  label="Query"
                  minRows={2}
                  maxRows={6}
                  placeholder="Ask a question, search for a phrase, or start a branching search..."
                  value={queryText}
                  onChange={(event) => setQueryText(event.currentTarget.value)}
                />
                <Group grow align="flex-end">
                  <Switch
                    label="Allow web search in chat"
                    checked={selectionState.includeWeb}
                    onChange={(event) =>
                      void refreshDesk({
                        includeWeb: event.currentTarget.checked,
                      })
                    }
                  />
                  <Switch
                    label="Rewrite search queries"
                    checked={selectionState.rewriteQuery}
                    onChange={(event) =>
                      void refreshDesk({
                        rewriteQuery: event.currentTarget.checked,
                      })
                    }
                  />
                </Group>
                <Group grow align="flex-end">
                  <NumberInput
                    label="Branch factor"
                    min={1}
                    max={6}
                    value={selectionState.branchFactor}
                    onChange={(value) =>
                      void refreshDesk({
                        branchFactor: normalizeInteger(value, selectionState.branchFactor),
                      })
                    }
                  />
                  <NumberInput
                    label="Depth"
                    min={1}
                    max={4}
                    value={selectionState.depth}
                    onChange={(value) =>
                      void refreshDesk({
                        depth: normalizeInteger(value, selectionState.depth),
                      })
                    }
                  />
                  <NumberInput
                    label="Max results"
                    min={1}
                    max={20}
                    value={selectionState.maxResults}
                    onChange={(value) =>
                      void refreshDesk({
                        maxResults: normalizeInteger(value, selectionState.maxResults),
                      })
                    }
                  />
                </Group>

                <Tabs
                  value={activeTab}
                  onChange={(value) => setActiveTab((value as MainTab) ?? "search")}
                >
                  <Tabs.List>
                    <Tabs.Tab value="search">File Search</Tabs.Tab>
                    <Tabs.Tab value="branch">Branch Search</Tabs.Tab>
                    <Tabs.Tab value="chat">Chat</Tabs.Tab>
                  </Tabs.List>

                  <Tabs.Panel value="search" pt="md">
                    <Group justify="space-between">
                      <Text size="sm" c="dimmed">
                        Raw vector-store retrieval within the current graph and tag scope.
                      </Text>
                      <Button
                        onClick={() => void runQuery("file_search")}
                        loading={busyState === "query"}
                      >
                        Search
                      </Button>
                    </Group>
                    <Divider my="md" />
                    <ResultHitList
                      hits={fileSearchResult?.hits ?? []}
                      onFocusNode={focusNode}
                    />
                  </Tabs.Panel>

                  <Tabs.Panel value="branch" pt="md">
                    <Group justify="space-between">
                      <Text size="sm" c="dimmed">
                        Expand one seed query into complementary branch queries.
                      </Text>
                      <Button
                        onClick={() => void runQuery("branch_search")}
                        loading={busyState === "query"}
                      >
                        Branch search
                      </Button>
                    </Group>
                    <Divider my="md" />
                    {branchSearchResult?.nodes.length ? (
                      <Stack gap="sm">
                        {branchSearchResult.nodes.map((node) => (
                          <Card key={node.id} className="result-card" withBorder>
                            <Group justify="space-between" align="flex-start">
                              <div>
                                <Text fw={600}>{node.query}</Text>
                                <Text c="dimmed" size="sm">
                                  Depth {node.depth}
                                  {node.rationale ? ` • ${node.rationale}` : ""}
                                </Text>
                              </div>
                              <Badge variant="light" color="blue">
                                {node.hits.length} hit(s)
                              </Badge>
                            </Group>
                            <ResultHitList hits={node.hits} onFocusNode={focusNode} />
                          </Card>
                        ))}
                      </Stack>
                    ) : (
                      <EmptyState
                        title="No branch search yet"
                        body="Run a branch search to generate related query branches from the current scope."
                      />
                    )}
                  </Tabs.Panel>

                  <Tabs.Panel value="chat" pt="md">
                    <Group justify="space-between">
                      <Text size="sm" c="dimmed">
                        Ask grounded questions against the selected graph scope.
                      </Text>
                      <Button
                        onClick={() => void runQuery("qa")}
                        loading={busyState === "query"}
                      >
                        Ask
                      </Button>
                    </Group>
                    <Divider my="md" />
                    {chatResult ? (
                      <Stack gap="sm">
                        <Card className="result-card" withBorder>
                          <Text>{chatResult.answer}</Text>
                        </Card>
                        <Group gap="xs">
                          {chatResult.citations.map((citation) => (
                            <Badge
                              key={`${citation.source}:${citation.label}`}
                              variant="light"
                              color={citation.source === "web" ? "orange" : "teal"}
                            >
                              {citation.label}
                            </Badge>
                          ))}
                        </Group>
                      </Stack>
                    ) : (
                      <EmptyState
                        title="No chat answer yet"
                        body="Ask a question to reuse the current graph, tag, and media filters."
                      />
                    )}
                  </Tabs.Panel>
                </Tabs>
              </Stack>
            </Card>
          </div>

          <div className="desk-side-column">
            <Card className="kb-card" shadow="sm" radius="lg" withBorder>
              <Stack gap="md">
                <div>
                  <Text className="section-kicker">Mutations</Text>
                  <Title order={3}>Command Bar</Title>
                </div>
                <Textarea
                  label="Command"
                  minRows={2}
                  placeholder="rename the selected node to Retrieval Map"
                  value={commandText}
                  onChange={(event) => setCommandText(event.currentTarget.value)}
                />
                <Text size="sm" c="dimmed">
                  Example commands:{" "}
                  <Code>rename the selected node to X</Code>,{" "}
                  <Code>add an edge from A to B labeled cites</Code>,{" "}
                  <Code>create tag research</Code>,{" "}
                  <Code>delete node Y</Code>.
                </Text>
                <Button
                  onClick={() => void executeCommand()}
                  loading={busyState === "command"}
                >
                  Run command
                </Button>
              </Stack>
            </Card>

            <Card className="kb-card" shadow="sm" radius="lg" withBorder>
              <Stack gap="md">
                <div>
                  <Text className="section-kicker">Upload</Text>
                  <Title order={3}>Add a Document Node</Title>
                </div>
                <FileInput
                  label="Document"
                  placeholder="Choose a file"
                  value={uploadFile}
                  onChange={setUploadFile}
                />
                <MultiSelect
                  label="Initial tags"
                  placeholder="Select tags"
                  value={uploadTagIds}
                  onChange={setUploadTagIds}
                  data={(knowledgeBase?.tags ?? []).map((tag) => ({
                    value: tag.id,
                    label: tag.name,
                  }))}
                />
                <Button
                  onClick={() => void uploadCurrentFile()}
                  disabled={!uploadFile}
                  loading={busyState === "upload"}
                >
                  Upload node
                </Button>
              </Stack>
            </Card>

            <Card className="kb-card" shadow="sm" radius="lg" withBorder>
              <Stack gap="md">
                <div>
                  <Text className="section-kicker">Filters</Text>
                  <Title order={3}>Graph and Retrieval Scope</Title>
                </div>
                <Text size="sm" c="dimmed">
                  {selectionSummary(knowledgeBase ?? null, selectedNode)}
                </Text>
                <Select
                  label="Tag match mode"
                  value={selectionState.tagMatchMode}
                  onChange={(value) =>
                    updateTagMatchMode((value as "all" | "any") ?? "all")
                  }
                  data={[
                    { value: "all", label: "All selected tags" },
                    { value: "any", label: "Any selected tag" },
                  ]}
                />
                <MultiSelect
                  label="Tag filters"
                  placeholder="Filter visible nodes by tag"
                  value={selectionState.selectedTagIds}
                  onChange={updateTagFilter}
                  data={(knowledgeBase?.tags ?? []).map((tag) => ({
                    value: tag.id,
                    label: `${tag.name} (${tag.node_count})`,
                  }))}
                />
                <MultiSelect
                  label="Media types"
                  placeholder="Limit retrieval by media type"
                  value={selectionState.selectedMediaTypes}
                  onChange={(values) =>
                    void refreshDesk({ selectedMediaTypes: values })
                  }
                  data={[
                    { value: "text/markdown", label: "Markdown" },
                    { value: "text/plain", label: "Plain text" },
                    { value: "image/jpeg", label: "JPEG" },
                    { value: "image/png", label: "PNG" },
                    { value: "audio/wav", label: "WAV audio" },
                    { value: "audio/mpeg", label: "MP3 audio" },
                    { value: "video/mp4", label: "MP4 video" },
                  ]}
                />
                <Group gap="xs">
                  <Badge variant="light" color="teal">
                    {knowledgeBase?.context.visible_node_ids.length ?? 0} visible
                  </Badge>
                  <Badge variant="light" color="blue">
                    {knowledgeBase?.context.scoped_node_ids.length ?? 0} scoped
                  </Badge>
                  <Badge variant="outline" color="gray">
                    {selectionState.graphSelectionMode}
                  </Badge>
                </Group>
              </Stack>
            </Card>

            <Card className="kb-card" shadow="sm" radius="lg" withBorder>
              <Stack gap="md">
                <div>
                  <Text className="section-kicker">Node</Text>
                  <Title order={3}>Selected Document Detail</Title>
                </div>
                {busyState === "refresh" ? <Loader size="sm" color="teal" /> : null}
                {nodeDetail ? (
                  <ScrollArea.Autosize mah={560} type="scroll">
                    <Stack gap="md">
                      <div>
                        <Title order={4}>{nodeDetail.display_title}</Title>
                        <Text c="dimmed">{nodeDetail.original_filename}</Text>
                      </div>
                      <Group gap="xs">
                        <Badge color={statusColor(nodeDetail.status)} variant="light">
                          {nodeDetail.status}
                        </Badge>
                        <Badge variant="outline" color="blue">
                          {nodeDetail.media_type}
                        </Badge>
                        <Badge variant="outline" color="gray">
                          {formatBytes(nodeDetail.byte_size)}
                        </Badge>
                      </Group>
                      <Text size="sm">
                        Created {formatDate(nodeDetail.created_at)} • Updated{" "}
                        {formatDate(nodeDetail.updated_at)}
                      </Text>
                      <Group gap="xs">
                        {nodeDetail.tags.map((tag) => (
                          <Badge key={tag.id} variant="light" color="teal">
                            {tag.name}
                          </Badge>
                        ))}
                      </Group>
                      <Text size="sm">
                        {nodeDetail.download_url ? (
                          <a href={nodeDetail.download_url} target="_blank" rel="noreferrer">
                            Open original file
                          </a>
                        ) : (
                          "Original file not available."
                        )}
                      </Text>
                      <Divider />
                      <div>
                        <Title order={5}>Outgoing edges</Title>
                        {nodeDetail.outgoing_edges.length ? (
                          <Stack gap="xs">
                            {nodeDetail.outgoing_edges.map((edge) => (
                              <Button
                                key={edge.id}
                                variant="subtle"
                                size="compact-md"
                                className="inline-button"
                                onClick={() => focusNode(edge.to_node_id)}
                              >
                                {edge.label} → {edge.to_node_title}
                              </Button>
                            ))}
                          </Stack>
                        ) : (
                          <Text c="dimmed" size="sm">
                            No outgoing edges yet.
                          </Text>
                        )}
                      </div>
                      <div>
                        <Title order={5}>Incoming edges</Title>
                        {nodeDetail.incoming_edges.length ? (
                          <Stack gap="xs">
                            {nodeDetail.incoming_edges.map((edge) => (
                              <Button
                                key={edge.id}
                                variant="subtle"
                                size="compact-md"
                                className="inline-button"
                                onClick={() => focusNode(edge.from_node_id)}
                              >
                                {edge.from_node_title} → {edge.label}
                              </Button>
                            ))}
                          </Stack>
                        ) : (
                          <Text c="dimmed" size="sm">
                            No incoming edges yet.
                          </Text>
                        )}
                      </div>
                      <div>
                        <Title order={5}>Derived artifacts</Title>
                        {nodeDetail.derived_artifacts.length ? (
                          <Stack gap="sm">
                            {nodeDetail.derived_artifacts.map((artifact) => (
                              <Card key={artifact.id} className="artifact-card" withBorder>
                                <Text fw={600}>{artifact.kind}</Text>
                                <Text size="sm">{excerpt(artifact.text_content, 420)}</Text>
                              </Card>
                            ))}
                          </Stack>
                        ) : (
                          <Text c="dimmed" size="sm">
                            No derived artifacts available.
                          </Text>
                        )}
                      </div>
                    </Stack>
                  </ScrollArea.Autosize>
                ) : (
                  <EmptyState
                    title="No node selected"
                    body="Click a visible node in the graph to inspect it and cycle its retrieval scope."
                  />
                )}
              </Stack>
            </Card>
          </div>
        </div>
      </div>
    </MantineProvider>
  );
}

export default function App(): ReactElement {
  return isStandaloneMode() ? (
    <StandaloneKnowledgeBaseDeskApp />
  ) : (
    <HostedKnowledgeBaseDeskApp />
  );
}
