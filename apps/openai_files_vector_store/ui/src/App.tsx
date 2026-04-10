import { applyDocumentTheme, applyHostFonts, applyHostStyleVariables, type McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import {
  Alert,
  AppShell,
  Badge,
  Button,
  Card,
  Code,
  Group,
  Loader,
  MantineProvider,
  ScrollArea,
  Select,
  Stack,
  Table,
  Tabs,
  Text,
  TextInput,
  Textarea,
  Title,
} from "@mantine/core";
import { startTransition, type CSSProperties, useDeferredValue, useEffect, useState } from "react";

import { createHostBridge, type VectorStoreConsoleBridge } from "./bridge";
import { createMockBridge } from "./mockBridge";
import { appCssVariablesResolver, appTheme } from "./theme";
import type {
  AskVectorStoreResult,
  AttributeValue,
  OpenVectorStoreConsoleResult,
  ScopeMode,
  SearchVectorStoreResult,
  VectorStoreFileSummary,
  VectorStoreListResult,
  VectorStoreStatusResult,
} from "./types";
import { isOpenVectorStoreConsoleResult } from "./types";

const IMPLEMENTATION = {
  name: "OpenAI Files Vector Store Console",
  version: "1.0.0",
};
const RESULT_SIZE_OPTIONS = ["3", "5", "8", "10"];
const RESERVED_ATTRIBUTE_KEYS = new Set(["openai_file_id", "filename"]);

type BusyState = "idle" | "refresh" | "status" | "search" | "ask";
type StatusTone = "danger" | "info" | "neutral" | "success";

function formatBytes(value: number): string {
  if (value < 1_024) {
    return `${value} B`;
  }
  if (value < 1_048_576) {
    return `${(value / 1_024).toFixed(1)} KB`;
  }
  return `${(value / 1_048_576).toFixed(1)} MB`;
}

function formatTimestamp(timestamp: number | null): string {
  if (timestamp === null) {
    return "n/a";
  }
  return new Date(timestamp * 1_000).toLocaleString();
}

function statusTone(status: string): StatusTone {
  switch (status) {
    case "completed":
      return "success";
    case "in_progress":
      return "info";
    case "failed":
      return "danger";
    case "cancelled":
      return "neutral";
    default:
      return "neutral";
  }
}

function extractSafeAreaStyle(hostContext?: McpUiHostContext): CSSProperties {
  return {
    paddingTop: hostContext?.safeAreaInsets?.top,
    paddingRight: hostContext?.safeAreaInsets?.right,
    paddingBottom: hostContext?.safeAreaInsets?.bottom,
    paddingLeft: hostContext?.safeAreaInsets?.left,
  };
}

function isStandaloneMode(): boolean {
  const params = new URLSearchParams(window.location.search);
  return window.parent === window || params.get("mock") === "1";
}

function isAttributeValue(value: unknown): value is AttributeValue {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

function isAttributeRecord(value: unknown): value is Record<string, AttributeValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && Object.values(value).every(isAttributeValue);
}

function extractFilename(file: VectorStoreFileSummary): string | null {
  const candidate = file.attributes?.filename;
  return typeof candidate === "string" ? candidate : null;
}

function getFileLabel(file: VectorStoreFileSummary): string {
  return extractFilename(file) ?? file.id;
}

function getEditableAttributes(file: VectorStoreFileSummary): Record<string, AttributeValue> {
  const editableEntries = Object.entries(file.attributes ?? {}).filter(([key]) => !RESERVED_ATTRIBUTE_KEYS.has(key));
  return Object.fromEntries(editableEntries);
}

function formatMetadataDraft(file: VectorStoreFileSummary): string {
  return JSON.stringify(getEditableAttributes(file), null, 2);
}

function parseMetadataDraft(input: string): Record<string, AttributeValue> {
  const trimmedInput = input.trim();
  if (!trimmedInput) {
    return {};
  }

  const parsed = JSON.parse(trimmedInput) as unknown;
  if (!isAttributeRecord(parsed)) {
    throw new Error("Metadata must be a flat JSON object with string, number, or boolean values.");
  }
  return parsed;
}

function getScopeFile(files: VectorStoreFileSummary[], fileId: string | null): VectorStoreFileSummary | null {
  if (fileId === null) {
    return null;
  }
  return files.find((file) => file.id === fileId) ?? null;
}

function resolveScopedFilename(files: VectorStoreFileSummary[], fileId: string | null, previousFilename: string | null): string | null {
  const scopedFile = getScopeFile(files, fileId);
  return scopedFile ? extractFilename(scopedFile) ?? previousFilename : null;
}

function HostedConsoleApp() {
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [initialState, setInitialState] = useState<OpenVectorStoreConsoleResult | null>(null);

  const { app, error, isConnected } = useApp({
    appInfo: IMPLEMENTATION,
    capabilities: {},
    onAppCreated: (createdApp) => {
      createdApp.ontoolresult = async (result) => {
        if (isOpenVectorStoreConsoleResult(result.structuredContent)) {
          setInitialState(result.structuredContent);
        }
      };
      createdApp.onhostcontextchanged = (params) => {
        setHostContext((previous) => ({ ...previous, ...params }));
      };
      createdApp.onerror = console.error;
    },
  });

  useEffect(() => {
    if (app) {
      setHostContext(app.getHostContext());
    }
  }, [app]);

  if (error) {
    return <CenteredState title="Unable to connect to the MCP host" description={error.message} tone="error" />;
  }

  if (!isConnected || !app || !initialState) {
    return <CenteredState title="Connecting to the MCP host" description="Waiting for the host and initial tool result." tone="loading" />;
  }

  return <ConsoleScaffold bridge={createHostBridge(app, initialState, hostContext)} hostContext={hostContext} />;
}

function StandaloneConsoleApp() {
  const bridge = createMockBridge();
  return <ConsoleScaffold bridge={bridge} hostContext={bridge.hostContext} />;
}

function CenteredState(props: { title: string; description: string; tone: "loading" | "error" }) {
  return (
    <MantineProvider cssVariablesResolver={appCssVariablesResolver} theme={appTheme} defaultColorScheme="light" forceColorScheme="light">
      <div
        style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: "2rem",
        }}
      >
        <Card className="vector-store-card" maw={560} radius="lg" shadow="md" withBorder>
          <Stack gap="md">
            <Group justify="space-between">
              <Title order={2}>{props.title}</Title>
              {props.tone === "loading" ? <Loader size="sm" /> : null}
            </Group>
            <Alert className="vector-store-alert" data-tone={props.tone === "error" ? "danger" : "info"} variant="light">
              {props.description}
            </Alert>
          </Stack>
        </Card>
      </div>
    </MantineProvider>
  );
}

function ConsoleScaffold(props: { bridge: VectorStoreConsoleBridge; hostContext?: McpUiHostContext }) {
  const [vectorStoreList, setVectorStoreList] = useState<VectorStoreListResult>(props.bridge.initial_state.vector_store_list);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(props.bridge.initial_state.selected_vector_store_id);
  const [selectedVectorStoreStatus, setSelectedVectorStoreStatus] = useState<VectorStoreStatusResult | null>(
    props.bridge.initial_state.selected_vector_store_status,
  );
  const [searchQuery, setSearchQuery] = useState(props.bridge.initial_state.search_panel.query);
  const [searchRewriteQuery, setSearchRewriteQuery] = useState(props.bridge.initial_state.search_panel.rewrite_query);
  const [searchMaxResults, setSearchMaxResults] = useState(String(props.bridge.initial_state.search_panel.max_num_results));
  const [searchScopeMode, setSearchScopeMode] = useState<ScopeMode>(props.bridge.initial_state.search_panel.scope);
  const [searchScopeFileId, setSearchScopeFileId] = useState<string | null>(props.bridge.initial_state.search_panel.file_id);
  const [searchScopeFilename, setSearchScopeFilename] = useState<string | null>(props.bridge.initial_state.search_panel.filename);
  const [question, setQuestion] = useState(props.bridge.initial_state.ask_panel.question);
  const [askMaxResults, setAskMaxResults] = useState(String(props.bridge.initial_state.ask_panel.max_num_results));
  const [askScopeMode, setAskScopeMode] = useState<ScopeMode>(props.bridge.initial_state.ask_panel.scope);
  const [askScopeFileId, setAskScopeFileId] = useState<string | null>(props.bridge.initial_state.ask_panel.file_id);
  const [askScopeFilename, setAskScopeFilename] = useState<string | null>(props.bridge.initial_state.ask_panel.filename);
  const [searchResult, setSearchResult] = useState<SearchVectorStoreResult | null>(null);
  const [askResult, setAskResult] = useState<AskVectorStoreResult | null>(null);
  const [editingFileId, setEditingFileId] = useState<string | null>(null);
  const [metadataDraft, setMetadataDraft] = useState("{}");
  const [savingMetadataFileId, setSavingMetadataFileId] = useState<string | null>(null);
  const [deletingFileId, setDeletingFileId] = useState<string | null>(null);
  const [busyState, setBusyState] = useState<BusyState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const deferredStatus = useDeferredValue(selectedVectorStoreStatus);
  const deferredSearchResult = useDeferredValue(searchResult);

  useEffect((): void => {
    applyDocumentTheme(props.hostContext?.theme ?? "light");

    if (props.hostContext?.styles?.variables) {
      applyHostStyleVariables(props.hostContext.styles.variables);
    }

    if (props.hostContext?.styles?.css?.fonts) {
      applyHostFonts(props.hostContext.styles.css.fonts);
    }
  }, [props.hostContext]);

  useEffect(() => {
    startTransition(() => {
      setVectorStoreList(props.bridge.initial_state.vector_store_list);
      setSelectedVectorStoreId(props.bridge.initial_state.selected_vector_store_id);
      setSelectedVectorStoreStatus(props.bridge.initial_state.selected_vector_store_status);
      setSearchQuery(props.bridge.initial_state.search_panel.query);
      setSearchRewriteQuery(props.bridge.initial_state.search_panel.rewrite_query);
      setSearchMaxResults(String(props.bridge.initial_state.search_panel.max_num_results));
      setSearchScopeMode(props.bridge.initial_state.search_panel.scope);
      setSearchScopeFileId(props.bridge.initial_state.search_panel.file_id);
      setSearchScopeFilename(props.bridge.initial_state.search_panel.filename);
      setQuestion(props.bridge.initial_state.ask_panel.question);
      setAskMaxResults(String(props.bridge.initial_state.ask_panel.max_num_results));
      setAskScopeMode(props.bridge.initial_state.ask_panel.scope);
      setAskScopeFileId(props.bridge.initial_state.ask_panel.file_id);
      setAskScopeFilename(props.bridge.initial_state.ask_panel.filename);
      setSearchResult(null);
      setAskResult(null);
      setEditingFileId(null);
      setMetadataDraft("{}");
      setSavingMetadataFileId(null);
      setDeletingFileId(null);
      setErrorMessage(null);
    });
  }, [props.bridge]);

  const selectedSummary = vectorStoreList.vector_stores.find((store) => store.id === selectedVectorStoreId) ?? null;
  const selectedFiles = deferredStatus?.files ?? [];
  const editingFile = getScopeFile(selectedFiles, editingFileId);

  const searchScopeOptions = [
    { value: "vector_store", label: "Whole vector store" },
    ...selectedFiles.map((file) => ({
      value: file.id,
      label: getFileLabel(file),
    })),
  ];
  const askScopeOptions = [
    { value: "vector_store", label: "Whole vector store" },
    ...selectedFiles.map((file) => ({
      value: file.id,
      label: getFileLabel(file),
    })),
  ];

  async function loadStatus(nextVectorStoreId: string | null): Promise<void> {
    if (nextVectorStoreId === null) {
      startTransition(() => {
        setSelectedVectorStoreId(null);
        setSelectedVectorStoreStatus(null);
        setSearchResult(null);
        setAskResult(null);
        setEditingFileId(null);
      });
      return;
    }

    setBusyState("status");
    setErrorMessage(null);
    try {
      const status = await props.bridge.get_vector_store_status({
        vector_store_id: nextVectorStoreId,
        file_limit: 20,
      });
      startTransition(() => {
        setSelectedVectorStoreId(nextVectorStoreId);
        setSelectedVectorStoreStatus(status);
        setSearchResult(null);
        setAskResult(null);
        setEditingFileId((current) => (getScopeFile(status.files, current) ? current : null));
        setSearchScopeFileId((current) => (getScopeFile(status.files, current) ? current : null));
        setSearchScopeFilename((current) => resolveScopedFilename(status.files, searchScopeFileId, current));
        setSearchScopeMode((current) => (getScopeFile(status.files, searchScopeFileId) ? current : "vector_store"));
        setAskScopeFileId((current) => (getScopeFile(status.files, current) ? current : null));
        setAskScopeFilename((current) => resolveScopedFilename(status.files, askScopeFileId, current));
        setAskScopeMode((current) => (getScopeFile(status.files, askScopeFileId) ? current : "vector_store"));
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to load vector store status.");
    } finally {
      setBusyState("idle");
    }
  }

  async function refreshVectorStores(): Promise<void> {
    setBusyState("refresh");
    setErrorMessage(null);
    try {
      const nextList = await props.bridge.list_vector_stores({ limit: 20 });
      const nextSelectedId =
        nextList.vector_stores.find((store) => store.id === selectedVectorStoreId)?.id ?? nextList.vector_stores[0]?.id ?? null;

      startTransition(() => {
        setVectorStoreList(nextList);
      });

      await loadStatus(nextSelectedId);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to refresh vector stores.");
      setBusyState("idle");
    }
  }

  function useFileForSearch(file: VectorStoreFileSummary): void {
    setSearchScopeMode("file");
    setSearchScopeFileId(file.id);
    setSearchScopeFilename(extractFilename(file));
    setErrorMessage(null);
  }

  function useFileForAsk(file: VectorStoreFileSummary): void {
    setAskScopeMode("file");
    setAskScopeFileId(file.id);
    setAskScopeFilename(extractFilename(file));
    setErrorMessage(null);
  }

  async function runSearch(): Promise<void> {
    if (!selectedVectorStoreId || !searchQuery.trim()) {
      setErrorMessage("Choose a vector store and enter a search query.");
      return;
    }

    if (searchScopeMode === "file" && searchScopeFileId === null) {
      setErrorMessage("Choose an attached file before running a file-scoped search.");
      return;
    }

    setBusyState("search");
    setErrorMessage(null);
    try {
      const result = await props.bridge.search_vector_store({
        vector_store_id: selectedVectorStoreId,
        query: searchQuery,
        max_num_results: Number(searchMaxResults),
        rewrite_query: searchRewriteQuery,
        file_id: searchScopeMode === "file" ? searchScopeFileId ?? undefined : undefined,
        filename: searchScopeMode === "file" ? searchScopeFilename ?? undefined : undefined,
      });
      startTransition(() => {
        setSearchResult(result);
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to search the vector store.");
    } finally {
      setBusyState("idle");
    }
  }

  async function runAsk(): Promise<void> {
    if (!selectedVectorStoreId || !question.trim()) {
      setErrorMessage("Choose a vector store and enter a grounded question.");
      return;
    }

    if (askScopeMode === "file" && askScopeFileId === null) {
      setErrorMessage("Choose an attached file before running a file-scoped grounded question.");
      return;
    }

    setBusyState("ask");
    setErrorMessage(null);
    try {
      const result = await props.bridge.ask_vector_store({
        vector_store_id: selectedVectorStoreId,
        question,
        max_num_results: Number(askMaxResults),
        file_id: askScopeMode === "file" ? askScopeFileId ?? undefined : undefined,
        filename: askScopeMode === "file" ? askScopeFilename ?? undefined : undefined,
      });
      startTransition(() => {
        setAskResult(result);
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to run grounded Q&A.");
    } finally {
      setBusyState("idle");
    }
  }

  function beginMetadataEdit(file: VectorStoreFileSummary): void {
    setEditingFileId(file.id);
    setMetadataDraft(formatMetadataDraft(file));
    setErrorMessage(null);
  }

  async function saveMetadata(): Promise<void> {
    if (!selectedVectorStoreId || editingFileId === null) {
      setErrorMessage("Choose an attached file before updating metadata.");
      return;
    }

    let attributes: Record<string, AttributeValue>;
    try {
      attributes = parseMetadataDraft(metadataDraft);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Invalid metadata JSON.");
      return;
    }

    setSavingMetadataFileId(editingFileId);
    setErrorMessage(null);
    try {
      await props.bridge.update_vector_store_file_attributes({
        vector_store_id: selectedVectorStoreId,
        file_id: editingFileId,
        attributes,
      });
      setEditingFileId(null);
      await loadStatus(selectedVectorStoreId);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to update file metadata.");
    } finally {
      setSavingMetadataFileId(null);
    }
  }

  async function deleteFile(file: VectorStoreFileSummary): Promise<void> {
    if (!window.confirm(`Delete ${getFileLabel(file)} globally from OpenAI Files?`)) {
      return;
    }

    setDeletingFileId(file.id);
    setErrorMessage(null);
    try {
      await props.bridge.delete_file({ file_id: file.id });
      if (editingFileId === file.id) {
        setEditingFileId(null);
      }
      if (searchScopeFileId === file.id) {
        setSearchScopeMode("vector_store");
        setSearchScopeFileId(null);
        setSearchScopeFilename(null);
      }
      if (askScopeFileId === file.id) {
        setAskScopeMode("vector_store");
        setAskScopeFileId(null);
        setAskScopeFilename(null);
      }
      await refreshVectorStores();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to delete the file.");
    } finally {
      setDeletingFileId(null);
    }
  }

  return (
    <MantineProvider
      cssVariablesResolver={appCssVariablesResolver}
      theme={appTheme}
      defaultColorScheme="light"
      forceColorScheme={props.hostContext?.theme ?? "light"}
    >
      <AppShell className="vector-store-shell" header={{ height: 76 }} navbar={{ width: 340, breakpoint: "md" }} padding="md">
        <AppShell.Header className="vector-store-header">
          <Group h="100%" justify="space-between" px="lg">
            <div>
              <Text className="vector-store-eyebrow" fw={700} size="sm">
                OPENAI FILES + VECTOR STORES
              </Text>
              <Title order={3}>Retrieval Console</Title>
            </div>
            <Group gap="sm">
              <Badge
                className="vector-store-badge vector-store-badge--solid"
                data-tone={props.bridge.mode === "host" ? "success" : "info"}
                size="lg"
              >
                {props.bridge.mode === "host" ? "MCP host" : "mock browser"}
              </Badge>
              <Button
                className="vector-store-button vector-store-button--secondary"
                loading={busyState === "refresh" || busyState === "status"}
                onClick={() => {
                  void refreshVectorStores();
                }}
                variant="default"
              >
                Refresh
              </Button>
            </Group>
          </Group>
        </AppShell.Header>

        <AppShell.Navbar className="vector-store-navbar" p="md">
          <Stack gap="md" h="100%">
            <Group justify="space-between">
              <div>
                <Text fw={700}>Vector Stores</Text>
                <Text c="dimmed" size="sm">
                  {vectorStoreList.total_returned} available
                </Text>
              </div>
              {busyState === "refresh" ? <Loader size="sm" /> : null}
            </Group>

            <ScrollArea h="100%">
              <Stack gap="sm">
                {vectorStoreList.vector_stores.map((store) => (
                  <Button
                    key={store.id}
                    className="vector-store-list-button"
                    data-selected={store.id === selectedVectorStoreId ? "true" : "false"}
                    fullWidth
                    justify="flex-start"
                    onClick={() => {
                      void loadStatus(store.id);
                    }}
                    styles={{ inner: { width: "100%" } }}
                    variant="default"
                  >
                    <Group justify="space-between" wrap="nowrap" w="100%">
                      <div>
                        <Text fw={700} lineClamp={1}>
                          {store.name}
                        </Text>
                        <Text className="vector-store-list-meta" size="xs">
                          {formatBytes(store.usage_bytes)}
                        </Text>
                      </div>
                      <Badge
                        className="vector-store-badge vector-store-badge--soft vector-store-status-badge"
                        data-tone={statusTone(store.status)}
                      >
                        {store.status}
                      </Badge>
                    </Group>
                  </Button>
                ))}
              </Stack>
            </ScrollArea>
          </Stack>
        </AppShell.Navbar>

        <AppShell.Main className="vector-store-main-panel">
          <div className="vector-store-safe-area" style={extractSafeAreaStyle(props.hostContext)}>
            <Stack className="vector-store-main">
              {errorMessage ? (
                <Alert className="vector-store-alert" data-tone="danger" variant="light">
                  {errorMessage}
                </Alert>
              ) : null}

              <div className="vector-store-main-grid">
                <Card className="vector-store-card" padding="lg" radius="lg" shadow="sm" withBorder>
                  <Stack gap="md">
                    <Group justify="space-between" align="flex-start">
                      <div>
                        <Text className="vector-store-eyebrow" size="sm" fw={700}>
                          SELECTED STORE
                        </Text>
                        <Title order={3}>{selectedSummary?.name ?? "No vector store selected"}</Title>
                      </div>
                      {busyState === "status" ? <Loader size="sm" /> : null}
                    </Group>

                    <Select
                      data={vectorStoreList.vector_stores.map((store) => ({
                        value: store.id,
                        label: store.name,
                      }))}
                      label="Jump to vector store"
                      onChange={(value) => {
                        void loadStatus(value);
                      }}
                      placeholder="Choose a vector store"
                      searchable
                      value={selectedVectorStoreId}
                    />

                    {deferredStatus ? (
                      <Stack gap="md">
                        <Group gap="sm">
                          <Badge
                            className="vector-store-badge vector-store-badge--soft"
                            data-tone={statusTone(deferredStatus.vector_store.status)}
                            size="lg"
                          >
                            {deferredStatus.vector_store.status}
                          </Badge>
                          <Badge className="vector-store-badge vector-store-badge--soft" data-tone="neutral">
                            {deferredStatus.vector_store.file_counts.completed}/{deferredStatus.vector_store.file_counts.total} files
                            complete
                          </Badge>
                        </Group>

                        <div className="vector-store-metrics-grid">
                          <Card className="vector-store-card vector-store-card--strong" padding="md" radius="md" withBorder>
                            <Stack gap={4}>
                              <Text c="dimmed" size="sm">
                                Vector store ID
                              </Text>
                              <Code>{deferredStatus.vector_store.id}</Code>
                            </Stack>
                          </Card>
                          <Card className="vector-store-card vector-store-card--strong" padding="md" radius="md" withBorder>
                            <Stack gap={4}>
                              <Text c="dimmed" size="sm">
                                Last active
                              </Text>
                              <Text>{formatTimestamp(deferredStatus.vector_store.last_active_at)}</Text>
                            </Stack>
                          </Card>
                          <Card className="vector-store-card vector-store-card--strong" padding="md" radius="md" withBorder>
                            <Stack gap={4}>
                              <Text c="dimmed" size="sm">
                                Usage
                              </Text>
                              <Text>{formatBytes(deferredStatus.vector_store.usage_bytes)}</Text>
                            </Stack>
                          </Card>
                        </div>

                        {deferredStatus.files.length > 0 ? (
                          <Table.ScrollContainer minWidth={960} type="native">
                            <Table className="vector-store-table" striped withTableBorder>
                              <Table.Thead>
                                <Table.Tr>
                                  <Table.Th>Attached file</Table.Th>
                                  <Table.Th>Status</Table.Th>
                                  <Table.Th>Bytes</Table.Th>
                                  <Table.Th>Created</Table.Th>
                                  <Table.Th>Metadata</Table.Th>
                                  <Table.Th>Actions</Table.Th>
                                </Table.Tr>
                              </Table.Thead>
                              <Table.Tbody>
                                {deferredStatus.files.map((file) => (
                                  <Table.Tr key={file.id}>
                                    <Table.Td>
                                      <Stack gap={4}>
                                        <Text fw={700}>{getFileLabel(file)}</Text>
                                        <Code>{file.id}</Code>
                                      </Stack>
                                    </Table.Td>
                                    <Table.Td>
                                      <Badge className="vector-store-badge vector-store-badge--soft" data-tone={statusTone(file.status)}>
                                        {file.status}
                                      </Badge>
                                    </Table.Td>
                                    <Table.Td>{formatBytes(file.usage_bytes)}</Table.Td>
                                    <Table.Td>{formatTimestamp(file.created_at)}</Table.Td>
                                    <Table.Td>
                                      {file.attributes && Object.keys(file.attributes).length > 0 ? (
                                        <div className="vector-store-attribute-list">
                                          {Object.entries(file.attributes).map(([key, value]) => (
                                            <Badge
                                              key={`${file.id}-${key}`}
                                              className="vector-store-badge vector-store-badge--soft"
                                              data-tone={RESERVED_ATTRIBUTE_KEYS.has(key) ? "info" : "neutral"}
                                            >
                                              {key}: {String(value)}
                                            </Badge>
                                          ))}
                                        </div>
                                      ) : (
                                        <Text c="dimmed" size="sm">
                                          No metadata
                                        </Text>
                                      )}
                                    </Table.Td>
                                    <Table.Td>
                                      <div className="vector-store-action-button-group">
                                        <Button
                                          className="vector-store-button vector-store-button--secondary"
                                          onClick={() => useFileForSearch(file)}
                                          size="xs"
                                          variant="default"
                                        >
                                          Use for search
                                        </Button>
                                        <Button
                                          className="vector-store-button vector-store-button--secondary"
                                          onClick={() => useFileForAsk(file)}
                                          size="xs"
                                          variant="default"
                                        >
                                          Use for ask
                                        </Button>
                                        <Button
                                          className="vector-store-button vector-store-button--secondary"
                                          loading={savingMetadataFileId === file.id}
                                          onClick={() => beginMetadataEdit(file)}
                                          size="xs"
                                          variant="default"
                                        >
                                          Edit metadata
                                        </Button>
                                        <Button
                                          className="vector-store-button vector-store-button--danger"
                                          loading={deletingFileId === file.id}
                                          onClick={() => {
                                            void deleteFile(file);
                                          }}
                                          size="xs"
                                        >
                                          Delete file
                                        </Button>
                                      </div>
                                    </Table.Td>
                                  </Table.Tr>
                                ))}
                              </Table.Tbody>
                            </Table>
                          </Table.ScrollContainer>
                        ) : (
                          <Alert className="vector-store-alert" data-tone="neutral" variant="light">
                            This vector store does not have any attached files yet.
                          </Alert>
                        )}

                        {editingFile ? (
                          <Card className="vector-store-card vector-store-card--strong" padding="md" radius="md" withBorder>
                            <Stack gap="sm">
                              <Group justify="space-between" align="flex-start">
                                <div>
                                  <Text className="vector-store-eyebrow" size="sm" fw={700}>
                                    EDIT FILE METADATA
                                  </Text>
                                  <Title order={5}>{getFileLabel(editingFile)}</Title>
                                </div>
                                <Code>{editingFile.id}</Code>
                              </Group>

                              <Alert className="vector-store-alert" data-tone="info" variant="light">
                                Reserved identity attributes stay managed by the server. Edit only the user-defined metadata below.
                              </Alert>

                              <Textarea
                                autosize
                                label="User-defined attributes (JSON object)"
                                minRows={8}
                                onChange={(event) => setMetadataDraft(event.currentTarget.value)}
                                value={metadataDraft}
                              />

                              <div className="vector-store-inline-actions">
                                <Button
                                  className="vector-store-button vector-store-button--primary"
                                  loading={savingMetadataFileId === editingFile.id}
                                  onClick={() => {
                                    void saveMetadata();
                                  }}
                                >
                                  Save metadata
                                </Button>
                                <Button
                                  className="vector-store-button vector-store-button--secondary"
                                  onClick={() => {
                                    setEditingFileId(null);
                                  }}
                                  variant="default"
                                >
                                  Cancel
                                </Button>
                              </div>
                            </Stack>
                          </Card>
                        ) : null}
                      </Stack>
                    ) : (
                      <Alert className="vector-store-alert" data-tone="info" variant="light">
                        Pick a vector store to inspect its file ingestion state.
                      </Alert>
                    )}
                  </Stack>
                </Card>

                <Card className="vector-store-card" padding="lg" radius="lg" shadow="sm" withBorder>
                  <Tabs className="vector-store-tabs" defaultValue="search">
                    <Tabs.List grow>
                      <Tabs.Tab value="search">Raw Search</Tabs.Tab>
                      <Tabs.Tab value="ask">Grounded Ask</Tabs.Tab>
                    </Tabs.List>

                    <Tabs.Panel pt="md" value="search">
                      <Stack gap="md">
                        <TextInput
                          label="Query"
                          onChange={(event) => setSearchQuery(event.currentTarget.value)}
                          placeholder="Search for a marker, topic, or phrase"
                          value={searchQuery}
                        />
                        <div className="vector-store-action-grid">
                          <Select
                            data={searchScopeOptions}
                            label="Search scope"
                            onChange={(value) => {
                              if (value === "vector_store" || value === null) {
                                setSearchScopeMode("vector_store");
                                setSearchScopeFileId(null);
                                setSearchScopeFilename(null);
                                return;
                              }
                              const file = getScopeFile(selectedFiles, value);
                              if (file) {
                                useFileForSearch(file);
                              }
                            }}
                            value={searchScopeMode === "file" ? searchScopeFileId : "vector_store"}
                          />
                          <Select
                            data={RESULT_SIZE_OPTIONS}
                            label="Max results"
                            onChange={(value) => {
                              if (value) {
                                setSearchMaxResults(value);
                              }
                            }}
                            value={searchMaxResults}
                          />
                          <Button
                            className="vector-store-button vector-store-button--secondary"
                            data-active={searchRewriteQuery ? "true" : undefined}
                            onClick={() => setSearchRewriteQuery((value) => !value)}
                            variant="default"
                          >
                            Rewrite query: {searchRewriteQuery ? "On" : "Off"}
                          </Button>
                          <Button
                            className="vector-store-button vector-store-button--primary"
                            loading={busyState === "search"}
                            onClick={() => {
                              void runSearch();
                            }}
                          >
                            Run search
                          </Button>
                        </div>

                        {searchScopeMode === "file" && searchScopeFileId ? (
                          <Alert className="vector-store-alert" data-tone="info" variant="light">
                            Searching within <Code>{searchScopeFilename ?? searchScopeFileId}</Code>.
                          </Alert>
                        ) : null}

                        {deferredSearchResult ? (
                          <Stack gap="md">
                            <Group justify="space-between" align="flex-start">
                              <Stack gap={4}>
                                <Text fw={700}>
                                  {deferredSearchResult.total_hits} hit(s) for <Code>{deferredSearchResult.query}</Code>
                                </Text>
                                {deferredSearchResult.file_id ? (
                                  <Text c="dimmed" size="sm">
                                    Scoped to {deferredSearchResult.filename ?? deferredSearchResult.file_id}
                                  </Text>
                                ) : null}
                              </Stack>
                            </Group>
                            <Table.ScrollContainer minWidth={720} type="native">
                              <Table className="vector-store-table" striped withTableBorder>
                                <Table.Thead>
                                  <Table.Tr>
                                    <Table.Th>File</Table.Th>
                                    <Table.Th>Score</Table.Th>
                                    <Table.Th>Snippet</Table.Th>
                                  </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                  {deferredSearchResult.hits.map((hit) => (
                                    <Table.Tr key={`${hit.file_id}-${hit.score}`}>
                                      <Table.Td>
                                        <Stack gap={2}>
                                          <Text fw={600}>{hit.filename}</Text>
                                          <Code>{hit.file_id}</Code>
                                        </Stack>
                                      </Table.Td>
                                      <Table.Td>{hit.score.toFixed(3)}</Table.Td>
                                      <Table.Td>
                                        <Text className="vector-store-hit-text" size="sm">
                                          {hit.text}
                                        </Text>
                                      </Table.Td>
                                    </Table.Tr>
                                  ))}
                                </Table.Tbody>
                              </Table>
                            </Table.ScrollContainer>
                          </Stack>
                        ) : (
                          <Alert className="vector-store-alert" data-tone="neutral" variant="light">
                            Run a raw search to inspect the exact chunks coming back from the selected store or from one attached file.
                          </Alert>
                        )}
                      </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel pt="md" value="ask">
                      <Stack gap="md">
                        <Textarea
                          autosize
                          label="Grounded question"
                          minRows={4}
                          onChange={(event) => setQuestion(event.currentTarget.value)}
                          placeholder="Ask a question that should be answered only from the selected vector store"
                          value={question}
                        />
                        <div className="vector-store-action-grid vector-store-action-grid--compact">
                          <Select
                            data={askScopeOptions}
                            label="Ask scope"
                            onChange={(value) => {
                              if (value === "vector_store" || value === null) {
                                setAskScopeMode("vector_store");
                                setAskScopeFileId(null);
                                setAskScopeFilename(null);
                                return;
                              }
                              const file = getScopeFile(selectedFiles, value);
                              if (file) {
                                useFileForAsk(file);
                              }
                            }}
                            value={askScopeMode === "file" ? askScopeFileId : "vector_store"}
                          />
                          <Select
                            data={RESULT_SIZE_OPTIONS}
                            label="Max search results"
                            onChange={(value) => {
                              if (value) {
                                setAskMaxResults(value);
                              }
                            }}
                            value={askMaxResults}
                          />
                          <Button
                            className="vector-store-button vector-store-button--primary"
                            loading={busyState === "ask"}
                            onClick={() => {
                              void runAsk();
                            }}
                          >
                            Ask vector store
                          </Button>
                        </div>

                        {askScopeMode === "file" && askScopeFileId ? (
                          <Alert className="vector-store-alert" data-tone="info" variant="light">
                            Asking only against <Code>{askScopeFilename ?? askScopeFileId}</Code>.
                          </Alert>
                        ) : null}

                        {askResult ? (
                          <Stack gap="md">
                            <Card className="vector-store-card vector-store-card--strong" padding="md" radius="md" withBorder>
                              <Stack gap="xs">
                                <Group justify="space-between">
                                  <Text fw={700}>Answer</Text>
                                  <Badge className="vector-store-badge vector-store-badge--soft" data-tone="neutral">
                                    {askResult.model}
                                  </Badge>
                                </Group>
                                {askResult.file_id ? (
                                  <Text c="dimmed" size="sm">
                                    Scoped to {askResult.filename ?? askResult.file_id}
                                  </Text>
                                ) : null}
                                <Text className="vector-store-answer">{askResult.answer}</Text>
                              </Stack>
                            </Card>

                            <Table.ScrollContainer minWidth={760} type="native">
                              <Table className="vector-store-table" striped withTableBorder>
                                <Table.Thead>
                                  <Table.Tr>
                                    <Table.Th>Search call</Table.Th>
                                    <Table.Th>Queries</Table.Th>
                                    <Table.Th>Supporting hits</Table.Th>
                                  </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                  {askResult.search_calls.map((searchCall) => (
                                    <Table.Tr key={searchCall.id}>
                                      <Table.Td>
                                        <Stack gap={2}>
                                          <Code>{searchCall.id}</Code>
                                          <Badge
                                            className="vector-store-badge vector-store-badge--soft"
                                            data-tone={statusTone(searchCall.status)}
                                          >
                                            {searchCall.status}
                                          </Badge>
                                        </Stack>
                                      </Table.Td>
                                      <Table.Td>
                                        {searchCall.queries.map((queryValue) => (
                                          <Text key={queryValue} size="sm">
                                            {queryValue}
                                          </Text>
                                        ))}
                                      </Table.Td>
                                      <Table.Td>
                                        <Stack gap="xs">
                                          {searchCall.results.map((result) => (
                                            <Card
                                              key={`${searchCall.id}-${result.file_id}-${result.score}`}
                                              className="vector-store-card vector-store-card--strong"
                                              padding="sm"
                                              radius="md"
                                              withBorder
                                            >
                                              <Stack gap={4}>
                                                <Group justify="space-between">
                                                  <Text fw={600} size="sm">
                                                    {result.filename}
                                                  </Text>
                                                  <Badge className="vector-store-badge vector-store-badge--soft" data-tone="success">
                                                    {result.score.toFixed(3)}
                                                  </Badge>
                                                </Group>
                                                <Text className="vector-store-hit-text" size="sm">
                                                  {result.text}
                                                </Text>
                                              </Stack>
                                            </Card>
                                          ))}
                                        </Stack>
                                      </Table.Td>
                                    </Table.Tr>
                                  ))}
                                </Table.Tbody>
                              </Table>
                            </Table.ScrollContainer>
                          </Stack>
                        ) : (
                          <Alert className="vector-store-alert" data-tone="info" variant="light">
                            Ask a grounded question to compare the synthesized answer with the supporting search hits.
                          </Alert>
                        )}
                      </Stack>
                    </Tabs.Panel>
                  </Tabs>
                </Card>
              </div>
            </Stack>
          </div>
        </AppShell.Main>
      </AppShell>
    </MantineProvider>
  );
}

export default function App() {
  return isStandaloneMode() ? <StandaloneConsoleApp /> : <HostedConsoleApp />;
}
