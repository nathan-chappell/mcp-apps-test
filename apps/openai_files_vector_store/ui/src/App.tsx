import type { McpUiHostContext } from "@modelcontextprotocol/ext-apps";
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
import {
  startTransition,
  type CSSProperties,
  useDeferredValue,
  useEffect,
  useState,
} from "react";

import { createHostBridge, type VectorStoreConsoleBridge } from "./bridge";
import { createMockBridge } from "./mockBridge";
import { appTheme } from "./theme";
import type {
  AskVectorStoreResult,
  OpenVectorStoreConsoleResult,
  SearchVectorStoreResult,
  VectorStoreListResult,
  VectorStoreStatusResult,
} from "./types";
import { isOpenVectorStoreConsoleResult } from "./types";

const IMPLEMENTATION = {
  name: "OpenAI Files Vector Store Console",
  version: "1.0.0",
};
const RESULT_SIZE_OPTIONS = ["3", "5", "8", "10"];

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

function statusColor(status: string): string {
  switch (status) {
    case "completed":
      return "teal";
    case "in_progress":
      return "blue";
    case "failed":
      return "red";
    case "cancelled":
      return "gray";
    default:
      return "sand";
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

function HostedConsoleApp() {
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [initialState, setInitialState] = useState<OpenVectorStoreConsoleResult | null>(
    null,
  );

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
    return (
      <CenteredState
        title="Unable to connect to the MCP host"
        description={error.message}
        tone="error"
      />
    );
  }

  if (!isConnected || !app || !initialState) {
    return (
      <CenteredState
        title="Connecting to the MCP host"
        description="Waiting for the host and initial tool result."
        tone="loading"
      />
    );
  }

  return (
    <ConsoleScaffold
      bridge={createHostBridge(app, initialState, hostContext)}
      hostContext={hostContext}
    />
  );
}

function StandaloneConsoleApp() {
  const bridge = createMockBridge();
  return <ConsoleScaffold bridge={bridge} hostContext={bridge.hostContext} />;
}

function CenteredState(props: {
  title: string;
  description: string;
  tone: "loading" | "error";
}) {
  return (
    <MantineProvider theme={appTheme} defaultColorScheme="light">
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
            <Alert color={props.tone === "error" ? "red" : "blue"} variant="light">
              {props.description}
            </Alert>
          </Stack>
        </Card>
      </div>
    </MantineProvider>
  );
}

function ConsoleScaffold(props: {
  bridge: VectorStoreConsoleBridge;
  hostContext?: McpUiHostContext;
}) {
  const [vectorStoreList, setVectorStoreList] = useState<VectorStoreListResult>(
    props.bridge.initial_state.vector_store_list,
  );
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(
    props.bridge.initial_state.selected_vector_store_id,
  );
  const [selectedVectorStoreStatus, setSelectedVectorStoreStatus] =
    useState<VectorStoreStatusResult | null>(
      props.bridge.initial_state.selected_vector_store_status,
    );
  const [searchQuery, setSearchQuery] = useState(
    props.bridge.initial_state.search_panel.query,
  );
  const [searchRewriteQuery, setSearchRewriteQuery] = useState(
    props.bridge.initial_state.search_panel.rewrite_query,
  );
  const [searchMaxResults, setSearchMaxResults] = useState(
    String(props.bridge.initial_state.search_panel.max_num_results),
  );
  const [question, setQuestion] = useState(props.bridge.initial_state.ask_panel.question);
  const [askMaxResults, setAskMaxResults] = useState(
    String(props.bridge.initial_state.ask_panel.max_num_results),
  );
  const [searchResult, setSearchResult] = useState<SearchVectorStoreResult | null>(null);
  const [askResult, setAskResult] = useState<AskVectorStoreResult | null>(null);
  const [busyState, setBusyState] = useState<
    "idle" | "refresh" | "status" | "search" | "ask"
  >("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const deferredStatus = useDeferredValue(selectedVectorStoreStatus);
  const deferredSearchResult = useDeferredValue(searchResult);

  useEffect(() => {
    startTransition(() => {
      setVectorStoreList(props.bridge.initial_state.vector_store_list);
      setSelectedVectorStoreId(props.bridge.initial_state.selected_vector_store_id);
      setSelectedVectorStoreStatus(props.bridge.initial_state.selected_vector_store_status);
      setSearchQuery(props.bridge.initial_state.search_panel.query);
      setSearchRewriteQuery(props.bridge.initial_state.search_panel.rewrite_query);
      setSearchMaxResults(
        String(props.bridge.initial_state.search_panel.max_num_results),
      );
      setQuestion(props.bridge.initial_state.ask_panel.question);
      setAskMaxResults(String(props.bridge.initial_state.ask_panel.max_num_results));
      setSearchResult(null);
      setAskResult(null);
      setErrorMessage(null);
    });
  }, [props.bridge]);

  const selectedSummary =
    vectorStoreList.vector_stores.find((store) => store.id === selectedVectorStoreId) ?? null;

  async function loadStatus(nextVectorStoreId: string | null): Promise<void> {
    if (nextVectorStoreId === null) {
      startTransition(() => {
        setSelectedVectorStoreId(null);
        setSelectedVectorStoreStatus(null);
        setSearchResult(null);
        setAskResult(null);
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
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to load vector store status.",
      );
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
        nextList.vector_stores.find((store) => store.id === selectedVectorStoreId)?.id ??
        nextList.vector_stores[0]?.id ??
        null;

      startTransition(() => {
        setVectorStoreList(nextList);
      });

      await loadStatus(nextSelectedId);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to refresh vector stores.",
      );
      setBusyState("idle");
    }
  }

  async function runSearch(): Promise<void> {
    if (!selectedVectorStoreId || !searchQuery.trim()) {
      setErrorMessage("Choose a vector store and enter a search query.");
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
      });
      startTransition(() => {
        setSearchResult(result);
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to search the vector store.",
      );
    } finally {
      setBusyState("idle");
    }
  }

  async function runAsk(): Promise<void> {
    if (!selectedVectorStoreId || !question.trim()) {
      setErrorMessage("Choose a vector store and enter a grounded question.");
      return;
    }

    setBusyState("ask");
    setErrorMessage(null);
    try {
      const result = await props.bridge.ask_vector_store({
        vector_store_id: selectedVectorStoreId,
        question,
        max_num_results: Number(askMaxResults),
      });
      startTransition(() => {
        setAskResult(result);
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to run grounded Q&A.",
      );
    } finally {
      setBusyState("idle");
    }
  }

  return (
    <MantineProvider theme={appTheme} defaultColorScheme="light">
      <AppShell
        className="vector-store-shell"
        header={{ height: 76 }}
        navbar={{ width: 340, breakpoint: "md" }}
        padding="md"
      >
        <AppShell.Header>
          <Group h="100%" justify="space-between" px="lg">
            <div>
              <Text fw={700} size="sm" c="dimmed">
                OPENAI FILES + VECTOR STORES
              </Text>
              <Title order={3}>Retrieval Console</Title>
            </div>
            <Group gap="sm">
              <Badge color={props.bridge.mode === "host" ? "teal" : "blue"} size="lg">
                {props.bridge.mode === "host" ? "MCP host" : "mock browser"}
              </Badge>
              <Button
                color="ink.7"
                loading={busyState === "refresh" || busyState === "status"}
                onClick={() => {
                  void refreshVectorStores();
                }}
                variant="light"
              >
                Refresh
              </Button>
            </Group>
          </Group>
        </AppShell.Header>

        <AppShell.Navbar p="md">
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
                    color={store.id === selectedVectorStoreId ? "teal" : "gray"}
                    justify="flex-start"
                    onClick={() => {
                      void loadStatus(store.id);
                    }}
                    styles={{ inner: { width: "100%" } }}
                    variant={store.id === selectedVectorStoreId ? "filled" : "light"}
                  >
                    <Group justify="space-between" wrap="nowrap" w="100%">
                      <div>
                        <Text fw={700} lineClamp={1}>
                          {store.name}
                        </Text>
                        <Text c={store.id === selectedVectorStoreId ? "white" : "dimmed"} size="xs">
                          {formatBytes(store.usage_bytes)}
                        </Text>
                      </div>
                      <Badge color={statusColor(store.status)} variant="white">
                        {store.status}
                      </Badge>
                    </Group>
                  </Button>
                ))}
              </Stack>
            </ScrollArea>
          </Stack>
        </AppShell.Navbar>

        <AppShell.Main style={extractSafeAreaStyle(props.hostContext)}>
          <Stack className="vector-store-main">
            {errorMessage ? (
              <Alert color="red" variant="light">
                {errorMessage}
              </Alert>
            ) : null}

            <div className="vector-store-main-grid">
              <Card className="vector-store-card" padding="lg" radius="lg" shadow="sm" withBorder>
                <Stack gap="md">
                  <Group justify="space-between" align="flex-start">
                    <div>
                      <Text c="dimmed" size="sm" fw={700}>
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
                        <Badge color={statusColor(deferredStatus.vector_store.status)} size="lg">
                          {deferredStatus.vector_store.status}
                        </Badge>
                        <Badge color="sand.6" variant="light">
                          {deferredStatus.vector_store.file_counts.completed}/
                          {deferredStatus.vector_store.file_counts.total} files complete
                        </Badge>
                      </Group>

                      <Group grow align="flex-start">
                        <Card className="vector-store-card" padding="md" radius="md" withBorder>
                          <Stack gap={4}>
                            <Text c="dimmed" size="sm">
                              Vector store ID
                            </Text>
                            <Code>{deferredStatus.vector_store.id}</Code>
                          </Stack>
                        </Card>
                        <Card className="vector-store-card" padding="md" radius="md" withBorder>
                          <Stack gap={4}>
                            <Text c="dimmed" size="sm">
                              Last active
                            </Text>
                            <Text>{formatTimestamp(deferredStatus.vector_store.last_active_at)}</Text>
                          </Stack>
                        </Card>
                        <Card className="vector-store-card" padding="md" radius="md" withBorder>
                          <Stack gap={4}>
                            <Text c="dimmed" size="sm">
                              Usage
                            </Text>
                            <Text>{formatBytes(deferredStatus.vector_store.usage_bytes)}</Text>
                          </Stack>
                        </Card>
                      </Group>

                      <Table striped withTableBorder>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Attached file</Table.Th>
                            <Table.Th>Status</Table.Th>
                            <Table.Th>Bytes</Table.Th>
                            <Table.Th>Created</Table.Th>
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          {deferredStatus.files.map((file) => (
                            <Table.Tr key={file.id}>
                              <Table.Td>
                                <Code>{file.id}</Code>
                              </Table.Td>
                              <Table.Td>
                                <Badge color={statusColor(file.status)}>{file.status}</Badge>
                              </Table.Td>
                              <Table.Td>{formatBytes(file.usage_bytes)}</Table.Td>
                              <Table.Td>{formatTimestamp(file.created_at)}</Table.Td>
                            </Table.Tr>
                          ))}
                        </Table.Tbody>
                      </Table>
                    </Stack>
                  ) : (
                    <Alert color="blue" variant="light">
                      Pick a vector store to inspect its file ingestion state.
                    </Alert>
                  )}
                </Stack>
              </Card>

              <Card className="vector-store-card" padding="lg" radius="lg" shadow="sm" withBorder>
                <Tabs defaultValue="search">
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
                      <Group grow align="end">
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
                          color={searchRewriteQuery ? "ink.7" : "gray"}
                          onClick={() => setSearchRewriteQuery((value) => !value)}
                          variant={searchRewriteQuery ? "filled" : "light"}
                        >
                          Rewrite query: {searchRewriteQuery ? "On" : "Off"}
                        </Button>
                        <Button
                          loading={busyState === "search"}
                          onClick={() => {
                            void runSearch();
                          }}
                        >
                          Run search
                        </Button>
                      </Group>

                      {deferredSearchResult ? (
                        <Stack gap="md">
                          <Group justify="space-between">
                            <Text fw={700}>
                              {deferredSearchResult.total_hits} hit(s) for{" "}
                              <Code>{deferredSearchResult.query}</Code>
                            </Text>
                          </Group>
                          <Table striped withTableBorder>
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
                        </Stack>
                      ) : (
                        <Alert color="sand.6" variant="light">
                          Run a raw search to inspect the exact chunks coming back from the selected store.
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
                      <Group grow align="end">
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
                          loading={busyState === "ask"}
                          onClick={() => {
                            void runAsk();
                          }}
                        >
                          Ask vector store
                        </Button>
                      </Group>

                      {askResult ? (
                        <Stack gap="md">
                          <Card className="vector-store-card" padding="md" radius="md" withBorder>
                            <Stack gap="xs">
                              <Group justify="space-between">
                                <Text fw={700}>Answer</Text>
                                <Badge color="ink.7" variant="light">
                                  {askResult.model}
                                </Badge>
                              </Group>
                              <Text className="vector-store-answer">{askResult.answer}</Text>
                            </Stack>
                          </Card>

                          <Table striped withTableBorder>
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
                                      <Badge color={statusColor(searchCall.status)}>
                                        {searchCall.status}
                                      </Badge>
                                    </Stack>
                                  </Table.Td>
                                  <Table.Td>
                                    {searchCall.queries.map((query) => (
                                      <Text key={query} size="sm">
                                        {query}
                                      </Text>
                                    ))}
                                  </Table.Td>
                                  <Table.Td>
                                    <Stack gap="xs">
                                      {searchCall.results.map((result) => (
                                        <Card
                                          key={`${searchCall.id}-${result.file_id}-${result.score}`}
                                          className="vector-store-card"
                                          padding="sm"
                                          radius="md"
                                          withBorder
                                        >
                                          <Stack gap={4}>
                                            <Group justify="space-between">
                                              <Text fw={600} size="sm">
                                                {result.filename}
                                              </Text>
                                              <Badge color="teal" variant="light">
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
                        </Stack>
                      ) : (
                        <Alert color="blue" variant="light">
                          Ask a grounded question to compare the synthesized answer with the supporting search hits.
                        </Alert>
                      )}
                    </Stack>
                  </Tabs.Panel>
                </Tabs>
              </Card>
            </div>
          </Stack>
        </AppShell.Main>
      </AppShell>
    </MantineProvider>
  );
}

export default function App() {
  return isStandaloneMode() ? <StandaloneConsoleApp /> : <HostedConsoleApp />;
}
