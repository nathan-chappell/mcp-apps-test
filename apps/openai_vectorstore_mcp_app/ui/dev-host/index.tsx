import { ClerkProvider, SignIn, SignedIn, SignedOut, UserButton, useAuth } from "@clerk/clerk-react";
import type { AppBridge as AppBridgeInstance } from "@modelcontextprotocol/ext-apps/app-bridge";
import type { Tool } from "@modelcontextprotocol/sdk/types.js";
import { createRoot } from "react-dom/client";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  connectToServer,
  createAppBridge,
  getToolDefaultInput,
  getToolUiUri,
  getVisibleTools,
  invokeToolWithOptionalUi,
  type AppMessage,
  type DisplayMode,
  type ModelContext,
  type ServerInfo,
} from "./mcpHost";

const MCP_SERVER_PATH = "/mcp";

interface DevAuthConfig {
  clerk_publishable_key: string | null;
  app_name: string;
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isUnauthorizedServerError(error: unknown): boolean {
  return /(^|[\s(])401([\s):]|$)|unauthorized/i.test(formatErrorMessage(error));
}

function formatServerConnectionError(serverUrl: string, error: unknown): string {
  if (!isUnauthorizedServerError(error)) {
    return `${serverUrl}: ${formatErrorMessage(error)}`;
  }

  return [
    `${serverUrl} rejected the MCP connection with 401 Unauthorized.`,
    "If Clerk showed a deployment or allowed-origin warning, open the dev host on http://localhost:8000/ and keep Clerk configured for localhost.",
  ].join(" ");
}

function formatPanelJson(value: unknown, emptyMessage: string): string {
  return value === null || value === undefined ? emptyMessage : JSON.stringify(value, null, 2);
}

function sleep(durationMs: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

async function waitForClerkSessionToken(
  getToken: (options?: { skipCache?: boolean }) => Promise<string | null>,
): Promise<string | null> {
  for (const delayMs of [0, 120, 320, 700]) {
    const token = await getToken({ skipCache: true });
    if (token) {
      return token;
    }
    if (delayMs > 0) {
      await sleep(delayMs);
    }
  }
  return (await getToken()) ?? null;
}

function getDefaultServerUrl(): string {
  return new URL(MCP_SERVER_PATH, window.location.origin).toString();
}

async function fetchDevAuthConfig(): Promise<DevAuthConfig> {
  const response = await fetch("/api/dev-auth-config");
  if (!response.ok) {
    throw new Error(`Unable to load Clerk auth config (${response.status}).`);
  }
  return (await response.json()) as DevAuthConfig;
}

function LoadingScreen(props: { title: string; description: string }) {
  return (
    <div className="host-shell">
      <section className="host-card host-hero">
        <div className="host-kicker">MCP App Dev Loop</div>
        <h1>{props.title}</h1>
        <p>{props.description}</p>
      </section>
    </div>
  );
}

function ErrorScreen(props: { title: string; description: string }) {
  return (
    <div className="host-shell">
      <section className="host-card host-hero">
        <div className="host-kicker">MCP App Dev Loop</div>
        <h1>{props.title}</h1>
        <div className="host-alert">{props.description}</div>
      </section>
    </div>
  );
}

function SignInScreen(props: { appName: string }) {
  return (
    <div className="host-shell">
      <section className="host-card host-hero">
        <div className="host-kicker">MCP App Dev Loop</div>
        <h1>Sign in to {props.appName}</h1>
        <p>
          The local dev host now mirrors the authenticated MCP flow. Sign in with Clerk
          here, then the host will attach your session token to MCP requests.
        </p>
      </section>

      <section className="host-card host-panel" style={{ maxWidth: 540, margin: "0 auto" }}>
        <SignIn oauthFlow="auto" routing="hash" withSignUp />
      </section>
    </div>
  );
}

function AuthenticatedDevHost(props: { serverUrl: string; appName: string }) {
  const { getToken, isLoaded, isSignedIn, sessionId } = useAuth();

  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [selectedToolName, setSelectedToolName] = useState("");
  const [inputJson, setInputJson] = useState("{}");
  const [isConnecting, setIsConnecting] = useState(true);
  const [isCalling, setIsCalling] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [callError, setCallError] = useState<string | null>(null);
  const [resultJson, setResultJson] = useState("");
  const [messages, setMessages] = useState<AppMessage[]>([]);
  const [modelContext, setModelContext] = useState<ModelContext>(null);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("inline");

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const bridgeRef = useRef<AppBridgeInstance | null>(null);

  const visibleTools = getVisibleTools(serverInfo);
  const selectedTool = visibleTools.find((tool) => tool.name === selectedToolName) ?? visibleTools[0] ?? null;
  const uiUri = getToolUiUri(selectedTool);

  let parsedInput: Record<string, unknown> | null = null;
  let inputValidationMessage: string | null = null;
  try {
    const candidate = JSON.parse(inputJson) as unknown;
    if (!isJsonObject(candidate)) {
      inputValidationMessage = "Tool input must be a JSON object.";
    } else {
      parsedInput = candidate;
    }
  } catch (error) {
    inputValidationMessage = formatErrorMessage(error);
  }

  const authTokenProvider = useMemo(() => {
    let initialTokenPromise: Promise<string | null> | null = null;
    return async () => {
      if (initialTokenPromise === null) {
        initialTokenPromise = waitForClerkSessionToken(getToken);
      }
      const bootstrapToken = await initialTokenPromise;
      if (bootstrapToken) {
        return bootstrapToken;
      }
      return (await getToken()) ?? null;
    };
  }, [getToken, sessionId]);

  useEffect(() => {
    if (!isLoaded || !isSignedIn) {
      return;
    }

    let isCancelled = false;

    async function loadServer(): Promise<void> {
      setIsConnecting(true);
      setLoadError(null);

      try {
        const bootstrapToken = await authTokenProvider();
        if (!bootstrapToken) {
          throw new Error(
            "Clerk sign-in succeeded, but no session token was available for the MCP connection yet. Refresh once and try again.",
          );
        }
        const connectedServer = await connectToServer(
          new URL(props.serverUrl),
          authTokenProvider,
        );
        if (isCancelled) {
          return;
        }

        setServerInfo(connectedServer);
        const firstTool = getVisibleTools(connectedServer)[0] ?? null;
        setSelectedToolName(firstTool?.name ?? "");
        setInputJson(getToolDefaultInput(firstTool));
      } catch (error) {
        if (!isCancelled) {
          setLoadError(formatServerConnectionError(props.serverUrl, error));
        }
      } finally {
        if (!isCancelled) {
          setIsConnecting(false);
        }
      }
    }

    void loadServer();

    return () => {
      isCancelled = true;
      if (bridgeRef.current !== null) {
        void bridgeRef.current.close().catch(() => undefined);
      }
    };
  }, [authTokenProvider, isLoaded, isSignedIn, props.serverUrl]);

  async function resetCurrentApp(): Promise<void> {
    if (bridgeRef.current !== null) {
      try {
        await bridgeRef.current.teardownResource({});
      } catch {
        // Ignore teardown races during rapid iteration.
      }

      await bridgeRef.current.close().catch(() => undefined);
      bridgeRef.current = null;
    }

    setDisplayMode("inline");

    if (iframeRef.current !== null) {
      iframeRef.current.src = "";
      iframeRef.current.style.height = "720px";
      iframeRef.current.style.minWidth = "0";
    }
  }

  function handleToolChange(nextToolName: string): void {
    setSelectedToolName(nextToolName);

    const nextTool = visibleTools.find((tool) => tool.name === nextToolName) ?? null;
    setInputJson(getToolDefaultInput(nextTool));
    setCallError(null);
    void resetCurrentApp();
  }

  async function handleCallTool(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (serverInfo === null || selectedTool === null || parsedInput === null) {
      return;
    }

    setIsCalling(true);
    setCallError(null);
    setResultJson("");
    setMessages([]);
    setModelContext(null);

    await resetCurrentApp();

    try {
      const shouldRenderUi = uiUri !== undefined && iframeRef.current !== null;
      if (shouldRenderUi && iframeRef.current !== null) {
        bridgeRef.current = createAppBridge(serverInfo, iframeRef.current, {
          onContextUpdate: setModelContext,
          onDisplayModeChange: setDisplayMode,
          onMessage: (message) => {
            setMessages((previous) => [...previous, message]);
          },
        });
      }

      const result = await invokeToolWithOptionalUi(
        serverInfo,
        selectedTool,
        parsedInput,
        iframeRef.current,
        bridgeRef.current,
      );
      setResultJson(JSON.stringify(result, null, 2));
    } catch (error) {
      setCallError(formatErrorMessage(error));
    } finally {
      setIsCalling(false);
    }
  }

  function exitFullscreen(): void {
    setDisplayMode("inline");
    if (bridgeRef.current !== null) {
      bridgeRef.current.sendHostContextChange({ displayMode: "inline" });
    }
  }

  const toolSupportsUi = uiUri !== undefined;

  return (
    <div className="host-shell">
      <section className="host-card host-hero">
        <div className="host-kicker">MCP App Dev Loop</div>
        <h1>Test the authenticated app flow without leaving `npm run dev`.</h1>
        <p>
          This local harness is adapted from the upstream `modelcontextprotocol/ext-apps`
          `basic-host` example. It now signs you in with Clerk first, then connects to
          the same-origin Python MCP server at <code>/mcp</code> with your session token.
        </p>
        <div className="host-surface-header" style={{ marginTop: 16 }}>
          <div className="host-tag">{props.appName}</div>
          <UserButton afterSignOutUrl="/" />
        </div>
      </section>

      <div className="host-grid">
        <form className="host-card host-controls" onSubmit={(event) => void handleCallTool(event)}>
          <div className="host-field">
            <label htmlFor="tool-select">Tool</label>
            <select
              id="tool-select"
              disabled={serverInfo === null || visibleTools.length === 0}
              value={selectedTool?.name ?? ""}
              onChange={(event) => handleToolChange(event.target.value)}
            >
              {visibleTools.map((tool) => (
                <option key={tool.name} value={tool.name}>
                  {tool.name}
                </option>
              ))}
            </select>
          </div>

          <div className="host-field">
            <label htmlFor="tool-input">Tool input</label>
            <textarea
              id="tool-input"
              aria-invalid={inputValidationMessage !== null}
              spellCheck={false}
              value={inputJson}
              onChange={(event) => setInputJson(event.target.value)}
            />
          </div>

          <div className="host-actions">
            <button className="host-button" disabled={isConnecting || isCalling || selectedTool === null || parsedInput === null} type="submit">
              {isCalling ? "Calling tool..." : "Call tool"}
            </button>
            <button className="host-button host-button--secondary" onClick={() => setInputJson(getToolDefaultInput(selectedTool))} type="button">
              Reset input
            </button>
          </div>

          <div className="host-status">
            {isConnecting
              ? "Connecting to /mcp..."
              : serverInfo === null
                ? "No server connected."
                : `Connected to ${serverInfo.url}`}
          </div>

          {inputValidationMessage !== null ? <div className="host-alert">{inputValidationMessage}</div> : null}
          {loadError !== null ? <div className="host-alert">{loadError}</div> : null}
          {callError !== null ? <div className="host-alert">{callError}</div> : null}
        </form>

        <section className="host-card host-surface">
          <div className="host-surface-header">
            <div>
              <h2>Rendered app</h2>
            </div>
            <div className="host-tag">
              {selectedTool === null ? "No tool selected" : toolSupportsUi ? `UI tool: ${selectedTool.name}` : `No UI resource on ${selectedTool.name}`}
            </div>
          </div>

          {toolSupportsUi ? (
            <div className={`host-preview${displayMode === "fullscreen" ? " fullscreen" : ""}`}>
              {displayMode === "fullscreen" ? (
                <div className="host-surface-header" style={{ padding: "14px 18px 0" }}>
                  <div className="host-tag">App requested fullscreen mode</div>
                  <button className="host-button host-button--secondary" onClick={exitFullscreen} type="button">
                    Exit fullscreen
                  </button>
                </div>
              ) : null}
              <iframe ref={iframeRef} title="MCP app preview" />
            </div>
          ) : (
            <div className="host-preview-empty">
              <div>
                This tool does not declare a UI resource.
                <br />
                Its JSON result still appears below so you can exercise the server from the same screen.
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="host-panels">
        <article className="host-card host-panel">
          <div className="host-surface-header">
            <h3>Tool result</h3>
            <div className="host-tag">{resultJson ? "latest call" : "waiting"}</div>
          </div>
          <p>The raw MCP result is always shown here, even for non-UI tools.</p>
          <pre className="host-code">{resultJson || "Call a tool to inspect its result."}</pre>
        </article>

        <article className="host-card host-panel">
          <div className="host-surface-header">
            <h3>App messages</h3>
            <div className="host-tag">{messages.length === 0 ? "none yet" : `${messages.length} message${messages.length === 1 ? "" : "s"}`}</div>
          </div>
          <p>Messages sent from the app to the host or model surface appear here.</p>
          <pre className="host-code">{formatPanelJson(messages.length === 0 ? null : messages, "No app messages yet.")}</pre>
        </article>

        <article className="host-card host-panel wide">
          <div className="host-surface-header">
            <h3>Model context</h3>
            <div className="host-tag">{modelContext === null ? "empty" : "updated"}</div>
          </div>
          <p>When the app calls `updateModelContext`, the host stores the most recent payload here for inspection.</p>
          <pre className="host-code">{formatPanelJson(modelContext, "The app has not published any model context yet.")}</pre>
        </article>
      </section>
    </div>
  );
}

function BootstrapApp() {
  const serverUrl = getDefaultServerUrl();
  const [authConfig, setAuthConfig] = useState<DevAuthConfig | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isCancelled = false;

    async function loadBootstrap(): Promise<void> {
      setIsLoading(true);
      setLoadError(null);
      try {
        const config = await fetchDevAuthConfig();
        if (isCancelled) {
          return;
        }
        setAuthConfig(config);
      } catch (error) {
        if (!isCancelled) {
          setLoadError(formatErrorMessage(error));
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
        }
      }
    }

    void loadBootstrap();
    return () => {
      isCancelled = true;
    };
  }, []);

  if (isLoading) {
    return (
      <LoadingScreen
        title="Loading dev host"
        description="Loading the Clerk configuration for the local /mcp dev host."
      />
    );
  }

  if (loadError !== null) {
    return <ErrorScreen title="Unable to load the dev host" description={loadError} />;
  }

  if (!authConfig?.clerk_publishable_key) {
    return (
      <ErrorScreen
        title="Clerk publishable key missing"
        description="Set CLERK_PUBLISHABLE_KEY in the server environment so the local dev host can sign in."
      />
    );
  }

  return (
    <ClerkProvider publishableKey={authConfig.clerk_publishable_key}>
      <SignedOut>
        <SignInScreen appName={authConfig.app_name} />
      </SignedOut>
      <SignedIn>
        <AuthenticatedDevHost appName={authConfig.app_name} serverUrl={serverUrl} />
      </SignedIn>
    </ClerkProvider>
  );
}

createRoot(document.getElementById("root")!).render(<BootstrapApp />);
