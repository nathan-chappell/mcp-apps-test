import { useEffect, useMemo, useRef, useState } from "react";
import { ChatKit, type UseChatKitOptions, useChatKit } from "@openai/chatkit-react";

import { authenticatedFetch, getChatKitConfig, setChatKitMetadataGetter } from "./api";

const CHATKIT_MODEL_CHOICES = [
  {
    id: "lightweight",
    label: "Lightweight",
    description: "Fastest pass for quick exploration",
  },
  {
    id: "balanced",
    label: "Balanced",
    description: "Best everyday tradeoff",
  },
  {
    id: "powerful",
    label: "Powerful",
    description: "Best available reasoning",
  },
] as const;

type ChatPaneProps = {
  selectedFileIds: string[];
  activeThreadId: string | null;
  onActiveThreadIdChange: (threadId: string | null) => void;
};

export function ChatPane({ selectedFileIds, activeThreadId, onActiveThreadIdChange }: ChatPaneProps) {
  const initialThreadRef = useRef<string | null>(activeThreadId);
  const threadIdRef = useRef<string | null>(activeThreadId);
  const [chatState, setChatState] = useState<"loading" | "ready" | "error">("loading");
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    setChatKitMetadataGetter(() => ({
      origin: "interactive",
      selected_file_ids: selectedFileIds,
    }));
    return () => {
      setChatKitMetadataGetter(null);
    };
  }, [selectedFileIds]);

  const chatKitConfig = getChatKitConfig();
  useEffect(() => {
    setChatState("loading");
    setChatError(null);
  }, [chatKitConfig.domainKey, chatKitConfig.url]);

  const options = useMemo<UseChatKitOptions>(() => {
    return {
      api: {
        url: chatKitConfig.url,
        domainKey: chatKitConfig.domainKey,
        fetch: authenticatedFetch,
      },
      initialThread: initialThreadRef.current,
      theme: {
        colorScheme: "light",
        radius: "round",
        density: "compact",
        typography: {
          baseSize: 14,
        },
      },
      history: {
        enabled: true,
        showDelete: false,
        showRename: false,
      },
      threadItemActions: {
        feedback: false,
      },
      header: {
        enabled: true,
        title: {
          enabled: true,
          text: "File Desk Chat",
        },
      },
      startScreen: {
        greeting: "Ask about the files on the left, or search across the whole library.",
        prompts: [
          {
            label: "Summarize selected files",
            prompt: "Summarize the files I have selected in the explorer and tell me what stands out.",
            icon: "document",
          },
          {
            label: "Find a document",
            prompt: "Search the library for the most relevant files about the topic I mention and explain why they matter.",
            icon: "bolt",
          },
          {
            label: "Read before answering",
            prompt: "Use the relevant file tools before answering. Quote or reference the files you relied on.",
            icon: "check-circle",
          },
        ],
      },
      composer: {
        placeholder: "Ask the assistant to search, inspect, or explain your files.",
        attachments: {
          enabled: false,
        },
        dictation: {
          enabled: false,
        },
        models: CHATKIT_MODEL_CHOICES.map((choice) => ({
          ...choice,
          default: choice.id === "balanced",
        })),
      },
      onThreadChange: ({ threadId }) => {
        threadIdRef.current = threadId ?? null;
        onActiveThreadIdChange(threadId ?? null);
      },
      onReady: () => {
        setChatState("ready");
        setChatError(null);
      },
      onError: (error) => {
        setChatState("error");
        setChatError(extractChatKitError(error));
      },
    };
  }, [chatKitConfig.domainKey, chatKitConfig.url, onActiveThreadIdChange]);

  const chatKit = useChatKit(options);

  useEffect(() => {
    if (!chatKit) {
      return;
    }
    if (threadIdRef.current === activeThreadId) {
      return;
    }
    threadIdRef.current = activeThreadId;
    void chatKit.setThreadId(activeThreadId ?? null);
  }, [activeThreadId, chatKit]);

  return (
    <div className="chatkit-pane-harness">
      <div className="chatkit-shell">
        <ChatKit control={chatKit.control} className="chatkit-element" />
        {chatState !== "ready" ? (
          <div className="chatkit-overlay">
            <div className="chatkit-placeholder">
              <p className="eyebrow">{chatState === "loading" ? "Connecting" : "Chat unavailable"}</p>
              <h3>{chatState === "loading" ? "Loading the ChatKit workspace" : "ChatKit could not initialize"}</h3>
              <p>
                {chatState === "loading"
                  ? "The assistant surface is wiring up to the authenticated ChatKit endpoint."
                  : (chatError ?? "Check the browser console and the /api/chatkit backend response.")}
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function extractChatKitError(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = error.message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return "The assistant UI did not finish initializing.";
}
