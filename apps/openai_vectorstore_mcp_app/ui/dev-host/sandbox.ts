import type { McpUiResourcePermissions, McpUiSandboxProxyReadyNotification, McpUiSandboxResourceReadyNotification } from "@modelcontextprotocol/ext-apps/app-bridge";
import { buildAllowAttribute } from "@modelcontextprotocol/ext-apps/app-bridge";

const ALLOWED_REFERRER_PATTERN = /^http:\/\/localhost(:|\/|$)/;
const DEV_HOST_PATH = "/";

function guessHostUrl(): string {
  const url = new URL(window.location.href);
  url.pathname = DEV_HOST_PATH;
  url.search = "";
  url.hash = "";
  return url.toString();
}

function renderStandaloneMessage(title: string, detail: string): void {
  document.body.innerHTML = `
    <main style="min-height:100vh;display:grid;place-items:center;padding:24px;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f6f7;color:#182327;">
      <section style="max-width:640px;padding:24px 28px;border:1px solid rgba(24,35,39,0.14);border-radius:16px;background:rgba(255,255,255,0.96);box-shadow:0 18px 48px rgba(20,35,42,0.12);">
        <h1 style="margin:0 0 12px;font-size:28px;line-height:1.1;">${title}</h1>
        <p style="margin:0 0 14px;line-height:1.6;">${detail}</p>
        <p style="margin:0;line-height:1.6;">
          Open the dev host instead:
          <a href="${guessHostUrl()}">${guessHostUrl()}</a>
        </p>
      </section>
    </main>
  `;
}

if (window.self === window.top) {
  renderStandaloneMessage(
    "This sandbox page is internal.",
    "The MCP app sandbox is meant to be loaded by the dev host inside an iframe, not opened directly."
  );
} else if (document.referrer.length === 0) {
  renderStandaloneMessage(
    "Missing host context.",
    "The sandbox could not verify the page that opened it."
  );
} else if (!ALLOWED_REFERRER_PATTERN.test(document.referrer)) {
  renderStandaloneMessage(
    "Unexpected embedding origin.",
    `The sandbox only allows the local dev host, but it was opened from ${document.referrer}.`
  );
} else {
  const expectedHostOrigin = new URL(document.referrer).origin;
  const ownOrigin = window.location.origin;
  const innerFrame = document.createElement("iframe");

  innerFrame.style.width = "100%";
  innerFrame.style.height = "100%";
  innerFrame.style.border = "0";
  innerFrame.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms");
  document.body.appendChild(innerFrame);

  const proxyReadyMethod: McpUiSandboxProxyReadyNotification["method"] = "ui/notifications/sandbox-proxy-ready";
  const resourceReadyMethod: McpUiSandboxResourceReadyNotification["method"] = "ui/notifications/sandbox-resource-ready";

  window.addEventListener("message", (event) => {
    if (event.source === window.parent) {
      if (event.origin !== expectedHostOrigin) {
        console.warn("[sandbox] Ignoring message from unexpected host origin:", event.origin);
        return;
      }

      if (event.data?.method === resourceReadyMethod) {
        const sandboxParameters = event.data.params as {
          html?: string;
          permissions?: McpUiResourcePermissions;
          sandbox?: string;
        };

        if (typeof sandboxParameters.sandbox === "string") {
          innerFrame.setAttribute("sandbox", sandboxParameters.sandbox);
        }

        const allowAttribute = buildAllowAttribute(sandboxParameters.permissions);
        if (allowAttribute) {
          innerFrame.setAttribute("allow", allowAttribute);
        }

        if (typeof sandboxParameters.html === "string") {
          const innerDocument = innerFrame.contentDocument ?? innerFrame.contentWindow?.document;
          if (innerDocument) {
            innerDocument.open();
            innerDocument.write(sandboxParameters.html);
            innerDocument.close();
          } else {
            innerFrame.srcdoc = sandboxParameters.html;
          }
        }

        return;
      }

      innerFrame.contentWindow?.postMessage(event.data, "*");
      return;
    }

    if (event.source === innerFrame.contentWindow) {
      if (event.origin !== ownOrigin) {
        console.warn("[sandbox] Ignoring message from unexpected inner origin:", event.origin);
        return;
      }

      window.parent.postMessage(event.data, expectedHostOrigin);
    }
  });

  window.parent.postMessage(
    {
      jsonrpc: "2.0",
      method: proxyReadyMethod,
      params: {},
    },
    expectedHostOrigin,
  );
}
