import "@mantine/core/styles.css";
import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";

createRoot(document.getElementById("vector-store-console-root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
