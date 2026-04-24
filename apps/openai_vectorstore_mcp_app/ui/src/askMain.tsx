import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { DocumentAskRoot } from "./App";

createRoot(document.getElementById("document-ask-root")!).render(
  <StrictMode>
    <DocumentAskRoot />
  </StrictMode>,
);
