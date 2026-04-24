import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { DocumentLibraryRoot } from "./App";

createRoot(document.getElementById("document-library-root")!).render(
  <StrictMode>
    <DocumentLibraryRoot />
  </StrictMode>,
);
