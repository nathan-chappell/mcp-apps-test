import { useApp } from "@modelcontextprotocol/ext-apps/react";
import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type ReactElement,
} from "react";

import {
  createAskHostBridge,
  createLibraryHostBridge,
  type DocumentLibraryBridge,
} from "./bridge";
import { createMockBridge } from "./mockBridge";
import type {
  DocumentAskResult,
  DocumentCitation,
  DocumentDetail,
  DocumentFilters,
  DocumentLibraryQueryResult,
  DocumentLibraryState,
  DocumentLibraryStateResult,
  DocumentSearchHit,
  KnowledgeTagSummary,
  QueryDocumentLibraryArguments,
} from "./types";
import {
  isDocumentLibraryQueryResult,
  isDocumentLibraryStateResult,
} from "./types";

const APP_INFO = {
  name: "Document Library",
  version: "1.0.0",
};

type EditableFilters = {
  tag_ids: string[];
  tag_match_mode: "all" | "any";
  filename_query: string;
  created_from: string;
  created_to: string;
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

function excerpt(value: string, limit = 280): string {
  return value.length > limit ? `${value.slice(0, limit - 3)}...` : value;
}

function filtersFromState(state: DocumentLibraryState | null): EditableFilters {
  const filters = state?.filters;
  return {
    tag_ids: filters?.tag_ids ?? [],
    tag_match_mode: filters?.tag_match_mode ?? "all",
    filename_query: filters?.filename_query ?? "",
    created_from: filters?.created_from ?? "",
    created_to: filters?.created_to ?? "",
  };
}

function buildStateArgs(filters: EditableFilters, detailDocumentId?: string): {
  tag_ids: string[];
  tag_match_mode: "all" | "any";
  filename_query?: string;
  created_from?: string;
  created_to?: string;
  detail_document_id?: string;
} {
  return {
    tag_ids: filters.tag_ids,
    tag_match_mode: filters.tag_match_mode,
    ...(filters.filename_query.trim()
      ? { filename_query: filters.filename_query.trim() }
      : {}),
    ...(filters.created_from ? { created_from: filters.created_from } : {}),
    ...(filters.created_to ? { created_to: filters.created_to } : {}),
    ...(detailDocumentId ? { detail_document_id: detailDocumentId } : {}),
  };
}

function buildQueryArgs(
  filters: EditableFilters,
  query: string,
  mode: "search" | "ask",
): QueryDocumentLibraryArguments {
  return {
    query,
    mode,
    ...buildStateArgs(filters),
  };
}

function syncTheme(theme?: "light" | "dark"): void {
  document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
}

function toggleId(current: string[], id: string): string[] {
  return current.includes(id)
    ? current.filter((currentId) => currentId !== id)
    : [...current, id];
}

function EmptyState(props: {
  title: string;
  body: string;
  loading?: boolean;
}): ReactElement {
  return (
    <div className="empty-state">
      {props.loading ? <div className="spinner" /> : null}
      <h2>{props.title}</h2>
      <p>{props.body}</p>
    </div>
  );
}

function StatusBanner(props: {
  tone: "neutral" | "success" | "danger";
  message: string;
}): ReactElement {
  return <div className={`status-banner tone-${props.tone}`}>{props.message}</div>;
}

function FilterPanel(props: {
  title: string;
  tags: KnowledgeTagSummary[];
  filters: EditableFilters;
  setFilters: React.Dispatch<React.SetStateAction<EditableFilters>>;
  onApply: () => Promise<void>;
  onClear: () => Promise<void>;
  busy: boolean;
}): ReactElement {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Filters</p>
          <h2>{props.title}</h2>
        </div>
        <div className="segmented-control">
          <button
            type="button"
            className={props.filters.tag_match_mode === "all" ? "active" : ""}
            onClick={() =>
              props.setFilters((current) => ({ ...current, tag_match_mode: "all" }))
            }
          >
            Match all
          </button>
          <button
            type="button"
            className={props.filters.tag_match_mode === "any" ? "active" : ""}
            onClick={() =>
              props.setFilters((current) => ({ ...current, tag_match_mode: "any" }))
            }
          >
            Match any
          </button>
        </div>
      </div>

      <div className="filter-grid">
        <label className="field">
          <span>Filename</span>
          <input
            type="text"
            value={props.filters.filename_query}
            onChange={(event) =>
              props.setFilters((current) => ({
                ...current,
                filename_query: event.target.value,
              }))
            }
            placeholder="invoice, runbook, notes..."
          />
        </label>
        <label className="field">
          <span>Created from</span>
          <input
            type="date"
            value={props.filters.created_from}
            onChange={(event) =>
              props.setFilters((current) => ({
                ...current,
                created_from: event.target.value,
              }))
            }
          />
        </label>
        <label className="field">
          <span>Created to</span>
          <input
            type="date"
            value={props.filters.created_to}
            onChange={(event) =>
              props.setFilters((current) => ({
                ...current,
                created_to: event.target.value,
              }))
            }
          />
        </label>
      </div>

      <div className="tag-picker">
        {props.tags.length === 0 ? <p className="muted">No tags yet.</p> : null}
        {props.tags.map((tag) => {
          const active = props.filters.tag_ids.includes(tag.id);
          return (
            <button
              key={tag.id}
              type="button"
              className={`tag-pill ${active ? "active" : ""}`}
              onClick={() =>
                props.setFilters((current) => ({
                  ...current,
                  tag_ids: toggleId(current.tag_ids, tag.id),
                }))
              }
            >
              <span
                className="tag-dot"
                style={{ backgroundColor: tag.color ?? "#0f766e" }}
              />
              <span>{tag.name}</span>
              <strong>{tag.node_count}</strong>
            </button>
          );
        })}
      </div>

      <div className="action-row">
        <button type="button" className="primary-button" onClick={props.onApply} disabled={props.busy}>
          {props.busy ? "Loading..." : "Apply filters"}
        </button>
        <button type="button" className="ghost-button" onClick={props.onClear} disabled={props.busy}>
          Clear
        </button>
      </div>
    </section>
  );
}

function DocumentList(props: {
  library: DocumentLibraryState;
  selectedDocumentId: string | null;
  onSelect: (documentId: string) => Promise<void>;
  busy: boolean;
}): ReactElement {
  if (props.library.documents.length === 0) {
    return (
      <section className="panel">
        <p className="eyebrow">Documents</p>
        <EmptyState
          title="No matching documents"
          body="Try clearing filters or upload a new file."
        />
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Documents</p>
          <h2>{props.library.library.filtered_document_count} in view</h2>
        </div>
        <p className="muted">
          {props.library.library.document_count} total in the library
        </p>
      </div>
      <div className="document-list">
        {props.library.documents.map((document) => {
          const active = props.selectedDocumentId === document.id;
          return (
            <button
              key={document.id}
              type="button"
              className={`document-card ${active ? "active" : ""}`}
              onClick={() => props.onSelect(document.id)}
              disabled={props.busy}
            >
              <div className="document-card-top">
                <h3>{document.title}</h3>
                <span className={`status-chip status-${document.status}`}>
                  {document.status}
                </span>
              </div>
              <p className="filename">{document.original_filename}</p>
              <p className="muted">
                {formatDate(document.created_at)} • {formatBytes(document.byte_size)}
              </p>
              <div className="tag-inline-row">
                {document.tags.map((tag) => (
                  <span key={tag.id} className="mini-tag">
                    {tag.name}
                  </span>
                ))}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function DocumentDetailPanel(props: {
  detail: DocumentDetail | null;
  allTags: KnowledgeTagSummary[];
  editingTagIds: string[];
  setEditingTagIds: React.Dispatch<React.SetStateAction<string[]>>;
  onSaveTags: () => Promise<void>;
  busy: boolean;
}): ReactElement {
  if (!props.detail) {
    return (
      <section className="panel">
        <p className="eyebrow">Detail</p>
        <EmptyState
          title="Pick a document"
          body="Select a document from the list to review metadata and edit tags."
        />
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Detail</p>
          <h2>{props.detail.title}</h2>
        </div>
        {props.detail.download_url ? (
          <a className="ghost-button" href={props.detail.download_url} target="_blank" rel="noreferrer">
            Download
          </a>
        ) : null}
      </div>

      <dl className="metadata-grid">
        <div>
          <dt>Filename</dt>
          <dd>{props.detail.original_filename}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{formatDate(props.detail.created_at)}</dd>
        </div>
        <div>
          <dt>Type</dt>
          <dd>{props.detail.media_type}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{formatBytes(props.detail.byte_size)}</dd>
        </div>
      </dl>

      <div className="editor-block">
        <div className="panel-header compact">
          <h3>Tags</h3>
          <button type="button" className="primary-button" onClick={props.onSaveTags} disabled={props.busy}>
            Save tags
          </button>
        </div>
        <div className="tag-picker">
          {props.allTags.map((tag) => {
            const active = props.editingTagIds.includes(tag.id);
            return (
              <button
                key={tag.id}
                type="button"
                className={`tag-pill ${active ? "active" : ""}`}
                onClick={() =>
                  props.setEditingTagIds((current) => toggleId(current, tag.id))
                }
              >
                <span
                  className="tag-dot"
                  style={{ backgroundColor: tag.color ?? "#0f766e" }}
                />
                <span>{tag.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="artifact-stack">
        {props.detail.derived_artifacts.length === 0 ? (
          <p className="muted">No derived text preview is available yet.</p>
        ) : null}
        {props.detail.derived_artifacts.map((artifact) => (
          <article key={artifact.id} className="artifact-card">
            <div className="panel-header compact">
              <h3>{artifact.kind}</h3>
              <span className="muted">{formatDate(artifact.updated_at)}</span>
            </div>
            <pre>{excerpt(artifact.text_content, 1200)}</pre>
          </article>
        ))}
      </div>
    </section>
  );
}

function UploadPanel(props: {
  allTags: KnowledgeTagSummary[];
  uploadTagIds: string[];
  setUploadTagIds: React.Dispatch<React.SetStateAction<string[]>>;
  uploadFile: File | null;
  setUploadFile: React.Dispatch<React.SetStateAction<File | null>>;
  newTagName: string;
  setNewTagName: React.Dispatch<React.SetStateAction<string>>;
  newTagColor: string;
  setNewTagColor: React.Dispatch<React.SetStateAction<string>>;
  onCreateTag: () => Promise<void>;
  onUpload: () => Promise<void>;
  busy: boolean;
}): ReactElement {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Ingest</p>
          <h2>Upload and tag</h2>
        </div>
      </div>

      <label className="field">
        <span>Choose file</span>
        <input
          type="file"
          onChange={(event: ChangeEvent<HTMLInputElement>) =>
            props.setUploadFile(event.target.files?.[0] ?? null)
          }
        />
      </label>

      <div className="tag-picker">
        {props.allTags.map((tag) => {
          const active = props.uploadTagIds.includes(tag.id);
          return (
            <button
              key={tag.id}
              type="button"
              className={`tag-pill ${active ? "active" : ""}`}
              onClick={() =>
                props.setUploadTagIds((current) => toggleId(current, tag.id))
              }
            >
              <span
                className="tag-dot"
                style={{ backgroundColor: tag.color ?? "#0f766e" }}
              />
              <span>{tag.name}</span>
            </button>
          );
        })}
      </div>

      <div className="action-row">
        <button type="button" className="primary-button" onClick={props.onUpload} disabled={props.busy || !props.uploadFile}>
          {props.busy ? "Uploading..." : "Upload document"}
        </button>
      </div>

      <div className="divider" />

      <div className="panel-header compact">
        <h3>Create tag</h3>
      </div>
      <div className="filter-grid">
        <label className="field">
          <span>Name</span>
          <input
            type="text"
            value={props.newTagName}
            onChange={(event) => props.setNewTagName(event.target.value)}
            placeholder="research, invoices, sprint-12"
          />
        </label>
        <label className="field">
          <span>Color</span>
          <input
            type="color"
            value={props.newTagColor}
            onChange={(event) => props.setNewTagColor(event.target.value)}
          />
        </label>
      </div>
      <div className="action-row">
        <button type="button" className="ghost-button" onClick={props.onCreateTag} disabled={props.busy || !props.newTagName.trim()}>
          Create tag
        </button>
      </div>
    </section>
  );
}

function SearchResultPanel(props: {
  hits: DocumentSearchHit[];
}): ReactElement {
  if (props.hits.length === 0) {
    return (
      <section className="panel">
        <p className="eyebrow">Results</p>
        <EmptyState
          title="No matches"
          body="Try broadening the filters or changing the query."
        />
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Results</p>
          <h2>{props.hits.length} matching documents</h2>
        </div>
      </div>
      <div className="result-list">
        {props.hits.map((hit) => (
          <article key={`${hit.document_id}:${hit.openai_file_id}`} className="result-card">
            <div className="panel-header compact">
              <div>
                <h3>{hit.document_title}</h3>
                <p className="filename">{hit.original_filename}</p>
              </div>
              <span className="score-chip">{Math.round(hit.score * 100)}%</span>
            </div>
            <p>{excerpt(hit.text)}</p>
            <div className="tag-inline-row">
              {hit.tags.map((tag) => (
                <span key={`${hit.document_id}:${tag}`} className="mini-tag">
                  {tag}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function CitationList(props: { citations: DocumentCitation[] }): ReactElement | null {
  if (props.citations.length === 0) {
    return null;
  }
  return (
    <section className="panel">
      <div className="panel-header compact">
        <h3>Citations</h3>
      </div>
      <div className="citation-list">
        {props.citations.map((citation, index) => (
          <article key={`${citation.label}:${index}`} className="citation-card">
            <strong>{citation.label}</strong>
            {citation.original_filename ? (
              <p className="filename">{citation.original_filename}</p>
            ) : null}
            {citation.quote ? <p>{excerpt(citation.quote, 220)}</p> : null}
            {citation.url ? (
              <a href={citation.url} target="_blank" rel="noreferrer">
                {citation.url}
              </a>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function LibraryScreen(props: {
  bridge: DocumentLibraryBridge;
  initialState: DocumentLibraryStateResult;
}): ReactElement {
  const [stateResult, setStateResult] = useState<DocumentLibraryStateResult>(
    props.initialState,
  );
  const [filters, setFilters] = useState<EditableFilters>(
    filtersFromState(props.initialState.document_library_state.library),
  );
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(
    props.initialState.document_detail?.id ?? null,
  );
  const [editingTagIds, setEditingTagIds] = useState<string[]>(
    props.initialState.document_detail?.tags.map((tag) => tag.id) ?? [],
  );
  const [uploadTagIds, setUploadTagIds] = useState<string[]>([]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [newTagName, setNewTagName] = useState("");
  const [newTagColor, setNewTagColor] = useState("#0f766e");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const library = stateResult.document_library_state.library;
  const detail = stateResult.document_detail;
  const tags = library?.tags ?? [];

  useEffect(() => {
    if (detail) {
      setSelectedDocumentId(detail.id);
      setEditingTagIds(detail.tags.map((tag) => tag.id));
    }
  }, [detail]);

  async function loadState(detailDocumentId?: string): Promise<void> {
    setBusy(true);
    setErrorMessage(null);
    try {
      const payload = await props.bridge.get_document_library_state(
        buildStateArgs(filters, detailDocumentId),
      );
      setStateResult(payload);
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function applyFilters(): Promise<void> {
    await loadState(selectedDocumentId ?? undefined);
  }

  async function clearFilters(): Promise<void> {
    const cleared: EditableFilters = {
      tag_ids: [],
      tag_match_mode: "all",
      filename_query: "",
      created_from: "",
      created_to: "",
    };
    setFilters(cleared);
    setSelectedDocumentId(null);
    setBusy(true);
    setErrorMessage(null);
    try {
      const payload = await props.bridge.get_document_library_state(
        buildStateArgs(cleared),
      );
      setStateResult(payload);
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function selectDocument(documentId: string): Promise<void> {
    setSelectedDocumentId(documentId);
    await loadState(documentId);
  }

  async function saveDocumentTags(): Promise<void> {
    if (!selectedDocumentId) {
      return;
    }
    setBusy(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      await props.bridge.update_document_library({
        action: "set_document_tags",
        document_id: selectedDocumentId,
        tag_ids: editingTagIds,
      });
      await loadState(selectedDocumentId);
      setStatusMessage("Document tags updated.");
    } catch (error) {
      setErrorMessage(String(error));
      setBusy(false);
    }
  }

  async function createTag(): Promise<void> {
    setBusy(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      await props.bridge.update_document_library({
        action: "create_tag",
        name: newTagName.trim(),
        color: newTagColor,
      });
      setNewTagName("");
      await loadState(selectedDocumentId ?? undefined);
      setStatusMessage("Tag created.");
    } catch (error) {
      setErrorMessage(String(error));
      setBusy(false);
    }
  }

  async function uploadDocument(): Promise<void> {
    if (!uploadFile) {
      return;
    }
    setBusy(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const payload = await props.bridge.upload_document({
        tag_ids: uploadTagIds,
        file: uploadFile,
      });
      setUploadFile(null);
      setUploadTagIds([]);
      await loadState(payload.document.id);
      setStatusMessage("Document uploaded.");
    } catch (error) {
      setErrorMessage(String(error));
      setBusy(false);
    }
  }

  if (!library) {
    return (
      <div className="page-shell">
        <EmptyState
          title="No library state"
          body={stateResult.document_library_state.access.message}
        />
      </div>
    );
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Document Library</p>
          <h1>{library.library.title}</h1>
          <p className="hero-copy">
            Browse files, review metadata, manage tags, and upload new source
            material without the old graph overhead.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-chip">
            <strong>{library.library.filtered_document_count}</strong>
            <span>In view</span>
          </div>
          <div className="stat-chip">
            <strong>{library.library.tag_count}</strong>
            <span>Tags</span>
          </div>
        </div>
      </header>

      {statusMessage ? (
        <StatusBanner tone="success" message={statusMessage} />
      ) : null}
      {errorMessage ? <StatusBanner tone="danger" message={errorMessage} /> : null}

      <div className="main-grid">
        <div className="main-column">
          <FilterPanel
            title="Filter by tag, filename, and created date"
            tags={tags}
            filters={filters}
            setFilters={setFilters}
            onApply={applyFilters}
            onClear={clearFilters}
            busy={busy}
          />
          <DocumentList
            library={library}
            selectedDocumentId={selectedDocumentId}
            onSelect={selectDocument}
            busy={busy}
          />
        </div>
        <div className="side-column">
          <DocumentDetailPanel
            detail={detail}
            allTags={tags}
            editingTagIds={editingTagIds}
            setEditingTagIds={setEditingTagIds}
            onSaveTags={saveDocumentTags}
            busy={busy}
          />
          <UploadPanel
            allTags={tags}
            uploadTagIds={uploadTagIds}
            setUploadTagIds={setUploadTagIds}
            uploadFile={uploadFile}
            setUploadFile={setUploadFile}
            newTagName={newTagName}
            setNewTagName={setNewTagName}
            newTagColor={newTagColor}
            setNewTagColor={setNewTagColor}
            onCreateTag={createTag}
            onUpload={uploadDocument}
            busy={busy}
          />
        </div>
      </div>
    </div>
  );
}

function AskScreen(props: {
  bridge: DocumentLibraryBridge;
  initialResult: DocumentLibraryQueryResult;
}): ReactElement {
  const [queryResult, setQueryResult] = useState<DocumentLibraryQueryResult>(
    props.initialResult,
  );
  const [filters, setFilters] = useState<EditableFilters>(
    filtersFromState(props.initialResult.document_library_state.library),
  );
  const [query, setQuery] = useState(
    props.initialResult.ask_result?.query ??
      props.initialResult.search_result?.query ??
      "",
  );
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const library = queryResult.document_library_state.library;
  const tags = library?.tags ?? [];
  const hits = queryResult.search_result?.hits ?? queryResult.ask_result?.hits ?? [];
  const citations = queryResult.ask_result?.citations ?? [];

  async function refreshState(): Promise<void> {
    setBusy(true);
    setErrorMessage(null);
    try {
      const payload = await props.bridge.get_document_library_state(
        buildStateArgs(filters),
      );
      setQueryResult((current) => ({
        ...current,
        document_library_state: payload.document_library_state,
      }));
      setStatusMessage("Filters updated.");
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function clearFilters(): Promise<void> {
    const cleared: EditableFilters = {
      tag_ids: [],
      tag_match_mode: "all",
      filename_query: "",
      created_from: "",
      created_to: "",
    };
    setFilters(cleared);
    setBusy(true);
    setErrorMessage(null);
    try {
      const payload = await props.bridge.get_document_library_state(
        buildStateArgs(cleared),
      );
      setQueryResult((current) => ({
        ...current,
        document_library_state: payload.document_library_state,
      }));
      setStatusMessage("Filters cleared.");
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function runQuery(mode: "search" | "ask"): Promise<void> {
    if (!query.trim()) {
      setErrorMessage("Enter a query first.");
      return;
    }

    setBusy(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await props.bridge.query_document_library(
        buildQueryArgs(filters, query.trim(), mode),
      );
      setQueryResult(payload);
      setStatusMessage(mode === "ask" ? "Grounded answer ready." : "Search complete.");
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      setBusy(false);
    }
  }

  if (!library) {
    return (
      <div className="page-shell">
        <EmptyState
          title="No library state"
          body={queryResult.document_library_state.access.message}
        />
      </div>
    );
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Document Ask</p>
          <h1>Search or ask with the same filters</h1>
          <p className="hero-copy">
            Narrow the corpus with tags, filename, and created date, then choose
            between a direct search result list or a grounded answer.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-chip">
            <strong>{library.library.filtered_document_count}</strong>
            <span>Eligible docs</span>
          </div>
          <div className="stat-chip">
            <strong>{hits.length}</strong>
            <span>Current hits</span>
          </div>
        </div>
      </header>

      {statusMessage ? (
        <StatusBanner tone="success" message={statusMessage} />
      ) : null}
      {errorMessage ? <StatusBanner tone="danger" message={errorMessage} /> : null}

      <div className="main-grid ask-layout">
        <div className="main-column">
          <FilterPanel
            title="Keep one filter model for search and ask"
            tags={tags}
            filters={filters}
            setFilters={setFilters}
            onApply={refreshState}
            onClear={clearFilters}
            busy={busy}
          />

          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Query</p>
                <h2>Run search or ask</h2>
              </div>
            </div>
            <label className="field">
              <span>Question or search text</span>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                rows={5}
                placeholder="What documents mention tag filtering for MVP search?"
              />
            </label>
            <div className="action-row">
              <button type="button" className="primary-button" onClick={() => runQuery("search")} disabled={busy}>
                {busy && queryResult.mode === "search" ? "Searching..." : "Search"}
              </button>
              <button type="button" className="ghost-button" onClick={() => runQuery("ask")} disabled={busy}>
                {busy && queryResult.mode === "ask" ? "Asking..." : "Ask"}
              </button>
            </div>
          </section>

          <SearchResultPanel hits={hits} />
        </div>

        <div className="side-column">
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Answer</p>
                <h2>Grounded response</h2>
              </div>
              <span className="mini-tag">{queryResult.mode}</span>
            </div>
            {queryResult.ask_result ? (
              <>
                <p className="answer-copy">{queryResult.ask_result.answer}</p>
                <p className="muted">
                  Model {queryResult.ask_result.model} • conversation{" "}
                  {queryResult.ask_result.conversation_id}
                </p>
              </>
            ) : (
              <EmptyState
                title="No answer yet"
                body="Run Ask to get a grounded summary with citations."
              />
            )}
          </section>
          <CitationList citations={citations} />
        </div>
      </div>
    </div>
  );
}

function HostedLibraryApp(): ReactElement {
  const [hostContext, setHostContext] = useState<
    { theme?: "light" | "dark" } | undefined
  >();
  const [initialState, setInitialState] =
    useState<DocumentLibraryStateResult | null>(null);
  const { app, error, isConnected } = useApp({
    appInfo: APP_INFO,
    capabilities: {
      availableDisplayModes: ["inline", "fullscreen"],
    },
    onAppCreated: (createdApp) => {
      createdApp.ontoolresult = async (result) => {
        if (isDocumentLibraryStateResult(result.structuredContent)) {
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
    syncTheme(hostContext?.theme);
  }, [hostContext?.theme]);

  const bridge = useMemo(() => {
    if (!app || !initialState) {
      return null;
    }
    return createLibraryHostBridge(app, initialState, app.getHostContext());
  }, [app, initialState]);

  if (error) {
    return (
      <div className="page-shell">
        <EmptyState title="Unable to connect" body={String(error)} />
      </div>
    );
  }

  if (!isConnected || !bridge || !initialState) {
    return (
      <div className="page-shell">
        <EmptyState
          title="Waiting for the MCP host"
          body="Open this app through open_document_library to load the initial state."
          loading
        />
      </div>
    );
  }

  return <LibraryScreen bridge={bridge} initialState={initialState} />;
}

function HostedAskApp(): ReactElement {
  const [hostContext, setHostContext] = useState<
    { theme?: "light" | "dark" } | undefined
  >();
  const [initialResult, setInitialResult] =
    useState<DocumentLibraryQueryResult | null>(null);
  const { app, error, isConnected } = useApp({
    appInfo: APP_INFO,
    capabilities: {
      availableDisplayModes: ["inline", "fullscreen"],
    },
    onAppCreated: (createdApp) => {
      createdApp.ontoolresult = async (result) => {
        if (isDocumentLibraryQueryResult(result.structuredContent)) {
          setInitialResult(result.structuredContent);
        }
      };
      createdApp.onhostcontextchanged = (params) => {
        setHostContext((previous) => ({ ...previous, ...params }));
      };
      createdApp.onerror = console.error;
    },
  });

  useEffect(() => {
    syncTheme(hostContext?.theme);
  }, [hostContext?.theme]);

  const bridge = useMemo(() => {
    if (!app || !initialResult) {
      return null;
    }
    return createAskHostBridge(app, initialResult, app.getHostContext());
  }, [app, initialResult]);

  if (error) {
    return (
      <div className="page-shell">
        <EmptyState title="Unable to connect" body={String(error)} />
      </div>
    );
  }

  if (!isConnected || !bridge || !initialResult) {
    return (
      <div className="page-shell">
        <EmptyState
          title="Waiting for the MCP host"
          body="Open this app through open_document_ask to load the initial state."
          loading
        />
      </div>
    );
  }

  return <AskScreen bridge={bridge} initialResult={initialResult} />;
}

export function DocumentLibraryRoot(): ReactElement {
  if (isStandaloneMode()) {
    syncTheme("light");
    const bridge = createMockBridge();
    return <LibraryScreen bridge={bridge} initialState={bridge.initial_library_state!} />;
  }
  return <HostedLibraryApp />;
}

export function DocumentAskRoot(): ReactElement {
  if (isStandaloneMode()) {
    syncTheme("light");
    const bridge = createMockBridge();
    return <AskScreen bridge={bridge} initialResult={bridge.initial_query_result!} />;
  }
  return <HostedAskApp />;
}
