import { useDeferredValue, useEffect, useEffectEvent, useRef, useState, startTransition, type ChangeEvent } from "react";
import { SignInButton, UserButton, useAuth, useUser } from "@clerk/react";

import {
  ApiError,
  deleteFile,
  getAuthenticatedUser,
  importArxivPaper,
  listFiles,
  searchArxiv,
  setClerkTokenGetter,
  uploadFile,
} from "./api";
import { ChatPane } from "./ChatPane";
import type { ArxivPaperCandidate, AuthUser, FileSummary } from "./types";

const PAGE_SIZE = 30;

export default function App() {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const { user } = useUser();
  const [authenticatedUser, setAuthenticatedUser] = useState<AuthUser | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthHydrating, setIsAuthHydrating] = useState(true);

  useEffect(() => {
    setClerkTokenGetter(async () => (await getToken()) ?? null);
    return () => {
      setClerkTokenGetter(null);
    };
  }, [getToken]);

  const hydrateAuthenticatedUser = useEffectEvent(async () => {
    if (!isLoaded) {
      return;
    }

    if (!isSignedIn) {
      setAuthenticatedUser(null);
      setAuthError(null);
      setIsAuthHydrating(false);
      return;
    }

    setIsAuthHydrating(true);
    try {
      const nextUser = await getAuthenticatedUser();
      setAuthenticatedUser(nextUser);
      setAuthError(null);
    } catch (error) {
      setAuthenticatedUser(null);
      if (error instanceof ApiError) {
        setAuthError(error.message);
      } else if (error instanceof Error) {
        setAuthError(error.message);
      } else {
        setAuthError("We could not verify your authenticated session.");
      }
    } finally {
      setIsAuthHydrating(false);
    }
  });

  useEffect(() => {
    void hydrateAuthenticatedUser();
  }, [isLoaded, isSignedIn]);

  if (!isLoaded || (isSignedIn && isAuthHydrating)) {
    return (
      <div className="screen-shell">
        <div className="status-card">
          <p className="eyebrow">Loading</p>
          <h1>Preparing your file desk</h1>
          <p>Checking your Clerk session, backend access, and ChatKit workspace.</p>
        </div>
      </div>
    );
  }

  if (!isSignedIn) {
    return (
      <div className="screen-shell">
        <div className="status-card">
          <p className="eyebrow">Sign In Required</p>
          <h1>Open the library with your Clerk session</h1>
          <p>
            This workspace uses the same Clerk-backed flow as the companion apps, so the explorer, uploads, and chat agent all stay scoped
            to your account.
          </p>
          <SignInButton mode="modal">
            <button className="primary-button" type="button">
              Sign in
            </button>
          </SignInButton>
        </div>
      </div>
    );
  }

  if (authError) {
    return (
      <div className="screen-shell">
        <div className="status-card">
          <p className="eyebrow">Access Check Failed</p>
          <h1>We could not open the file desk yet</h1>
          <p>{authError}</p>
          <div className="status-actions">
            <button className="primary-button" type="button" onClick={() => void hydrateAuthenticatedUser()}>
              Retry
            </button>
            <UserButton />
          </div>
        </div>
      </div>
    );
  }

  if (!authenticatedUser?.active) {
    return (
      <div className="screen-shell">
        <div className="status-card">
          <p className="eyebrow">Access Pending</p>
          <h1>Your Clerk sign-in worked</h1>
          <p>
            Your account is signed in, but the backend still marks it as inactive. Ask an admin to set the active flag in Clerk private
            metadata, then refresh this page.
          </p>
          <div className="status-actions">
            <button className="primary-button" type="button" onClick={() => void hydrateAuthenticatedUser()}>
              Check again
            </button>
            <UserButton />
          </div>
        </div>
      </div>
    );
  }

  return (
    <Workspace
      userLabel={
        authenticatedUser.display_name ||
        user?.fullName ||
        authenticatedUser.primary_email ||
        user?.primaryEmailAddress?.emailAddress ||
        "Signed-in user"
      }
    />
  );
}

function Workspace({ userLabel }: { userLabel: string }) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [files, setFiles] = useState<FileSummary[]>([]);
  const [isArxivPanelOpen, setIsArxivPanelOpen] = useState(false);
  const [arxivQuery, setArxivQuery] = useState("");
  const [arxivResults, setArxivResults] = useState<ArxivPaperCandidate[]>([]);
  const [isArxivSearching, setIsArxivSearching] = useState(false);
  const [isArxivImportingId, setIsArxivImportingId] = useState<string | null>(null);
  const [arxivErrorMessage, setArxivErrorMessage] = useState<string | null>(null);
  const [selectedFileIds, setSelectedFileIds] = useState<string[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [isLibraryLoading, setIsLibraryLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadLibrary = useEffectEvent(async () => {
    setIsLibraryLoading(true);
    setErrorMessage(null);
    try {
      const response = await listFiles({
        query: deferredQuery || undefined,
        sort: "newest",
        page,
        pageSize: PAGE_SIZE,
      });
      setFiles(response.files);
      setTotalCount(response.total_count);
      setSelectedFileIds((current) => current.filter((fileId) => response.files.some((file) => file.id === fileId)));
    } catch (error) {
      setFiles([]);
      setTotalCount(0);
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Could not load your files.");
      }
    } finally {
      setIsLibraryLoading(false);
    }
  });

  useEffect(() => {
    void loadLibrary();
  }, [deferredQuery, page]);

  const refreshLibraryAfterMutation = useEffectEvent(async () => {
    if (page !== 1) {
      setPage(1);
      return;
    }
    await loadLibrary();
  });

  function toggleSelectedFile(fileId: string): void {
    setSelectedFileIds((current) => (current.includes(fileId) ? current.filter((value) => value !== fileId) : [...current, fileId]));
  }

  async function handleUploadSelection(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const chosenFiles = Array.from(event.currentTarget.files ?? []);
    if (!chosenFiles.length) {
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);
    try {
      for (const file of chosenFiles) {
        await uploadFile(file, []);
      }
      await refreshLibraryAfterMutation();
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Upload failed.");
      }
    } finally {
      setIsUploading(false);
      event.currentTarget.value = "";
    }
  }

  async function handleDelete(file: FileSummary): Promise<void> {
    const confirmed = window.confirm(`Delete "${file.display_title}" from the library?`);
    if (!confirmed) {
      return;
    }
    setErrorMessage(null);
    try {
      await deleteFile(file.id);
      setSelectedFileIds((current) => current.filter((value) => value !== file.id));
      await refreshLibraryAfterMutation();
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Delete failed.");
      }
    }
  }

  async function handleArxivSearch(): Promise<void> {
    const normalizedQuery = arxivQuery.trim();
    if (!normalizedQuery) {
      setArxivResults([]);
      setArxivErrorMessage(null);
      return;
    }

    setIsArxivSearching(true);
    setArxivErrorMessage(null);
    try {
      const response = await searchArxiv(normalizedQuery);
      setArxivResults(response.results);
    } catch (error) {
      setArxivResults([]);
      if (error instanceof ApiError) {
        setArxivErrorMessage(error.message);
      } else {
        setArxivErrorMessage("Could not search arXiv right now.");
      }
    } finally {
      setIsArxivSearching(false);
    }
  }

  async function handleArxivImport(paper: ArxivPaperCandidate): Promise<void> {
    setIsArxivImportingId(paper.arxiv_id);
    setArxivErrorMessage(null);
    setErrorMessage(null);
    try {
      await importArxivPaper(paper);
      await refreshLibraryAfterMutation();
      setIsArxivPanelOpen(false);
      setArxivQuery("");
      setArxivResults([]);
    } catch (error) {
      if (error instanceof ApiError) {
        setArxivErrorMessage(error.message);
      } else {
        setArxivErrorMessage("Could not import that arXiv paper.");
      }
    } finally {
      setIsArxivImportingId(null);
    }
  }

  const selectedCount = selectedFileIds.length;

  return (
    <div className="app-shell">
      <div className="workspace">
        <aside className="explorer-panel">
          <div className="explorer-header">
            <div>
              <p className="eyebrow">File Desk</p>
              <h1>Explorer</h1>
            </div>
            <div className="explorer-actions">
              <button
                className={isArxivPanelOpen ? "ghost-button ghost-button--active" : "ghost-button"}
                type="button"
                onClick={() => {
                  setIsArxivPanelOpen((current) => !current);
                  setArxivErrorMessage(null);
                }}
              >
                {isArxivPanelOpen ? "Hide arXiv" : "Add arXiv"}
              </button>
              <button className="ghost-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
                {isUploading ? "Uploading..." : "Upload files"}
              </button>
              <UserButton />
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden-input"
            multiple
            onChange={(event) => {
              void handleUploadSelection(event);
            }}
          />

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

          {isArxivPanelOpen ? (
            <section className="explorer-card arxiv-panel">
              <div className="section-heading section-heading--tight">
                <div>
                  <h2>Add from arXiv</h2>
                  <p>Search arxiv.org, then import the PDF into your library.</p>
                </div>
              </div>

              <form
                className="arxiv-search-row"
                onSubmit={(event) => {
                  event.preventDefault();
                  void handleArxivSearch();
                }}
              >
                <input
                  className="search-input search-input--flush"
                  type="search"
                  value={arxivQuery}
                  onChange={(event) => setArxivQuery(event.target.value)}
                  placeholder="Find papers on arxiv.org"
                />
                <button className="ghost-button ghost-button--compact" type="submit" disabled={isArxivSearching}>
                  {isArxivSearching ? "Searching..." : "Search"}
                </button>
              </form>

              {arxivErrorMessage ? <div className="error-banner error-banner--compact">{arxivErrorMessage}</div> : null}

              {arxivResults.length ? (
                <div className="arxiv-results">
                  {arxivResults.map((paper) => {
                    const isImporting = isArxivImportingId === paper.arxiv_id;
                    return (
                      <article key={paper.arxiv_id} className="arxiv-result">
                        <div className="arxiv-result__main">
                          <div className="arxiv-result__title-row">
                            <h3>{paper.title}</h3>
                            <span>{paper.arxiv_id}</span>
                          </div>
                          {paper.authors.length ? <p className="arxiv-result__authors">{paper.authors.join(", ")}</p> : null}
                          {paper.summary ? <p className="arxiv-result__summary">{paper.summary}</p> : null}
                          <div className="arxiv-result__links">
                            <a href={paper.abs_url} target="_blank" rel="noreferrer">
                              Abstract
                            </a>
                            <a href={paper.pdf_url} target="_blank" rel="noreferrer">
                              PDF
                            </a>
                          </div>
                        </div>
                        <button
                          className="primary-button primary-button--compact"
                          type="button"
                          disabled={isImporting}
                          onClick={() => {
                            void handleArxivImport(paper);
                          }}
                        >
                          {isImporting ? "Importing..." : "Import"}
                        </button>
                      </article>
                    );
                  })}
                </div>
              ) : arxivQuery.trim() && !isArxivSearching && !arxivErrorMessage ? (
                <div className="empty-state empty-state--compact">
                  <p>No arXiv matches yet. Try a broader topic or author name.</p>
                </div>
              ) : null}
            </section>
          ) : null}

          <div className="explorer-toolbar">
            <div className="section-heading">
              <div>
                <h2>Files</h2>
                <p>{isLibraryLoading ? "Loading..." : `${totalCount} file${totalCount === 1 ? "" : "s"} in your library`}</p>
              </div>
              <div className="section-heading__actions">
                <div className="selection-pill">{selectedCount} in chat</div>
              </div>
            </div>
            <input
              id="file-search"
              className="search-input search-input--flush"
              type="search"
              value={query}
              onChange={(event) => {
                startTransition(() => {
                  setPage(1);
                  setQuery(event.target.value);
                });
              }}
              placeholder="Filter by filename, type, or tag"
            />
          </div>

          <section className="explorer-card explorer-card--stretch">
            <div className="file-list-toolbar">
              <span>{isLibraryLoading ? "Loading files..." : `Showing ${files.length} of ${totalCount}`}</span>
              <span>Newest first</span>
            </div>

            <div className="file-table">
              <div className="file-table__header file-table__grid" aria-hidden="true">
                <span>Chat</span>
                <span>Filename</span>
                <span>Size</span>
                <span>Added</span>
                <span>Type</span>
                <span></span>
              </div>

              <div className="file-list">
                {files.map((file) => {
                  const isSelected = selectedFileIds.includes(file.id);
                  return (
                    <article key={file.id} className={isSelected ? "file-row file-row--selected" : "file-row"}>
                      <div className="file-table__grid">
                        <button
                          className={isSelected ? "select-toggle select-toggle--active" : "select-toggle"}
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleSelectedFile(file.id);
                          }}
                        >
                          {isSelected ? "On" : "Add"}
                        </button>
                        <div className="file-row__name" title={file.original_filename}>
                          <span className="file-row__filename">{file.original_filename}</span>
                          {file.status !== "ready" ? (
                            <span className={`status-badge status-badge--${file.status}`}>{file.status}</span>
                          ) : null}
                        </div>
                        <span className="file-cell file-cell--numeric" title={formatBytes(file.byte_size)}>
                          {formatBytes(file.byte_size)}
                        </span>
                        <span className="file-cell" title={formatDate(file.created_at)}>
                          {formatTableDate(file.created_at)}
                        </span>
                        <span className="file-cell" title={file.media_type}>
                          {formatTypeLabel(file)}
                        </span>
                        <button
                          className="delete-link delete-link--table"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            void handleDelete(file);
                          }}
                        >
                          Del
                        </button>
                      </div>
                    </article>
                  );
                })}
                {!isLibraryLoading && !files.length ? (
                  <div className="empty-state">
                    <h3>No matching files yet</h3>
                    <p>Upload a few files or clear your filters to widen the library view.</p>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="pagination-row">
              <button
                className="ghost-button"
                type="button"
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={page === 1 || isLibraryLoading}
              >
                Previous
              </button>
              <span>Page {page}</span>
              <button
                className="ghost-button"
                type="button"
                onClick={() => setPage((current) => current + 1)}
                disabled={isLibraryLoading || page * PAGE_SIZE >= totalCount}
              >
                Next
              </button>
            </div>
          </section>
        </aside>

        <main className="chat-panel">
          <div className="chat-header">
            <div>
              <p className="eyebrow">Assistant</p>
              <h2>{userLabel}</h2>
              <p className="panel-copy">
                The chat can see the files you explicitly select in the explorer, and it can widen out to the full MCP-backed library when
                needed.
              </p>
            </div>
            <div className="chat-summary">
              <span>{selectedCount}</span>
              <small>files in current chat context</small>
            </div>
          </div>
          <ChatPane selectedFileIds={selectedFileIds} activeThreadId={activeThreadId} onActiveThreadIdChange={setActiveThreadId} />
        </main>
      </div>
    </div>
  );
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
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function formatTableDate(value: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function formatTypeLabel(file: FileSummary): string {
  const lastDot = file.original_filename.lastIndexOf(".");
  if (lastDot > -1 && lastDot < file.original_filename.length - 1) {
    const extension = file.original_filename.slice(lastDot + 1).trim();
    if (extension && extension.length <= 5) {
      return extension.toUpperCase();
    }
  }

  const slashIndex = file.media_type.indexOf("/");
  if (slashIndex > -1 && slashIndex < file.media_type.length - 1) {
    return file.media_type.slice(slashIndex + 1).toUpperCase();
  }

  return file.source_kind.toUpperCase();
}
