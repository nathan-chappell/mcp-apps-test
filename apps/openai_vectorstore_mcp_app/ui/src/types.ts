import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

export interface UserSummary {
  clerk_user_id: string;
  display_name: string;
  primary_email: string | null;
  active: boolean;
  role: string | null;
}

export interface KnowledgeAccessState {
  status: "active" | "pending_access";
  message: string;
  user: UserSummary;
}

export interface KnowledgeTagSummary {
  id: string;
  name: string;
  slug: string;
  color: string | null;
  node_count: number;
}

export interface DerivedArtifactSummary {
  id: string;
  kind: string;
  openai_file_id: string | null;
  text_content: string;
  structured_payload: unknown;
  created_at: string;
  updated_at: string;
}

export interface DocumentFilters {
  tag_ids: string[];
  selected_tag_names: string[];
  tag_match_mode: "all" | "any";
  filename_query: string | null;
  created_from: string | null;
  created_to: string | null;
  matching_document_ids: string[];
}

export interface DocumentSummary {
  id: string;
  title: string;
  original_filename: string;
  media_type: string;
  source_kind: "document" | "audio" | "image" | "video" | "other";
  status: "processing" | "ready" | "failed";
  byte_size: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  tags: KnowledgeTagSummary[];
  derived_kinds: string[];
  openai_original_file_id: string | null;
  download_url: string | null;
}

export interface DocumentDetail extends DocumentSummary {
  original_mime_type: string | null;
  derived_artifacts: DerivedArtifactSummary[];
}

export interface DocumentLibrarySummary {
  id: string;
  title: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  document_count: number;
  filtered_document_count: number;
  tag_count: number;
  vector_store_ready: boolean;
}

export interface DocumentLibraryState {
  library: DocumentLibrarySummary;
  tags: KnowledgeTagSummary[];
  documents: DocumentSummary[];
  filters: DocumentFilters;
}

export interface DocumentLibraryCapabilities {
  upload_url: string;
  upload_token_ttl_seconds: number;
  supports_video_audio_extraction: boolean;
  accepted_hint: string;
}

export interface DocumentLibraryViewState {
  access: KnowledgeAccessState;
  library: DocumentLibraryState | null;
  capabilities: DocumentLibraryCapabilities;
}

export interface DocumentLibraryStateResult {
  document_library_state: DocumentLibraryViewState;
  document_detail: DocumentDetail | null;
}

export interface DocumentSearchHit {
  document_id: string;
  document_title: string;
  original_filename: string;
  derived_artifact_id: string | null;
  openai_file_id: string;
  original_openai_file_id: string | null;
  media_type: string;
  source_kind: string;
  score: number;
  text: string;
  tags: string[];
}

export interface DocumentCitation {
  label: string;
  document_id: string | null;
  document_title: string | null;
  original_filename: string | null;
  quote: string | null;
  url: string | null;
  source: "document_library" | "web";
}

export interface DocumentSearchResult {
  query: string;
  hits: DocumentSearchHit[];
  total_hits: number;
}

export interface DocumentAskResult {
  query: string;
  answer: string;
  model: string;
  conversation_id: string;
  citations: DocumentCitation[];
  hits: DocumentSearchHit[];
}

export interface DocumentLibraryQueryResult {
  mode: "search" | "ask";
  document_library_state: DocumentLibraryViewState;
  search_result: DocumentSearchResult | null;
  ask_result: DocumentAskResult | null;
}

export interface UploadSessionResult {
  upload_url: string;
  upload_token: string;
  expires_at: number;
}

export interface DocumentUploadFinalizeResult {
  document: DocumentSummary;
}

export interface UpdateDocumentLibraryResult {
  action: "prepare_upload" | "create_tag" | "set_document_tags";
  upload_session: UploadSessionResult | null;
  tag: KnowledgeTagSummary | null;
  document: DocumentSummary | null;
}

export interface GetDocumentLibraryStateArguments {
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  filename_query?: string;
  created_from?: string;
  created_to?: string;
  detail_document_id?: string;
}

export interface QueryDocumentLibraryArguments {
  query: string;
  mode?: "search" | "ask";
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  filename_query?: string;
  created_from?: string;
  created_to?: string;
}

export interface UpdateDocumentLibraryArguments {
  action: "prepare_upload" | "create_tag" | "set_document_tags";
  document_id?: string;
  tag_ids?: string[];
  name?: string;
  color?: string;
}

export type ToolResultName =
  | "get_document_library_state"
  | "query_document_library"
  | "update_document_library";

export function getStructuredContent<T>(result: CallToolResult): T {
  if (!("structuredContent" in result) || result.structuredContent == null) {
    throw new Error("Tool result did not include structured content.");
  }
  return result.structuredContent as T;
}

export function isDocumentLibraryStateResult(
  value: unknown,
): value is DocumentLibraryStateResult {
  if (!value || typeof value !== "object") {
    return false;
  }
  return "document_library_state" in value && "document_detail" in value;
}

export function isDocumentLibraryQueryResult(
  value: unknown,
): value is DocumentLibraryQueryResult {
  if (!value || typeof value !== "object") {
    return false;
  }
  return (
    "document_library_state" in value &&
    "mode" in value &&
    "search_result" in value &&
    "ask_result" in value
  );
}
