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

export interface KnowledgeEdgeSummary {
  id: string;
  from_node_id: string;
  to_node_id: string;
  from_node_title: string;
  to_node_title: string;
  label: string;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeNodeSummary {
  id: string;
  display_title: string;
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
  outgoing_edge_count: number;
  incoming_edge_count: number;
}

export interface KnowledgeNodeDetail extends KnowledgeNodeSummary {
  original_mime_type: string | null;
  derived_artifacts: DerivedArtifactSummary[];
  outgoing_edges: KnowledgeEdgeSummary[];
  incoming_edges: KnowledgeEdgeSummary[];
}

export interface KnowledgeBaseSummary {
  id: string;
  title: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  node_count: number;
  tag_count: number;
  edge_count: number;
  vector_store_ready: boolean;
}

export interface KnowledgeBaseContext {
  selected_node_id: string | null;
  graph_selection_mode: "self" | "children" | "descendants";
  tag_ids: string[];
  selected_tag_names: string[];
  tag_match_mode: "all" | "any";
  media_types: string[];
  include_web: boolean;
  rewrite_query: boolean;
  branch_factor: number;
  depth: number;
  max_results: number;
  visible_node_ids: string[];
  scoped_node_ids: string[];
}

export interface KnowledgeBaseState {
  knowledge_base: KnowledgeBaseSummary;
  tags: KnowledgeTagSummary[];
  nodes: KnowledgeNodeSummary[];
  edges: KnowledgeEdgeSummary[];
  context: KnowledgeBaseContext;
}

export interface KnowledgeBaseCapabilities {
  upload_url: string;
  upload_token_ttl_seconds: number;
  confirmation_token_ttl_seconds: number;
  supports_video_audio_extraction: boolean;
  accepted_hint: string;
}

export interface KnowledgeBaseDeskState {
  access: KnowledgeAccessState;
  knowledge_base: KnowledgeBaseState | null;
  capabilities: KnowledgeBaseCapabilities;
}

export interface SearchHit {
  node_id: string;
  node_title: string;
  original_filename: string;
  derived_artifact_id: string | null;
  openai_file_id: string;
  original_openai_file_id: string | null;
  media_type: string;
  source_kind: string;
  score: number;
  text: string;
  tags: string[];
  attributes: Record<string, string | number | boolean> | null;
}

export interface FileSearchCallSummary {
  id: string;
  status: string;
  queries: string[];
  results: SearchHit[];
}

export interface WebSearchCallSummary {
  id: string;
  status: string;
  query: string;
  sources: string[];
}

export interface KnowledgeFileSearchResult {
  knowledge_base_id: string;
  query: string;
  context: KnowledgeBaseContext;
  hits: SearchHit[];
  total_hits: number;
}

export interface BranchSearchNode {
  id: string;
  parent_id: string | null;
  depth: number;
  query: string;
  rationale: string | null;
  hits: SearchHit[];
  children: string[];
}

export interface KnowledgeBranchSearchResult {
  knowledge_base_id: string;
  seed_query: string;
  context: KnowledgeBaseContext;
  nodes: BranchSearchNode[];
  merged_hits: SearchHit[];
}

export interface KnowledgeAnswerCitation {
  source: "knowledge_base" | "web";
  label: string;
  node_id: string | null;
  node_title: string | null;
  original_filename: string | null;
  url: string | null;
  quote: string | null;
}

export interface KnowledgeChatResult {
  knowledge_base_id: string;
  question: string;
  answer: string;
  model: string;
  include_web: boolean;
  conversation_id: string;
  context: KnowledgeBaseContext;
  search_calls: FileSearchCallSummary[];
  web_search_calls: WebSearchCallSummary[];
  citations: KnowledgeAnswerCitation[];
}

export type KnowledgeQueryMode = "qa" | "file_search" | "branch_search";
export type KnowledgeQueryKind =
  | "knowledge_base"
  | "qa"
  | "file_search"
  | "branch_search";

export interface KnowledgeBaseQueryArguments {
  query?: string;
  mode?: KnowledgeQueryMode;
  selected_node_id?: string;
  graph_selection_mode?: "self" | "children" | "descendants";
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  media_types?: string[];
  include_web?: boolean;
  rewrite_query?: boolean;
  branch_factor?: number;
  depth?: number;
  max_results?: number;
}

export interface KnowledgeQueryResult {
  kind: KnowledgeQueryKind;
  knowledge_base_state: KnowledgeBaseDeskState;
  file_search_result: KnowledgeFileSearchResult | null;
  branch_search_result: KnowledgeBranchSearchResult | null;
  chat_result: KnowledgeChatResult | null;
}

export interface KnowledgeBaseInfoArguments {
  selected_node_id?: string;
  graph_selection_mode?: "self" | "children" | "descendants";
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  media_types?: string[];
  include_web?: boolean;
  rewrite_query?: boolean;
  branch_factor?: number;
  depth?: number;
  max_results?: number;
  detail_node_id?: string;
}

export interface KnowledgeInfoResult {
  knowledge_base_state: KnowledgeBaseDeskState;
  node_detail: KnowledgeNodeDetail | null;
}

export type UpdateKnowledgeBaseAction =
  | "prepare_upload"
  | "rename_node"
  | "create_tag"
  | "set_node_tags"
  | "upsert_edge"
  | "delete_edge"
  | "delete_node";

export interface UpdateKnowledgeBaseArguments {
  action: UpdateKnowledgeBaseAction;
  node_id?: string;
  edge_id?: string;
  from_node_id?: string;
  to_node_id?: string;
  tag_ids?: string[];
  title?: string;
  name?: string;
  color?: string;
  label?: string;
}

export interface UploadSessionResult {
  upload_url: string;
  upload_token: string;
  expires_at: number;
}

export interface UpdateKnowledgeBaseResult {
  action: UpdateKnowledgeBaseAction;
  knowledge_base_state: KnowledgeBaseDeskState | null;
  node: KnowledgeNodeSummary | null;
  edge: KnowledgeEdgeSummary | null;
  tag: KnowledgeTagSummary | null;
  deleted_node_id: string | null;
  deleted_edge_id: string | null;
  upload_session: UploadSessionResult | null;
}

export interface UploadFinalizeResult {
  node: KnowledgeNodeSummary;
}

export interface PendingCommandResult {
  token: string;
  prompt: string;
  summary: string;
  expires_at: number;
}

export interface RunKnowledgeBaseCommandArguments {
  command: string;
  selected_node_id?: string;
  graph_selection_mode?: "self" | "children" | "descendants";
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  media_types?: string[];
  include_web?: boolean;
  rewrite_query?: boolean;
  branch_factor?: number;
  depth?: number;
  max_results?: number;
}

export interface ConfirmKnowledgeBaseCommandArguments {
  token: string;
  selected_node_id?: string;
  graph_selection_mode?: "self" | "children" | "descendants";
  tag_ids?: string[];
  tag_match_mode?: "all" | "any";
  media_types?: string[];
  include_web?: boolean;
  rewrite_query?: boolean;
  branch_factor?: number;
  depth?: number;
  max_results?: number;
}

export interface KnowledgeBaseCommandResult {
  status: "executed" | "pending_confirmation" | "rejected";
  message: string;
  action: string | null;
  parser: "agent" | "fallback" | "manual";
  raw_command: string;
  knowledge_base_state: KnowledgeBaseDeskState;
  pending_confirmation: PendingCommandResult | null;
  node: KnowledgeNodeSummary | null;
  edge: KnowledgeEdgeSummary | null;
  tag: KnowledgeTagSummary | null;
}

export type ToolResultName =
  | "query_knowledge_base"
  | "get_knowledge_base_info"
  | "update_knowledge_base"
  | "run_knowledge_base_command"
  | "confirm_knowledge_base_command";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function getStructuredContent<T>(result: CallToolResult): T {
  if (!isRecord(result.structuredContent)) {
    throw new Error("Expected structuredContent in the tool result.");
  }
  return result.structuredContent as T;
}

export function isKnowledgeBaseDeskState(
  value: unknown,
): value is KnowledgeBaseDeskState {
  return (
    isRecord(value) &&
    "access" in value &&
    "capabilities" in value &&
    "knowledge_base" in value
  );
}

export function isKnowledgeQueryResult(
  value: unknown,
): value is KnowledgeQueryResult {
  return isRecord(value) && "kind" in value && "knowledge_base_state" in value;
}
