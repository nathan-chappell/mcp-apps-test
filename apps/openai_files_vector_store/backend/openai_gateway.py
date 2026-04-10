from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import NotFoundError, OpenAI
from openai.types.file_purpose import FilePurpose
from openai.types.shared_params.comparison_filter import ComparisonFilter
from openai.types.shared_params.compound_filter import CompoundFilter
from openai.types.vector_stores.vector_store_file import VectorStoreFile

from .schemas import (
    AttachFilesResult,
    DeletedFileResult,
    FileListResult,
    FileSummary,
    OpenAIAttributes,
    SearchHit,
    SearchVectorStoreResult,
    ToolAttributes,
    UploadFileResult,
    VectorStoreBatchSummary,
    VectorStoreFileSummary,
    VectorStoreListResult,
    VectorStoreMetadata,
    VectorStoreStatusResult,
    VectorStoreSummary,
)
from .settings import AppSettings

RESERVED_FILE_ID_ATTRIBUTE = "openai_file_id"
RESERVED_FILENAME_ATTRIBUTE = "filename"


class OpenAIFilesVectorStoreGateway:
    """Live gateway for OpenAI file and vector-store operations."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._logger = logging.getLogger(__name__)

    def upload_file(
        self,
        *,
        local_path: str,
        vector_store_id: str | None,
        purpose: FilePurpose,
        attributes: ToolAttributes | None,
    ) -> UploadFileResult:
        started_at = perf_counter()
        resolved_path = self._resolve_local_path(local_path)

        with resolved_path.open("rb") as file_handle:
            uploaded_file = self._client.files.create(file=file_handle, purpose=purpose)

        attached_file: VectorStoreFileSummary | None = None
        normalized_attributes = self._normalize_attributes(attributes)
        if vector_store_id is not None:
            attach_kwargs: dict[str, Any] = {
                "file_id": uploaded_file.id,
                "vector_store_id": vector_store_id,
                "poll_interval_ms": self._settings.openai_poll_interval_ms,
            }
            self._client.vector_stores.files.create_and_poll(
                **attach_kwargs
            )
            attached_file = self._update_vector_store_file_attributes(
                vector_store_id=vector_store_id,
                file_id=uploaded_file.id,
                filename=uploaded_file.filename,
                attributes=normalized_attributes,
            )

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_file_upload file_id=%s vector_store_id=%s status=%s duration_ms=%.1f",
            uploaded_file.id,
            vector_store_id,
            attached_file.status if attached_file else uploaded_file.status,
            duration_ms,
        )

        return UploadFileResult(
            uploaded_file=FileSummary.from_openai(uploaded_file),
            vector_store_id=vector_store_id,
            attached_file=attached_file,
        )

    def list_files(
        self,
        *,
        limit: int,
        purpose: str | None,
    ) -> FileListResult:
        started_at = perf_counter()
        page = self._client.files.list(limit=limit)
        files = [FileSummary.from_openai(file_object) for file_object in page.data]
        if purpose is not None:
            files = [
                file_object for file_object in files if file_object.purpose == purpose
            ]

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_file_list returned=%s purpose_filter=%s duration_ms=%.1f",
            len(files),
            purpose,
            duration_ms,
        )

        return FileListResult(
            files=files,
            total_returned=len(files),
            purpose_filter=purpose,
        )

    def create_vector_store(
        self,
        *,
        name: str | None,
        description: str | None,
        metadata: VectorStoreMetadata | None,
    ) -> VectorStoreSummary:
        started_at = perf_counter()
        create_kwargs: dict[str, Any] = {}
        if name is not None:
            create_kwargs["name"] = name
        if description is not None:
            create_kwargs["description"] = description
        if metadata:
            create_kwargs["metadata"] = metadata

        vector_store = self._client.vector_stores.create(**create_kwargs)

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_create vector_store_id=%s status=%s duration_ms=%.1f",
            vector_store.id,
            vector_store.status,
            duration_ms,
        )

        return VectorStoreSummary.from_openai(vector_store)

    def list_vector_stores(self, *, limit: int) -> VectorStoreListResult:
        started_at = perf_counter()
        page = self._client.vector_stores.list(limit=limit)
        vector_stores = [
            VectorStoreSummary.from_openai(vector_store) for vector_store in page.data
        ]

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_list returned=%s duration_ms=%.1f",
            len(vector_stores),
            duration_ms,
        )

        return VectorStoreListResult(
            vector_stores=vector_stores,
            total_returned=len(vector_stores),
        )

    def attach_files_to_vector_store(
        self,
        *,
        vector_store_id: str,
        file_ids: list[str] | None,
        local_paths: list[str] | None,
        attributes: ToolAttributes | None,
    ) -> AttachFilesResult:
        started_at = perf_counter()
        existing_file_ids = list(dict.fromkeys(file_ids or []))
        resolved_local_paths = [
            str(self._resolve_local_path(local_path))
            for local_path in (local_paths or [])
        ]
        normalized_attributes = self._normalize_attributes(attributes)
        total_files = len(existing_file_ids) + len(resolved_local_paths)

        if total_files == 0:
            raise ValueError("Provide at least one file_id or local_path.")

        batch_summary: VectorStoreBatchSummary | None = None
        attached_files: list[VectorStoreFileSummary]

        if total_files == 1 and existing_file_ids:
            attach_kwargs: dict[str, Any] = {
                "file_id": existing_file_ids[0],
                "vector_store_id": vector_store_id,
                "poll_interval_ms": self._settings.openai_poll_interval_ms,
            }
            self._client.vector_stores.files.create_and_poll(
                **attach_kwargs
            )
            file_object = self._client.files.retrieve(existing_file_ids[0])
            attached_files = [
                self._update_vector_store_file_attributes(
                    vector_store_id=vector_store_id,
                    file_id=file_object.id,
                    filename=file_object.filename,
                    attributes=normalized_attributes,
                )
            ]
        elif total_files == 1 and resolved_local_paths:
            with Path(resolved_local_paths[0]).open("rb") as file_handle:
                upload_kwargs: dict[str, Any] = {
                    "vector_store_id": vector_store_id,
                    "file": file_handle,
                    "poll_interval_ms": self._settings.openai_poll_interval_ms,
                }
                vector_store_file = self._client.vector_stores.files.upload_and_poll(
                    **upload_kwargs
                )
            file_object = self._client.files.retrieve(vector_store_file.id)
            attached_files = [
                self._update_vector_store_file_attributes(
                    vector_store_id=vector_store_id,
                    file_id=file_object.id,
                    filename=file_object.filename,
                    attributes=normalized_attributes,
                )
            ]
        else:
            with ExitStack() as exit_stack:
                file_handles = [
                    exit_stack.enter_context(Path(local_path).open("rb"))
                    for local_path in resolved_local_paths
                ]
                batch_kwargs: dict[str, Any] = {
                    "vector_store_id": vector_store_id,
                    "files": file_handles,
                    "file_ids": existing_file_ids,
                    "max_concurrency": min(total_files, 5),
                    "poll_interval_ms": self._settings.openai_poll_interval_ms,
                }
                batch = self._client.vector_stores.file_batches.upload_and_poll(
                    **batch_kwargs
                )

            batch_summary = VectorStoreBatchSummary.from_openai(batch)
            batch_file_page = self._client.vector_stores.file_batches.list_files(
                batch.id,
                vector_store_id=vector_store_id,
                limit=min(max(total_files, 20), 100),
            )
            attached_files = [
                self._update_vector_store_file_attributes(
                    vector_store_id=vector_store_id,
                    file_id=file_object.id,
                    filename=file_object.filename,
                    attributes=normalized_attributes,
                )
                for vector_store_file in batch_file_page.data
                for file_object in [self._client.files.retrieve(vector_store_file.id)]
            ]

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_attach vector_store_id=%s existing_files=%s local_files=%s batch_id=%s duration_ms=%.1f",
            vector_store_id,
            len(existing_file_ids),
            len(resolved_local_paths),
            batch_summary.id if batch_summary else None,
            duration_ms,
        )

        return AttachFilesResult(
            vector_store_id=vector_store_id,
            file_ids=existing_file_ids,
            local_paths=resolved_local_paths,
            attached_files=attached_files,
            batch=batch_summary,
        )

    def get_vector_store_status(
        self,
        *,
        vector_store_id: str,
        file_limit: int,
        batch_id: str | None,
    ) -> VectorStoreStatusResult:
        started_at = perf_counter()
        vector_store = self._client.vector_stores.retrieve(vector_store_id)
        vector_store_file_page = self._client.vector_stores.files.list(
            vector_store_id,
            limit=file_limit,
        )
        files = self._existing_vector_store_files(
            vector_store_id=vector_store_id,
            vector_store_files=vector_store_file_page.data,
        )

        batch_summary: VectorStoreBatchSummary | None = None
        batch_files: list[VectorStoreFileSummary] = []
        if batch_id is not None:
            batch = self._client.vector_stores.file_batches.retrieve(
                batch_id,
                vector_store_id=vector_store_id,
            )
            batch_summary = VectorStoreBatchSummary.from_openai(batch)
            batch_file_page = self._client.vector_stores.file_batches.list_files(
                batch_id,
                vector_store_id=vector_store_id,
                limit=file_limit,
            )
            batch_files = self._existing_vector_store_files(
                vector_store_id=vector_store_id,
                vector_store_files=batch_file_page.data,
            )

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_status vector_store_id=%s batch_id=%s file_count=%s duration_ms=%.1f",
            vector_store_id,
            batch_id,
            len(files),
            duration_ms,
        )

        return VectorStoreStatusResult(
            vector_store=VectorStoreSummary.from_openai(vector_store),
            files=files,
            batch=batch_summary,
            batch_files=batch_files,
        )

    def search_vector_store(
        self,
        *,
        vector_store_id: str,
        query: str,
        max_num_results: int,
        rewrite_query: bool,
        file_id: str | None,
        filename: str | None,
    ) -> SearchVectorStoreResult:
        started_at = perf_counter()
        filters = self._build_search_filters(file_id=file_id, filename=filename)
        page = self._client.vector_stores.search(
            vector_store_id,
            query=query,
            max_num_results=max_num_results,
            rewrite_query=rewrite_query,
            filters=filters,
        )
        hits = [SearchHit.from_openai(search_result) for search_result in page.data]

        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_search vector_store_id=%s file_id=%s filename=%s hits=%s duration_ms=%.1f",
            vector_store_id,
            file_id,
            filename,
            len(hits),
            duration_ms,
        )

        return SearchVectorStoreResult(
            vector_store_id=vector_store_id,
            query=query,
            file_id=file_id,
            filename=filename,
            hits=hits,
            total_hits=len(hits),
        )

    def update_vector_store_file_attributes(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        attributes: ToolAttributes | None,
    ) -> VectorStoreFileSummary:
        started_at = perf_counter()
        file_object = self._client.files.retrieve(file_id)
        normalized_attributes = self._normalize_attributes(attributes)
        updated_file = self._update_vector_store_file_attributes(
            vector_store_id=vector_store_id,
            file_id=file_object.id,
            filename=file_object.filename,
            attributes=normalized_attributes,
        )
        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_vector_store_file_attributes_update vector_store_id=%s file_id=%s attribute_count=%s duration_ms=%.1f",
            vector_store_id,
            file_id,
            len(updated_file.attributes or {}),
            duration_ms,
        )
        return updated_file

    def delete_file(self, *, file_id: str) -> DeletedFileResult:
        started_at = perf_counter()
        deleted_file = self._client.files.delete(file_id)
        duration_ms = (perf_counter() - started_at) * 1000
        self._logger.info(
            "openai_file_delete file_id=%s deleted=%s duration_ms=%.1f",
            file_id,
            deleted_file.deleted,
            duration_ms,
        )
        return DeletedFileResult(file_id=file_id, deleted=deleted_file.deleted)

    def _resolve_local_path(self, local_path: str) -> Path:
        resolved_path = Path(local_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Local file not found: {resolved_path}")
        return resolved_path

    def _normalize_attributes(
        self,
        attributes: ToolAttributes | None,
    ) -> OpenAIAttributes | None:
        if not attributes:
            return None

        normalized_attributes: OpenAIAttributes = {}
        for key, value in attributes.items():
            if isinstance(value, bool):
                normalized_attributes[key] = value
            elif isinstance(value, int):
                normalized_attributes[key] = float(value)
            else:
                normalized_attributes[key] = value
        return normalized_attributes

    def _update_vector_store_file_attributes(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        filename: str,
        attributes: OpenAIAttributes | None,
    ) -> VectorStoreFileSummary:
        merged_attributes = {
            **(attributes or {}),
            RESERVED_FILE_ID_ATTRIBUTE: file_id,
            RESERVED_FILENAME_ATTRIBUTE: filename,
        }
        vector_store_file = self._client.vector_stores.files.update(
            file_id,
            vector_store_id=vector_store_id,
            attributes=merged_attributes,
        )
        return VectorStoreFileSummary.from_openai(vector_store_file)

    def _build_search_filters(
        self,
        *,
        file_id: str | None,
        filename: str | None,
    ) -> ComparisonFilter | CompoundFilter | None:
        filters: list[ComparisonFilter] = []
        if file_id is not None:
            filters.append(
                {
                    "key": RESERVED_FILE_ID_ATTRIBUTE,
                    "type": "eq",
                    "value": file_id,
                }
            )
        if filename is not None:
            filters.append(
                {
                    "key": RESERVED_FILENAME_ATTRIBUTE,
                    "type": "eq",
                    "value": filename,
                }
            )

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"type": "and", "filters": filters}

    def _existing_vector_store_files(
        self,
        *,
        vector_store_id: str,
        vector_store_files: list[VectorStoreFile],
    ) -> list[VectorStoreFileSummary]:
        existing_files: list[VectorStoreFileSummary] = []
        missing_file_ids: list[str] = []
        for vector_store_file in vector_store_files:
            try:
                self._client.files.retrieve(vector_store_file.id)
            except NotFoundError:
                missing_file_ids.append(vector_store_file.id)
                continue
            refreshed_vector_store_file = self._client.vector_stores.files.retrieve(
                vector_store_file.id,
                vector_store_id=vector_store_id,
            )
            existing_files.append(
                VectorStoreFileSummary.from_openai(refreshed_vector_store_file)
            )

        if missing_file_ids:
            self._logger.warning(
                "openai_vector_store_status_skipping_missing_files file_ids=%s",
                missing_file_ids,
            )
        return existing_files
