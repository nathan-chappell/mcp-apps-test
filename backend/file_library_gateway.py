from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import mimetypes
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, cast
from urllib.parse import unquote, urlparse

import httpx
from openai import AsyncOpenAI
from openai.types.file_purpose import FilePurpose
from openai.types.shared_params.comparison_filter import ComparisonFilter
from openai.types.shared_params.compound_filter import CompoundFilter
from pydantic import BaseModel, Field

from .openai_tracing import extract_openai_trace_refs
from .schemas import ArxivPaperCandidate, ImageDescriptionPayload, SearchHit, TagMatchMode
from .settings import AppSettings

logger = logging.getLogger(__name__)
ARXIV_ALLOWED_HOSTS = {"arxiv.org", "www.arxiv.org"}
ARXIV_ID_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")


class _ArxivWebSearchPaper(BaseModel):
    title: str
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    url: str


class _ArxivWebSearchPayload(BaseModel):
    papers: list[_ArxivWebSearchPaper] = Field(default_factory=list)


@dataclass(slots=True)
class DownloadedRemoteFile:
    local_path: Path
    filename: str
    media_type: str
    source_url: str


class OpenAIFileLibraryGateway:
    """OpenAI-backed file, search, and multimodal ingestion operations."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._http_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._logger = logging.getLogger(__name__)

    async def close(self) -> None:
        await self._client.close()
        await self._http_client.aclose()

    async def create_vector_store(
        self,
        *,
        name: str,
        description: str | None,
        metadata: dict[str, str] | None,
    ) -> str:
        started_at = perf_counter()
        create_arguments: dict[str, object] = {
            "name": name,
            "metadata": metadata,
        }
        if description is not None:
            create_arguments["description"] = description
        vector_store = await cast(Any, self._client.vector_stores.create)(**create_arguments)
        self._logger.info(
            "file_library_vector_store_created vector_store_id=%s name=%s duration_ms=%.1f",
            vector_store.id,
            name,
            (perf_counter() - started_at) * 1000,
        )
        return vector_store.id

    async def upload_original_file(
        self,
        *,
        local_path: Path,
        purpose: FilePurpose,
    ) -> str:
        started_at = perf_counter()
        with local_path.open("rb") as file_handle:
            file_object = await self._client.files.create(file=file_handle, purpose=purpose)
        self._logger.info(
            "file_library_original_file_uploaded file_id=%s filename=%s purpose=%s duration_ms=%.1f",
            file_object.id,
            file_object.filename,
            purpose,
            (perf_counter() - started_at) * 1000,
        )
        return file_object.id

    async def create_text_artifact_and_attach(
        self,
        *,
        vector_store_id: str,
        filename: str,
        text_content: str,
        attributes: dict[str, str | float | bool],
    ) -> str:
        started_at = perf_counter()
        with NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text_content)

        try:
            with temp_path.open("rb") as file_handle:
                uploaded_file = await self._client.files.create(
                    file=file_handle,
                    purpose="assistants",
                )
            await self._client.vector_stores.files.create_and_poll(
                file_id=uploaded_file.id,
                vector_store_id=vector_store_id,
                attributes=attributes,
                poll_interval_ms=self._settings.openai_poll_interval_ms,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        self._logger.info(
            "file_library_text_artifact_attached vector_store_id=%s file_id=%s filename=%s duration_ms=%.1f",
            vector_store_id,
            uploaded_file.id,
            filename,
            (perf_counter() - started_at) * 1000,
        )
        return uploaded_file.id

    async def attach_existing_file_to_vector_store(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, str | float | bool],
    ) -> str:
        started_at = perf_counter()
        await self._client.vector_stores.files.create_and_poll(
            file_id=file_id,
            vector_store_id=vector_store_id,
            attributes=attributes,
            poll_interval_ms=self._settings.openai_poll_interval_ms,
        )
        self._logger.info(
            "file_library_existing_file_attached vector_store_id=%s file_id=%s duration_ms=%.1f",
            vector_store_id,
            file_id,
            (perf_counter() - started_at) * 1000,
        )
        return file_id

    async def describe_image(
        self,
        *,
        openai_file_id: str,
    ) -> ImageDescriptionPayload:
        started_at = perf_counter()
        response = await self._client.responses.parse(
            model=self._settings.openai_vision_model,
            text_format=ImageDescriptionPayload,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this uploaded image for a searchable file library. "
                                "Capture objects, scene, layout, visible text, and practical "
                                "details a later user might search for."
                            ),
                        },
                        {
                            "type": "input_image",
                            "file_id": openai_file_id,
                            "detail": "high",
                        },
                    ],
                }
            ],
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Expected structured image description output from OpenAI.")

        trace_refs = extract_openai_trace_refs(response)
        self._logger.info(
            "file_library_image_described file_id=%s model=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s duration_ms=%.1f",
            openai_file_id,
            self._settings.openai_vision_model,
            trace_refs.response_id,
            trace_refs.response_log_url,
            trace_refs.conversation_id,
            trace_refs.conversation_log_url,
            (perf_counter() - started_at) * 1000,
        )
        return parsed

    async def transcribe_audio(
        self,
        *,
        local_path: Path,
    ) -> tuple[str, dict[str, object]]:
        started_at = perf_counter()
        with local_path.open("rb") as file_handle:
            # The SDK runtime accepts newer diarization options than the generated stubs model.
            transcription = await cast(Any, self._client.audio.transcriptions).create(
                file=file_handle,
                model=self._settings.openai_audio_transcription_model,
                response_format="diarized_json",
                chunking_strategy="auto",
            )

        segments = [
            {
                "id": segment.id,
                "speaker": segment.speaker,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "type": segment.type,
            }
            for segment in transcription.segments
        ]
        searchable_text = "\n".join(f"[{segment['speaker']}] {segment['text']}" for segment in segments).strip()
        payload: dict[str, object] = {
            "duration": transcription.duration,
            "task": transcription.task,
            "text": transcription.text,
            "segments": segments,
        }
        self._logger.info(
            "file_library_audio_transcribed filename=%s model=%s segments=%s duration_ms=%.1f",
            local_path.name,
            self._settings.openai_audio_transcription_model,
            len(segments),
            (perf_counter() - started_at) * 1000,
        )
        return searchable_text or transcription.text, payload

    async def transcribe_video(
        self,
        *,
        local_path: Path,
    ) -> tuple[str, dict[str, object]]:
        started_at = perf_counter()
        audio_path = await self._extract_audio_track(local_path)
        try:
            transcript_text, payload = await self.transcribe_audio(local_path=audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        payload["video_filename"] = local_path.name
        self._logger.info(
            "file_library_video_transcribed filename=%s duration_ms=%.1f",
            local_path.name,
            (perf_counter() - started_at) * 1000,
        )
        return transcript_text, payload

    async def search_vector_store(
        self,
        *,
        vector_store_id: str,
        query: str,
        max_results: int,
        rewrite_query: bool,
        filters: ComparisonFilter | CompoundFilter | None,
    ) -> list[SearchHit]:
        started_at = perf_counter()
        search_arguments: dict[str, object] = {
            "vector_store_id": vector_store_id,
            "query": query,
            "max_num_results": max_results,
            "rewrite_query": rewrite_query,
        }
        if filters is not None:
            search_arguments["filters"] = filters
        page = await cast(Any, self._client.vector_stores.search)(**search_arguments)
        hits = [SearchHit.from_openai(search_result) for search_result in page.data]
        self._logger.info(
            "file_library_vector_store_search vector_store_id=%s query=%s hits=%s duration_ms=%.1f",
            vector_store_id,
            query,
            len(hits),
            (perf_counter() - started_at) * 1000,
        )
        return hits

    async def search_arxiv_papers(
        self,
        *,
        query: str,
        max_results: int,
    ) -> list[ArxivPaperCandidate]:
        started_at = perf_counter()
        response = await self._client.responses.parse(
            model=self._settings.openai_agent_model,
            text_format=_ArxivWebSearchPayload,
            include=["web_search_call.action.sources"],
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": ["arxiv.org"]},
                    "search_context_size": "medium",
                }
            ],
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Search arXiv for papers relevant to the user's query. "
                                f"Return up to {max_results} real papers from arxiv.org only. "
                                "For each paper, return the exact arXiv abstract page URL in the url field, "
                                "a concise one-paragraph summary, and author names when available."
                                f"\n\nQuery: {query}"
                            ),
                        }
                    ],
                }
            ],
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Expected structured arXiv search output from OpenAI.")

        normalized_results: list[ArxivPaperCandidate] = []
        seen_ids: set[str] = set()
        for paper in parsed.papers:
            try:
                normalized = _normalize_arxiv_candidate(paper)
            except ValueError:
                continue
            if normalized.arxiv_id in seen_ids:
                continue
            seen_ids.add(normalized.arxiv_id)
            normalized_results.append(normalized)
            if len(normalized_results) >= max_results:
                break

        trace_refs = extract_openai_trace_refs(response)
        self._logger.info(
            "file_library_arxiv_search query=%s requested=%s returned=%s response_id=%s response_log_url=%s conversation_id=%s conversation_log_url=%s duration_ms=%.1f",
            query,
            max_results,
            len(normalized_results),
            trace_refs.response_id,
            trace_refs.response_log_url,
            trace_refs.conversation_id,
            trace_refs.conversation_log_url,
            (perf_counter() - started_at) * 1000,
        )
        return normalized_results

    async def download_arxiv_pdf(
        self,
        *,
        paper: ArxivPaperCandidate,
    ) -> DownloadedRemoteFile:
        started_at = perf_counter()
        pdf_url = _canonical_arxiv_pdf_url(paper.pdf_url)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            pdf_prefix = bytearray()
            bytes_written = 0
            content_disposition: str | None = None
            async with self._http_client.stream("GET", pdf_url) as response:
                response.raise_for_status()
                content_disposition = response.headers.get("content-disposition")
                with temp_path.open("wb") as output_file:
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        if len(pdf_prefix) < 5:
                            remaining = 5 - len(pdf_prefix)
                            pdf_prefix.extend(chunk[:remaining])
                        output_file.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written == 0:
                raise RuntimeError("Downloaded arXiv PDF was empty.")
            if bytes(pdf_prefix) != b"%PDF-":
                raise RuntimeError("Downloaded arXiv content was not a PDF.")

            resolved_filename = _preferred_arxiv_download_filename(
                paper=paper,
                content_disposition=content_disposition,
            )
            self._logger.info(
                "file_library_arxiv_pdf_downloaded arxiv_id=%s filename=%s bytes=%s duration_ms=%.1f",
                paper.arxiv_id,
                resolved_filename,
                bytes_written,
                (perf_counter() - started_at) * 1000,
            )
            return DownloadedRemoteFile(
                local_path=temp_path,
                filename=resolved_filename,
                media_type="application/pdf",
                source_url=pdf_url,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    async def delete_file(self, *, file_id: str) -> None:
        started_at = perf_counter()
        await self._client.files.delete(file_id)
        self._logger.info(
            "file_library_file_deleted file_id=%s duration_ms=%.1f",
            file_id,
            (perf_counter() - started_at) * 1000,
        )

    async def read_file_bytes(self, *, file_id: str) -> bytes:
        started_at = perf_counter()
        response = await self._client.files.content(file_id)
        content = response.content
        self._logger.info(
            "file_library_file_read file_id=%s bytes=%s duration_ms=%.1f",
            file_id,
            len(content),
            (perf_counter() - started_at) * 1000,
        )
        return content

    @staticmethod
    def choose_original_file_purpose(*, source_kind: str) -> FilePurpose:
        if source_kind == "image":
            return "vision"
        return "assistants"

    async def _extract_audio_track(self, local_path: Path) -> Path:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_path = Path(temp_file.name)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(local_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            audio_path.unlink(missing_ok=True)
            error_message = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg audio extraction failed: {error_message}")
        return audio_path


def guess_media_type(local_path: Path, declared_media_type: str | None) -> str:
    if declared_media_type:
        return declared_media_type
    guessed_media_type, _ = mimetypes.guess_type(local_path.name)
    return guessed_media_type or "application/octet-stream"


def build_filter_groups(
    *,
    file_ids: list[str],
    media_types: list[str],
    tag_slugs: list[str],
    tag_match_mode: TagMatchMode,
) -> ComparisonFilter | CompoundFilter | None:
    groups: list[ComparisonFilter | CompoundFilter] = []
    if file_ids:
        groups.append(
            _or_group("file_id", file_ids)
            if len(file_ids) > 1
            else {"type": "eq", "key": "file_id", "value": file_ids[0]}
        )
    if media_types:
        groups.append(
            _or_group("media_type", media_types)
            if len(media_types) > 1
            else {"type": "eq", "key": "media_type", "value": media_types[0]}
        )
    if tag_slugs:
        tag_filters: list[ComparisonFilter] = [
            {"type": "eq", "key": f"tag__{slug}", "value": True} for slug in tag_slugs
        ]
        if len(tag_filters) == 1:
            groups.append(tag_filters[0])
        else:
            groups.append(
                {
                    "type": "and" if tag_match_mode == "all" else "or",
                    "filters": tag_filters,
                }
            )

    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]
    return {"type": "and", "filters": groups}


def build_searchable_attributes(
    *,
    file_library_id: str,
    file_id: str,
    file_title: str,
    derived_artifact_id: str | None,
    source_kind: str,
    media_type: str,
    derived_kind: str,
    original_openai_file_id: str | None,
    original_filename: str,
    tag_names: list[str],
    tag_slugs: list[str],
) -> dict[str, str | float | bool]:
    attributes: dict[str, str | float | bool] = {
        "file_library_id": file_library_id,
        "file_id": file_id,
        "file_title": file_title,
        "source_kind": source_kind,
        "media_type": media_type,
        "derived_kind": derived_kind,
        "original_filename": original_filename,
        "tag_names": ",".join(tag_names),
    }
    if derived_artifact_id is not None:
        attributes["derived_artifact_id"] = derived_artifact_id
    if original_openai_file_id is not None:
        attributes["original_openai_file_id"] = original_openai_file_id
    for slug in tag_slugs:
        attributes[f"tag__{slug}"] = True
    return attributes


def _or_group(key: str, values: list[str]) -> CompoundFilter:
    return {
        "type": "or",
        "filters": [{"type": "eq", "key": key, "value": value} for value in values],
    }


def _normalize_arxiv_candidate(paper: _ArxivWebSearchPaper) -> ArxivPaperCandidate:
    abs_url = _canonical_arxiv_abs_url(paper.url)
    arxiv_id = _arxiv_id_from_url(abs_url)
    return ArxivPaperCandidate(
        arxiv_id=arxiv_id,
        title=paper.title.strip() or arxiv_id,
        summary=paper.summary.strip(),
        authors=[author.strip() for author in paper.authors if author.strip()],
        abs_url=abs_url,
        pdf_url=_canonical_arxiv_pdf_url(abs_url),
    )


def _canonical_arxiv_abs_url(url: str) -> str:
    arxiv_id = _arxiv_id_from_url(url)
    return f"https://arxiv.org/abs/{arxiv_id}"


def _canonical_arxiv_pdf_url(url: str) -> str:
    arxiv_id = _arxiv_id_from_url(url)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _arxiv_id_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Expected an HTTP arXiv URL.")
    if parsed.netloc.lower() not in ARXIV_ALLOWED_HOSTS:
        raise ValueError("Expected an arXiv URL.")

    path = parsed.path.strip()
    if path.startswith("/abs/"):
        raw_identifier = path.removeprefix("/abs/")
    elif path.startswith("/pdf/"):
        raw_identifier = path.removeprefix("/pdf/")
    else:
        raise ValueError("Expected an arXiv abstract or PDF URL.")

    normalized_identifier = raw_identifier.removesuffix(".pdf").strip("/")
    if not normalized_identifier or not ARXIV_ID_PATTERN.match(normalized_identifier):
        raise ValueError("Could not parse an arXiv identifier from the URL.")
    return normalized_identifier


def _display_filename_from_paper(paper: ArxivPaperCandidate) -> str:
    normalized_title = re.sub(r"[^a-z0-9]+", "-", paper.title.lower()).strip("-")
    if not normalized_title:
        normalized_title = paper.arxiv_id.lower().replace("/", "-")
    return f"{normalized_title[:120]}.pdf"


def _preferred_arxiv_download_filename(
    *,
    paper: ArxivPaperCandidate,
    content_disposition: str | None,
) -> str:
    preferred_filename = _display_filename_from_paper(paper)
    if preferred_filename != ".pdf":
        return preferred_filename
    return _filename_from_content_disposition(content_disposition) or f"{paper.arxiv_id.lower().replace('/', '-')}.pdf"


def _filename_from_content_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None
    filename_star_match = re.search(r"filename\\*=UTF-8''([^;]+)", content_disposition, flags=re.IGNORECASE)
    if filename_star_match:
        candidate = unquote(filename_star_match.group(1)).strip().strip('"')
        return candidate or None
    filename_match = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if filename_match:
        candidate = filename_match.group(1).strip()
        return candidate or None
    return None
