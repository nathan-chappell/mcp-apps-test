from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.file_library_gateway import OpenAIFileLibraryGateway
from backend.schemas import ArxivPaperCandidate
from backend.settings import AppSettings


class _FakeStreamResponse:
    def __init__(self, *, headers: dict[str, str], chunks: list[bytes]) -> None:
        self.headers = headers
        self._chunks = chunks

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>File Desk</div></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-clerk-secret")
    monkeypatch.setenv("APP_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("CLERK_ISSUER_URL", "https://clerk.example.com")
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'file-desk.db'}",
    )
    monkeypatch.setenv("STATIC_DIR", str(static_dir))
    return AppSettings()


@pytest.mark.asyncio
async def test_arxiv_download_prefers_title_based_filename_over_arxiv_header(
    configured_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = OpenAIFileLibraryGateway(configured_settings)
    paper = ArxivPaperCandidate(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        summary="Transformer paper.",
        authors=["Ashish Vaswani"],
        abs_url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    )

    def fake_stream(method: str, url: str) -> _FakeStreamContext:
        assert method == "GET"
        assert url == "https://arxiv.org/pdf/1706.03762.pdf"
        return _FakeStreamContext(
            _FakeStreamResponse(
                headers={"content-disposition": 'attachment; filename="1706.03762v7.pdf"'},
                chunks=[b"%PDF-1.7\n", b"1 0 obj\n<<>>\nendobj\n%%EOF"],
            )
        )

    monkeypatch.setattr(gateway._http_client, "stream", fake_stream)

    try:
        downloaded = await gateway.download_arxiv_pdf(paper=paper)
    finally:
        await gateway.close()

    try:
        assert downloaded.filename == "attention-is-all-you-need.pdf"
        assert downloaded.local_path.exists()
    finally:
        downloaded.local_path.unlink(missing_ok=True)
