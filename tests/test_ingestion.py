from pathlib import Path

from enterprise_rag.ingestion.chunker import ChunkingConfig, DocumentChunker
from enterprise_rag.ingestion.cleaner import DocumentCleaner
from enterprise_rag.ingestion.loaders import load_file
from enterprise_rag.ingestion.models import EnterpriseDocument


def test_cleaner_normalizes_whitespace_and_unicode():
    cleaned = DocumentCleaner().clean_text("A\u00a0  title\r\n\r\n\r\nB")
    assert cleaned == "A title\n\nB"


def test_section_chunker_preserves_heading():
    document = EnterpriseDocument("d1", "# Policy\n\nImportant rule.", "Policy", "x.md")
    chunks = DocumentChunker(ChunkingConfig(strategy="section", chunk_size=100, overlap=10)).chunk(document)
    assert len(chunks) == 1
    assert chunks[0].section == "Policy"
    assert chunks[0].document_id == "d1"


def test_fixed_chunker_rejects_invalid_overlap():
    document = EnterpriseDocument("d1", "abcdef", "Policy", "x.md")
    try:
        DocumentChunker(ChunkingConfig(strategy="fixed", chunk_size=5, overlap=5)).chunk(document)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap should fail")


def test_markdown_loader_reads_front_matter(tmp_path: Path):
    path = tmp_path / "policy.md"
    path.write_text("---\ndocument_id: p1\ntitle: Test Policy\n---\n\n# Rules\n\nUse MFA.", encoding="utf-8")
    document = load_file(path)
    assert document.document_id == "p1"
    assert document.title == "Test Policy"
    assert "Use MFA" in document.text

