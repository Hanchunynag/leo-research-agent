import json
from pathlib import Path

from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.knowledge.corpus import corpus_summary
from app.storage import write_jsonl_atomic


def test_corpus_summary_separates_metadata_only_and_indexed_papers(
    tmp_path: Path,
) -> None:
    knowledge = tmp_path / "data" / "knowledge"
    canonical = tmp_path / "data" / "canonical" / "P_indexed"
    structures = knowledge / "structures"
    canonical.mkdir(parents=True)
    structures.mkdir(parents=True)
    (canonical / "paper.json").write_text(
        json.dumps({"paper_id": "P_indexed"}), encoding="utf-8"
    )
    papers = [
        {"paper_id": "P_indexed", "title": "Indexed"},
        {"paper_id": "EXT_abstract", "title": "Abstract only"},
    ]
    write_jsonl_atomic(knowledge / "paper_records.jsonl", papers)
    chunks = [
        {
            "chunk_id": "C1",
            "paper_id": "P_indexed",
            "document_id": "D_indexed",
            "content": "evidence",
            "section_path": ["Results"],
            "parent_contexts": [],
            "overlap_context": None,
        }
    ]
    write_jsonl_atomic(knowledge / "chunks.jsonl", chunks)
    (structures / "D_indexed.structure.json").write_text(
        json.dumps({"sections": [{"section_id": "S1"}]}), encoding="utf-8"
    )
    write_bm25_index(tmp_path, build_bm25_index(chunks))

    summary = corpus_summary(tmp_path)

    assert summary.paper_count == 2
    assert summary.canonical_document_count == 1
    assert summary.indexed_paper_count == 1
    assert summary.indexed_document_count == 1
    assert summary.section_count == 1
    assert summary.chunk_count == 1
    assert summary.searchable_chunk_count == 1
    assert summary.bm25_index_count == 1
    assert summary.metadata_only_paper_ids == ["EXT_abstract"]
