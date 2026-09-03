"""Bootstrap 的真实 Literature/Parse Provider 适配与持久化 Job 组合。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from app.academic_mcp.models import DiscoveryResult, FullTextResult, PaperRecord
from app.jobs import (
    JobRecord,
    LongTaskSubmitter,
    PersistentJobRepository,
    PersistentJobWorker,
)
from app.parsing.pipeline import PaperParseConfig, PaperParseResult, parse_paper
from app.storage import write_json_atomic


def _sync(value: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("同步 Provider Adapter 不能在运行中的 event loop 内调用。")


class AcademicBackend(Protocol):
    async def search(self, query: str, limit: int = 20) -> DiscoveryResult: ...

    async def find_fulltext(
        self,
        doi: str | None = None,
        openalex_id: str | None = None,
        arxiv_id: str | None = None,
    ) -> FullTextResult: ...

    async def download_open_pdf(
        self, download_token: str, filename: str | None = None
    ) -> dict[str, str | int | bool]: ...


class CandidateKnowledgeRepository:
    """外部候选只保存元数据，不写入 Canonical Corpus。"""

    def __init__(self, project_root: Path) -> None:
        self.path = (
            project_root.expanduser().resolve()
            / "data"
            / "candidate_knowledge"
            / "papers.json"
        )

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": "1.0", "papers": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
            raise ValueError("Candidate Knowledge schema 不受支持。")
        return payload

    @staticmethod
    def candidate_id(paper: PaperRecord) -> str:
        identity = paper.doi or paper.arxiv_id or paper.title.casefold()
        return f"CK_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"

    def put(self, paper: PaperRecord) -> dict[str, Any]:
        candidate_id = self.candidate_id(paper)
        value = {"paper_id": candidate_id, "status": "candidate", **paper.to_dict()}
        payload = self._load()
        payload["papers"][candidate_id] = value
        write_json_atomic(self.path, payload)
        return value

    def get(self, candidate_id: str) -> dict[str, Any]:
        value = self._load()["papers"].get(candidate_id)
        if not isinstance(value, dict):
            raise KeyError(f"Candidate paper 不存在：{candidate_id}")
        return dict(value)


class JobResultStore:
    """完整工具结果保存为文件引用，不进入 Job 表。"""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.expanduser().resolve()
        self.directory = self.root / "data" / "jobs" / "results"

    def write(self, job_id: str, result: Mapping[str, Any]) -> str:
        path = self.directory / f"{job_id}.json"
        write_json_atomic(path, dict(result))
        return path.relative_to(self.root).as_posix()

    def read(self, reference: str) -> Mapping[str, Any]:
        path = (self.root / reference).resolve()
        if self.directory.resolve() not in path.parents:
            raise PermissionError("Job result_reference 越界。")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Job result 必须为 JSON 对象。")
        return value


class AcademicLiteratureProviderAdapter:
    def __init__(
        self, backend: AcademicBackend, candidates: CandidateKnowledgeRepository
    ) -> None:
        self.backend = backend
        self.candidates = candidates

    def search(
        self, arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        limit = min(10, max(1, int(arguments.get("limit") or 10)))
        result = _sync(self.backend.search(str(arguments["query"]), limit=limit))
        papers = [self.candidates.put(value) for value in result.papers[:10]]
        return {
            "results": papers,
            "result_count": len(papers),
            "provider_failures": [value.to_dict() for value in result.failures],
        }

    def get_metadata(
        self, arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        value = self.candidates.get(str(arguments["paper_id"]))
        return {**value, "lightweight": True}

    def resolve_publication_date(
        self, arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Resolve one title to publication metadata for timeline ordering."""

        title = str(arguments["title"]).strip()
        if not title:
            raise ValueError("论文标题不能为空。")
        resolver = getattr(self.backend, "resolve", None)
        if not callable(resolver):
            raise RuntimeError("当前 Academic backend 不支持按标题解析论文元数据。")
        result = _sync(resolver(title, limit=5))
        papers = getattr(result, "papers", None)
        failures = getattr(result, "failures", None)
        values = [paper.to_dict() for paper in papers[:5]] if isinstance(papers, list) else []
        best = values[0] if values else {}
        return {
            "query_title": title,
            "matched_title": str(best.get("title") or ""),
            "publication_year": best.get("publication_year"),
            "doi": best.get("doi"),
            "sources": list(best.get("sources") or []),
            "match_score": best.get("match_score"),
            "candidates": values,
            "provider_failures": [
                value.to_dict() for value in failures
            ] if isinstance(failures, list) else [],
        }

    def download(self, paper_id: str) -> Mapping[str, Any]:
        paper = self.candidates.get(paper_id)
        external = paper.get("external_ids")
        external_ids = external if isinstance(external, Mapping) else {}
        fulltext = _sync(
            self.backend.find_fulltext(
                doi=str(paper["doi"]) if paper.get("doi") else None,
                openalex_id=(
                    str(external_ids["openalex"])
                    if external_ids.get("openalex")
                    else None
                ),
                arxiv_id=(
                    str(paper["arxiv_id"]) if paper.get("arxiv_id") else None
                ),
            )
        )
        location = next(
            (value for value in fulltext.locations if value.download_token), None
        )
        if location is None or location.download_token is None:
            raise FileNotFoundError("候选论文没有可下载的开放 PDF。")
        return _sync(
            self.backend.download_open_pdf(
                location.download_token, filename=f"{paper_id}.pdf"
            )
        )


ParseProvider = Callable[[Path, PaperParseConfig], PaperParseResult]


class CanonicalDocumentParseProviderAdapter:
    def __init__(
        self,
        project_root: Path,
        *,
        parser: ParseProvider = parse_paper,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.parser = parser

    def parse(self, path: str, mode: str) -> Mapping[str, Any]:
        source = Path(path).expanduser().resolve()
        result = self.parser(source, PaperParseConfig(project_root=self.project_root))
        payload = json.loads(result.paper_json.read_text(encoding="utf-8"))
        identity = payload.get("identity") if isinstance(payload, dict) else None
        document_id = identity.get("document_id") if isinstance(identity, dict) else None
        if not document_id:
            raise ValueError("Canonical parse 未生成 document_id。")
        return {
            "document_id": str(document_id),
            "paper_id": result.paper_id,
            "content_hash": result.sha256,
            "paper_json": result.paper_json.relative_to(self.project_root).as_posix(),
            "mode": mode,
        }


@dataclass(frozen=True, slots=True)
class BootstrapProviderComposition:
    gateway_handlers: Mapping[str, Callable[..., Mapping[str, Any]]]
    worker: PersistentJobWorker
    repository: PersistentJobRepository


def build_bootstrap_provider_composition(
    project_root: Path,
    backend: AcademicBackend,
    *,
    parser: ParseProvider = parse_paper,
    worker_id: str = "bootstrap-worker",
) -> BootstrapProviderComposition:
    root = project_root.expanduser().resolve()
    repository = PersistentJobRepository(root)
    results = JobResultStore(root)
    literature = AcademicLiteratureProviderAdapter(
        backend, CandidateKnowledgeRepository(root)
    )
    document_parser = CanonicalDocumentParseProviderAdapter(root, parser=parser)
    submitter = LongTaskSubmitter(repository)

    def download_job(record: JobRecord, context: Any) -> str:
        context.raise_if_cancelled()
        result = literature.download(str(record.payload["paper_id"]))
        return results.write(record.job_id, result)

    def parse_job(record: JobRecord, context: Any) -> str:
        context.raise_if_cancelled()
        result = document_parser.parse(
            str(record.payload["path"]), str(record.payload.get("mode") or "formal")
        )
        return results.write(record.job_id, result)

    def status(
        arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        record = repository.get(str(arguments["job_id"]))
        result = (
            results.read(record.result_reference)
            if record.status == "SUCCEEDED" and record.result_reference
            else None
        )
        return {
            "job_id": record.job_id,
            "status": record.status.casefold(),
            "result": result,
            "error_type": record.error_type,
            "error_summary": record.error_summary,
        }

    return BootstrapProviderComposition(
        gateway_handlers={
            "literature.search": literature.search,
            "literature.get_metadata": literature.get_metadata,
            "literature.resolve_publication_date": literature.resolve_publication_date,
            "literature.download": submitter.handler("literature.download"),
            "document.parse": submitter.handler("document.parse"),
            "job.get_status": status,
        },
        worker=PersistentJobWorker(
            repository,
            {
                "literature.download": download_job,
                "document.parse": parse_job,
            },
            worker_id=worker_id,
        ),
        repository=repository,
    )
