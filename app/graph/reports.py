"""Source-bound community report generation and dense indexing."""

from __future__ import annotations

import json
import uuid
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.generation.openai_compatible import parse_json_object
from app.graph.models import CommunityReport
from app.indexing.qdrant_collections import (
    COMMUNITIES_ALIAS, VECTOR_NAME, versioned_collection_name,
)


REPORT_NAMESPACE = uuid.UUID("70008a68-15c6-4f52-a249-745def8b5bf5")
COMMUNITY_REPORT_PROMPT = """Produce a source-grounded scientific community report.
Use only the supplied RelationClaims. Every finding must cite source_claim_ids from the input.
Preserve contradictions and conditions. Never add an unsupported global conclusion. Return
exactly one JSON object matching the schema and no markdown."""


class ReportProvider(Protocol):
    model_name: str
    def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]: ...


def generate_community_report(
    provider: ReportProvider, *, community_id: str, entities: list[dict[str, Any]],
    claims: list[dict[str, Any]], max_tokens: int = 4096,
) -> CommunityReport:
    known = {str(value["claim_id"]) for value in claims if value.get("claim_id")}
    payload = provider.chat_completion([
        {"role": "system", "content": COMMUNITY_REPORT_PROMPT},
        {"role": "user", "content": json.dumps({
            "schema": CommunityReport.model_json_schema(), "community_id": community_id,
            "entities": entities, "relation_claims": claims,
        }, ensure_ascii=False)},
    ], max_tokens=max_tokens)
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("community report response lacks content") from error
    report = CommunityReport.model_validate(parse_json_object(str(content)))
    cited = set(report.source_claim_ids)
    cited.update(source for finding in report.findings for source in finding.source_claim_ids)
    if not cited.issubset(known):
        raise ValueError("community report cites unknown claim ids")
    if any(not finding.source_claim_ids for finding in report.findings):
        raise ValueError("every community finding requires sources")
    return report


def report_text(report: CommunityReport) -> str:
    return "\n".join((report.title, report.summary,
                       *(finding.statement for finding in report.findings),
                       *("Contradiction: " + value for value in report.contradictions)))


def embed_community_reports(
    client: QdrantClient, provider: EmbeddingProvider, reports: list[tuple[str, str, CommunityReport]],
    epoch: int,
) -> dict[str, Any]:
    if not reports:
        return {"embedded_count": 0, "upserted_count": 0}
    model_name = str(getattr(provider, "model_name"))
    revision = getattr(provider, "revision", None)
    collection = versioned_collection_name(COMMUNITIES_ALIAS, model_name, revision)
    texts = [report_text(report) for _, _, report in reports]
    vectors = provider.embed_documents(texts)
    if len(vectors) != len(reports):
        raise RuntimeError("community embedding count mismatch")
    existing = {value.name for value in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(collection_name=collection, vectors_config={
            VECTOR_NAME: models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
        })
    points = []
    for (community_id, fingerprint, report), vector in zip(reports, vectors, strict=True):
        point_id = str(uuid.uuid5(REPORT_NAMESPACE,
            f"{community_id}\x1f{fingerprint}\x1f{model_name}\x1f{revision or ''}"))
        points.append(models.PointStruct(id=point_id, vector={VECTOR_NAME: vector}, payload={
            "community_id": community_id, "fingerprint": fingerprint,
            "title": report.title, "summary": report.summary,
            "full_report": report.model_dump_json(),
            "source_claim_ids": report.source_claim_ids,
            "valid_from_epoch": epoch, "valid_to_epoch": None,
        }))
    client.upsert(collection_name=collection, points=points, wait=True)
    return {"collection_name": collection, "embedded_count": len(points),
            "upserted_count": len(points)}
